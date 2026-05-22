# scripts/benchmark_final.py
"""
Benchmark: B-spline+linear LUT vs RBF+linear LUT vs RBF+Hermite LUT
on standard KAN target functions.

Methodology (Kuznetsov 2026, extended):
  - B-spline: BSplineKANSingleLayerAdapter trained via lstsq (exact solution)
  - RBF:      RBFKANLayerTorch trained via Adam (500 epochs)
  - LUT:      compiled with sweep K x L x interp
  - Metrics:  float_err, task_err, phi_err, task_degrad% over N_SEEDS seeds

Architecture: single layer [in, out] — Kuznetsov-style.
  1D functions: [1,1], 2D function: [2,1]

h_init: midpoint of conditioning interval h = (x_max-x_min)/(G-1)
  From Noorizadegan & Wang (2026): epsilon in [h/2, 3h/2].

B-spline training: lstsq on design matrix (exact minimum-norm solution).
  base_kind='none', sb=0 (pure spline, no SiLU residual).

Target functions (Liu et al. 2024 benchmark):
  sin2pi:   sin(2*pi*x)          smooth
  cusp:     |x - 0.5|            cusp — derivative discontinuous at x=0.5
  tanh5:    tanh(5*(x-0.5))      saturating
  sincos2d: sin(pi*x)*cos(pi*y)  smooth 2D
"""
from __future__ import annotations

import sys
import csv
from pathlib import Path
from itertools import product

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.rbf_adapter import RBFKANLayerTorch, RBFKANSingleLayerAdapter
from src.models.bspline_adapter import BSplineKANSingleLayerAdapter, _bspline_eval
from src.quant.lut_builder import build_lut_for_edges
from src.quant.lut_builder_rbf import build_rbf_lut_for_edges
from src.kernels.lut_contract import pack_dense_layer
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy
from src.kernels.lut_backend_dense_numpy_rbf import (
    forward_dense_numpy_rbf, pack_rbf_dense_layer
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SEEDS  = 5
N_TRAIN  = 500
N_TEST   = 500
N_EPOCHS = 500
LR       = 0.01
G_RBF    = 20
DOMAIN   = (0.0, 1.0)
K_GRID   = [4, 8, 16]
L_GRID   = [4, 8, 16]
OUT_DIR  = Path("outputs/benchmark_final")

# B-spline config (matches Kuznetsov default)
BSP_GRID_POINTS  = 5   # interior knots
BSP_DEGREE       = 3   # cubic


# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------
def f_sin2pi(x):   return np.sin(2 * np.pi * x).astype(np.float32)
def f_cusp(x):     return np.abs(x - 0.5).astype(np.float32)
def f_tanh5(x):    return np.tanh(5 * (x - 0.5)).astype(np.float32)
def f_sincos2d(x): return (np.sin(np.pi*x[:,0]) * np.cos(np.pi*x[:,1])).astype(np.float32)

FUNCTIONS_1D = {"sin2pi": f_sin2pi, "cusp": f_cusp, "tanh5": f_tanh5}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_data_1d(f, N, seed):
    rng = np.random.default_rng(seed)
    # Clip slightly inside domain to avoid B-spline boundary issues
    x = rng.uniform(0.001, 0.999, size=(N, 1)).astype(np.float32)
    y = f(x[:, 0]).reshape(-1, 1)
    return x, y

def make_data_2d(N, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.001, 0.999, size=(N, 2)).astype(np.float32)
    y = f_sincos2d(x).reshape(-1, 1)
    return x, y


# ---------------------------------------------------------------------------
# B-spline training via lstsq
# ---------------------------------------------------------------------------
def train_bspline(in_dim, out_dim, x_train, y_train):
    """
    Fit BSplineKANSingleLayerAdapter coefficients via least squares.
    Pure spline (sb=0, no SiLU residual).
    Returns trained adapter.
    """
    adapter = BSplineKANSingleLayerAdapter.from_arch({
        'in_dim': in_dim, 'out_dim': out_dim,
        'degree': BSP_DEGREE, 'grid_points': BSP_GRID_POINTS,
        'x_min': 0.0, 'x_max': 1.0, 'base_kind': 'none',
    })
    adapter.sb[:] = 0.0  # disable base branch

    num_coef = adapter.coef.shape[2]

    for i in range(in_dim):
        xi = x_train[:, i]
        # Build design matrix: Phi[n,k] = B_k(xi[n])
        Phi = np.zeros((len(xi), num_coef), dtype=np.float32)
        for k in range(num_coef):
            ek = np.zeros(num_coef, dtype=np.float32)
            ek[k] = 1.0
            Phi[:, k] = _bspline_eval(xi, ek, adapter.knots_aug, adapter.degree)

        for j in range(out_dim):
            yj = y_train[:, j] / out_dim  # distribute output equally
            coef_opt, _, _, _ = np.linalg.lstsq(Phi, yj, rcond=None)
            adapter.coef[i, j, :] = coef_opt.astype(np.float32)

    return adapter


# ---------------------------------------------------------------------------
# RBF training via Adam
# ---------------------------------------------------------------------------
class RBFModel(nn.Module):
    def __init__(self, in_dim, out_dim, G, h_init):
        super().__init__()
        self.layer = RBFKANLayerTorch(
            in_dim, out_dim, G=G, h_init=h_init, h_learnable=True)

    def forward(self, x):
        return self.layer(x)

    def freeze_h(self):
        self.layer.freeze_h()

    def get_h(self):
        return self.layer.get_h()


def train_rbf(in_dim, out_dim, x_train, y_train, seed):
    G = G_RBF
    h_init = 1.0 / (G - 1)
    torch.manual_seed(seed)
    model = RBFModel(in_dim, out_dim, G=G, h_init=h_init)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train)
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        ((model(x_t) - y_t)**2).mean().backward()
        opt.step()
    model.eval()
    model.freeze_h()
    return model


# ---------------------------------------------------------------------------
# LUT evaluation
# ---------------------------------------------------------------------------
def eval_bspline_lut(adapter, x_te, y_te, K, L):
    """Returns (float_err, task_err, phi_err, mem)."""
    y_float = adapter.forward_float(x_te)
    float_err = float(np.mean(np.abs(y_te - y_float)))

    edges = adapter.extract_edges()
    knots = np.linspace(0.0, 1.0, K+1, dtype=np.float32)
    for e in edges:
        object.__setattr__(e, 'knots', knots)

    art = build_lut_for_edges(
        edges=edges, L=L, interp="linear",
        y_range_method="minmax", lower_pct=0.0, upper_pct=100.0,
        dtype="uint8", scheme="asymmetric", qmin=0, qmax=255,
        value_representation="phi",
    )
    packed = pack_dense_layer(
        art, edges=edges,
        in_dim=adapter.in_dim, out_dim=adapter.out_dim,
    )
    y_lut = forward_dense_numpy(x_te, packed)

    task_err = float(np.mean(np.abs(y_te - y_lut)))
    phi_err  = float(np.mean(np.abs(y_float - y_lut)))
    mem = K * L * 1
    return float_err, task_err, phi_err, mem


def eval_rbf_lut(rbf_model, x_te, y_te, K, L, interp):
    """Returns (float_err, task_err, phi_err, mem)."""
    with torch.no_grad():
        y_float = rbf_model(torch.from_numpy(x_te)).numpy()
    float_err = float(np.mean(np.abs(y_te - y_float)))

    layer   = rbf_model.layer
    adapter = RBFKANSingleLayerAdapter.from_trained_layer(layer)
    edges   = adapter.extract_edges()
    knots   = np.linspace(0.0, 1.0, K+1, dtype=np.float32)
    for e in edges:
        object.__setattr__(e, 'knots', knots)

    art = build_rbf_lut_for_edges(
        edges=edges, L=L, interp=interp,
        y_range_method="minmax", lower_pct=0.0, upper_pct=100.0,
        dtype="uint8", scheme="asymmetric", qmin=0, qmax=255,
    )
    packed = pack_rbf_dense_layer(
        art, edges=edges,
        in_dim=adapter.in_dim, out_dim=adapter.out_dim,
    )
    y_lut = forward_dense_numpy_rbf(x_te, packed)

    task_err = float(np.mean(np.abs(y_te - y_lut)))
    phi_err  = float(np.mean(np.abs(y_float - y_lut)))
    mem = K * L * (2 if interp == "hermite" else 1)
    return float_err, task_err, phi_err, mem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_benchmark():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    all_tasks = list(FUNCTIONS_1D.items()) + [("sincos2d", None)]

    for fname, f in all_tasks:
        is_2d = (fname == "sincos2d")
        in_dim = 2 if is_2d else 1

        print(f"\n{'='*60}\nFunction: {fname}  in_dim={in_dim}\n{'='*60}")

        for seed in range(N_SEEDS):
            if is_2d:
                x_tr, y_tr = make_data_2d(N_TRAIN, seed*100)
                x_te, y_te = make_data_2d(N_TEST,  seed*100+1)
            else:
                x_tr, y_tr = make_data_1d(f, N_TRAIN, seed*100)
                x_te, y_te = make_data_1d(f, N_TEST,  seed*100+1)

            # Train models
            bsp_adapter = train_bspline(in_dim, 1, x_tr, y_tr)
            rbf_model   = train_rbf(in_dim, 1, x_tr, y_tr, seed)

            # Float errors
            fe_bsp = float(np.mean(np.abs(y_te - bsp_adapter.forward_float(x_te))))
            with torch.no_grad():
                fe_rbf = float(np.mean(np.abs(y_te - rbf_model(torch.from_numpy(x_te)).numpy())))

            print(f"  seed={seed}  float_err: bspline={fe_bsp:.5f}  rbf={fe_rbf:.5f}")

            for K, L in product(K_GRID, L_GRID):
                # B-spline + linear
                fe, te, pe, mem = eval_bspline_lut(bsp_adapter, x_te, y_te, K, L)
                results.append(dict(
                    function=fname, model="bspline", interp="linear",
                    seed=seed, K=K, L=L, mem=mem,
                    float_err=fe, task_err=te, phi_err=pe,
                    task_degrad=(te-fe)/max(fe,1e-10),
                ))

                # RBF + linear
                fe, te, pe, mem = eval_rbf_lut(rbf_model, x_te, y_te, K, L, "linear")
                results.append(dict(
                    function=fname, model="rbf", interp="linear",
                    seed=seed, K=K, L=L, mem=mem,
                    float_err=fe, task_err=te, phi_err=pe,
                    task_degrad=(te-fe)/max(fe,1e-10),
                ))

                # RBF + hermite
                fe, te, pe, mem = eval_rbf_lut(rbf_model, x_te, y_te, K, L, "hermite")
                results.append(dict(
                    function=fname, model="rbf", interp="hermite",
                    seed=seed, K=K, L=L, mem=mem,
                    float_err=fe, task_err=te, phi_err=pe,
                    task_degrad=(te-fe)/max(fe,1e-10),
                ))

    # Save
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")
    _print_summary(results)


def _print_summary(results):
    import collections
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        key = (r["function"], r["model"], r["interp"], r["K"], r["L"])
        for m in ("float_err","task_err","phi_err","task_degrad"):
            groups[key][m].append(r[m])

    # Table 1: task_err at K=8 L=8
    print(f"\n{'='*80}")
    print("Table 1: task_err mean+-std (K=8, L=8) — model quality after LUT")
    print(f"{'='*80}")
    print(f"{'function':>10} {'model':>8} {'interp':>8}  "
          f"{'float_err':>10}  {'task_err':>10}  {'degrad%':>8}")
    print("-"*60)
    for fname in ["sin2pi","cusp","tanh5","sincos2d"]:
        for model, interp in [("bspline","linear"),("rbf","linear"),("rbf","hermite")]:
            key = (fname, model, interp, 8, 8)
            if key not in groups: continue
            fe = np.mean(groups[key]["float_err"])
            te = np.mean(groups[key]["task_err"])
            td = np.mean(groups[key]["task_degrad"])*100
            print(f"{fname:>10} {model:>8} {interp:>8}  "
                  f"{fe:>10.5f}  {te:>10.5f}  {td:>7.1f}%")
        print()

    # Table 2: phi_err — Hermite vs linear on RBF
    print(f"\n{'='*70}")
    print("Table 2: phi_err — Hermite vs linear (RBF only)")
    print(f"{'='*70}")
    print(f"{'function':>10} {'K':>3} {'L':>3}  "
          f"{'linear':>10}  {'hermite':>10}  {'improvement':>12}")
    print("-"*55)
    for fname in ["sin2pi","cusp","tanh5"]:
        for K, L in [(4,4),(8,4),(8,8),(16,4)]:
            kl = (fname,"rbf","linear",K,L)
            kh = (fname,"rbf","hermite",K,L)
            if kl not in groups or kh not in groups: continue
            pl = np.mean(groups[kl]["phi_err"])
            ph = np.mean(groups[kh]["phi_err"])
            impr = (1 - ph/pl)*100
            print(f"{fname:>10} {K:>3} {L:>3}  {pl:>10.6f}  {ph:>10.6f}  {impr:>11.1f}%")
        print()


if __name__ == "__main__":
    run_benchmark()
