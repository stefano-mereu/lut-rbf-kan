"""
B-spline (PyKAN) multi-layer LUT eval on DoS — linear AND Hermite.

Completes the 2x2 table (basis x interp). The previous lut_eval_bspline.py
silently fell back to linear for hermite because PyKAN EdgeSpec lacks
eval_phi_and_deriv. Here we wrap each PyKAN edge into an RBFEdgeSpec whose
derivative is computed by central finite differences on the full edge
function (sb*silu + ss*spline). The B-spline edge is C2, so FD is accurate
everywhere except exactly at knots (measure zero); this is numerically
equivalent to the analytic Cox-de Boor derivative for LUT compilation.

Both interps go through the SAME extended pipeline
(build_rbf_lut_for_edges + forward_dense_numpy_rbf), so the linear rows
double as a cross-check against the classic-backend numbers.

Knots are widened to each layer's real input domain (from model.acts),
reproducing PyKAN's extrapolation instead of clipping.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from kan import KAN

LUT_KAN_PATH = str(Path(__file__).resolve().parents[2])  # repo root
if LUT_KAN_PATH not in sys.path:
    sys.path.insert(0, LUT_KAN_PATH)

from src.models.kan_wrapper import PyKANSingleLayerAdapter
from src.models.rbf_adapter import RBFEdgeSpec
from src.quant.lut_builder_rbf import build_rbf_lut_for_edges
from src.kernels.lut_backend_dense_numpy_rbf import (
    forward_dense_numpy_rbf, pack_rbf_dense_layer
)

FD_EPS = 1e-3  # central finite-difference step for edge derivative



def packed_totals(packed_list):
    """Measured bytes and edge count over a list of packed layers."""
    import numpy as _np
    tot, edges = 0, 0
    for pk in packed_list:
        qf = pk.q_flat
        try:
            e = int(qf.shape[0]) * int(qf.shape[1])
        except AttributeError:
            e = len(qf) * int(qf[0].shape[0])
        edges += e
        ab = getattr(pk, "art_bytes", None)
        if ab is not None:
            tot += int(ab)
            continue
        for name in ("knots", "q_flat", "scale", "y_min",
                     "dq_flat", "dscale", "dy_min"):
            a = getattr(pk, name, None)
            if a is None:
                continue
            if isinstance(a, _np.ndarray):
                tot += int(a.nbytes)
            elif isinstance(a, (list, tuple)):
                for x in a:
                    if x is not None:
                        tot += int(x.nbytes)
    return tot, edges

def wrap_edges_with_deriv(edges, knots):
    """
    Wrap PyKAN EdgeSpec list into RBFEdgeSpec list with FD derivative.
    knots: widened knot vector (same for all edges of the layer).
    """
    wrapped = []
    dummy = np.zeros(1, dtype=np.float32)
    for e in edges:
        phi_fn = e.eval_phi  # full edge: sb*silu + ss*spline (PyKAN adapter)

        def _eval_phi(x, f=phi_fn):
            return f(np.asarray(x, dtype=np.float32))

        def _eval_phi_and_deriv(x, f=phi_fn):
            x = np.asarray(x, dtype=np.float32)
            phi = f(x)
            dphi = (f(x + FD_EPS) - f(x - FD_EPS)) / (2.0 * FD_EPS)
            return phi.astype(np.float32), dphi.astype(np.float32)

        wrapped.append(RBFEdgeSpec(
            edge_id=e.edge_id,
            src_idx=e.src_idx,
            dst_idx=e.dst_idx,
            knots=knots,
            domain=(float(knots[0]), float(knots[-1])),
            eval_phi=_eval_phi,
            eval_phi_and_deriv=_eval_phi_and_deriv,
            centers=dummy, sigmas=dummy, coef=dummy, h=0.0,
        ))
    return wrapped


def compile_layer(model, layer_idx, x_min, x_max, K, L, interp):
    adapter = PyKANSingleLayerAdapter(model, layer_idx=layer_idx)
    edges = adapter.extract_edges()
    knots = np.linspace(x_min, x_max, K + 1, dtype=np.float32)
    wrapped = wrap_edges_with_deriv(edges, knots)

    art = build_rbf_lut_for_edges(
        edges=wrapped, L=L, interp=interp,
        y_range_method="minmax", lower_pct=0.0, upper_pct=100.0,
        dtype="uint8", scheme="asymmetric", qmin=0, qmax=255,
        meta_dtype="float16",
    )
    packed = pack_rbf_dense_layer(
        art, edges=wrapped, in_dim=adapter.in_dim, out_dim=adapter.out_dim
    )
    from src.quant.lut_builder_rbf import rbf_artifact_memory_bytes
    try:
        packed.art_bytes = int(rbf_artifact_memory_bytes(art))
    except Exception:
        object.__setattr__(packed, "art_bytes", int(rbf_artifact_memory_bytes(art)))
    return packed


def get_layer_domains(model, x):
    with torch.no_grad():
        _ = model(x)
    domains = []
    for i in range(len(model.acts) - 1):
        a = model.acts[i].detach().cpu().numpy()
        lo, hi = float(a.min()), float(a.max())
        span = hi - lo
        domains.append((lo - 0.02 * span, hi + 0.02 * span))
    return domains


def forward_cascade(x_np, packed_layers):
    x = x_np.astype(np.float32)
    for packed in packed_layers:
        x = forward_dense_numpy_rbf(x, packed)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dos_data/dataset.pt")
    ap.add_argument("--model", default="dos_data/bspline_model/bspline_kan.pt")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--L_list", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--check_L", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    model = KAN(width=ckpt["width"], grid=ckpt["grid"], k=ckpt["k"], seed=42)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded B-spline PyKAN float F1={ckpt['final_metrics']['f1']:.4f}")

    d = torch.load(args.data, map_location="cpu")
    x_te = d["test_input"].numpy()
    y_te = d["test_label"].numpy().ravel().astype(int)
    if 0 < args.n_test < len(x_te):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_te), args.n_test, replace=False)
        x_te, y_te = x_te[idx], y_te[idx]
    print(f"Test samples: {len(x_te)}")

    x_cal = d["train_input"].numpy()
    if len(x_cal) > 20000:
        rng_cal = np.random.default_rng(0)
        x_cal = x_cal[rng_cal.choice(len(x_cal), 20000, replace=False)]
    domains = get_layer_domains(model, torch.from_numpy(x_cal))
    print(f"LUT domains calibrated on {len(x_cal)} TRAIN samples (frozen)")
    for i, (lo, hi) in enumerate(domains):
        print(f"  layer {i} domain: [{lo:.2f}, {hi:.2f}]")

    with torch.no_grad():
        logits_float = model(torch.from_numpy(x_te)).numpy().ravel()
    f1_float = f1_score(y_te, (logits_float > 0).astype(int))
    print(f"Float F1 (this subset): {f1_float:.4f}\n")

    n_layers = len(domains)

    print(f"=== Validation at L={args.check_L} ===")
    for interp in ("linear", "hermite"):
        packed = [compile_layer(model, li, *domains[li], args.K,
                                args.check_L, interp) for li in range(n_layers)]
        logits = forward_cascade(x_te, packed).ravel()
        mae = float(np.mean(np.abs(logits_float - logits)))
        f1 = f1_score(y_te, (logits > 0).astype(int))
        print(f"  {interp:>8}: MAE_logit={mae:.5f}  F1={f1:.4f}")
    print()

    print(f"{'interp':>8} {'L':>3} {'mem/edge':>9} {'F1':>8} "
          f"{'dF1':>9} {'MAE_logit':>10}")
    print("-" * 55)
    for interp in ("linear", "hermite"):
        for L in args.L_list:
            packed = [compile_layer(model, li, *domains[li], args.K, L, interp)
                      for li in range(n_layers)]
            logits = forward_cascade(x_te, packed).ravel()
            f1 = f1_score(y_te, (logits > 0).astype(int))
            mae = float(np.mean(np.abs(logits_float - logits)))
            mem_tot, n_edges = packed_totals(packed)
            mem = mem_tot // max(n_edges, 1)
            print(f"{interp:>8} {L:>3} {mem:>9} {f1:>8.4f} "
                  f"{f1 - f1_float:>+9.4f} {mae:>10.5f}")
        print()


if __name__ == "__main__":
    main()
