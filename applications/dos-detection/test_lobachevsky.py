"""
Test the Lobachevsky order-3 interpolator against linear and Hermite on the
same synthetic harness as benchmark_final.py (Kuznetsov's target functions).

Lobachevsky3 is implemented and wired into the extended backend
(lut_backend_dense_numpy_rbf supports interp='lobachevsky3', using the same
int8 derivative table as Hermite -> same memory cost, K*L*2 bytes/edge)
but was never benchmarked. This answers that question honestly, whatever
the outcome.

Setup identical to benchmark_final.py: single-layer RBF [1,1], G=20,
h_init=1/(G-1), Adam 500 epochs, 5 seeds, phi_err = MAE(float, LUT).
"""
from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_final import (
    FUNCTIONS_1D, make_data_1d, train_rbf, eval_rbf_lut,
)

N_SEEDS = 5
K_GRID = [4, 8, 16]
L_GRID = [4, 8, 16]
INTERPS = ("linear", "hermite", "lobachevsky3")


def main():
    results = {}

    for fname, f in FUNCTIONS_1D.items():
        for seed in range(N_SEEDS):
            x_tr, y_tr = make_data_1d(f, 500, seed * 100)
            x_te, y_te = make_data_1d(f, 500, seed * 100 + 1)
            model = train_rbf(1, 1, x_tr, y_tr, seed)

            for K, L in product(K_GRID, L_GRID):
                for interp in INTERPS:
                    try:
                        _, _, pe, _ = eval_rbf_lut(model, x_te, y_te, K, L, interp)
                    except Exception as ex:
                        pe = float("nan")
                        print(f"  ERROR {fname} seed={seed} K={K} L={L} "
                              f"{interp}: {ex}")
                    results.setdefault((fname, interp, K, L), []).append(pe)
        print(f"{fname}: done ({N_SEEDS} seeds)")

    print()
    print(f"phi_err mean over {N_SEEDS} seeds "
          f"(mem: linear=K*L, hermite/loba3=K*L*2 bytes/edge)")
    print(f"{'function':>8} {'K':>3} {'L':>3}  "
          f"{'linear':>10} {'hermite':>10} {'loba3':>10}  "
          f"{'loba3 vs herm':>13}")
    print("-" * 65)
    for fname in FUNCTIONS_1D:
        for K in K_GRID:
            for L in L_GRID:
                vals = {}
                for interp in INTERPS:
                    v = results[(fname, interp, K, L)]
                    vals[interp] = float(np.nanmean(v))
                herm, loba = vals["hermite"], vals["lobachevsky3"]
                verdict = ("loba wins" if loba < herm * 0.99
                           else "herm wins" if herm < loba * 0.99
                           else "~equal")
                print(f"{fname:>8} {K:>3} {L:>3}  "
                      f"{vals['linear']:>10.6f} {herm:>10.6f} {loba:>10.6f}  "
                      f"{verdict:>13}")
            print()


if __name__ == "__main__":
    main()
