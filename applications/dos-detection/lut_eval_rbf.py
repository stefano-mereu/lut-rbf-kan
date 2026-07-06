"""
Compile the trained multi-layer RBF-KAN to LUT and evaluate on DoS test set.

For each (K, L, interp) config:
  - compile every RBF layer's edges to a LUT over [0,1]
  - forward pass: for each layer, apply frozen minmax01 (float) then LUT lookup
  - errors propagate through the cascade (this is what we measure)

Metrics per config:
  - F1 (the headline number; may stay ~0.99 due to large decision margin)
  - MAE on logits vs the float model (fine-grained; where Hermite vs linear shows)
  - memory per edge (bytes): K*L for linear, K*L*2 for hermite

Compares RBF+linear vs RBF+Hermite across an L sweep.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

LUT_RBF_KAN_PATH = str(Path(__file__).resolve().parents[2])  # repo root
if LUT_RBF_KAN_PATH not in sys.path:
    sys.path.insert(0, LUT_RBF_KAN_PATH)

from src.models.rbf_adapter import RBFKANSingleLayerAdapter
from src.quant.lut_builder_rbf import build_rbf_lut_for_edges
from src.kernels.lut_backend_dense_numpy_rbf import (
    forward_dense_numpy_rbf, pack_rbf_dense_layer
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rbf_multilayer import RBFKANMultiLayer


def compile_layer_lut(rbf_layer, K, L, interp):
    """Compile one RBFKANLayerTorch to a packed LUT over [0,1]."""
    adapter = RBFKANSingleLayerAdapter.from_trained_layer(rbf_layer)
    edges = adapter.extract_edges()
    knots = np.linspace(0.0, 1.0, K + 1, dtype=np.float32)
    for e in edges:
        object.__setattr__(e, "knots", knots)
    art = build_rbf_lut_for_edges(
        edges=edges, L=L, interp=interp,
        y_range_method="minmax", lower_pct=0.0, upper_pct=100.0,
        dtype="uint8", scheme="asymmetric", qmin=0, qmax=255,
    )
    packed = pack_rbf_dense_layer(
        art, edges=edges, in_dim=adapter.in_dim, out_dim=adapter.out_dim
    )
    return packed


def minmax01_np(x, x_min, x_max):
    span = np.clip(x_max - x_min, 1e-6, None)
    return np.clip((x - x_min) / span, 0.0, 1.0).astype(np.float32)


def forward_lut_multilayer(model, x_np, packed_layers):
    """
    Full multi-layer LUT forward.
    model: RBFKANMultiLayer (for frozen minmax stats)
    packed_layers: list of packed LUTs, one per block
    """
    x = x_np.astype(np.float32)
    for blk, packed in zip(model.blocks, packed_layers):
        x_min = blk.norm.x_min.cpu().numpy()
        x_max = blk.norm.x_max.cpu().numpy()
        x01 = minmax01_np(x, x_min, x_max)
        x = forward_dense_numpy_rbf(x01, packed)
    return x  # logits (N, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dos_data/dataset.pt")
    ap.add_argument("--model", default="dos_data/rbf_model/rbf_kan.pt")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--L_list", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--n_test", type=int, default=20000,
                    help="subsample test for speed (0 = all)")
    args = ap.parse_args()

    # Load model
    ckpt = torch.load(args.model, map_location="cpu")
    model = RBFKANMultiLayer(ckpt["width"], G=ckpt["G"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded RBF-KAN {ckpt['width']} G={ckpt['G']}  "
          f"float F1={ckpt['final_metrics']['f1']:.4f}")

    # Load test data
    d = torch.load(args.data, map_location="cpu")
    x_te = d["test_input"].numpy()
    y_te = d["test_label"].numpy().ravel().astype(int)
    if args.n_test > 0 and args.n_test < len(x_te):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_te), args.n_test, replace=False)
        x_te, y_te = x_te[idx], y_te[idx]
    print(f"Test samples: {len(x_te)}")

    # Float reference (logits + F1)
    with torch.no_grad():
        logits_float = model(torch.from_numpy(x_te)).numpy().ravel()
    f1_float = f1_score(y_te, (logits_float > 0).astype(int))
    print(f"Float F1 (recomputed on this subset): {f1_float:.4f}\n")

    print(f"{'interp':>8} {'L':>3} {'mem/edge':>9} {'F1':>8} "
          f"{'dF1':>9} {'MAE_logit':>10}")
    print("-" * 55)

    for interp in ("linear", "hermite"):
        for L in args.L_list:
            packed_layers = [
                compile_layer_lut(blk.rbf, args.K, L, interp)
                for blk in model.blocks
            ]
            logits_lut = forward_lut_multilayer(model, x_te, packed_layers).ravel()

            preds = (logits_lut > 0).astype(int)
            f1 = f1_score(y_te, preds)
            mae_logit = float(np.mean(np.abs(logits_float - logits_lut)))
            mem = args.K * L * (2 if interp == "hermite" else 1)

            print(f"{interp:>8} {L:>3} {mem:>9} {f1:>8.4f} "
                  f"{f1 - f1_float:>+9.4f} {mae_logit:>10.5f}")
        print()


if __name__ == "__main__":
    main()
