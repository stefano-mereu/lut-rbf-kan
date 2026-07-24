"""
Bisect the RBF+hermite L=2 collapse on ToN_IoT (F1 0.31 vs linear 0.91).

Facts established so far:
  - Float-math hermite at K=8, L=2 is FINE (better than linear) on these edges
  - Builder/backend sampling conventions are consistent (closed linspace,
    dx = segw/(L-1))
  => the failure lives in the compiled pipeline. Localize it empirically.

Axes:
  A) Mixed cascade: compile ONLY layer j (others float) x {linear, hermite}
     -> which layer, and value-path vs derivative-path
  B) Full cascade at K=16, L=2 -> does finer segmentation recover?
  C) Full cascade at K=8, L=3 -> is the cliff exactly at L=2?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lut_eval_rbf import compile_layer_lut, minmax01_np
from src.kernels.lut_backend_dense_numpy_rbf import forward_dense_numpy_rbf
from rbf_multilayer import RBFKANMultiLayer

DATA = "/mnt/disk1/stefano/stefano/scripts/dos-rbf/ton_data/dataset.pt"
MODEL = "/mnt/disk1/stefano/stefano/scripts/dos-rbf/ton_data/rbf_model/rbf_kan.pt"

ckpt = torch.load(MODEL, map_location="cpu")
model = RBFKANMultiLayer(ckpt["width"], G=ckpt["G"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

d = torch.load(DATA, map_location="cpu")
x_te = d["test_input"].numpy()
y_te = d["test_label"].numpy().ravel().astype(int)

with torch.no_grad():
    logits_float = model(torch.from_numpy(x_te)).numpy().ravel()
f1_float = f1_score(y_te, (logits_float > 0).astype(int))
print(f"Float F1: {f1_float:.4f}   n={len(x_te)}")
print(f"h per layer: {[f'{b.rbf.get_h():.3f}' for b in model.blocks]}")
print()


def forward_mixed(x_np, lut_layer_idx, packed):
    """Cascade with only layer lut_layer_idx compiled; others float torch."""
    x = x_np.astype(np.float32)
    for j, blk in enumerate(model.blocks):
        if j == lut_layer_idx:
            x_min = blk.norm.x_min.cpu().numpy()
            x_max = blk.norm.x_max.cpu().numpy()
            x01 = minmax01_np(x, x_min, x_max)
            x = forward_dense_numpy_rbf(x01, packed)
        else:
            with torch.no_grad():
                x = blk(torch.from_numpy(x)).numpy()
    return x.ravel()


def forward_full(x_np, packed_layers):
    x = x_np.astype(np.float32)
    for blk, packed in zip(model.blocks, packed_layers):
        x_min = blk.norm.x_min.cpu().numpy()
        x_max = blk.norm.x_max.cpu().numpy()
        x = forward_dense_numpy_rbf(minmax01_np(x, x_min, x_max), packed)
    return x.ravel()


print("=== A) Single-layer compilation, K=8, L=2 ===")
print(f"{'layer':>5} {'interp':>8} {'F1':>8} {'MAE_logit':>10}")
for j in range(len(model.blocks)):
    for interp in ("linear", "hermite"):
        packed = compile_layer_lut(model.blocks[j].rbf, 8, 2, interp)
        lg = forward_mixed(x_te, j, packed)
        f1 = f1_score(y_te, (lg > 0).astype(int))
        mae = float(np.mean(np.abs(logits_float - lg)))
        print(f"{j:>5} {interp:>8} {f1:>8.4f} {mae:>10.5f}")
print()

print("=== B) Full cascade, K=16, L=2 ===")
for interp in ("linear", "hermite"):
    packed = [compile_layer_lut(b.rbf, 16, 2, interp) for b in model.blocks]
    lg = forward_full(x_te, packed)
    f1 = f1_score(y_te, (lg > 0).astype(int))
    mae = float(np.mean(np.abs(logits_float - lg)))
    print(f"  {interp:>8}: F1={f1:.4f}  MAE={mae:.5f}")
print()

print("=== C) Full cascade, K=8, L=3 ===")
for interp in ("linear", "hermite"):
    packed = [compile_layer_lut(b.rbf, 8, 3, interp) for b in model.blocks]
    lg = forward_full(x_te, packed)
    f1 = f1_score(y_te, (lg > 0).astype(int))
    mae = float(np.mean(np.abs(logits_float - lg)))
    print(f"  {interp:>8}: F1={f1:.4f}  MAE={mae:.5f}")
