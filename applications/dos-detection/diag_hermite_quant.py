"""
Diagnostic: is the non-monotonic Hermite MAE_logit (worse at L=8,16 than L=4)
caused by int8 quantization of the derivative table, or a bug?

Test: on the FIRST RBF layer of the trained model, reconstruct phi via Hermite
with (a) int8-quantized derivatives (what the builder does) and (b) exact float32
derivatives. Sweep L. If the non-monotonicity appears only with int8, it's the
quantization floor, not a bug.
"""
import sys
from pathlib import Path

import numpy as np
import torch

LUT_RBF_KAN_PATH = str(Path(__file__).resolve().parents[2])  # repo root
if LUT_RBF_KAN_PATH not in sys.path:
    sys.path.insert(0, LUT_RBF_KAN_PATH)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.rbf_adapter import RBFKANSingleLayerAdapter
from src.kernels.lut_interp_advanced import (
    hermite_interp, quantize_deriv_table, dequant_deriv
)
from rbf_multilayer import RBFKANMultiLayer

K = 8
L_LIST = [2, 4, 8, 16, 32, 64]

ckpt = torch.load("dos_data/rbf_model/rbf_kan.pt", map_location="cpu")
model = RBFKANMultiLayer(ckpt["width"], G=ckpt["G"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Take the first RBF layer, pick one representative edge
adapter = RBFKANSingleLayerAdapter.from_trained_layer(model.blocks[0].rbf)
edges = adapter.extract_edges()

# Evaluation points on [0,1]
rng = np.random.default_rng(0)
x_eval = rng.uniform(0.001, 0.999, 5000).astype(np.float32)

def reconstruct(edge, K, L, x_eval, quantize):
    """Hermite reconstruction of one edge over K segments, L samples each."""
    seg_w = 1.0 / K
    # sample grid: K segments x L points
    x_grid = np.zeros(K * L, dtype=np.float32)
    for s in range(K):
        x_grid[s*L:(s+1)*L] = np.linspace(s*seg_w, (s+1)*seg_w, L)
    phi_g, dphi_g = edge.eval_phi_and_deriv(x_grid)

    if quantize:
        # int8 symmetric quantization per segment (as builder does)
        dphi_seg = dphi_g.reshape(1, K, L)
        dq, dscale, _ = quantize_deriv_table(dphi_seg)
        dphi_g = dequant_deriv(dq, dscale[:, :, None]).reshape(-1)

    dx = seg_w / max(L - 1, 1)
    y = np.zeros(len(x_eval), dtype=np.float32)
    for s in range(K):
        x0 = s * seg_w
        m = (x_eval >= x0) & (x_eval <= x0 + seg_w)
        if not m.any():
            continue
        xs = x_eval[m]
        pos = (xs - x0) / seg_w * (L - 1)
        r0 = np.clip(np.floor(pos).astype(int), 0, L - 2)
        tl = (pos - r0).astype(np.float32)
        base = s * L
        i0, i1 = base + r0, base + r0 + 1
        y[m] = hermite_interp(tl, phi_g[i0], phi_g[i1],
                              dphi_g[i0], dphi_g[i1], dx)
    return y

edge = edges[0]
phi_true = edge.eval_phi(x_eval)

print(f"Edge 0 of layer 0, K={K}")
print(f"{'L':>4} {'MAE_int8':>12} {'MAE_float':>12} {'int8/float':>11}")
print("-" * 42)
for L in L_LIST:
    y_int8 = reconstruct(edge, K, L, x_eval, quantize=True)
    y_flt  = reconstruct(edge, K, L, x_eval, quantize=False)
    mae_int8 = float(np.mean(np.abs(phi_true - y_int8)))
    mae_flt  = float(np.mean(np.abs(phi_true - y_flt)))
    ratio = mae_int8 / max(mae_flt, 1e-12)
    print(f"{L:>4} {mae_int8:>12.6f} {mae_flt:>12.6f} {ratio:>10.1f}x")

print()
print("If MAE_float is monotonic (keeps decreasing) but MAE_int8 flattens/rises")
print("at large L, the non-monotonicity is the int8 derivative quantization floor.")
