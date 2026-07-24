"""
Final probe: compare the COMPILED tables (values + derivatives) of layer 0
(ToN model, K=8, L=2) against the analytic edge functions, per edge.

Decides among the last suspects:
  - value table wrong somewhere -> linear would suffer too (it doesn't much)
  - derivative table wrong (magnitude or EDGE LAYOUT scrambled) -> explains
    hermite-only collapse scaling with dx
  - both tables correct -> mechanism is in the forward composition
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lut_eval_rbf import compile_layer_lut
from src.models.rbf_adapter import RBFKANSingleLayerAdapter
from rbf_multilayer import RBFKANMultiLayer

MODEL = "/mnt/disk1/stefano/stefano/scripts/dos-rbf/ton_data/rbf_model/rbf_kan.pt"
K, L = 8, 2

ckpt = torch.load(MODEL, map_location="cpu")
model = RBFKANMultiLayer(ckpt["width"], G=ckpt["G"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

blk = model.blocks[0]
adapter = RBFKANSingleLayerAdapter.from_trained_layer(blk.rbf)
edges = adapter.extract_edges()
in_dim, out_dim = adapter.in_dim, adapter.out_dim
print(f"Layer 0: in={in_dim} out={out_dim} edges={len(edges)}  K={K} L={L}")

packed = compile_layer_lut(blk.rbf, K, L, "hermite")
print("Packed attrs:", [a for a in dir(packed) if not a.startswith("_")][:20])

# Sample grid identical to the builder (closed linspace per segment)
segw = 1.0 / K
x_grid = np.empty((K, L), dtype=np.float32)
for k in range(K):
    x_grid[k] = np.linspace(k * segw, (k + 1) * segw, L)

def stored_v(i, j):
    """Dequantized value table for edge (src=i, dst=j): [K, L]."""
    q = packed.q_flat[i][j].reshape(K, L).astype(np.float32)
    y0 = packed.y_min[i][j].astype(np.float32)[:, None]
    sc = packed.scale[i][j].astype(np.float32)[:, None]
    return y0 + sc * q

def stored_d(i, j):
    dq = packed.dq_flat[i][j].reshape(K, L).astype(np.float32)
    ds = packed.dscale[i][j].astype(np.float32)[:, None]
    return ds * dq

# Map edge list to (src, dst)
v_err, d_err, d_scr = [], [], []
for e in edges:
    i, j = e.src_idx, e.dst_idx
    phi, dphi = e.eval_phi_and_deriv(x_grid.ravel())
    phi = phi.reshape(K, L); dphi = dphi.reshape(K, L)
    sv, sd = stored_v(i, j), stored_d(i, j)
    v_err.append(np.max(np.abs(sv - phi)))
    d_err.append(np.max(np.abs(sd - dphi)))
    d_scr.append(np.max(np.abs(dphi)))

v_err = np.array(v_err); d_err = np.array(d_err); d_scr = np.array(d_scr)
print()
print(f"VALUE table  |stored - analytic|  median={np.median(v_err):.5f}  max={v_err.max():.5f}")
print(f"DERIV table  |stored - analytic|  median={np.median(d_err):.5f}  max={d_err.max():.5f}")
print(f"(analytic |dphi| median of per-edge max: {np.median(d_scr):.3f})")
print()
bad = (d_err > 0.5 * np.maximum(d_scr, 1e-6)).sum()
print(f"Edges where deriv-table error > 50% of the edge's own derivative scale: {bad}/{len(edges)}")
if bad > len(edges) // 4:
    print("=> DERIVATIVE TABLE IS WRONG (layout scrambled or mis-scaled)")
elif np.median(d_err) < 0.1 and np.median(v_err) < 0.01:
    print("=> Both tables CORRECT: mechanism is in the forward composition")
