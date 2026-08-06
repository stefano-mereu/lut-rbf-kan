# src/kernels/lut_backend_dense_numpy_rbf.py
"""
Extended dense NumPy backend supporting Hermite and Lobachevsky3 interpolation.

Drop-in alongside lut_backend_dense_numpy.py.
Handles interp modes: "linear", "nearest", "hermite", "lobachevsky3".

For "hermite" and "lobachevsky3":
  - Requires a PackedRBFLUT (extends PackedLUT with dq_flat, dscale, dy_min)
  - Falls back gracefully to "linear" if derivative table is absent

Usage:
    from src.kernels.lut_backend_dense_numpy_rbf import forward_dense_numpy_rbf
    y = forward_dense_numpy_rbf(x, packed)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.kernels.lut_contract import PackedLUT
from src.kernels.lut_math import (
    clip_for_indexing,
    in_domain_mask,
    lut_interp_indices,
    segment_params_nonuniform,
    segment_params_uniform,
    dequant,
)
from src.kernels.lut_interp_advanced import hermite_interp, lobachevsky3_interp


# ---------------------------------------------------------------------------
# Extended packed structure (adds derivative arrays)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackedRBFLUT(PackedLUT):
    """
    Extends PackedLUT with derivative quantization tables.

    Additional arrays:
      dq_flat:  [in_dim, out_dim, K*L]  int8 — quantized phi'(x)
      dscale:   [in_dim, out_dim, K]    float32
      dy_min:   [in_dim, out_dim, K]    float32 (all zeros, symmetric)
    """
    dq_flat: Optional[np.ndarray] = None   # int8 [in_dim, out_dim, K*L]
    dscale: Optional[np.ndarray] = None    # float32 [in_dim, out_dim, K]
    dy_min: Optional[np.ndarray] = None    # float32 [in_dim, out_dim, K]

    def has_derivatives(self) -> bool:
        return self.dq_flat is not None


def pack_rbf_dense_layer(
    art,           # RBFLUTArtifact
    *,
    edges,
    in_dim: int,
    out_dim: int,
    boundary_mode: str = "half_open",
) -> PackedRBFLUT:
    """
    Pack an RBFLUTArtifact into PackedRBFLUT for inference.
    Extends pack_dense_layer() with derivative table packing.
    """
    from src.kernels.lut_contract import pack_dense_layer, _edge_matrix

    # Pack base (value table) using existing infrastructure
    base_packed = pack_dense_layer(
        art, edges=edges, in_dim=in_dim, out_dim=out_dim,
        boundary_mode=boundary_mode,
    )

    # Pack derivative tables (if present)
    dq_flat = None
    dscale_dense = None
    dy_min_dense = None

    if getattr(art, "dq_table", None) is not None:
        edge_ids = _edge_matrix(edges, in_dim=in_dim, out_dim=out_dim)
        K = base_packed.K
        L = base_packed.L
        E = art.dq_table.shape[0]

        dq_dense = np.empty((in_dim, out_dim, K, L), dtype=np.int8)
        ds_dense = np.empty((in_dim, out_dim, K), dtype=np.float32)

        dscale_art = np.asarray(art.dscale, dtype=np.float32)
        dq_art = np.asarray(art.dq_table)

        for i in range(in_dim):
            for j in range(out_dim):
                eid = int(edge_ids[i, j])
                dq_dense[i, j, :, :] = dq_art[eid, :, :]
                ds_dense[i, j, :] = dscale_art[eid, :]

        dq_flat = dq_dense.reshape(in_dim, out_dim, K * L)
        dscale_dense = ds_dense
        dy_min_dense = None

    # Build PackedRBFLUT by copying all fields from base + adding deriv arrays
    return PackedRBFLUT(
        q_flat=base_packed.q_flat,
        scale=base_packed.scale,
        y_min=base_packed.y_min,
        knots=base_packed.knots,
        L=base_packed.L,
        K=base_packed.K,
        interp=base_packed.interp,
        q_dtype=base_packed.q_dtype,
        base_kind=base_packed.base_kind,
        coef_base=base_packed.coef_base,
        coef_lut=base_packed.coef_lut,
        coef_out=base_packed.coef_out,
        value_representation=base_packed.value_representation,
        oob_behavior=base_packed.oob_behavior,
        boundary_mode=base_packed.boundary_mode,
        x_min=base_packed.x_min,
        x_max=base_packed.x_max,
        uniform_dx=base_packed.uniform_dx,
        # derivative arrays
        dq_flat=dq_flat,
        dscale=dscale_dense,
        dy_min=dy_min_dense,
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_dense_numpy_rbf(x: np.ndarray, packed: PackedLUT) -> np.ndarray:
    """
    Dense NumPy forward supporting linear, nearest, hermite, lobachevsky3.

    For hermite/lobachevsky3:
      - Uses derivative tables from packed (PackedRBFLUT)
      - Falls back to linear if derivative table absent

    Args:
        x:      (N, in_dim) float32
        packed: PackedLUT or PackedRBFLUT

    Returns:
        y: (N, out_dim) float32
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [N, in_dim], got {x.shape}")

    N, in_dim = x.shape
    in_dim_p, out_dim, KL = packed.q_flat.shape
    if in_dim != in_dim_p:
        raise ValueError(f"in_dim mismatch: x={in_dim}, packed={in_dim_p}")

    y = np.zeros((N, out_dim), dtype=np.float32)
    interp = (packed.interp or "linear").lower().strip()

    # Check if we have derivative tables for advanced interpolation
    use_deriv = (
        interp in ("hermite", "lobachevsky3")
        and isinstance(packed, PackedRBFLUT)
        and packed.has_derivatives()
    )
    if interp in ("hermite", "lobachevsky3") and not use_deriv:
        # Graceful fallback
        interp = "linear"

    for i in range(in_dim):
        xi = x[:, i]
        xi_clip = clip_for_indexing(xi, packed.x_min, packed.x_max, packed.boundary_mode)

        if packed.uniform_dx is not None:
            k, u = segment_params_uniform(xi_clip, packed.x_min, packed.uniform_dx, packed.K)
            # dx for Hermite = spacing between adjacent LUT entries within a segment.
            # The LUT is sampled on a linspace of L points over [x_k, x_{k+1}],
            # so the spacing is segment_width / (L-1).
            dx = float(packed.uniform_dx) / float(packed.L - 1)
        else:
            k, u = segment_params_nonuniform(xi_clip, packed.knots)
            dx_arr = ((packed.knots[k + 1] - packed.knots[k]) /
                      float(packed.L - 1)).astype(np.float32)
            dx = None   # handled per-sample below for non-uniform

        r0, r1, w = lut_interp_indices(u, packed.L)
        idx0 = (k * packed.L + r0).astype(np.int32)
        idx1 = (k * packed.L + r1).astype(np.int32)

        # Value table for all j: [out_dim, KL]
        qij = packed.q_flat[i]        # [out_dim, KL]
        q0 = qij[:, idx0]             # [out_dim, N]
        q1 = qij[:, idx1]

        y_min_seg = packed.y_min[i][:, k]    # [out_dim, N]
        scale_seg = packed.scale[i][:, k]

        v0 = y_min_seg + scale_seg * q0.astype(np.float32)
        v1 = y_min_seg + scale_seg * q1.astype(np.float32)

        if interp == "nearest":
            lut_val = np.where(w[None, :] < 0.5, v0, v1)

        elif interp == "linear":
            w2 = w.astype(np.float32)[None, :]
            lut_val = (1.0 - w2) * v0 + w2 * v1

        elif interp in ("hermite", "lobachevsky3"):
            # Derivative table: [out_dim, KL]
            dqij = packed.dq_flat[i]
            dq0 = dqij[:, idx0]
            dq1 = dqij[:, idx1]

            dscale_seg = packed.dscale[i][:, k]   # [out_dim, N]
            # symmetric: dy_min = 0
            d0 = dscale_seg * dq0.astype(np.float32)
            d1 = dscale_seg * dq1.astype(np.float32)

            t = w.astype(np.float32)              # [N]

            if dx is not None:
                # uniform: scalar dx
                if interp == "hermite":
                    lut_val = hermite_interp(t, v0, v1, d0, d1, dx)
                else:
                    lut_val = lobachevsky3_interp(t, v0, v1, d0, d1, dx)
            else:
                # non-uniform: per-sample dx — loop (rare path)
                lut_val = np.empty_like(v0)
                for n in range(N):
                    dx_n = float(dx_arr[n])
                    if interp == "hermite":
                        lut_val[:, n] = hermite_interp(
                            t[n:n+1], v0[:, n:n+1], v1[:, n:n+1],
                            d0[:, n:n+1], d1[:, n:n+1], dx_n
                        )[:, 0]
                    else:
                        lut_val[:, n] = lobachevsky3_interp(
                            t[n:n+1], v0[:, n:n+1], v1[:, n:n+1],
                            d0[:, n:n+1], d1[:, n:n+1], dx_n
                        )[:, 0]
        else:
            raise ValueError(f"Unknown interp mode: '{interp}'")

        # OOB masking
        if packed.oob_behavior == "zero":
            mask = in_domain_mask(xi, packed.x_min, packed.x_max,
                                   packed.boundary_mode).astype(np.float32)
            lut_val = lut_val * mask[None, :]

        y += lut_val.T

    return y
