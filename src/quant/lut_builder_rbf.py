# src/quant/lut_builder_rbf.py
"""
Extended LUT builder for RBF-KAN with derivative sampling.

Extends build_lut_for_edges() to also sample phi'(x) at the same grid points,
enabling Hermite and Lobachevsky interpolation at inference.

The derivative table is stored alongside the value table in an extended
LUTArtifact subclass (RBFLUTArtifact), fully backward compatible with
the existing PackedLUT infrastructure.

Key design:
  - For RBF edges: eval_phi_and_deriv() is called ONCE per grid point,
    obtaining both phi and dphi at zero extra exp() cost.
  - For non-RBF edges: dphi is estimated via finite differences (fallback).
  - dq_table is quantized with symmetric int8 (derivatives are zero-mean).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

from src.quant.lut_builder import (
    LUTArtifact,
    BoundaryMode,
    InterpMode,
    OOBBehavior,
    QuantDType,
    QuantScheme,
    build_segment_grid,
    _quant_params_from_range,
)
from src.kernels.lut_interp_advanced import quantize_deriv_table


@dataclass
class RBFLUTArtifact(LUTArtifact):
    """
    Extended LUT artifact that also stores derivative tables.

    Additional arrays (vs LUTArtifact):
      dq_table: int8  [E, K, L] — quantized phi'(x) values
      dscale:   float32 [E, K]  — per-segment derivative scale
      dy_min:   float32 [E, K]  — always 0.0 (symmetric quantization)

    When dq_table is None, falls back to standard linear interpolation.
    """
    dq_table: Optional[np.ndarray] = None   # int8 [E, K, L]
    dscale: Optional[np.ndarray] = None     # float32 [E, K]
    dy_min: Optional[np.ndarray] = None     # float32 [E, K]  (all zeros)

    def has_derivatives(self) -> bool:
        return self.dq_table is not None


def _finite_diff_deriv(
    eval_phi,
    x_grid: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Estimate phi'(x) via central differences when analytic derivative unavailable.
    Used as fallback for non-RBF edges.
    """
    x_grid = np.asarray(x_grid, dtype=np.float32)
    eps32 = np.float32(eps)
    return (
        (eval_phi(x_grid + eps32) - eval_phi(x_grid - eps32)) / (2.0 * eps32)
    ).astype(np.float32)


def build_rbf_lut_for_edges(
    edges: List,
    L: int,
    interp: InterpMode,
    y_range_method: Literal["minmax", "percentile"],
    lower_pct: float,
    upper_pct: float,
    dtype: QuantDType,
    scheme: QuantScheme,
    qmin: int,
    qmax: int,
    meta_dtype: Literal["float16", "float32"] = "float32",
    oob_behavior: OOBBehavior = "clip",
    boundary_mode: BoundaryMode = "half_open",
    deriv_finite_diff_eps: float = 1e-4,
) -> RBFLUTArtifact:
    """
    Build extended LUT artifact with derivative table for Hermite/Lobachevsky interpolation.

    For each edge:
      - If edge has eval_phi_and_deriv attribute: uses it (free for RBF gaussians)
      - Otherwise: estimates derivative via finite differences

    Args:
        edges:    list of EdgeSpec or RBFEdgeSpec objects
        L:        LUT resolution (samples per segment)
        interp:   stored in artifact; use "hermite" or "lobachevsky3" at inference
        ... (same as build_lut_for_edges)
        deriv_finite_diff_eps: step for finite difference fallback

    Returns:
        RBFLUTArtifact with both q_table and dq_table populated
    """
    if not edges:
        raise ValueError("edges must be non-empty")
    if L < 2:
        raise ValueError("L must be >= 2")

    knots = np.asarray(edges[0].knots, dtype=np.float32)
    K = int(knots.size - 1)
    E = int(len(edges))

    # Use linspace (closed: includes both endpoints) for ALL interpolation modes.
    # This ensures that for Hermite, adjacent LUT entries r and r+1 are evenly
    # spaced with dx_lut = segment_width / (L-1), and both segment endpoints
    # are included in the table.
    #
    # Note: the original build_segment_grid uses half-open [x_k, x_{k+1}),
    # which works for linear but causes Hermite to misread the segment boundary.
    # We use linspace for all modes for consistency.
    x_grid = np.empty((K, L), dtype=np.float32)
    for k in range(K):
        x_grid[k, :] = np.linspace(
            float(knots[k]), float(knots[k + 1]), L, dtype=np.float32
        )

    float_lut = np.empty((E, K, L), dtype=np.float32)
    dfloat_lut = np.empty((E, K, L), dtype=np.float32)

    for ei, e in enumerate(edges):
        eval_fn = e.eval_phi
        eval_and_deriv = getattr(e, "eval_phi_and_deriv", None)

        for k in range(K):
            xk = x_grid[k, :]
            if eval_and_deriv is not None:
                # RBF path: phi and dphi for free
                phi_k, dphi_k = eval_and_deriv(xk)
            else:
                # Fallback: finite differences
                phi_k = eval_fn(xk).astype(np.float32)
                dphi_k = _finite_diff_deriv(eval_fn, xk, deriv_finite_diff_eps)

            float_lut[ei, k, :] = phi_k.astype(np.float32)
            dfloat_lut[ei, k, :] = dphi_k.astype(np.float32)

    # --- Quantize values (same as original builder) ---
    if y_range_method == "percentile":
        y_lo = np.percentile(float_lut, lower_pct, axis=2).astype(np.float32)
        y_hi = np.percentile(float_lut, upper_pct, axis=2).astype(np.float32)
    else:
        y_lo = np.min(float_lut, axis=2).astype(np.float32)
        y_hi = np.max(float_lut, axis=2).astype(np.float32)

    scale, y_min_arr = _quant_params_from_range(
        y_lo, y_hi, scheme=scheme, dtype=dtype, qmin=qmin, qmax=qmax
    )

    scale_b = scale[:, :, None]
    y_min_b = y_min_arr[:, :, None]
    q_float = (float_lut - y_min_b) / scale_b
    q = np.clip(np.rint(q_float), qmin, qmax)

    if dtype == "uint8":
        q_table = q.astype(np.uint8)
    else:
        q_table = q.astype(np.int8)

    md = np.float32   # always float32 for extended artifact

    # --- Quantize derivatives (always symmetric int8) ---
    dq_table, dscale, dy_min_arr = quantize_deriv_table(
        dfloat_lut, qmin=-127, qmax=127
    )

    return RBFLUTArtifact(
        format_version=2,           # new version for extended format
        knots=knots,
        L=int(L),
        interp=interp,
        boundary_mode=boundary_mode,
        oob_behavior=oob_behavior,
        q_table=q_table,
        scale=scale.astype(md),
        y_min=y_min_arr.astype(md),
        dtype=dtype,
        scheme=scheme,
        qmin=int(qmin),
        qmax=int(qmax),
        value_representation="phi",
        base_kind="none",
        # derivative tables
        dq_table=dq_table,
        dscale=dscale.astype(md),
        dy_min=dy_min_arr.astype(md),
    )
