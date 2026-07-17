# src/kernels/lut_interp_advanced.py
"""
Advanced interpolation methods for LUT inference.

Extends lut_math.py with two new interpolators:

1. HERMITE CUBIC
   Stores phi(x) and phi'(x) at each LUT entry.
   Uses the standard cubic Hermite polynomial between two adjacent nodes.
   For Gaussian RBF edges: phi'(x) is FREE at compile time (no extra exp()).
   Reduces required LUT resolution L by ~4x for the same accuracy vs linear.

2. LOBACHEVSKY SPLINE (order 3)
   Uses the order-3 Lobachevsky spline as local interpolator.
   Λ₃(t) is quadratic piecewise, C¹ smooth, and converges to a Gaussian.
   Natural match for RBF-KAN: same functional family as the edge basis.
   Compact support → only evaluates over [0,1] local coordinate.

Both methods share the same LUT compile API:
    compile_hermite_lut(phi_vals, dphi_vals) -> (q_table, dq_table, scale, dscale, y_min, dy_min)
    compile_lobachevsky_lut(phi_vals, dphi_vals) -> same signature

And the same inference API:
    hermite_interp(u, v0, v1, d0, d1, dx) -> scalar
    lobachevsky3_interp(u, v0, v1, d0, d1, dx) -> scalar
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Cubic Hermite interpolation
# ---------------------------------------------------------------------------

def hermite_basis(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The four cubic Hermite basis polynomials evaluated at t in [0, 1].

    H00(t) = 2t³ - 3t² + 1       (value at left endpoint)
    H10(t) = t³ - 2t² + t         (derivative at left endpoint, scaled by dx)
    H01(t) = -2t³ + 3t²           (value at right endpoint)
    H11(t) = t³ - t²              (derivative at right endpoint, scaled by dx)

    Returns:
        H00, H10, H01, H11: each (N,) float32
    """
    t = np.asarray(t, dtype=np.float32)
    t2 = t * t
    t3 = t2 * t
    H00 = 2.0 * t3 - 3.0 * t2 + 1.0
    H10 = t3 - 2.0 * t2 + t
    H01 = -2.0 * t3 + 3.0 * t2
    H11 = t3 - t2
    return (H00.astype(np.float32), H10.astype(np.float32),
            H01.astype(np.float32), H11.astype(np.float32))


def hermite_interp(
    t: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    d0: np.ndarray,
    d1: np.ndarray,
    dx: float,
) -> np.ndarray:
    """
    Cubic Hermite interpolation between two adjacent LUT entries.

    Args:
        t:   (N,) local coordinate in [0, 1]
        v0:  (N,) dequantized value at left node
        v1:  (N,) dequantized value at right node
        d0:  (N,) dequantized derivative at left node  (phi'(x_k))
        d1:  (N,) dequantized derivative at right node (phi'(x_{k+1}))
        dx:  segment width (scalar) — needed to scale the derivative terms

    Returns:
        y:   (N,) interpolated values
    """
    H00, H10, H01, H11 = hermite_basis(t)
    dx32 = np.float32(dx)
    return (H00 * v0 + H10 * dx32 * d0 + H01 * v1 + H11 * dx32 * d1).astype(np.float32)


# ---------------------------------------------------------------------------
# Lobachevsky spline order 3 interpolation
# ---------------------------------------------------------------------------

def lobachevsky3(t: np.ndarray) -> np.ndarray:
    """
    Lobachevsky spline of order 3, evaluated at t.

    Λ₃(t) is quadratic piecewise, C¹, support [-3/2, 3/2]:
        Λ₃(t) = 3/4 - t²              for |t| <= 1/2
        Λ₃(t) = 1/2 * (3/2 - |t|)²   for 1/2 < |t| <= 3/2
        Λ₃(t) = 0                      otherwise

    This is the interpolation KERNEL, not the basis used for training.
    We evaluate it at the local coordinate t ∈ [0, 1] mapped to
    the standard support by centering at 0.5.
    """
    t = np.asarray(t, dtype=np.float32)
    s = np.abs(t)
    result = np.where(
        s <= 0.5,
        np.float32(0.75) - s * s,
        np.where(
            s <= 1.5,
            np.float32(0.5) * (np.float32(1.5) - s) ** 2,
            np.float32(0.0),
        ),
    )
    return result.astype(np.float32)


def lobachevsky3_deriv(t: np.ndarray) -> np.ndarray:
    """
    Derivative of Λ₃(t):
        Λ₃'(t) = -2t               for |t| <= 1/2
        Λ₃'(t) = -(3/2 - |t|) * sign(t)  for 1/2 < |t| <= 3/2
        Λ₃'(t) = 0                  otherwise
    """
    t = np.asarray(t, dtype=np.float32)
    s = np.abs(t)
    sign_t = np.sign(t).astype(np.float32)
    result = np.where(
        s <= 0.5,
        -2.0 * t,
        np.where(
            s <= 1.5,
            -(np.float32(1.5) - s) * sign_t,
            np.float32(0.0),
        ),
    )
    return result.astype(np.float32)


def lobachevsky3_interp(
    t: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    d0: np.ndarray,
    d1: np.ndarray,
    dx: float,
) -> np.ndarray:
    """
    Lobachevsky order-3 interpolation between two adjacent LUT entries.

    Uses the 4 nearest nodes (centered at t=0.5 in local coords):
        node at t=0   -> local coord -0.5  -> Λ₃(-0.5) contributes v0
        node at t=1   -> local coord +0.5  -> Λ₃(+0.5) contributes v1
        derivative at t=0 -> via Λ₃'(-0.5) * dx
        derivative at t=1 -> via Λ₃'(+0.5) * dx

    This is a 4-point Hermite-like scheme using Λ₃ basis weights.

    Args:
        t:   (N,) local coordinate in [0, 1]
        v0:  (N,) value at left node
        v1:  (N,) value at right node
        d0:  (N,) derivative at left node
        d1:  (N,) derivative at right node
        dx:  segment width

    Returns:
        y:   (N,) interpolated values
    """
    # 4-node quasi-interpolation with exact partition of unity.
    # Lambda3 has support [-1.5, 1.5]: for t in [0,1] the contributing lattice
    # nodes are -1, 0, 1, 2. Ghost values at -1 and 2 are Taylor estimates
    # from the stored derivatives (available at no extra memory cost):
    #     v[-1] ~ v0 - dx*d0,   v[2] ~ v1 + dx*d1
    # Sum of the four Lambda3 weights is exactly 1 on [0,1], so the
    # reconstruction is unbiased and converges (order 2).
    t32 = np.asarray(t, dtype=np.float32)
    dx32 = np.float32(dx)

    w_m1 = lobachevsky3(t32 + np.float32(1.0))
    w_0  = lobachevsky3(t32)
    w_1  = lobachevsky3(t32 - np.float32(1.0))
    w_2  = lobachevsky3(t32 - np.float32(2.0))

    v_m1 = v0 - dx32 * d0
    v_2  = v1 + dx32 * d1

    return (w_m1 * v_m1 + w_0 * v0 + w_1 * v1 + w_2 * v_2).astype(np.float32)


# ---------------------------------------------------------------------------
# Quantization helpers for derivative tables
# ---------------------------------------------------------------------------

def quantize_deriv_table(
    dphi_vals: np.ndarray,
    qmin: int = -127,
    qmax: int = 127,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize derivative values per-segment, symmetric int8.

    Derivatives are symmetric around zero by nature (zero-mean for gaussians),
    so symmetric quantization is always appropriate.

    Args:
        dphi_vals: (E, K, L) float32 derivative values
        qmin, qmax: int8 symmetric range

    Returns:
        dq_table: (E, K, L) int8
        dscale:   (E, K) float32
        dy_min:   (E, K) float32 — always 0.0 for symmetric
    """
    dphi_vals = np.asarray(dphi_vals, dtype=np.float32)
    assert dphi_vals.ndim == 3, "Expected (E, K, L)"

    max_abs = np.max(np.abs(dphi_vals), axis=2)           # [E, K]
    denom = float(max(abs(qmin), abs(qmax)))
    dscale = np.where(max_abs > 0, max_abs / denom, 1.0).astype(np.float32)
    dy_min = np.zeros_like(dscale, dtype=np.float32)

    # Quantize
    dq_float = dphi_vals / dscale[:, :, None]
    dq = np.clip(np.rint(dq_float), qmin, qmax).astype(np.int8)

    return dq, dscale, dy_min


def dequant_deriv(
    dq: np.ndarray,
    dscale: np.ndarray,
) -> np.ndarray:
    """
    Dequantize derivative: dphi = dscale * dq
    (symmetric: dy_min = 0 always)
    """
    return (dscale.astype(np.float32) * dq.astype(np.float32)).astype(np.float32)
