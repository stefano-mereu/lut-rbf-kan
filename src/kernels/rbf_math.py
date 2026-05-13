# src/kernels/rbf_math.py
"""
RBF basis functions for KAN edges.

Implements Gaussian RBF:
    psi(x; mu, sigma) = exp( -(x - mu)^2 / (2 * sigma^2) )

Key property exploited for Hermite LUT:
    psi'(x) = -(x - mu) / sigma^2 * psi(x)

So the derivative is FREE once psi(x) is computed — no extra exp() calls.

For a full edge function:
    phi(x) = sum_k  c_k * psi(x; mu_k, sigma_k)
    phi'(x) = sum_k  c_k * psi'(x; mu_k, sigma_k)
            = sum_k  c_k * [-(x - mu_k) / sigma_k^2] * psi(x; mu_k, sigma_k)

Both are computed together in eval_phi_and_deriv() at no extra cost.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Single gaussian basis
# ---------------------------------------------------------------------------

def gaussian_rbf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Evaluate Gaussian RBF: exp(-(x-mu)^2 / (2*sigma^2))

    Args:
        x:     (N,) input points
        mu:    center
        sigma: width (> 0)

    Returns:
        psi: (N,) values in [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    z = (x - np.float32(mu)) / np.float32(sigma)
    return np.exp(-0.5 * z * z, dtype=np.float32)


def gaussian_rbf_deriv(x: np.ndarray, mu: float, sigma: float,
                        psi: np.ndarray | None = None) -> np.ndarray:
    """
    Derivative of Gaussian RBF: -(x-mu)/sigma^2 * psi(x)

    If psi is provided (already computed), reuses it — no extra exp().

    Args:
        x:    (N,) input points
        mu:   center
        sigma: width
        psi:  (N,) optional pre-computed psi(x; mu, sigma)

    Returns:
        dpsi: (N,) derivative values
    """
    x = np.asarray(x, dtype=np.float32)
    if psi is None:
        psi = gaussian_rbf(x, mu, sigma)
    factor = -(x - np.float32(mu)) / np.float32(sigma ** 2)
    return (factor * psi).astype(np.float32)


# ---------------------------------------------------------------------------
# Full edge function: mixture of K gaussians
# ---------------------------------------------------------------------------

def rbf_edge_eval(
    x: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    """
    Evaluate RBF edge function: phi(x) = sum_k c_k * psi(x; mu_k, sigma_k)

    Args:
        x:       (N,) input points
        centers: (K,) gaussian centers mu_k
        sigmas:  (K,) gaussian widths sigma_k
        coef:    (K,) mixture coefficients c_k

    Returns:
        phi: (N,) output values
    """
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)
    coef = np.asarray(coef, dtype=np.float32)

    K = len(centers)
    if len(sigmas) != K or len(coef) != K:
        raise ValueError("centers, sigmas, coef must have the same length K")

    # Vectorized: z[k, n] = (x[n] - mu_k) / sigma_k
    z = (x[None, :] - centers[:, None]) / sigmas[:, None]   # [K, N]
    psi = np.exp(-0.5 * z * z, dtype=np.float32)             # [K, N]
    return (coef[:, None] * psi).sum(axis=0).astype(np.float32)


def rbf_edge_eval_and_deriv(
    x: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    coef: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate RBF edge function AND its derivative simultaneously.

    phi(x)  = sum_k c_k * psi_k(x)
    phi'(x) = sum_k c_k * [-(x - mu_k) / sigma_k^2] * psi_k(x)

    The derivative comes FREE: psi_k(x) is computed once, reused for both.

    Returns:
        phi:   (N,) function values
        dphi:  (N,) derivative values
    """
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)
    coef = np.asarray(coef, dtype=np.float32)

    K = len(centers)
    z = (x[None, :] - centers[:, None]) / sigmas[:, None]          # [K, N]
    psi = np.exp(-0.5 * z * z, dtype=np.float32)                    # [K, N]

    # phi(x)
    phi = (coef[:, None] * psi).sum(axis=0)                         # [N]

    # phi'(x) = sum_k c_k * (-z_k / sigma_k) * psi_k
    # factor_k(x) = -(x - mu_k) / sigma_k^2
    factors = -(x[None, :] - centers[:, None]) / (sigmas[:, None] ** 2)  # [K, N]
    dphi = (coef[:, None] * factors * psi).sum(axis=0)              # [N]

    return phi.astype(np.float32), dphi.astype(np.float32)


# ---------------------------------------------------------------------------
# Utility: build uniform centers and sigmas for a domain [a, b]
# ---------------------------------------------------------------------------

def build_uniform_rbf_params(
    K: int,
    x_min: float,
    x_max: float,
    overlap: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place K gaussian centers uniformly in [x_min, x_max].
    sigma = overlap * spacing, where spacing = (x_max - x_min) / (K - 1).

    Args:
        K:       number of basis functions
        x_min:   domain left boundary
        x_max:   domain right boundary
        overlap: controls width relative to spacing (1.0 = adjacent gaussians
                 cross at ~0.6 height; 0.5 = narrower, more local)

    Returns:
        centers: (K,) float32
        sigmas:  (K,) float32 (uniform)
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    centers = np.linspace(x_min, x_max, K, dtype=np.float32)
    spacing = float(x_max - x_min) / float(K - 1)
    sigma = float(overlap) * spacing
    sigmas = np.full(K, sigma, dtype=np.float32)
    return centers, sigmas
