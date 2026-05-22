# src/models/rbf_adapter.py
"""
GKAN-style RBF-KAN adapter with adaptive shape parameter.

Architecture (Noorizadegan & Wang 2025, Section 3):
    phi_ij(x) = einsum over k: coeffs[i,j,k] * psi(|x - c_k| / h)

    psi(r) = exp(-r^2)      # Gaussian kernel (r is normalized distance)

Key design decisions (from Haider & Mereu experiments, May 2026):
  - Centers on [0, 1]  (NOT [-2, 2] — different parametrization)
  - Single global log_h parameter shared across all layers
  - h initialized via LOOCV in range [0.8/(G-1), 2.5/(G-1)]
  - h optionally learnable (nn.Parameter) during training
  - Pure RBF branch only — NO SiLU residual branch

LUT quantization interface (Kuznetsov LUT-KAN):
  - After training: freeze h, then compile LUT over domain [0, 1]
  - eval_phi_and_deriv() provides phi and dphi simultaneously at no extra cost
  - dphi available analytically: d/dx[exp(-r^2)] = -2r/h * exp(-r^2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.kernels.rbf_math import (
    rbf_edge_eval,
    rbf_edge_eval_and_deriv,
)


# ---------------------------------------------------------------------------
# LOOCV shape parameter initialization (Rippa 1999)
# ---------------------------------------------------------------------------

def loocv_optimal_h(
    x_data: np.ndarray,
    y_data: np.ndarray,
    G: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    search_range: Optional[Tuple[float, float]] = None,
    num_candidates: int = 30,
) -> float:
    """
    Find optimal epsilon via Leave-One-Out Cross-Validation (Rippa 1999).

    Conditioning interval from Noorizadegan & Wang (2026):
        epsilon in [h/2, 3h/2]  where h = (x_max - x_min) / (G - 1)

    This keeps h/epsilon in [2/3, 2], ensuring well-conditioned feature matrix.
    The formula used is phi(x) = exp(-((x-mu)/epsilon)^2) — no 0.5 factor.

    This is efficient: LOOCV score = sum_i (c_i / A_ii^{-1})^2
    where c = A^{-1} y and A_ii^{-1} is the i-th diagonal of A^{-1}.
    No re-fitting needed for each left-out point.

    Args:
        x_data:       (N,) training inputs
        y_data:       (N,) training targets
        G:            number of RBF centers
        x_min:        domain lower bound (default 0.0)
        x_max:        domain upper bound (default 1.0)
        search_range: override (eps_min, eps_max) — defaults to [h/2, 3h/2]
        num_candidates: number of epsilon values to try (log-spaced)

    Returns:
        eps_opt: float, optimal shape parameter
    """
    x_data = np.asarray(x_data, dtype=np.float64).ravel()
    y_data = np.asarray(y_data, dtype=np.float64).ravel()
    N = len(x_data)

    # Conditioning interval: epsilon in [h/2, 3h/2]
    h_spacing = (x_max - x_min) / (G - 1)
    if search_range is None:
        h_min = h_spacing / 2.0
        h_max = 3.0 * h_spacing / 2.0
    else:
        h_min, h_max = search_range

    centers = np.linspace(0.0, 1.0, G)
    h_candidates = np.logspace(np.log10(h_min), np.log10(h_max), num_candidates)

    best_h = float(h_candidates[len(h_candidates) // 2])
    best_score = np.inf

    for h in h_candidates:
        # Build feature matrix Phi: (N, G)
        r = np.abs(x_data[:, None] - centers[None, :]) / h
        Phi = np.exp(-r ** 2)

        # Solve A c = y where A = Phi^T Phi + tiny regularization
        A = Phi.T @ Phi + 1e-10 * np.eye(G)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue

        # Coefficients
        c = A_inv @ (Phi.T @ y_data)

        # LOOCV score via Rippa's formula
        # residual vector e = y - Phi c
        e = y_data - Phi @ c

        # Hat matrix diagonal: h_ii = Phi_i^T (A^{-1}) Phi_i
        # LOOCV_i = e_i / (1 - h_ii)
        h_diag = np.sum((Phi @ A_inv) * Phi, axis=1)   # (N,)
        denom = 1.0 - h_diag
        # Clip to avoid division by near-zero
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        loocv_errors = e / denom
        score = float(np.mean(loocv_errors ** 2))

        if score < best_score:
            best_score = score
            best_h = float(h)

    return best_h


def theoretical_h_init(G: int, x_min: float = 0.0, x_max: float = 1.0) -> float:
    """
    Midpoint of conditioning interval [h/2, 3h/2] where h = (x_max-x_min)/(G-1).

    From Noorizadegan & Wang (2026): keeps h/epsilon in [2/3, 2].
    The midpoint h corresponds to h/epsilon = 1 — optimal conditioning.
    """
    h_spacing = (x_max - x_min) / (G - 1)
    return h_spacing  # midpoint of [h/2, 3h/2]


# ---------------------------------------------------------------------------
# Pure NumPy RBF-KAN layer (for LUT compilation, no torch needed)
# ---------------------------------------------------------------------------

class RBFKANLayerNumpy:
    """
    Pure NumPy RBF-KAN layer for LUT compilation and testing.

    phi_ij(x) = sum_k  coeffs[i,j,k] * exp(-((x - c_k)/h)^2)

    Centers on [0, 1], h is a scalar (global).
    """

    def __init__(
        self,
        coef: np.ndarray,
        *,
        G: int,
        h: float,
        x_min: float = 0.0,
        x_max: float = 1.0,
        grid_segments: int = 16,
    ):
        self.coef = np.asarray(coef, dtype=np.float32)
        self.G = int(G)
        self.h = float(h)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.grid_segments = int(grid_segments)

        if self.coef.ndim != 3:
            raise ValueError(f"coef must be [in_dim, out_dim, G], got {self.coef.shape}")

        self.in_dim = self.coef.shape[0]
        self.out_dim = self.coef.shape[1]

        # Centers on [0, 1] — this is the canonical parametrization
        self.centers = np.linspace(0.0, 1.0, G, dtype=np.float32)

        # Sigmas: sigma = h (uniform, global)
        self.sigmas = np.full(G, h, dtype=np.float32)

        # LUT knots over [x_min, x_max]
        self.knots = np.linspace(x_min, x_max, grid_segments + 1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected [N, {self.in_dim}], got {x.shape}")
        N = x.shape[0]
        y = np.zeros((N, self.out_dim), dtype=np.float32)
        for i in range(self.in_dim):
            xi = x[:, i]
            for j in range(self.out_dim):
                y[:, j] += rbf_edge_eval(xi, self.centers, self.sigmas, self.coef[i, j])
        return y


# ---------------------------------------------------------------------------
# EdgeSpec for LUT builder (compatible with build_rbf_lut_for_edges)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RBFEdgeSpec:
    """
    Edge spec for GKAN-style RBF-KAN, compatible with LUT builder.

    eval_phi_and_deriv returns (phi, dphi) simultaneously — free for Gaussians.
    """
    edge_id: int
    src_idx: int
    dst_idx: int
    knots: np.ndarray
    domain: Tuple[float, float]
    eval_phi: Callable[[np.ndarray], np.ndarray]
    eval_phi_and_deriv: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    # stored for inspection
    centers: np.ndarray
    sigmas: np.ndarray
    coef: np.ndarray
    h: float
    # PyKAN-style compat
    eval_spline: Optional[Callable] = None
    base_kind: str = "none"
    sb: float = 1.0
    ss: float = 1.0
    m: float = 1.0


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class RBFKANSingleLayerAdapter:
    """
    GKAN-style RBF-KAN adapter.

    Two construction modes:
      from_arch()         — random init, use before training
      from_trained_layer()— from a trained RBFKANLayerTorch after h is frozen

    Workflow for LUT-KAN:
      1. Initialize h via LOOCV:   h = loocv_optimal_h(x_train, y_train, G)
      2. Train with learnable h:   RBFKANLayerTorch(h_init=h, h_learnable=True)
      3. Freeze h after training:  layer.freeze_h()
      4. Export for LUT:           adapter = RBFKANSingleLayerAdapter.from_trained_layer(layer)
      5. Build LUT:                build_rbf_lut_for_edges(adapter.extract_edges(), ...)
    """

    def __init__(
        self,
        coef: np.ndarray,
        *,
        G: int,
        h: float,
        x_min: float = 0.0,
        x_max: float = 1.0,
        grid_segments: int = 16,
    ):
        self.layer = RBFKANLayerNumpy(
            coef=coef, G=G, h=h,
            x_min=x_min, x_max=x_max,
            grid_segments=grid_segments,
        )

    @property
    def in_dim(self): return self.layer.in_dim
    @property
    def out_dim(self): return self.layer.out_dim
    @property
    def G(self): return self.layer.G
    @property
    def h(self): return self.layer.h
    @property
    def grid_segments(self): return self.layer.grid_segments

    @staticmethod
    def from_arch(
        arch: dict,
        *,
        seed: int = 0,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> "RBFKANSingleLayerAdapter":
        """
        Build adapter from arch config, with optional LOOCV h initialization.

        Config keys:
            in_dim, out_dim  (required)
            G                number of RBF centers (default: 20)
            h_init           if provided, skips LOOCV (use theoretical_h_init(G) as default)
            h_mode           "loocv" | "theoretical" | "fixed" (default: "theoretical")
            x_min, x_max     domain (default: 0.0, 1.0)
            grid_segments    LUT segments (default: 16)
        """
        in_dim = int(arch["in_dim"])
        out_dim = int(arch["out_dim"])
        G = int(arch.get("G", 20))
        x_min = float(arch.get("x_min", 0.0))
        x_max = float(arch.get("x_max", 1.0))
        grid_segments = int(arch.get("grid_segments", 16))
        h_mode = str(arch.get("h_mode", "theoretical")).lower()

        # Shape parameter initialization
        if "h_init" in arch:
            h = float(arch["h_init"])
        elif h_mode == "loocv" and x_train is not None and y_train is not None:
            h = loocv_optimal_h(x_train, y_train, G)
            print(f"LOOCV h_init = {h:.6f} (range [{0.8/(G-1):.4f}, {2.5/(G-1):.4f}])")
        else:
            h = theoretical_h_init(G)
            print(f"Theoretical h_init = {h:.6f} = 1.5/{G-1}")

        rng = np.random.default_rng(seed)
        coef = rng.normal(0.0, 0.1 / np.sqrt(G),
                          size=(in_dim, out_dim, G)).astype(np.float32)

        return RBFKANSingleLayerAdapter(
            coef=coef, G=G, h=h,
            x_min=x_min, x_max=x_max,
            grid_segments=grid_segments,
        )

    @staticmethod
    def from_trained_layer(layer) -> "RBFKANSingleLayerAdapter":
        """
        Build adapter from a trained RBFKANLayerTorch (after h is frozen).

        Call layer.freeze_h() before this to ensure h is fixed.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required for from_trained_layer()")

        coef_np = layer.coeffs.detach().cpu().numpy().astype(np.float32)
        h = float(torch.exp(layer.log_h).item())
        G = int(layer.num_grid)
        x_min = float(getattr(layer, "x_min", 0.0))
        x_max = float(getattr(layer, "x_max", 1.0))
        grid_segments = int(getattr(layer, "grid_segments", 16))

        return RBFKANSingleLayerAdapter(
            coef=coef_np, G=G, h=h,
            x_min=x_min, x_max=x_max,
            grid_segments=grid_segments,
        )

    def extract_edges(self) -> List[RBFEdgeSpec]:
        """Extract edge specs for LUT compilation."""
        layer = self.layer
        edges = []
        edge_id = 0

        for out_idx in range(layer.out_dim):
            for in_idx in range(layer.in_dim):
                c = layer.coef[in_idx, out_idx, :].copy()
                ctrs = layer.centers.copy()
                sigs = layer.sigmas.copy()

                def _eval_phi(x, c=c, ctrs=ctrs, sigs=sigs):
                    return rbf_edge_eval(
                        np.asarray(x, np.float32).ravel(), ctrs, sigs, c)

                def _eval_and_deriv(x, c=c, ctrs=ctrs, sigs=sigs):
                    return rbf_edge_eval_and_deriv(
                        np.asarray(x, np.float32).ravel(), ctrs, sigs, c)

                edges.append(RBFEdgeSpec(
                    edge_id=edge_id,
                    src_idx=in_idx,
                    dst_idx=out_idx,
                    knots=layer.knots.copy(),
                    domain=(layer.x_min, layer.x_max),
                    eval_phi=_eval_phi,
                    eval_phi_and_deriv=_eval_and_deriv,
                    centers=ctrs,
                    sigmas=sigs,
                    coef=c,
                    h=layer.h,
                ))
                edge_id += 1

        return edges

    def forward_float(self, x: np.ndarray) -> np.ndarray:
        return self.layer.forward(x)


# ---------------------------------------------------------------------------
# PyTorch training layer (optional — requires torch)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class RBFKANLayerTorch(nn.Module):
        """
        GKAN-style RBF-KAN layer for PyTorch training.

        Implements exactly:
            h = exp(log_h)
            r = |x_expanded - centers| / h
            phi = exp(-r^2)
            output = einsum('nig,iog->no', phi, coeffs)

        Phase 1 (LOOCV/BO): set h_learnable=False, use h_init from loocv_optimal_h()
        Phase 2 (gradient refinement): set h_learnable=True, h drifts 6-25% toward optimum

        Args:
            input_dim:   number of input features
            output_dim:  number of output features
            G:           number of RBF centers (on [0, 1])
            h_init:      initial shape parameter (use loocv_optimal_h or theoretical_h_init)
            h_learnable: if True, log_h is nn.Parameter (updated by Adam)
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            G: int,
            h_init: float,
            h_learnable: bool = True,
            x_min: float = 0.0,
            x_max: float = 1.0,
            grid_segments: int = 16,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_grid = G
            self.x_min = x_min
            self.x_max = x_max
            self.grid_segments = grid_segments

            # Centers fixed on [0, 1]
            centers = torch.linspace(0.0, 1.0, G)
            self.register_buffer("centers", centers)

            # Shape parameter
            log_h_val = torch.tensor(float(np.log(h_init)))
            if h_learnable:
                self.log_h = nn.Parameter(log_h_val)
            else:
                self.register_buffer("log_h", log_h_val)
            self.h_learnable = h_learnable

            # Coefficients
            self.coeffs = nn.Parameter(
                torch.empty(input_dim, output_dim, G)
            )
            nn.init.normal_(self.coeffs, 0.0, 0.1 / np.sqrt(G))

        def freeze_h(self):
            """Call before LUT compilation. Converts log_h from Parameter to buffer."""
            if self.h_learnable:
                h_val = self.log_h.data.clone()
                del self.log_h
                self.register_buffer("log_h", h_val)
                self.h_learnable = False

        def get_h(self) -> float:
            return float(torch.exp(self.log_h).item())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (N, input_dim) in [0, 1]
            Returns:
                y: (N, output_dim)
            """
            h = torch.exp(self.log_h)
            # x: (N, in, 1), centers: (1, 1, G)
            x_exp = x.unsqueeze(-1)                         # (N, in, 1)
            c_exp = self.centers.view(1, 1, -1)             # (1, 1, G)
            r = torch.abs(x_exp - c_exp) / h               # (N, in, G)
            phi = torch.exp(-r * r)                         # (N, in, G)
            # coeffs: (in, out, G) -> einsum nig,iog->no
            return torch.einsum("nig,iog->no", phi, self.coeffs)

    class RBFKANTorch(nn.Module):
        """
        Multi-layer GKAN with shared log_h.

        Architecture: [input_dim, *hidden_dims, output_dim]
        Recommended: [d, 12, 12, 1] for general tasks
                     [d, 16, 1] for periodic/HF tasks (Shallow)
        """

        def __init__(
            self,
            dims: List[int],
            G: int,
            h_init: float,
            h_learnable: bool = True,
        ):
            super().__init__()
            self.layers = nn.ModuleList([
                RBFKANLayerTorch(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    G=G,
                    h_init=h_init,
                    h_learnable=False,   # only first layer has learnable h
                )
                for i in range(len(dims) - 1)
            ])
            # Shared log_h — first-layer dominance (Noorizadegan & Wang 2025)
            log_h_val = torch.tensor(float(np.log(h_init)))
            if h_learnable:
                self.log_h = nn.Parameter(log_h_val)
            else:
                self.register_buffer("log_h", log_h_val)
            self.h_learnable = h_learnable
            self._sync_h()

        def _sync_h(self):
            """Sync shared log_h to all layers."""
            for layer in self.layers:
                if hasattr(self, "log_h"):
                    layer.log_h = self.log_h

        def freeze_h(self):
            """Freeze h before LUT compilation."""
            if self.h_learnable:
                h_val = self.log_h.data.clone()
                del self.log_h
                self.register_buffer("log_h", h_val)
                self.h_learnable = False
            for layer in self.layers:
                layer.freeze_h()

        def get_h(self) -> float:
            return float(torch.exp(self.log_h).item())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x
