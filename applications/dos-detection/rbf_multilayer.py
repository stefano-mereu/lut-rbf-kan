"""
Multi-layer RBF-KAN for CICIDS2017 DoS detection.

Design rationale
----------------
RBFKANLayerTorch has centers fixed on [0,1] and evaluates r = |x - c| / h.
Inputs outside [0,1] fall in the tail of every Gaussian (all bases ~0), so
every layer's input MUST be mapped into [0,1] before the RBF forward.

We therefore wrap each RBF layer with a min-max normalization to [0,1]:

    block(x) = RBF( minmax01(x) )

- Layer 0 input: StandardScaler features (range ~[-4.5, 6.2]).
  minmax uses per-feature (min, max) measured on the training set.
- Inner layers: input is the previous block's output, whose range emerges
  from training. minmax uses running statistics (updated in train mode,
  frozen for eval / LUT compilation).

This keeps every RBF layer in its valid [0,1] regime and makes the eventual
LUT compilation identical to the validated single-layer case: each RBF layer
compiles over [0,1], and the cheap affine minmax (2 params/feature) is applied
before lookup.

The minmax statistics are stored as buffers so they persist and can be frozen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make the lut-rbf-kan package importable (adjust path as needed)
LUT_RBF_KAN_PATH = str(Path(__file__).resolve().parents[2])  # repo root
if LUT_RBF_KAN_PATH not in sys.path:
    sys.path.insert(0, LUT_RBF_KAN_PATH)

from src.models.rbf_adapter import RBFKANLayerTorch, theoretical_h_init


class MinMax01(nn.Module):
    """
    Affine map to [0,1] using per-feature (min, max).

    Two modes:
      - "fixed":   min/max set once from data (buffers), never updated.
      - "running": min/max tracked as running estimates in train mode
                   (like BatchNorm running stats), frozen in eval mode.

    Output is clipped to [0,1] so downstream RBF centers always see valid range.
    A small margin expands [min,max] slightly so training-time extremes don't
    saturate exactly at the boundary.
    """

    def __init__(self, num_features: int, mode: str = "running",
                 margin: float = 0.05, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.mode = mode
        self.margin = margin
        self.momentum = momentum
        self.register_buffer("x_min", torch.zeros(num_features))
        self.register_buffer("x_max", torch.ones(num_features))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        self.frozen = False

    @torch.no_grad()
    def set_from_data(self, x: torch.Tensor):
        """Fix min/max from a data batch (for mode='fixed' or init)."""
        lo = x.min(dim=0).values
        hi = x.max(dim=0).values
        span = (hi - lo).clamp_min(1e-6)
        self.x_min.copy_(lo - self.margin * span)
        self.x_max.copy_(hi + self.margin * span)
        self.initialized.fill_(True)

    def freeze(self):
        self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "running" and self.training and not self.frozen:
            with torch.no_grad():
                lo = x.min(dim=0).values
                hi = x.max(dim=0).values
                span = (hi - lo).clamp_min(1e-6)
                lo = lo - self.margin * span
                hi = hi + self.margin * span
                if not bool(self.initialized):
                    self.x_min.copy_(lo)
                    self.x_max.copy_(hi)
                    self.initialized.fill_(True)
                else:
                    m = self.momentum
                    self.x_min.mul_(1 - m).add_(m * lo)
                    self.x_max.mul_(1 - m).add_(m * hi)

        span = (self.x_max - self.x_min).clamp_min(1e-6)
        z = (x - self.x_min) / span
        return z.clamp(0.0, 1.0)


class RBFBlock(nn.Module):
    """minmax01 -> RBF layer."""

    def __init__(self, in_dim, out_dim, G, h_init, norm_mode="running"):
        super().__init__()
        self.norm = MinMax01(in_dim, mode=norm_mode)
        self.rbf = RBFKANLayerTorch(
            input_dim=in_dim, output_dim=out_dim, G=G,
            h_init=h_init, h_learnable=True,
            x_min=0.0, x_max=1.0,  # RBF always sees [0,1] after norm
        )

    def forward(self, x):
        return self.rbf(self.norm(x))

    def freeze(self):
        self.norm.freeze()
        self.rbf.freeze_h()


class RBFKANMultiLayer(nn.Module):
    """
    Multi-layer RBF-KAN: width e.g. [78, 32, 16, 1].

    Each layer is minmax01 -> RBF. The first layer's norm is initialized from
    the actual input data; inner layers use running statistics.
    """

    def __init__(self, width, G=20, norm_mode="running"):
        super().__init__()
        self.width = width
        self.G = G
        h_init = theoretical_h_init(G)  # midpoint of [h/2, 3h/2] on [0,1]

        self.blocks = nn.ModuleList([
            RBFBlock(width[i], width[i + 1], G, h_init, norm_mode=norm_mode)
            for i in range(len(width) - 1)
        ])

    @torch.no_grad()
    def init_first_norm(self, x_sample: torch.Tensor):
        """Fix the first layer's minmax from real input data."""
        self.blocks[0].norm.set_from_data(x_sample)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def freeze(self):
        """Freeze all norm stats and h before LUT compilation."""
        for blk in self.blocks:
            blk.freeze()


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    model = RBFKANMultiLayer([78, 32, 16, 1], G=20)
    x = torch.randn(256, 78) * 2.0  # StandardScaler-like range
    model.init_first_norm(x)
    y = model(x)
    print(f"Input {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
