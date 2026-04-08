"""Node encoders for layer-1 graph entities."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((0, self.net[-1].out_features), dtype=torch.float32, device=x.device)
        return self.net(x)
