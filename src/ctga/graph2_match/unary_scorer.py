"""Unary scorer for object-track candidate edges."""

from __future__ import annotations

import torch
import torch.nn as nn


class UnaryScorer(nn.Module):
    def __init__(self, in_dim: int = 20, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, unary_feat: torch.FloatTensor) -> torch.FloatTensor:
        if unary_feat.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=unary_feat.device)
        return self.net(unary_feat).squeeze(-1)
