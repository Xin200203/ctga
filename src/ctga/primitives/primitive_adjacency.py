"""Primitive adjacency helpers."""

from __future__ import annotations

import torch


def primitive_adjacency(center_a: torch.Tensor, center_b: torch.Tensor, radius: float) -> bool:
    return bool(torch.linalg.norm(center_a - center_b).item() <= radius)
