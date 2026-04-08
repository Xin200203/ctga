"""Utilities for 2D mask feature normalization."""

from __future__ import annotations

import torch


def normalize_mask_feature(feature: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(feature).clamp_min(1e-6)
    return feature / norm
