"""Primitive feature utilities."""

from __future__ import annotations

import torch


def aggregate_primitive_feature(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.zeros(0, dtype=torch.float32)
    stacked = torch.stack(values, dim=0)
    return torch.cat([stacked.mean(dim=0), stacked.max(dim=0).values], dim=0)
