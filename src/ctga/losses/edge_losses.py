"""Losses for layer-1 edge prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=logits.device if logits.numel() else "cpu")
    target = labels[valid].float()
    return F.binary_cross_entropy_with_logits(logits[valid], target)


def layer1_edge_loss(logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = [
        masked_bce_with_logits(logits["logit_mp"], labels["y_mp"]),
        masked_bce_with_logits(logits["logit_pt"], labels["y_pt"]),
        masked_bce_with_logits(logits["logit_mt"], labels["y_mt"]),
        masked_bce_with_logits(logits["logit_pp"], labels["y_pp"]),
    ]
    return torch.stack(losses).mean()
