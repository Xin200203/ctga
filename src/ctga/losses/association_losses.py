"""Losses for layer-2 association training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def unary_loss(unary_logits: torch.Tensor, unary_labels: torch.Tensor) -> torch.Tensor:
    if unary_logits.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=unary_logits.device if unary_logits.numel() else "cpu")
    return F.binary_cross_entropy_with_logits(unary_logits, unary_labels.float())


def pairwise_loss(
    pairwise_scores: dict[tuple[int, int], torch.Tensor],
    pairwise_labels: dict[tuple[int, int, int, int], int],
    candidate_map: dict[int, list[int]],
) -> torch.Tensor:
    terms = []
    for (obj_a, obj_b), matrix in pairwise_scores.items():
        cand_a = candidate_map.get(obj_a, [])
        cand_b = candidate_map.get(obj_b, [])
        for ia, trk_a in enumerate(cand_a):
            for ib, trk_b in enumerate(cand_b):
                label = pairwise_labels.get((obj_a, obj_b, trk_a, trk_b))
                if label is None:
                    continue
                target = torch.tensor(float(label), dtype=torch.float32, device=matrix.device)
                terms.append(F.binary_cross_entropy_with_logits(matrix[ia, ib], target))
    if not terms:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.stack(terms).mean()
