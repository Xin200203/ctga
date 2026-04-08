"""Relation scorer for candidate-gated second-order matching."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn


class RelationScorer(nn.Module):
    def __init__(self, feature_dim: int = 32) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        obj_edge_feat: torch.FloatTensor,
        trk_edge_feat: torch.FloatTensor,
        obj_edge_index: torch.LongTensor,
        trk_edge_index: torch.LongTensor,
        candidate_map: dict[int, list[int]],
    ) -> dict[tuple[int, int], torch.FloatTensor]:
        trk_lookup = self._track_edge_lookup(trk_edge_index, trk_edge_feat)
        compat: dict[tuple[int, int], torch.FloatTensor] = {}
        for edge_id in range(obj_edge_index.shape[1]):
            obj_a = int(obj_edge_index[0, edge_id].item())
            obj_b = int(obj_edge_index[1, edge_id].item())
            cand_a = candidate_map.get(obj_a, [])
            cand_b = candidate_map.get(obj_b, [])
            if not cand_a or not cand_b:
                continue

            matrix = torch.zeros((len(cand_a), len(cand_b)), dtype=torch.float32)
            obj_rel = obj_edge_feat[edge_id]
            for ia, trk_a in enumerate(cand_a):
                for ib, trk_b in enumerate(cand_b):
                    if trk_a == trk_b:
                        matrix[ia, ib] = -1e4
                        continue
                    trk_rel = trk_lookup.get((min(trk_a, trk_b), max(trk_a, trk_b)))
                    if trk_rel is None:
                        matrix[ia, ib] = -0.5
                        continue
                    if trk_a > trk_b:
                        trk_rel = self._swap_relation(trk_rel)
                    fused = torch.cat([obj_rel, trk_rel, torch.abs(obj_rel - trk_rel), obj_rel * trk_rel], dim=0)
                    matrix[ia, ib] = self.proj(fused).squeeze()
            compat[(obj_a, obj_b)] = matrix
        return compat

    def _track_edge_lookup(
        self, edge_index: torch.Tensor, edge_feat: torch.Tensor
    ) -> dict[tuple[int, int], torch.Tensor]:
        lookup: dict[tuple[int, int], torch.Tensor] = {}
        for edge_id in range(edge_index.shape[1]):
            a = int(edge_index[0, edge_id].item())
            b = int(edge_index[1, edge_id].item())
            lookup[(a, b)] = edge_feat[edge_id]
        return lookup

    def _swap_relation(self, feat: torch.Tensor) -> torch.Tensor:
        swapped = feat.clone()
        swapped[0:3] = -swapped[0:3]
        return swapped
