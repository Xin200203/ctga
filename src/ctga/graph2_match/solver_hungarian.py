"""Hungarian fallback solver."""

from __future__ import annotations

from functools import lru_cache

import torch


class HungarianFallbackSolver:
    def solve(self, unary_cost_matrix: torch.FloatTensor) -> dict[int, int]:
        num_objs, num_cols = unary_cost_matrix.shape
        num_tracks = max(num_cols - 1, 0)

        @lru_cache(maxsize=None)
        def dp(obj_idx: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
            if obj_idx == num_objs:
                return 0.0, ()

            best_score = float("-inf")
            best_assign: tuple[int, ...] = ()

            newborn_score, newborn_rest = dp(obj_idx + 1, used_mask)
            newborn_score += float(unary_cost_matrix[obj_idx, num_tracks].item())
            best_score = newborn_score
            best_assign = (-1,) + newborn_rest

            for track_idx in range(num_tracks):
                if used_mask & (1 << track_idx):
                    continue
                score_rest, assign_rest = dp(obj_idx + 1, used_mask | (1 << track_idx))
                score = float(unary_cost_matrix[obj_idx, track_idx].item()) + score_rest
                if score > best_score:
                    best_score = score
                    best_assign = (track_idx,) + assign_rest
            return best_score, best_assign

        _, assignment = dp(0, 0)
        return {obj_idx: assignment[obj_idx] for obj_idx in range(num_objs)}
