"""Assign primitives to teacher GT instances."""

from __future__ import annotations

import torch

from ctga.common.types import Primitive3D


class PrimitiveGTAssigner:
    def __init__(self, min_overlap_ratio: float = 0.5, max_leak_ratio: float = 0.1) -> None:
        self.min_overlap_ratio = min_overlap_ratio
        self.max_leak_ratio = max_leak_ratio

    def assign(
        self,
        primitives: list[Primitive3D],
        gt_instance_voxel_ids: dict[int, torch.LongTensor],
    ) -> dict[int, int | None]:
        assignments: dict[int, int | None] = {}
        gt_sets = {gt_id: set(int(v) for v in voxels.tolist()) for gt_id, voxels in gt_instance_voxel_ids.items()}
        for primitive in primitives:
            prim_set = set(int(v) for v in primitive.voxel_ids.tolist())
            if not prim_set:
                assignments[primitive.prim_id] = None
                continue
            best_gt = None
            best_overlap = 0.0
            total_overlap = 0.0
            for gt_id, gt_set in gt_sets.items():
                overlap = len(prim_set & gt_set) / max(len(prim_set), 1)
                total_overlap += overlap
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gt = gt_id
            leak = max(total_overlap - best_overlap, 0.0)
            if best_overlap >= self.min_overlap_ratio and leak <= self.max_leak_ratio:
                assignments[primitive.prim_id] = best_gt
            else:
                assignments[primitive.prim_id] = None
        return assignments
