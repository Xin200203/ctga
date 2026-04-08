"""Build 3D primitives from the active frustum."""

from __future__ import annotations

from collections import defaultdict

import torch

from ctga.common.geometry import bbox_from_points, voxelize
from ctga.common.types import Primitive3D


class PrimitiveBuilder:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.voxel_size = float(self.cfg.get("voxel_size", 0.05))
        self.color_merge_threshold = float(self.cfg.get("color_merge_threshold", 0.25))
        self.min_points_per_primitive = int(self.cfg.get("min_points_per_primitive", 1))

    def build(
        self,
        active_points_xyz: torch.FloatTensor,
        active_points_rgb: torch.FloatTensor,
        active_voxel_ids: torch.LongTensor,
    ) -> list[Primitive3D]:
        if active_points_xyz.numel() == 0:
            return []

        voxel_coords = voxelize(active_points_xyz, self.voxel_size)
        point_by_voxel: dict[int, list[int]] = defaultdict(list)
        coord_by_voxel: dict[int, torch.Tensor] = {}
        for idx, voxel_id in enumerate(active_voxel_ids.tolist()):
            point_by_voxel[voxel_id].append(idx)
            coord_by_voxel[voxel_id] = voxel_coords[idx]

        clusters = self._cluster_voxels(point_by_voxel, coord_by_voxel, active_points_rgb)
        primitives: list[Primitive3D] = []
        for prim_id, cluster_voxel_ids in enumerate(clusters):
            point_indices: list[int] = []
            coords = []
            for voxel_id in cluster_voxel_ids:
                point_indices.extend(point_by_voxel[voxel_id])
                coords.append(coord_by_voxel[voxel_id])
            if len(point_indices) < self.min_points_per_primitive:
                continue

            idx_t = torch.tensor(point_indices, dtype=torch.long)
            xyz = active_points_xyz[idx_t]
            rgb = active_points_rgb[idx_t]
            bbox = bbox_from_points(xyz)
            feat = torch.cat(
                [
                    xyz.mean(dim=0),
                    xyz.std(dim=0).nan_to_num(0.0),
                    rgb.mean(dim=0),
                    rgb.std(dim=0).nan_to_num(0.0),
                    (bbox[3:] - bbox[:3]),
                    torch.tensor([float(idx_t.numel())], dtype=torch.float32),
                ]
            )
            primitives.append(
                Primitive3D(
                    prim_id=prim_id,
                    voxel_coords=torch.stack(coords, dim=0).to(torch.int32),
                    voxel_ids=torch.tensor(cluster_voxel_ids, dtype=torch.long),
                    world_xyz=xyz.float(),
                    center_xyz=xyz.mean(dim=0).float(),
                    bbox_xyzxyz=bbox.float(),
                    normal_mean=self._estimate_normal(xyz),
                    color_mean=rgb.mean(dim=0).float(),
                    feat3d_raw=feat.float(),
                    visible_pixel_count=int(idx_t.numel()),
                )
            )
        return primitives

    def _cluster_voxels(
        self,
        point_by_voxel: dict[int, list[int]],
        coord_by_voxel: dict[int, torch.Tensor],
        rgb: torch.Tensor,
    ) -> list[list[int]]:
        voxel_ids = list(point_by_voxel.keys())
        parent = {voxel_id: voxel_id for voxel_id in voxel_ids}

        coord_to_voxel = {tuple(int(v.item()) for v in coord): voxel_id for voxel_id, coord in coord_by_voxel.items()}
        mean_color = {
            voxel_id: rgb[torch.tensor(indices, dtype=torch.long)].mean(dim=0)
            for voxel_id, indices in point_by_voxel.items()
        }

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for voxel_id, coord in coord_by_voxel.items():
            cx, cy, cz = (int(v.item()) for v in coord)
            for ncoord in (
                (cx + 1, cy, cz),
                (cx - 1, cy, cz),
                (cx, cy + 1, cz),
                (cx, cy - 1, cz),
                (cx, cy, cz + 1),
                (cx, cy, cz - 1),
            ):
                other = coord_to_voxel.get(ncoord)
                if other is None:
                    continue
                color_delta = torch.linalg.norm(mean_color[voxel_id] - mean_color[other]).item()
                if color_delta <= self.color_merge_threshold:
                    union(voxel_id, other)

        groups: dict[int, list[int]] = defaultdict(list)
        for voxel_id in voxel_ids:
            groups[find(voxel_id)].append(voxel_id)
        return list(groups.values())

    def _estimate_normal(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[0] < 3:
            return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        centered = xyz - xyz.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(xyz.shape[0] - 1, 1)
        _, _, vh = torch.linalg.svd(cov)
        normal = vh[-1]
        return normal / torch.linalg.norm(normal).clamp_min(1e-6)
