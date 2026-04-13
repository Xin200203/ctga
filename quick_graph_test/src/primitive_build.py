"""Single-frame primitive over-segmentation.

Task 2 implements a simple pixel-grid over-segmentation using depth, normal,
and color consistency, then lifts each connected region into 3D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .common_types import FramePacket, Primitive3D
from .geometry import bbox_from_points, estimate_normals_from_depth, unproject_depth, voxelize


@dataclass
class PrimitiveConfig:
    tau_z: float = 0.05
    tau_n_deg: float = 30.0
    tau_c: float = 35.0
    connectivity: int = 8
    min_depth: float = 0.1
    max_depth: float = 8.0
    min_pixels: int = 32
    voxel_size: float = 0.05
    max_points_per_primitive: int = 4096
    max_primitives: int = 4096


class PrimitiveBuilder:
    def __init__(self, cfg: PrimitiveConfig | None = None) -> None:
        self.cfg = cfg or PrimitiveConfig()

    def build(self, frame: FramePacket) -> list[Primitive3D]:
        depth = frame.depth.astype(np.float32)
        rgb = frame.rgb.astype(np.uint8)
        valid = (depth > self.cfg.min_depth) & (depth < self.cfg.max_depth)
        normals, normal_valid = estimate_normals_from_depth(depth, frame.K)

        h, w = depth.shape
        num_pixels = h * w
        parent = np.arange(num_pixels, dtype=np.int32)
        rank = np.zeros(num_pixels, dtype=np.int8)

        def index(y: int, x: int) -> int:
            return y * w + x

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        neighbor_offsets = [(0, 1), (1, 0)]
        if self.cfg.connectivity == 8:
            neighbor_offsets.extend([(1, 1), (1, -1)])

        for y in range(h):
            for x in range(w):
                if not valid[y, x]:
                    continue
                idx = index(y, x)
                for dy, dx in neighbor_offsets:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w or not valid[ny, nx]:
                        continue
                    if self._can_connect(
                        depth=depth,
                        rgb=rgb,
                        normals=normals,
                        normal_valid=normal_valid,
                        y0=y,
                        x0=x,
                        y1=ny,
                        x1=nx,
                    ):
                        union(idx, index(ny, nx))

        groups: dict[int, list[int]] = {}
        for y in range(h):
            for x in range(w):
                if not valid[y, x]:
                    continue
                root = find(index(y, x))
                groups.setdefault(root, []).append(index(y, x))

        primitives: list[Primitive3D] = []
        for root_id, flat_indices in groups.items():
            if len(flat_indices) < self.cfg.min_pixels:
                continue
            ys = np.array([flat // w for flat in flat_indices], dtype=np.int32)
            xs = np.array([flat % w for flat in flat_indices], dtype=np.int32)
            pixel_idx = np.stack([ys, xs], axis=1)
            if pixel_idx.shape[0] > self.cfg.max_points_per_primitive:
                choice = np.linspace(0, pixel_idx.shape[0] - 1, self.cfg.max_points_per_primitive).astype(np.int32)
                pixel_idx = pixel_idx[choice]
                ys = pixel_idx[:, 0]
                xs = pixel_idx[:, 1]

            mask = np.zeros((h, w), dtype=bool)
            mask[ys, xs] = True
            xyz, _ = unproject_depth(depth, frame.K, frame.pose_c2w, valid_mask=mask)
            if xyz.shape[0] == 0:
                continue
            voxel_ids = self._voxel_ids(xyz)
            color_mean = rgb[ys, xs].astype(np.float32).mean(axis=0)
            valid_normals = normals[ys, xs][normal_valid[ys, xs]]
            if valid_normals.size == 0:
                normal_mean = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                normal_mean = valid_normals.mean(axis=0).astype(np.float32)
                norm = np.linalg.norm(normal_mean)
                if norm > 1e-6:
                    normal_mean = normal_mean / norm

            primitives.append(
                Primitive3D(
                    prim_id=len(primitives),
                    pixel_idx=pixel_idx.astype(np.int32),
                    xyz=xyz.astype(np.float32),
                    voxel_ids=voxel_ids.astype(np.int64),
                    center_xyz=xyz.mean(axis=0).astype(np.float32),
                    bbox_xyzxyz=bbox_from_points(xyz).astype(np.float32),
                    normal_mean=normal_mean.astype(np.float32),
                    color_mean=color_mean.astype(np.float32),
                )
            )
            if len(primitives) >= self.cfg.max_primitives:
                break

        return primitives

    def _can_connect(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        normals: np.ndarray,
        normal_valid: np.ndarray,
        y0: int,
        x0: int,
        y1: int,
        x1: int,
    ) -> bool:
        if abs(float(depth[y0, x0]) - float(depth[y1, x1])) > self.cfg.tau_z:
            return False
        color_delta = np.linalg.norm(rgb[y0, x0].astype(np.float32) - rgb[y1, x1].astype(np.float32))
        if float(color_delta) > self.cfg.tau_c:
            return False
        if normal_valid[y0, x0] and normal_valid[y1, x1]:
            cosine = float(np.clip(np.dot(normals[y0, x0], normals[y1, x1]), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cosine)))
            if angle_deg > self.cfg.tau_n_deg:
                return False
        return True

    def _voxel_ids(self, xyz: np.ndarray) -> np.ndarray:
        coords = voxelize(xyz.astype(np.float32), self.cfg.voxel_size)
        if coords.size == 0:
            return np.zeros((0,), dtype=np.int64)
        unique_coords = np.unique(coords, axis=0)
        scale = np.array([73856093, 19349663, 83492791], dtype=np.int64)
        return (unique_coords.astype(np.int64) * scale[None, :]).sum(axis=1)
