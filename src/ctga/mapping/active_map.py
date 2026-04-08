"""Active map scaffold for online integration and frustum queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctga.common.geometry import points_in_image, project_points, unproject_depth, voxelize
from ctga.common.types import FramePacket


@dataclass
class _VoxelRecord:
    voxel_id: int
    coord: tuple[int, int, int]
    xyz_sum: torch.Tensor
    rgb_sum: torch.Tensor
    count: int
    last_seen: int


class ActiveMap:
    def __init__(
        self,
        voxel_size: float,
        integration_stride: int = 4,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
    ) -> None:
        self.voxel_size = voxel_size
        self.integration_stride = integration_stride
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._coord_to_id: dict[tuple[int, int, int], int] = {}
        self._voxels: dict[int, _VoxelRecord] = {}
        self._next_voxel_id = 0

    def integrate(self, frame: FramePacket) -> None:
        world_xyz, pixels = unproject_depth(
            frame.depth,
            frame.K,
            frame.pose_c2w,
            stride=self.integration_stride,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        if world_xyz.numel() == 0:
            return

        rgb = frame.rgb[pixels[:, 0], pixels[:, 1]].float() / 255.0
        voxel_coords = voxelize(world_xyz, self.voxel_size)

        for coord_t, point, color in zip(voxel_coords, world_xyz, rgb):
            coord = tuple(int(v.item()) for v in coord_t)
            voxel_id = self._coord_to_id.get(coord)
            if voxel_id is None:
                voxel_id = self._next_voxel_id
                self._next_voxel_id += 1
                self._coord_to_id[coord] = voxel_id
                self._voxels[voxel_id] = _VoxelRecord(
                    voxel_id=voxel_id,
                    coord=coord,
                    xyz_sum=point.clone(),
                    rgb_sum=color.clone(),
                    count=1,
                    last_seen=frame.frame_id,
                )
                continue

            record = self._voxels[voxel_id]
            record.xyz_sum += point
            record.rgb_sum += color
            record.count += 1
            record.last_seen = frame.frame_id

    def query_active_frustum(self, frame: FramePacket) -> dict[str, torch.Tensor]:
        if not self._voxels:
            empty_long = torch.zeros(0, dtype=torch.long)
            empty_int = torch.zeros((0, 3), dtype=torch.int32)
            empty_float = torch.zeros((0, 3), dtype=torch.float32)
            return {
                "active_voxel_ids": empty_long,
                "active_voxel_coords": empty_int,
                "active_points_xyz": empty_float,
                "active_points_rgb": empty_float,
            }

        voxel_ids = []
        voxel_coords = []
        points_xyz = []
        points_rgb = []
        for voxel_id, record in self._voxels.items():
            mean_xyz = record.xyz_sum / max(record.count, 1)
            mean_rgb = record.rgb_sum / max(record.count, 1)
            voxel_ids.append(voxel_id)
            voxel_coords.append(record.coord)
            points_xyz.append(mean_xyz)
            points_rgb.append(mean_rgb)

        voxel_ids_t = torch.tensor(voxel_ids, dtype=torch.long)
        voxel_coords_t = torch.tensor(voxel_coords, dtype=torch.int32)
        points_xyz_t = torch.stack(points_xyz, dim=0).float()
        points_rgb_t = torch.stack(points_rgb, dim=0).float()

        uv, depth = project_points(points_xyz_t, frame.pose_c2w, frame.K)
        height, width = frame.depth.shape
        valid = points_in_image(uv, depth, height, width)

        return {
            "active_voxel_ids": voxel_ids_t[valid],
            "active_voxel_coords": voxel_coords_t[valid],
            "active_points_xyz": points_xyz_t[valid],
            "active_points_rgb": points_rgb_t[valid],
        }
