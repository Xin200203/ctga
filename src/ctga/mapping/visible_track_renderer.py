"""Render active tracks into the current frame for visibility support."""

from __future__ import annotations

import torch

from ctga.common.geometry import project_points
from ctga.common.types import FramePacket, TrackState


class VisibleTrackRenderer:
    def render(
        self,
        tracks: list[TrackState],
        frame: FramePacket,
    ) -> dict[int, torch.BoolTensor]:
        height, width = frame.depth.shape
        rendered: dict[int, torch.BoolTensor] = {}
        for track in tracks:
            mask = torch.zeros((height, width), dtype=torch.bool)
            corners = self._bbox_corners(track.bbox_xyzxyz)
            uv, depth = project_points(corners, frame.pose_c2w, frame.K)
            valid = depth > 0
            if valid.sum() == 0:
                rendered[track.track_id] = mask
                continue
            uv = uv[valid]
            x0 = int(torch.floor(uv[:, 0].min()).clamp(0, width - 1).item())
            x1 = int(torch.ceil(uv[:, 0].max()).clamp(0, width - 1).item())
            y0 = int(torch.floor(uv[:, 1].min()).clamp(0, height - 1).item())
            y1 = int(torch.ceil(uv[:, 1].max()).clamp(0, height - 1).item())
            if x1 >= x0 and y1 >= y0:
                mask[y0 : y1 + 1, x0 : x1 + 1] = True
            rendered[track.track_id] = mask
        return rendered

    def _bbox_corners(self, bbox_xyzxyz: torch.Tensor) -> torch.Tensor:
        x0, y0, z0, x1, y1, z1 = bbox_xyzxyz.tolist()
        return torch.tensor(
            [
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z0],
                [x1, y1, z1],
            ],
            dtype=torch.float32,
        )
