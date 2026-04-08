"""Pairwise relation feature builders for graph matching."""

from __future__ import annotations

import torch

from ctga.common.geometry import angle_to_camera, bbox_iou_3d, relative_order_scalar
from ctga.common.types import CurrentObjectHypothesis, FramePacket, TrackState


class RelationFeatureBuilder:
    def build_object_relation(
        self,
        obj_a: CurrentObjectHypothesis,
        obj_b: CurrentObjectHypothesis,
        frame: FramePacket,
    ) -> torch.FloatTensor:
        return self._build_relation(obj_a.center_xyz, obj_a.bbox_xyzxyz, obj_b.center_xyz, obj_b.bbox_xyzxyz, frame)

    def build_track_relation(
        self,
        trk_a: TrackState,
        trk_b: TrackState,
        frame: FramePacket,
    ) -> torch.FloatTensor:
        return self._build_relation(trk_a.center_xyz, trk_a.bbox_xyzxyz, trk_b.center_xyz, trk_b.bbox_xyzxyz, frame)

    def _build_relation(
        self,
        center_a: torch.Tensor,
        bbox_a: torch.Tensor,
        center_b: torch.Tensor,
        bbox_b: torch.Tensor,
        frame: FramePacket,
    ) -> torch.FloatTensor:
        delta = (center_b - center_a).float()
        size_a = (bbox_a[3:] - bbox_a[:3]).float()
        size_b = (bbox_b[3:] - bbox_b[:3]).float()
        log_ratio = torch.log(size_b.clamp_min(1e-6) / size_a.clamp_min(1e-6))
        overlap = bbox_iou_3d(bbox_a.float(), bbox_b.float())
        support_contact = 1.0 if abs(float(bbox_a[5] - bbox_b[2])) < 0.05 or abs(float(bbox_b[5] - bbox_a[2])) < 0.05 else 0.0
        feat = torch.tensor(
            [
                float(delta[0]),
                float(delta[1]),
                float(delta[2]),
                float(log_ratio[0]),
                float(log_ratio[1]),
                float(log_ratio[2]),
                float(center_b[1] - center_a[1]),
                float(support_contact),
                float(overlap),
                float(relative_order_scalar(float(center_a[0]), float(center_b[0]))),
                float(relative_order_scalar(float(center_a[2]), float(center_b[2]))),
                float(angle_to_camera(center_a.float(), frame.pose_c2w.float())),
                float(angle_to_camera(center_b.float(), frame.pose_c2w.float())),
            ],
            dtype=torch.float32,
        )
        if feat.shape[0] < 32:
            feat = torch.cat([feat, torch.zeros(32 - feat.shape[0], dtype=torch.float32)], dim=0)
        return feat[:32]
