"""Unary feature builders for object-track association."""

from __future__ import annotations

import torch

from ctga.common.geometry import bbox_iou_3d, center_distance, safe_cosine_similarity
from ctga.common.types import CurrentObjectHypothesis, FramePacket, TrackState


class UnaryFeatureBuilder:
    def build(
        self,
        obj: CurrentObjectHypothesis,
        track: TrackState,
        frame: FramePacket,
    ) -> torch.FloatTensor:
        obj_size = (obj.bbox_xyzxyz[3:] - obj.bbox_xyzxyz[:3]).float()
        trk_size = (track.bbox_xyzxyz[3:] - track.bbox_xyzxyz[:3]).float()
        size_ratio = obj_size / trk_size.clamp_min(1e-6)
        center_dist = center_distance(obj.center_xyz.float(), track.center_xyz.float())
        bbox_iou = bbox_iou_3d(obj.bbox_xyzxyz.float(), track.bbox_xyzxyz.float())
        feat_sim = safe_cosine_similarity(obj.feat_obj.float(), track.feat_track.float())
        last_seen_gap = max(frame.frame_id - track.last_seen, 0)
        hist_support = 1.0 if track.track_id in obj.support_track_ids else 0.0
        return torch.tensor(
            [
                float(bbox_iou),
                float(center_dist),
                float(size_ratio.mean()),
                float(size_ratio.min()),
                float(size_ratio.max()),
                float(feat_sim),
                float(track.confidence),
                float(track.age),
                float(track.miss_count),
                float(last_seen_gap),
                float(hist_support),
                float(obj.center_xyz[2]),
                float(track.center_xyz[2]),
                float(obj_size.mean()),
                float(trk_size.mean()),
                float(len(obj.primitive_ids)),
                float(len(obj.support_mask_ids)),
                float(len(obj.support_track_ids)),
                1.0 if obj.center_xyz[0] < track.center_xyz[0] else -1.0,
                1.0,
            ],
            dtype=torch.float32,
        )
