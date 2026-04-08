"""Feature builders for layer-1 edges."""

from __future__ import annotations

import torch

from ctga.common.geometry import (
    bbox_iou_2d,
    bbox_iou_3d,
    center_distance,
    containment_ratio,
    distance_decay,
    mask_iou,
    project_points,
    safe_cosine_similarity,
)
from ctga.common.types import Mask2D, Primitive3D, TrackState


class EdgeFeatureBuilderL1:
    def __init__(self, sigma_depth: float = 0.25, sigma_center: float = 0.5, adjacency_radius: float = 0.5) -> None:
        self.sigma_depth = sigma_depth
        self.sigma_center = sigma_center
        self.adjacency_radius = adjacency_radius

    def build_mp_features(
        self,
        mask: Mask2D,
        primitive: Primitive3D,
        frame_K: torch.Tensor,
        frame_pose: torch.Tensor,
    ) -> torch.FloatTensor:
        proj_mask, proj_bbox, depth_stats = self._project_primitive(primitive, mask.bitmap.shape, frame_pose, frame_K)
        bbox_iou = bbox_iou_2d(mask.bbox_xyxy.float(), proj_bbox)
        overlap = mask_iou(mask.bitmap, proj_mask)
        contain_m2p = containment_ratio(mask.bitmap, proj_mask)
        contain_p2m = containment_ratio(proj_mask, mask.bitmap)
        depth_gap = abs(mask.depth_median - depth_stats[0])
        feat_sim = safe_cosine_similarity(mask.feat2d_raw.float(), primitive.feat3d_raw.float().flatten())
        boundary_penalty = 1.0 - overlap
        return torch.tensor(
            [
                float(overlap),
                float(bbox_iou),
                float(contain_m2p),
                float(contain_p2m),
                float(distance_decay(torch.tensor(depth_gap), self.sigma_depth)),
                float(feat_sim),
                float(boundary_penalty),
                float(mask.score),
                float(mask.area),
                float(depth_gap),
                float(depth_stats[1]),
                float(depth_stats[2]),
                float(proj_mask.sum().item()),
                float(proj_bbox[2] - proj_bbox[0]),
                float(proj_bbox[3] - proj_bbox[1]),
                1.0,
            ],
            dtype=torch.float32,
        )

    def build_pt_features(self, primitive: Primitive3D, track: TrackState) -> torch.FloatTensor:
        vote_ratio = self._voxel_vote_ratio(primitive.voxel_ids, track.voxel_ids)
        bbox_iou = bbox_iou_3d(primitive.bbox_xyzxyz.float(), track.bbox_xyzxyz.float())
        center_dist = center_distance(primitive.center_xyz.float(), track.center_xyz.float())
        size_ratio = (primitive.bbox_xyzxyz[3:] - primitive.bbox_xyzxyz[:3]).prod() / (
            (track.bbox_xyzxyz[3:] - track.bbox_xyzxyz[:3]).prod().clamp_min(1e-6)
        )
        feat_sim = safe_cosine_similarity(primitive.feat3d_raw.float(), track.feat3d_ema.float())
        return torch.tensor(
            [
                float(vote_ratio),
                float(bbox_iou),
                float(distance_decay(center_dist, self.sigma_center)),
                float(center_dist),
                float(size_ratio),
                float(feat_sim),
                float(track.confidence),
                float(track.age),
                float(track.miss_count),
                float(primitive.visible_pixel_count),
                float((primitive.bbox_xyzxyz[3:] - primitive.bbox_xyzxyz[:3]).mean()),
                float((track.bbox_xyzxyz[3:] - track.bbox_xyzxyz[:3]).mean()),
                float(primitive.center_xyz[2]),
                float(track.center_xyz[2]),
                1.0,
                0.0,
            ],
            dtype=torch.float32,
        )

    def build_mt_features(self, mask: Mask2D, rendered_track_mask: torch.Tensor, track: TrackState) -> torch.FloatTensor:
        if rendered_track_mask.numel() == 0:
            rendered_track_mask = torch.zeros_like(mask.bitmap)
        track_bbox = self._bbox_from_bitmap(rendered_track_mask)
        bbox_iou = bbox_iou_2d(mask.bbox_xyxy.float(), track_bbox)
        overlap = mask_iou(mask.bitmap, rendered_track_mask)
        contain_mask = containment_ratio(mask.bitmap, rendered_track_mask)
        contain_track = containment_ratio(rendered_track_mask, mask.bitmap)
        feat_sim = safe_cosine_similarity(mask.feat2d_raw.float(), track.feat2d_ema.float())
        vis_ratio = rendered_track_mask.sum(dtype=torch.float32) / float(mask.bitmap.numel())
        return torch.tensor(
            [
                float(overlap),
                float(bbox_iou),
                float(contain_mask),
                float(contain_track),
                float(feat_sim),
                float(vis_ratio),
                float(track.confidence),
                float(track.age),
                float(track.miss_count),
                float(mask.score),
                float(mask.area),
                1.0,
            ],
            dtype=torch.float32,
        )

    def build_pp_features(self, prim_a: Primitive3D, prim_b: Primitive3D) -> torch.FloatTensor:
        center_dist = center_distance(prim_a.center_xyz.float(), prim_b.center_xyz.float())
        adjacency = 1.0 if center_dist <= self.adjacency_radius else 0.0
        normal_sim = safe_cosine_similarity(prim_a.normal_mean.float(), prim_b.normal_mean.float())
        color_sim = safe_cosine_similarity(prim_a.color_mean.float(), prim_b.color_mean.float())
        bbox_iou = bbox_iou_3d(prim_a.bbox_xyzxyz.float(), prim_b.bbox_xyzxyz.float())
        return torch.tensor(
            [
                float(adjacency),
                float(distance_decay(center_dist, self.sigma_center)),
                float(center_dist),
                float(normal_sim),
                float(color_sim),
                float(bbox_iou),
                float(prim_a.visible_pixel_count),
                float(prim_b.visible_pixel_count),
                float((prim_a.bbox_xyzxyz[3:] - prim_a.bbox_xyzxyz[:3]).mean()),
                float((prim_b.bbox_xyzxyz[3:] - prim_b.bbox_xyzxyz[:3]).mean()),
                1.0 if prim_a.center_xyz[2] < prim_b.center_xyz[2] else -1.0,
                1.0,
            ],
            dtype=torch.float32,
        )

    def _project_primitive(
        self,
        primitive: Primitive3D,
        shape_hw: tuple[int, int],
        pose_c2w: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = shape_hw
        mask = torch.zeros((h, w), dtype=torch.bool)
        if primitive.world_xyz.numel() == 0:
            return mask, torch.zeros(4, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)
        uv, depth = project_points(primitive.world_xyz.float(), pose_c2w.float(), K.float())
        valid = (
            (depth > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        if valid.sum() == 0:
            return mask, torch.zeros(4, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)
        uv = uv[valid]
        depth = depth[valid]
        x = uv[:, 0].long().clamp(0, w - 1)
        y = uv[:, 1].long().clamp(0, h - 1)
        mask[y, x] = True
        bbox = torch.tensor(
            [
                float(x.min()),
                float(y.min()),
                float(x.max() + 1),
                float(y.max() + 1),
            ],
            dtype=torch.float32,
        )
        depth_stats = torch.tensor(
            [
                float(depth.median()),
                float(depth.min()),
                float(depth.max()),
            ],
            dtype=torch.float32,
        )
        return mask, bbox, depth_stats

    def _voxel_vote_ratio(self, voxels_a: torch.Tensor, voxels_b: torch.Tensor) -> float:
        set_a = set(int(v) for v in voxels_a.tolist())
        set_b = set(int(v) for v in voxels_b.tolist())
        if not set_a:
            return 0.0
        return len(set_a & set_b) / max(len(set_a), 1)

    def _bbox_from_bitmap(self, bitmap: torch.Tensor) -> torch.Tensor:
        coords = torch.nonzero(bitmap)
        if coords.numel() == 0:
            return torch.zeros(4, dtype=torch.float32)
        min_yx = coords.min(dim=0).values
        max_yx = coords.max(dim=0).values
        return torch.tensor([min_yx[1], min_yx[0], max_yx[1] + 1, max_yx[0] + 1], dtype=torch.float32)
