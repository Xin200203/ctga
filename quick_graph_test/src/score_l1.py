"""Layer-1 hand-crafted edge scoring for the quick graph test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common_types import FramePacket, Mask2D, Primitive3D, TrackState


@dataclass
class Layer1Config:
    mp_weight_cover: float = 0.40
    mp_weight_contain: float = 0.30
    mp_weight_depth: float = 0.20
    mp_weight_color: float = 0.10
    mp_depth_sigma: float = 0.20
    mp_color_sigma: float = 45.0
    mp_positive_thresh: float = 0.10

    pp_weight_adj: float = 0.50
    pp_weight_normal: float = 0.30
    pp_weight_color: float = 0.20
    pp_gap_sigma: float = 0.08
    pp_color_sigma: float = 40.0
    pp_positive_thresh: float = 0.10
    pp_max_center_dist: float = 1.20

    pt_weight_vote: float = 0.50
    pt_weight_bbox: float = 0.30
    pt_weight_center: float = 0.20
    pt_center_sigma: float = 0.60
    pt_positive_thresh: float = 0.08

    lambda_pp: float = 0.45
    lambda_mask: float = 0.75
    lambda_track: float = 0.55
    lambda_mask_conflict: float = 0.85
    lambda_track_conflict: float = 1.10

    strong_mp_thresh: float = 0.35
    strong_pt_thresh: float = 0.20


@dataclass
class Layer1Graph:
    mp_scores: np.ndarray
    pp_scores: np.ndarray
    pt_scores: np.ndarray
    positive_weights: np.ndarray
    negative_weights: np.ndarray
    merge_scores: np.ndarray
    top_mask_ids: np.ndarray
    top_mask_scores: np.ndarray
    top_track_ids: np.ndarray
    top_track_scores: np.ndarray


def _color_similarity(color_a: np.ndarray, color_b: np.ndarray, sigma: float) -> float:
    dist = float(np.linalg.norm(color_a.astype(np.float32) - color_b.astype(np.float32)))
    return float(np.exp(-dist / max(float(sigma), 1e-6)))


def _bbox_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    mins = np.maximum(box_a[:3], box_b[:3])
    maxs = np.minimum(box_a[3:], box_b[3:])
    overlap = np.maximum(maxs - mins, 0.0)
    inter = float(np.prod(overlap))
    if inter <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.maximum(box_a[3:] - box_a[:3], 0.0)))
    vol_b = float(np.prod(np.maximum(box_b[3:] - box_b[:3], 0.0)))
    return inter / max(vol_a + vol_b - inter, 1e-6)


def _bbox_gap_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    gap = np.maximum(np.maximum(box_a[:3] - box_b[3:], box_b[:3] - box_a[3:]), 0.0)
    return float(np.linalg.norm(gap))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter <= 0:
        return 0.0
    union = int(mask_a.sum()) + int(mask_b.sum()) - inter
    return inter / max(union, 1)


def _center_score(center_a: np.ndarray, center_b: np.ndarray, sigma: float) -> float:
    dist = float(np.linalg.norm(center_a.astype(np.float32) - center_b.astype(np.float32)))
    return float(np.exp(-dist / max(float(sigma), 1e-6)))


class Layer1Scorer:
    def __init__(self, cfg: Layer1Config | None = None) -> None:
        self.cfg = cfg or Layer1Config()

    def build_graph(
        self,
        frame: FramePacket,
        masks: list[Mask2D],
        primitives: list[Primitive3D],
        active_tracks: list[TrackState] | None = None,
    ) -> Layer1Graph:
        active_tracks = active_tracks or []
        num_masks = len(masks)
        num_prims = len(primitives)
        num_tracks = len(active_tracks)

        mp_scores = np.zeros((num_masks, num_prims), dtype=np.float32)
        pp_scores = np.zeros((num_prims, num_prims), dtype=np.float32)
        pt_scores = np.zeros((num_prims, num_tracks), dtype=np.float32)

        mask_depth_mean = np.zeros((num_masks,), dtype=np.float32)
        mask_color_mean = np.zeros((num_masks, 3), dtype=np.float32)
        for mask_idx, mask in enumerate(masks):
            bitmap = mask.bitmap.astype(bool)
            valid = bitmap & (frame.depth > 0)
            if np.any(valid):
                mask_depth_mean[mask_idx] = float(frame.depth[valid].mean())
                mask_color_mean[mask_idx] = frame.rgb[valid].astype(np.float32).mean(axis=0)

        prim_depth_mean = np.zeros((num_prims,), dtype=np.float32)
        for prim_idx, primitive in enumerate(primitives):
            coords = primitive.pixel_idx.astype(np.int32)
            if coords.size == 0:
                continue
            ys = np.clip(coords[:, 0], 0, frame.depth.shape[0] - 1)
            xs = np.clip(coords[:, 1], 0, frame.depth.shape[1] - 1)
            prim_depth_mean[prim_idx] = float(frame.depth[ys, xs].mean())

        for mask_idx, mask in enumerate(masks):
            bitmap = mask.bitmap.astype(bool)
            for prim_idx, primitive in enumerate(primitives):
                coords = primitive.pixel_idx.astype(np.int32)
                if coords.size == 0:
                    continue
                ys = np.clip(coords[:, 0], 0, bitmap.shape[0] - 1)
                xs = np.clip(coords[:, 1], 0, bitmap.shape[1] - 1)
                inside = bitmap[ys, xs]
                inter = int(inside.sum())
                if inter <= 0:
                    continue
                prim_area = max(int(coords.shape[0]), 1)
                cover = inter / prim_area
                contain = inter / max(int(mask.area), 1)
                depth_delta = abs(float(mask_depth_mean[mask_idx]) - float(prim_depth_mean[prim_idx]))
                depth_cons = float(np.exp(-depth_delta / max(self.cfg.mp_depth_sigma, 1e-6)))
                color_sim = _color_similarity(mask_color_mean[mask_idx], primitive.color_mean, self.cfg.mp_color_sigma)
                score = (
                    self.cfg.mp_weight_cover * cover
                    + self.cfg.mp_weight_contain * contain
                    + self.cfg.mp_weight_depth * depth_cons
                    + self.cfg.mp_weight_color * color_sim
                )
                if mask.mask_id in primitive.support_mask_ids:
                    score = min(1.0, score + 0.15)
                mp_scores[mask_idx, prim_idx] = float(score)

        for prim_idx_a, primitive_a in enumerate(primitives):
            for prim_idx_b in range(prim_idx_a + 1, num_prims):
                primitive_b = primitives[prim_idx_b]
                center_dist = float(np.linalg.norm(primitive_a.center_xyz - primitive_b.center_xyz))
                if center_dist > self.cfg.pp_max_center_dist:
                    continue
                gap = _bbox_gap_3d(primitive_a.bbox_xyzxyz, primitive_b.bbox_xyzxyz)
                adj = float(np.exp(-gap / max(self.cfg.pp_gap_sigma, 1e-6)))
                normal_sim = float(
                    np.clip(np.dot(primitive_a.normal_mean, primitive_b.normal_mean), -1.0, 1.0)
                )
                normal_sim = (normal_sim + 1.0) * 0.5
                color_sim = _color_similarity(primitive_a.color_mean, primitive_b.color_mean, self.cfg.pp_color_sigma)
                score = (
                    self.cfg.pp_weight_adj * adj
                    + self.cfg.pp_weight_normal * normal_sim
                    + self.cfg.pp_weight_color * color_sim
                )
                pp_scores[prim_idx_a, prim_idx_b] = float(score)
                pp_scores[prim_idx_b, prim_idx_a] = float(score)

        for prim_idx, primitive in enumerate(primitives):
            for track_idx, track in enumerate(active_tracks):
                vote = self._voxel_vote(primitive.voxel_ids, track.voxel_ids)
                bbox_iou = _bbox_iou_3d(primitive.bbox_xyzxyz, track.bbox_xyzxyz)
                center_sim = _center_score(primitive.center_xyz, track.center_xyz, self.cfg.pt_center_sigma)
                score = (
                    self.cfg.pt_weight_vote * vote
                    + self.cfg.pt_weight_bbox * bbox_iou
                    + self.cfg.pt_weight_center * center_sim
                )
                pt_scores[prim_idx, track_idx] = float(score)

        top_mask_ids = np.full((num_prims,), -1, dtype=np.int32)
        top_mask_scores = np.zeros((num_prims,), dtype=np.float32)
        if num_masks > 0 and num_prims > 0:
            top_mask_choice = mp_scores.argmax(axis=0)
            top_mask_ids = np.array([masks[idx].mask_id for idx in top_mask_choice], dtype=np.int32)
            top_mask_scores = mp_scores[top_mask_choice, np.arange(num_prims)].astype(np.float32)

        top_track_ids = np.full((num_prims,), -1, dtype=np.int32)
        top_track_scores = np.zeros((num_prims,), dtype=np.float32)
        if num_tracks > 0 and num_prims > 0:
            top_track_choice = pt_scores.argmax(axis=1)
            top_track_ids = np.array([active_tracks[idx].track_id for idx in top_track_choice], dtype=np.int32)
            top_track_scores = pt_scores[np.arange(num_prims), top_track_choice].astype(np.float32)

        positive = np.zeros((num_prims, num_prims), dtype=np.float32)
        negative = np.zeros((num_prims, num_prims), dtype=np.float32)

        for prim_idx_a in range(num_prims):
            for prim_idx_b in range(prim_idx_a + 1, num_prims):
                same_mask_support = float(np.dot(mp_scores[:, prim_idx_a], mp_scores[:, prim_idx_b])) if num_masks else 0.0
                same_track_support = float(np.dot(pt_scores[prim_idx_a], pt_scores[prim_idx_b])) if num_tracks else 0.0
                pos = (
                    self.cfg.lambda_pp * pp_scores[prim_idx_a, prim_idx_b]
                    + self.cfg.lambda_mask * same_mask_support
                    + self.cfg.lambda_track * same_track_support
                )
                neg = (
                    self.cfg.lambda_mask_conflict * self._mask_conflict(
                        prim_idx_a,
                        prim_idx_b,
                        masks,
                        top_mask_ids,
                        top_mask_scores,
                    )
                    + self.cfg.lambda_track_conflict * self._track_conflict(
                        prim_idx_a,
                        prim_idx_b,
                        active_tracks,
                        top_track_ids,
                        top_track_scores,
                    )
                )
                positive[prim_idx_a, prim_idx_b] = positive[prim_idx_b, prim_idx_a] = float(pos)
                negative[prim_idx_a, prim_idx_b] = negative[prim_idx_b, prim_idx_a] = float(neg)

        merge_scores = positive - negative
        np.fill_diagonal(positive, 0.0)
        np.fill_diagonal(negative, 0.0)
        np.fill_diagonal(merge_scores, 0.0)

        return Layer1Graph(
            mp_scores=mp_scores,
            pp_scores=pp_scores,
            pt_scores=pt_scores,
            positive_weights=positive,
            negative_weights=negative,
            merge_scores=merge_scores,
            top_mask_ids=top_mask_ids,
            top_mask_scores=top_mask_scores,
            top_track_ids=top_track_ids,
            top_track_scores=top_track_scores,
        )

    def _voxel_vote(self, prim_voxels: np.ndarray, track_voxels: np.ndarray) -> float:
        if prim_voxels.size == 0 or track_voxels.size == 0:
            return 0.0
        inter = np.intersect1d(prim_voxels.astype(np.int64), track_voxels.astype(np.int64), assume_unique=False)
        return float(inter.size / max(int(np.unique(prim_voxels).size), 1))

    def _mask_conflict(
        self,
        prim_idx_a: int,
        prim_idx_b: int,
        masks: list[Mask2D],
        top_mask_ids: np.ndarray,
        top_mask_scores: np.ndarray,
    ) -> float:
        if not masks:
            return 0.0
        mask_id_a = int(top_mask_ids[prim_idx_a])
        mask_id_b = int(top_mask_ids[prim_idx_b])
        if mask_id_a < 0 or mask_id_b < 0 or mask_id_a == mask_id_b:
            return 0.0
        score_a = float(top_mask_scores[prim_idx_a])
        score_b = float(top_mask_scores[prim_idx_b])
        if score_a < self.cfg.strong_mp_thresh or score_b < self.cfg.strong_mp_thresh:
            return 0.0
        mask_lookup = {mask.mask_id: mask for mask in masks}
        if mask_id_a not in mask_lookup or mask_id_b not in mask_lookup:
            return 0.0
        iou = _mask_iou(mask_lookup[mask_id_a].bitmap.astype(bool), mask_lookup[mask_id_b].bitmap.astype(bool))
        return score_a * score_b * (1.0 - iou)

    def _track_conflict(
        self,
        prim_idx_a: int,
        prim_idx_b: int,
        active_tracks: list[TrackState],
        top_track_ids: np.ndarray,
        top_track_scores: np.ndarray,
    ) -> float:
        if not active_tracks:
            return 0.0
        track_id_a = int(top_track_ids[prim_idx_a])
        track_id_b = int(top_track_ids[prim_idx_b])
        if track_id_a < 0 or track_id_b < 0 or track_id_a == track_id_b:
            return 0.0
        score_a = float(top_track_scores[prim_idx_a])
        score_b = float(top_track_scores[prim_idx_b])
        if score_a < self.cfg.strong_pt_thresh or score_b < self.cfg.strong_pt_thresh:
            return 0.0
        track_lookup = {track.track_id: track for track in active_tracks}
        if track_id_a not in track_lookup or track_id_b not in track_lookup:
            return 0.0
        iou = _bbox_iou_3d(track_lookup[track_id_a].bbox_xyzxyz, track_lookup[track_id_b].bbox_xyzxyz)
        return score_a * score_b * (1.0 - iou)
