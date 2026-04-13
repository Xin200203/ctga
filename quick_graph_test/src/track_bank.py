"""Minimal track-bank interface for the quick graph test."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common_types import CurrentObject, Primitive3D, TrackState


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


def _center_score(center_a: np.ndarray, center_b: np.ndarray, sigma: float) -> float:
    dist = float(np.linalg.norm(center_a.astype(np.float32) - center_b.astype(np.float32)))
    return float(np.exp(-dist / max(float(sigma), 1e-6)))


def _voxel_vote(voxels_a: np.ndarray, voxels_b: np.ndarray) -> float:
    if voxels_a.size == 0 or voxels_b.size == 0:
        return 0.0
    inter = np.intersect1d(voxels_a.astype(np.int64), voxels_b.astype(np.int64), assume_unique=False)
    return float(inter.size / max(int(np.unique(voxels_a).size), 1))


@dataclass
class QuickTrackBankConfig:
    match_weight_iou: float = 0.45
    match_weight_center: float = 0.35
    match_weight_vote: float = 0.20
    center_sigma: float = 0.75
    match_thresh: float = 0.14
    max_miss: int = 3


@dataclass
class QuickTrackBank:
    active: dict[int, TrackState] = field(default_factory=dict)
    dormant: dict[int, TrackState] = field(default_factory=dict)
    next_track_id: int = 0
    cfg: QuickTrackBankConfig = field(default_factory=QuickTrackBankConfig)

    def query_active(self) -> list[TrackState]:
        return sorted(self.active.values(), key=lambda item: item.track_id)

    def update_from_current_objects(
        self,
        current_objects: list[CurrentObject],
        primitives: list[Primitive3D],
        frame_id: int,
    ) -> dict[str, object]:
        primitive_lookup = {primitive.prim_id: primitive for primitive in primitives}
        active_tracks = self.query_active()
        candidate_pairs: list[tuple[float, int, int]] = []
        for obj_idx, current_object in enumerate(current_objects):
            for track_idx, track in enumerate(active_tracks):
                score = self._match_score(current_object, track)
                if score >= self.cfg.match_thresh:
                    candidate_pairs.append((score, obj_idx, track_idx))
        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        matched_objects: set[int] = set()
        matched_tracks: set[int] = set()
        assignments: dict[int, int] = {}
        match_scores: dict[int, float] = {}
        for score, obj_idx, track_idx in candidate_pairs:
            if obj_idx in matched_objects or track_idx in matched_tracks:
                continue
            matched_objects.add(obj_idx)
            matched_tracks.add(track_idx)
            track_id = active_tracks[track_idx].track_id
            assignments[obj_idx] = track_id
            match_scores[obj_idx] = float(score)

        new_active: dict[int, TrackState] = {}
        matched_track_ids = {active_tracks[idx].track_id for idx in matched_tracks}
        for obj_idx, current_object in enumerate(current_objects):
            track_id = assignments.get(obj_idx)
            confidence = match_scores.get(obj_idx, 0.55)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                confidence = 0.55
            new_active[track_id] = self._track_from_object(
                track_id=track_id,
                current_object=current_object,
                primitive_lookup=primitive_lookup,
                frame_id=frame_id,
                confidence=confidence,
                previous=self.active.get(track_id),
            )

        for track in active_tracks:
            if track.track_id in matched_track_ids:
                continue
            updated = TrackState(
                track_id=track.track_id,
                voxel_ids=track.voxel_ids.copy(),
                center_xyz=track.center_xyz.copy(),
                bbox_xyzxyz=track.bbox_xyzxyz.copy(),
                feat_color_mean=track.feat_color_mean.copy(),
                last_seen=track.last_seen,
                age=track.age,
                miss_count=track.miss_count + 1,
                confidence=max(track.confidence * 0.90, 0.05),
                status="active" if track.miss_count + 1 <= self.cfg.max_miss else "dormant",
            )
            if updated.status == "active":
                new_active[track.track_id] = updated
            else:
                self.dormant[track.track_id] = updated

        self.active = new_active
        return {
            "num_active": len(self.active),
            "matched_pairs": [
                {
                    "obj_id_local": int(current_objects[obj_idx].obj_id_local),
                    "track_id": int(track_id),
                    "score": float(match_scores[obj_idx]),
                }
                for obj_idx, track_id in assignments.items()
            ],
            "newborn_track_ids": [
                int(track.track_id)
                for track in self.active.values()
                if track.age == 1 and track.last_seen == frame_id
            ],
        }

    def _match_score(self, current_object: CurrentObject, track: TrackState) -> float:
        bbox_iou = _bbox_iou_3d(current_object.bbox_xyzxyz, track.bbox_xyzxyz)
        center_sim = _center_score(current_object.center_xyz, track.center_xyz, self.cfg.center_sigma)
        vote = _voxel_vote(current_object.voxel_ids, track.voxel_ids)
        return (
            self.cfg.match_weight_iou * bbox_iou
            + self.cfg.match_weight_center * center_sim
            + self.cfg.match_weight_vote * vote
        )

    def _track_from_object(
        self,
        track_id: int,
        current_object: CurrentObject,
        primitive_lookup: dict[int, Primitive3D],
        frame_id: int,
        confidence: float,
        previous: TrackState | None,
    ) -> TrackState:
        color_values = [
            primitive_lookup[prim_id].color_mean.astype(np.float32)
            for prim_id in current_object.primitive_ids
            if prim_id in primitive_lookup
        ]
        feat_color_mean = (
            np.stack(color_values, axis=0).mean(axis=0).astype(np.float32)
            if color_values
            else np.zeros((3,), dtype=np.float32)
        )
        age = 1 if previous is None else previous.age + 1
        return TrackState(
            track_id=int(track_id),
            voxel_ids=current_object.voxel_ids.astype(np.int64),
            center_xyz=current_object.center_xyz.astype(np.float32),
            bbox_xyzxyz=current_object.bbox_xyzxyz.astype(np.float32),
            feat_color_mean=feat_color_mean,
            last_seen=int(frame_id),
            age=int(age),
            miss_count=0,
            confidence=float(confidence),
            status="active",
        )
