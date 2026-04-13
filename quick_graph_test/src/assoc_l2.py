"""Unary current-to-memory matching for the quick graph test."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy may be unavailable locally
    linear_sum_assignment = None

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


def _color_similarity(color_a: np.ndarray, color_b: np.ndarray, sigma: float) -> float:
    dist = float(np.linalg.norm(color_a.astype(np.float32) - color_b.astype(np.float32)))
    return float(np.exp(-dist / max(float(sigma), 1e-6)))


def _bbox_volume(box: np.ndarray) -> float:
    side = np.maximum(box[3:] - box[:3], 0.0)
    return float(np.prod(side))


@dataclass
class UnaryAssocConfig:
    geom_weight_iou: float = 0.45
    geom_weight_center: float = 0.35
    geom_weight_vote: float = 0.20
    geom_center_sigma: float = 0.75

    unary_weight_geom: float = 0.50
    unary_weight_app: float = 0.30
    unary_weight_hist: float = 0.20
    app_color_sigma: float = 45.0

    gating_max_center_dist: float = 1.80
    gating_min_size_ratio: float = 0.12
    gating_max_size_ratio: float = 8.00
    top_k: int = 5
    match_thresh: float = 0.18


@dataclass
class UnaryAssociationResult:
    score_matrix: np.ndarray
    candidate_mask: np.ndarray
    assignments: dict[int, int] = field(default_factory=dict)  # obj_idx -> track_col
    assigned_track_ids: dict[int, int] = field(default_factory=dict)  # obj_idx -> track_id
    match_scores: dict[int, float] = field(default_factory=dict)  # obj_idx -> unary score
    solver: str = "none"

    @property
    def unmatched_object_indices(self) -> list[int]:
        num_objects = int(self.score_matrix.shape[0])
        return [idx for idx in range(num_objects) if idx not in self.assignments]

    def unmatched_track_indices(self, num_tracks: int) -> list[int]:
        matched = set(self.assignments.values())
        return [idx for idx in range(num_tracks) if idx not in matched]


class UnaryAssociator:
    def __init__(self, cfg: UnaryAssocConfig | None = None) -> None:
        self.cfg = cfg or UnaryAssocConfig()

    def match(
        self,
        current_objects: list[CurrentObject],
        active_tracks: list[TrackState],
        primitives: list[Primitive3D],
    ) -> UnaryAssociationResult:
        num_objects = len(current_objects)
        num_tracks = len(active_tracks)
        score_matrix = np.zeros((num_objects, num_tracks), dtype=np.float32)
        candidate_mask = np.zeros((num_objects, num_tracks), dtype=bool)

        if num_objects == 0 or num_tracks == 0:
            return UnaryAssociationResult(
                score_matrix=score_matrix,
                candidate_mask=candidate_mask,
                solver="none",
            )

        primitive_lookup = {primitive.prim_id: primitive for primitive in primitives}
        object_colors = [self._object_color_mean(obj, primitive_lookup) for obj in current_objects]
        track_id_to_col = {track.track_id: idx for idx, track in enumerate(active_tracks)}

        for obj_idx, current_object in enumerate(current_objects):
            valid_cols: list[int] = []
            raw_scores: list[tuple[float, int]] = []
            for track_idx, track in enumerate(active_tracks):
                geom = self._geom_score(current_object, track)
                app = _color_similarity(object_colors[obj_idx], track.feat_color_mean, self.cfg.app_color_sigma)
                hist = 1.0 if int(track.track_id) in set(current_object.support_track_ids) else 0.0
                score = (
                    self.cfg.unary_weight_geom * geom
                    + self.cfg.unary_weight_app * app
                    + self.cfg.unary_weight_hist * hist
                )
                score_matrix[obj_idx, track_idx] = float(score)
                if self._passes_gating(current_object, track):
                    valid_cols.append(track_idx)
                    raw_scores.append((float(score), track_idx))

            raw_scores.sort(key=lambda item: item[0], reverse=True)
            for _, track_idx in raw_scores[: max(int(self.cfg.top_k), 0)]:
                candidate_mask[obj_idx, track_idx] = True

            for support_track_id in current_object.support_track_ids:
                track_idx = track_id_to_col.get(int(support_track_id))
                if track_idx is not None and self._passes_gating(current_object, active_tracks[track_idx]):
                    candidate_mask[obj_idx, track_idx] = True

            # If top-k was empty but a gated track exists, keep the best gated one.
            if not np.any(candidate_mask[obj_idx]) and raw_scores:
                candidate_mask[obj_idx, raw_scores[0][1]] = True

        assignments, match_scores, solver = self._solve(score_matrix=score_matrix, candidate_mask=candidate_mask)
        assigned_track_ids = {
            obj_idx: int(active_tracks[track_idx].track_id)
            for obj_idx, track_idx in assignments.items()
        }
        return UnaryAssociationResult(
            score_matrix=score_matrix,
            candidate_mask=candidate_mask,
            assignments=assignments,
            assigned_track_ids=assigned_track_ids,
            match_scores=match_scores,
            solver=solver,
        )

    def _geom_score(self, current_object: CurrentObject, track: TrackState) -> float:
        bbox_iou = _bbox_iou_3d(current_object.bbox_xyzxyz, track.bbox_xyzxyz)
        center_sim = _center_score(current_object.center_xyz, track.center_xyz, self.cfg.geom_center_sigma)
        vote = _voxel_vote(current_object.voxel_ids, track.voxel_ids)
        return (
            self.cfg.geom_weight_iou * bbox_iou
            + self.cfg.geom_weight_center * center_sim
            + self.cfg.geom_weight_vote * vote
        )

    def _passes_gating(self, current_object: CurrentObject, track: TrackState) -> bool:
        center_dist = float(np.linalg.norm(current_object.center_xyz - track.center_xyz))
        if center_dist > self.cfg.gating_max_center_dist:
            return False

        obj_vol = _bbox_volume(current_object.bbox_xyzxyz)
        track_vol = _bbox_volume(track.bbox_xyzxyz)
        size_ratio = max(obj_vol, 1e-6) / max(track_vol, 1e-6)
        if size_ratio < self.cfg.gating_min_size_ratio or size_ratio > self.cfg.gating_max_size_ratio:
            return False
        return True

    def _object_color_mean(
        self,
        current_object: CurrentObject,
        primitive_lookup: dict[int, Primitive3D],
    ) -> np.ndarray:
        colors = [
            primitive_lookup[prim_id].color_mean.astype(np.float32)
            for prim_id in current_object.primitive_ids
            if prim_id in primitive_lookup
        ]
        if not colors:
            return np.zeros((3,), dtype=np.float32)
        return np.stack(colors, axis=0).mean(axis=0).astype(np.float32)

    def _solve(
        self,
        score_matrix: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[dict[int, int], dict[int, float], str]:
        if linear_sum_assignment is not None:
            return self._solve_hungarian(score_matrix=score_matrix, candidate_mask=candidate_mask)
        return self._solve_greedy(score_matrix=score_matrix, candidate_mask=candidate_mask)

    def _solve_hungarian(
        self,
        score_matrix: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[dict[int, int], dict[int, float], str]:
        num_objects, num_tracks = score_matrix.shape
        dummy_cost = 1.0 - float(self.cfg.match_thresh)
        invalid_cost = 2.0
        cost = np.full((num_objects, num_tracks + num_objects), dummy_cost, dtype=np.float32)
        for obj_idx in range(num_objects):
            for track_idx in range(num_tracks):
                if candidate_mask[obj_idx, track_idx]:
                    cost[obj_idx, track_idx] = 1.0 - float(np.clip(score_matrix[obj_idx, track_idx], 0.0, 1.0))
                else:
                    cost[obj_idx, track_idx] = invalid_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        assignments: dict[int, int] = {}
        match_scores: dict[int, float] = {}
        for obj_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            if col_idx >= num_tracks:
                continue
            if not candidate_mask[obj_idx, col_idx]:
                continue
            score = float(score_matrix[obj_idx, col_idx])
            if score < self.cfg.match_thresh:
                continue
            assignments[int(obj_idx)] = int(col_idx)
            match_scores[int(obj_idx)] = score
        return assignments, match_scores, "hungarian"

    def _solve_greedy(
        self,
        score_matrix: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[dict[int, int], dict[int, float], str]:
        num_objects, num_tracks = score_matrix.shape
        pairs: list[tuple[float, int, int]] = []
        for obj_idx in range(num_objects):
            for track_idx in range(num_tracks):
                if not candidate_mask[obj_idx, track_idx]:
                    continue
                score = float(score_matrix[obj_idx, track_idx])
                if score < self.cfg.match_thresh:
                    continue
                pairs.append((score, obj_idx, track_idx))
        pairs.sort(key=lambda item: item[0], reverse=True)

        assignments: dict[int, int] = {}
        match_scores: dict[int, float] = {}
        matched_objects: set[int] = set()
        matched_tracks: set[int] = set()
        for score, obj_idx, track_idx in pairs:
            if obj_idx in matched_objects or track_idx in matched_tracks:
                continue
            matched_objects.add(obj_idx)
            matched_tracks.add(track_idx)
            assignments[int(obj_idx)] = int(track_idx)
            match_scores[int(obj_idx)] = float(score)
        return assignments, match_scores, "greedy"
