"""Track bank scaffold."""

from __future__ import annotations

import torch

from ctga.common.types import CurrentObjectHypothesis, FramePacket, TrackState
from ctga.memory.feature_fusion import TrackFeatureFusion
from ctga.memory.lifecycle import LifecycleManager


class TrackBank:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.feat2d_dim = int(self.cfg.get("feat2d_dim", 128))
        self.feat3d_dim = int(self.cfg.get("feat3d_dim", 64))
        self.track_dim = int(self.cfg.get("track_dim", 128))
        self.use_reactivation = bool(self.cfg.get("use_reactivation", True))
        self.active: dict[int, TrackState] = {}
        self.dormant: dict[int, TrackState] = {}
        self.dead: dict[int, TrackState] = {}
        self.next_track_id = 0
        self.feature_fusion = TrackFeatureFusion(feat2d_dim=self.feat2d_dim)
        self.lifecycle = LifecycleManager(
            dormant_after=int(self.cfg.get("dormant_after", 10)),
            dead_after=int(self.cfg.get("dead_after", 30)),
        )

    def query_active(self, frame: FramePacket) -> list[TrackState]:
        tracks = list(self.active.values())
        if self.use_reactivation:
            tracks.extend(self.dormant.values())
        return sorted(tracks, key=lambda item: item.track_id)

    def apply_assignments(
        self,
        frame_id: int,
        matched: dict[int, int],
        current_objects: list[CurrentObjectHypothesis],
    ) -> None:
        matched_track_ids = set()
        for obj_local_id, track_id in matched.items():
            obj = current_objects[obj_local_id]
            if track_id == -1 or track_id not in self.active and track_id not in self.dormant:
                self.create_newborn(frame_id, [obj])
                continue

            track = self.active.pop(track_id, None)
            if track is None:
                track = self.dormant.pop(track_id)
            track = self.feature_fusion.update(track, obj)
            track.last_seen = frame_id
            track.miss_count = 0
            track.status = "active"
            self.active[track.track_id] = track
            matched_track_ids.add(track.track_id)

        matched_obj_ids = set(matched.keys())
        newborn_objects = [obj for obj in current_objects if obj.obj_id_local not in matched_obj_ids]
        if newborn_objects:
            self.create_newborn(frame_id, newborn_objects)

        for track_id, track in list(self.active.items()):
            if track_id not in matched_track_ids:
                self.active[track_id] = self.lifecycle.step(track, frame_id)

    def step_lifecycle(self, frame_id: int) -> None:
        for track_id, track in list(self.active.items()):
            updated = self.lifecycle.step(track, frame_id)
            if updated.status == "dormant":
                self.dormant[track_id] = updated
                del self.active[track_id]
            elif updated.status == "dead":
                self.dead[track_id] = updated
                del self.active[track_id]
        for track_id, track in list(self.dormant.items()):
            updated = self.lifecycle.step(track, frame_id)
            if updated.status == "dead":
                self.dead[track_id] = updated
                del self.dormant[track_id]

    def create_newborn(self, frame_id: int, newborn_objects: list[CurrentObjectHypothesis]) -> None:
        for obj in newborn_objects:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.active[track_id] = TrackState(
                track_id=track_id,
                status="active",
                voxel_ids=torch.unique(obj.voxel_ids.long()),
                center_xyz=obj.center_xyz.float(),
                bbox_xyzxyz=obj.bbox_xyzxyz.float(),
                feat2d_ema=torch.zeros(self.feat2d_dim, dtype=torch.float32),
                feat3d_ema=self._fit_dim(obj.feat_obj.float(), self.feat3d_dim),
                feat_track=self._fit_dim(obj.feat_obj.float(), self.track_dim),
                last_seen=frame_id,
                age=1,
                miss_count=0,
                confidence=0.5,
                keyview_frame_ids=[frame_id],
                keyview_scores=[1.0],
            )

    def _fit_dim(self, feat: torch.Tensor, dim: int) -> torch.Tensor:
        if feat.shape[0] >= dim:
            return feat[:dim].float()
        return torch.cat([feat.float(), torch.zeros(dim - feat.shape[0], dtype=torch.float32)], dim=0)
