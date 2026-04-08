"""Teacher replay utilities for track-memory supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ctga.common.types import CurrentObjectHypothesis, TrackState


class TeacherTrackReplay:
    def replay(
        self,
        current_objects_per_frame: list[list[CurrentObjectHypothesis]],
        gt_map_per_frame: list[dict[int, int | None]],
    ) -> list[dict]:
        active_tracks: dict[int, TrackState] = {}
        snapshots: list[dict] = []
        for frame_id, (objects, gt_map) in enumerate(zip(current_objects_per_frame, gt_map_per_frame)):
            for obj in objects:
                gt_id = gt_map.get(obj.obj_id_local)
                if gt_id is None:
                    continue
                existing = active_tracks.get(gt_id)
                if existing is None:
                    active_tracks[gt_id] = TrackState(
                        track_id=gt_id,
                        status="active",
                        voxel_ids=obj.voxel_ids.clone().long(),
                        center_xyz=obj.center_xyz.clone().float(),
                        bbox_xyzxyz=obj.bbox_xyzxyz.clone().float(),
                        feat2d_ema=torch.zeros(128, dtype=torch.float32),
                        feat3d_ema=obj.feat_obj[:64].clone().float() if obj.feat_obj.shape[0] >= 64 else F.pad(obj.feat_obj.float(), (0, 64 - obj.feat_obj.shape[0])),
                        feat_track=obj.feat_obj[:128].clone().float() if obj.feat_obj.shape[0] >= 128 else F.pad(obj.feat_obj.float(), (0, 128 - obj.feat_obj.shape[0])),
                        last_seen=frame_id,
                        age=1,
                        miss_count=0,
                        confidence=1.0,
                        keyview_frame_ids=[frame_id],
                        keyview_scores=[1.0],
                    )
                else:
                    existing.voxel_ids = torch.unique(torch.cat([existing.voxel_ids, obj.voxel_ids.long()], dim=0))
                    existing.center_xyz = obj.center_xyz.clone().float()
                    existing.bbox_xyzxyz = obj.bbox_xyzxyz.clone().float()
                    existing.last_seen = frame_id
                    existing.miss_count = 0
                    existing.age += 1
            snapshots.append(
                {
                    "active_tracks_teacher": [track for track in active_tracks.values()],
                    "track_gt_ids": {track_id: track_id for track_id in active_tracks.keys()},
                }
            )
        return snapshots
