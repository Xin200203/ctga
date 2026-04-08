"""Feature fusion for track state updates."""

from __future__ import annotations

import torch

from ctga.common.types import CurrentObjectHypothesis, TrackState


class TrackFeatureFusion:
    def __init__(self, ema_momentum: float = 0.9, feat2d_dim: int = 128) -> None:
        self.ema_momentum = ema_momentum
        self.feat2d_dim = feat2d_dim

    def update(self, track: TrackState, obj: CurrentObjectHypothesis) -> TrackState:
        feat_obj = obj.feat_obj.float()
        feat3d = self._match_dim(feat_obj, track.feat3d_ema.shape[0])
        feat_track = self._match_dim(feat_obj, track.feat_track.shape[0])
        feat2d = torch.zeros(track.feat2d_ema.shape[0] if track.feat2d_ema.numel() else self.feat2d_dim, dtype=torch.float32)

        track.voxel_ids = torch.unique(torch.cat([track.voxel_ids.long(), obj.voxel_ids.long()], dim=0))
        track.center_xyz = obj.center_xyz.float()
        track.bbox_xyzxyz = obj.bbox_xyzxyz.float()
        track.feat2d_ema = self._ema(track.feat2d_ema.float(), feat2d)
        track.feat3d_ema = self._ema(track.feat3d_ema.float(), feat3d)
        track.feat_track = self._ema(track.feat_track.float(), feat_track)
        track.confidence = min(1.0, 0.5 * track.confidence + 0.5)
        return track

    def _ema(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old.numel() == 0:
            return new.float()
        return self.ema_momentum * old + (1.0 - self.ema_momentum) * new

    def _match_dim(self, feat: torch.Tensor, dim: int) -> torch.Tensor:
        if feat.shape[0] >= dim:
            return feat[:dim].float()
        return torch.cat([feat.float(), torch.zeros(dim - feat.shape[0], dtype=torch.float32)], dim=0)
