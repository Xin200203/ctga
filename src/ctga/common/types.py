"""Core typed containers used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class FramePacket:
    frame_id: int
    rgb: torch.ByteTensor
    depth: torch.FloatTensor
    pose_c2w: torch.FloatTensor
    K: torch.FloatTensor
    scene_id: str


@dataclass
class Mask2D:
    mask_id: int
    bitmap: torch.BoolTensor
    bbox_xyxy: torch.FloatTensor
    area: int
    score: float
    feat2d_raw: torch.FloatTensor
    depth_median: float
    depth_minmax: torch.FloatTensor


@dataclass
class Primitive3D:
    prim_id: int
    voxel_coords: torch.IntTensor
    voxel_ids: torch.LongTensor
    world_xyz: torch.FloatTensor
    center_xyz: torch.FloatTensor
    bbox_xyzxyz: torch.FloatTensor
    normal_mean: torch.FloatTensor
    color_mean: torch.FloatTensor
    feat3d_raw: torch.FloatTensor
    visible_pixel_count: int


@dataclass
class TrackState:
    track_id: int
    status: str
    voxel_ids: torch.LongTensor
    center_xyz: torch.FloatTensor
    bbox_xyzxyz: torch.FloatTensor
    feat2d_ema: torch.FloatTensor
    feat3d_ema: torch.FloatTensor
    feat_track: torch.FloatTensor
    last_seen: int
    age: int
    miss_count: int
    confidence: float
    keyview_frame_ids: list[int] = field(default_factory=list)
    keyview_scores: list[float] = field(default_factory=list)


@dataclass
class CurrentObjectHypothesis:
    obj_id_local: int
    primitive_ids: list[int]
    support_mask_ids: list[int]
    support_track_ids: list[int]
    voxel_ids: torch.LongTensor
    center_xyz: torch.FloatTensor
    bbox_xyzxyz: torch.FloatTensor
    feat_obj: torch.FloatTensor
