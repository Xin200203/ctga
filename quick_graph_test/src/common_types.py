"""Shared dataclasses for the quick graph test prototype."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    pose_c2w: np.ndarray
    K: np.ndarray
    scene_id: str


@dataclass
class Mask2D:
    mask_id: int
    bitmap: np.ndarray
    bbox_xyxy: np.ndarray
    area: int
    score: float
    feat2d: np.ndarray | None = None


@dataclass
class Primitive3D:
    prim_id: int
    pixel_idx: np.ndarray
    xyz: np.ndarray
    voxel_ids: np.ndarray
    center_xyz: np.ndarray
    bbox_xyzxyz: np.ndarray
    normal_mean: np.ndarray
    color_mean: np.ndarray
    support_mask_ids: list[int] = field(default_factory=list)


@dataclass
class TrackState:
    track_id: int
    voxel_ids: np.ndarray
    center_xyz: np.ndarray
    bbox_xyzxyz: np.ndarray
    feat_color_mean: np.ndarray
    last_seen: int
    age: int
    miss_count: int
    confidence: float
    status: str


@dataclass
class CurrentObject:
    obj_id_local: int
    primitive_ids: list[int]
    support_mask_ids: list[int]
    support_track_ids: list[int]
    voxel_ids: np.ndarray
    center_xyz: np.ndarray
    bbox_xyzxyz: np.ndarray
