"""NumPy geometry helpers for the quick diagnostic prototype."""

from __future__ import annotations

import math

import numpy as np


def bbox_from_points(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros(6, dtype=np.float32)
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    return np.concatenate([mins, maxs], axis=0).astype(np.float32)


def bbox_from_mask(bitmap: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(bitmap)
    if ys.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    pose_c2w: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    if valid_mask is None:
        valid_mask = depth > 0
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    z = depth[ys, xs].astype(np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    points_world = (points_cam @ pose_c2w.T)[:, :3]
    pixels = np.stack([ys, xs], axis=1).astype(np.int32)
    return points_world.astype(np.float32), pixels


def project_points(points_world: np.ndarray, pose_c2w: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    pose_w2c = np.linalg.inv(pose_c2w)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    points_cam = (points_h @ pose_w2c.T)[:, :3]
    z = np.clip(points_cam[:, 2], 1e-6, None)
    uvw = points_cam @ K.T
    uv = uvw[:, :2] / z[:, None]
    return uv.astype(np.float32), z.astype(np.float32)


def voxelize(points_world: np.ndarray, voxel_size: float) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.floor(points_world / float(voxel_size)).astype(np.int32)


def points_camera_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.indices((h, w))
    z = depth.astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y = (ys.astype(np.float32) - cy) * z / max(fy, 1e-6)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def estimate_normals_from_depth(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-pixel normals in camera coordinates.

    Returns:
        normals: [H, W, 3]
        valid_mask: [H, W] bool
    """

    points = points_camera_from_depth(depth, K)
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)

    if h < 3 or w < 3:
        return normals, valid

    dx = points[:, 2:, :] - points[:, :-2, :]
    dy = points[2:, :, :] - points[:-2, :, :]
    cross = np.cross(dx[1:-1, :, :], dy[:, 1:-1, :])
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    stable = (
        (depth[1:-1, 1:-1] > 0)
        & (depth[1:-1, :-2] > 0)
        & (depth[1:-1, 2:] > 0)
        & (depth[:-2, 1:-1] > 0)
        & (depth[2:, 1:-1] > 0)
        & (norm[..., 0] > 1e-6)
    )
    normals_inner = np.zeros_like(cross, dtype=np.float32)
    normals_inner[stable] = cross[stable] / norm[stable]
    normals[1:-1, 1:-1, :] = normals_inner
    valid[1:-1, 1:-1] = stable
    return normals, valid


def normal_angle_deg(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    denom = np.linalg.norm(normal_a) * np.linalg.norm(normal_b)
    if denom <= 1e-8:
        return 180.0
    cosine = float(np.clip(np.dot(normal_a, normal_b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)
