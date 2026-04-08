"""Geometry helpers for projection, overlap, and spatial queries."""

from __future__ import annotations

import math

import torch


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=keepdim).clamp_min(1e-8)


def safe_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    min_dim = min(a.shape[-1], b.shape[-1])
    if min_dim == 0:
        return torch.tensor(0.0, device=a.device if a.numel() else b.device)
    a_cut = a[..., :min_dim]
    b_cut = b[..., :min_dim]
    return torch.sum(a_cut * b_cut, dim=-1) / (safe_norm(a_cut) * safe_norm(b_cut))


def bbox_from_points(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return torch.zeros(6, dtype=torch.float32)
    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    return torch.cat([mins, maxs], dim=0)


def bbox_size(bbox_xyzxyz: torch.Tensor) -> torch.Tensor:
    return (bbox_xyzxyz[3:] - bbox_xyzxyz[:3]).clamp_min(1e-6)


def bbox_volume(bbox_xyzxyz: torch.Tensor) -> torch.Tensor:
    return bbox_size(bbox_xyzxyz).prod()


def bbox_iou_3d(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    inter_min = torch.maximum(box_a[:3], box_b[:3])
    inter_max = torch.minimum(box_a[3:], box_b[3:])
    inter_size = (inter_max - inter_min).clamp_min(0.0)
    inter_vol = inter_size.prod()
    union = bbox_volume(box_a) + bbox_volume(box_b) - inter_vol
    return inter_vol / union.clamp_min(1e-6)


def bbox_iou_2d(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    inter_min = torch.maximum(box_a[:2], box_b[:2])
    inter_max = torch.minimum(box_a[2:], box_b[2:])
    inter_size = (inter_max - inter_min).clamp_min(0.0)
    inter_area = inter_size.prod()
    area_a = ((box_a[2:] - box_a[:2]).clamp_min(0.0)).prod()
    area_b = ((box_b[2:] - box_b[:2]).clamp_min(0.0)).prod()
    union = area_a + area_b - inter_area
    return inter_area / union.clamp_min(1e-6)


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    inter = torch.logical_and(mask_a, mask_b).sum(dtype=torch.float32)
    union = torch.logical_or(mask_a, mask_b).sum(dtype=torch.float32)
    return inter / union.clamp_min(1.0)


def containment_ratio(container: torch.Tensor, containee: torch.Tensor) -> torch.Tensor:
    inter = torch.logical_and(container, containee).sum(dtype=torch.float32)
    denom = containee.sum(dtype=torch.float32).clamp_min(1.0)
    return inter / denom


def world_to_camera(points_world: torch.Tensor, pose_c2w: torch.Tensor) -> torch.Tensor:
    pose_w2c = torch.linalg.inv(pose_c2w)
    ones = torch.ones((points_world.shape[0], 1), dtype=points_world.dtype, device=points_world.device)
    points_h = torch.cat([points_world, ones], dim=1)
    cam_h = points_h @ pose_w2c.T
    return cam_h[:, :3]


def camera_to_world(points_camera: torch.Tensor, pose_c2w: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points_camera.shape[0], 1), dtype=points_camera.dtype, device=points_camera.device)
    points_h = torch.cat([points_camera, ones], dim=1)
    world_h = points_h @ pose_c2w.T
    return world_h[:, :3]


def project_points(
    points_world: torch.Tensor,
    pose_c2w: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cam = world_to_camera(points_world, pose_c2w)
    z = cam[:, 2].clamp_min(1e-6)
    uvw = cam @ K.T
    uv = uvw[:, :2] / z.unsqueeze(1)
    return uv, z


def points_in_image(
    uv: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    return (
        (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )


def unproject_depth(
    depth: torch.Tensor,
    K: torch.Tensor,
    pose_c2w: torch.Tensor,
    stride: int = 4,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = depth.shape
    ys = torch.arange(0, h, stride, device=depth.device)
    xs = torch.arange(0, w, stride, device=depth.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sampled_depth = depth[grid_y, grid_x]
    valid = (sampled_depth > min_depth) & (sampled_depth < max_depth)
    if valid.sum() == 0:
        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 2), dtype=torch.long)

    x = grid_x[valid].float()
    y = grid_y[valid].float()
    z = sampled_depth[valid].float()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x3 = (x - cx) * z / fx
    y3 = (y - cy) * z / fy
    points_cam = torch.stack([x3, y3, z], dim=1)
    points_world = camera_to_world(points_cam, pose_c2w)
    pixels = torch.stack([y.long(), x.long()], dim=1)
    return points_world, pixels


def voxelize(points_world: torch.Tensor, voxel_size: float) -> torch.Tensor:
    if points_world.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.int32, device=points_world.device)
    return torch.floor(points_world / voxel_size).to(torch.int32)


def voxel_neighbors(coord: torch.Tensor) -> torch.Tensor:
    deltas = torch.tensor(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        dtype=coord.dtype,
        device=coord.device,
    )
    return coord.unsqueeze(0) + deltas


def distance_decay(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-distance / max(sigma, 1e-6))


def center_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a - b)


def relative_order_scalar(a: float, b: float) -> float:
    return 1.0 if a < b else -1.0 if a > b else 0.0


def angle_to_camera(point_world: torch.Tensor, pose_c2w: torch.Tensor) -> torch.Tensor:
    cam_pos = pose_c2w[:3, 3]
    view = point_world - cam_pos
    forward = pose_c2w[:3, 2]
    cosine = torch.dot(view, forward) / (safe_norm(view, dim=0) * safe_norm(forward, dim=0))
    return torch.arccos(cosine.clamp(-1.0, 1.0)) / math.pi
