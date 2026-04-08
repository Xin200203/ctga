"""GT projection helpers."""

from __future__ import annotations

import torch

from ctga.common.geometry import project_points


def project_gt_points(points_world: torch.Tensor, pose_c2w: torch.Tensor, K: torch.Tensor):
    return project_points(points_world, pose_c2w, K)
