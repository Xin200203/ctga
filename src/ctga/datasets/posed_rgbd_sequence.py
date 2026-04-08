"""Dataset definition for posed RGB-D sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ctga.common.io import list_sorted_files
from ctga.common.types import FramePacket


class PosedRGBDSequence(Dataset):
    """Dataset scaffold matching the ESAM-style custom data layout."""

    def __init__(
        self,
        root: str,
        interval: int = 1,
        depth_scale: float = 1000.0,
        scene_id: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.interval = max(interval, 1)
        self.depth_scale = depth_scale
        self.scene_id = scene_id or self.root.name

        self.color_files = list_sorted_files(self.root / "color", (".png", ".jpg", ".jpeg"))
        self.depth_files = list_sorted_files(self.root / "depth", (".png", ".npy"))
        self.pose_files = list_sorted_files(self.root / "pose", (".txt", ".npy"))

        if not (self.color_files and self.depth_files and self.pose_files):
            raise FileNotFoundError(
                f"Expected color/depth/pose folders under {self.root}, but files are missing."
            )

        length = min(len(self.color_files), len(self.depth_files), len(self.pose_files))
        self.indices = list(range(0, length, self.interval))
        self.K = self._load_intrinsic(self.root / "intrinsic.txt")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> FramePacket:
        frame_id = self.indices[idx]
        color = self._load_color(self.color_files[frame_id])
        depth = self._load_depth(self.depth_files[frame_id])
        pose = self._load_pose(self.pose_files[frame_id])
        return FramePacket(
            frame_id=frame_id,
            rgb=color,
            depth=depth,
            pose_c2w=pose,
            K=self.K.clone(),
            scene_id=self.scene_id,
        )

    def _load_intrinsic(self, path: Path) -> torch.FloatTensor:
        matrix = np.loadtxt(path).astype(np.float32)
        matrix = matrix.reshape(3, 3)
        return torch.from_numpy(matrix)

    def _load_color(self, path: Path) -> torch.ByteTensor:
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(image)

    def _load_depth(self, path: Path) -> torch.FloatTensor:
        if path.suffix.lower() == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.array(Image.open(path), dtype=np.float32) / self.depth_scale
        return torch.from_numpy(depth)

    def _load_pose(self, path: Path) -> torch.FloatTensor:
        if path.suffix.lower() == ".npy":
            pose = np.load(path).astype(np.float32)
        else:
            pose = np.loadtxt(path).astype(np.float32)
        return torch.from_numpy(pose.reshape(4, 4))
