"""Sequence IO for posed RGB-D quick tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .common_types import FramePacket


def _sorted_files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in suffixes)


def _numeric_stem(path: Path, fallback: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else fallback


class PosedRGBDSequence:
    """Read a short posed RGB-D sequence from ESAM-style folders."""

    def __init__(
        self,
        scene_root: str | Path,
        interval: int = 1,
        depth_scale: float = 1000.0,
        scene_id: str | None = None,
    ) -> None:
        self.scene_root = Path(scene_root)
        self.interval = max(int(interval), 1)
        self.depth_scale = float(depth_scale)
        self.scene_id = scene_id or self.scene_root.name

        self.color_files = _sorted_files(self.scene_root / "color", (".png", ".jpg", ".jpeg"))
        self.depth_files = _sorted_files(self.scene_root / "depth", (".png", ".npy"))
        self.pose_files = _sorted_files(self.scene_root / "pose", (".txt", ".npy"))
        if not (self.color_files and self.depth_files and self.pose_files):
            raise FileNotFoundError(
                f"Expected color/depth/pose folders under {self.scene_root}, but one or more are missing."
            )

        self.length = min(len(self.color_files), len(self.depth_files), len(self.pose_files))
        self.indices = list(range(0, self.length, self.interval))
        self.K = self._load_intrinsic(self.scene_root / "intrinsic.txt")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> FramePacket:
        raw_index = self.indices[index]
        color_path = self.color_files[raw_index]
        depth_path = self.depth_files[raw_index]
        pose_path = self.pose_files[raw_index]
        return FramePacket(
            frame_id=_numeric_stem(color_path, raw_index),
            rgb=self._load_color(color_path),
            depth=self._load_depth(depth_path),
            pose_c2w=self._load_pose(pose_path),
            K=self.K.copy(),
            scene_id=self.scene_id,
        )

    def iter_frames(self):
        for index in range(len(self)):
            yield self[index]

    def _load_intrinsic(self, path: Path) -> np.ndarray:
        matrix = np.loadtxt(path, dtype=np.float32)
        matrix = matrix.reshape(-1, matrix.shape[-1])
        if matrix.shape == (3, 3):
            return matrix
        if matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
            return matrix[:3, :3]
        raise ValueError(f"Unsupported intrinsic shape at {path}: {matrix.shape}")

    def _load_color(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _load_depth(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.asarray(Image.open(path), dtype=np.float32) / self.depth_scale
        return depth

    def _load_pose(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            pose = np.load(path).astype(np.float32)
        else:
            pose = np.loadtxt(path, dtype=np.float32)
        return pose.reshape(4, 4)
