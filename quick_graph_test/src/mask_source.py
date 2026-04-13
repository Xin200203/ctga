"""Mask cache readers for quick graph tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common_types import Mask2D


def _bbox_from_bitmap(bitmap: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(bitmap)
    if ys.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _as_numpy(array_like: Any) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if hasattr(array_like, "detach") and hasattr(array_like, "cpu"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


class BaseMaskSource:
    def load_masks(self, scene_id: str, frame_id: int, image_shape: tuple[int, int] | None = None) -> list[Mask2D]:
        raise NotImplementedError


class EmptyMaskSource(BaseMaskSource):
    def load_masks(self, scene_id: str, frame_id: int, image_shape: tuple[int, int] | None = None) -> list[Mask2D]:
        return []


class OracleMaskSource(BaseMaskSource):
    def __init__(self, gt_root: str | Path) -> None:
        self.gt_root = Path(gt_root)

    def load_masks(self, scene_id: str, frame_id: int, image_shape: tuple[int, int] | None = None) -> list[Mask2D]:
        raise NotImplementedError(
            "Oracle mask mode is reserved for debug comparisons and depends on the GT file format. "
            "Implement this after the scene-specific GT layout is confirmed."
        )


class CacheMaskSource(BaseMaskSource):
    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)

    def load_masks(self, scene_id: str, frame_id: int, image_shape: tuple[int, int] | None = None) -> list[Mask2D]:
        frame_stem = f"frame_{frame_id:06d}"
        numeric_stem = str(frame_id)
        candidates = [
            self.cache_root / f"{numeric_stem}.pt",
            self.cache_root / f"{numeric_stem}.npz",
            self.cache_root / frame_stem,
            self.cache_root / scene_id / f"{frame_stem}_masks.pt",
            self.cache_root / scene_id / f"{frame_stem}.pt",
            self.cache_root / scene_id / f"{frame_stem}_masks.npz",
            self.cache_root / scene_id / f"{frame_stem}.npz",
            self.cache_root / scene_id / f"{numeric_stem}.pt",
            self.cache_root / scene_id / f"{numeric_stem}.npz",
            self.cache_root / scene_id / "fastsam_masks" / f"{numeric_stem}.pt",
            self.cache_root / scene_id / "fastsam_masks" / f"{numeric_stem}.npz",
            self.cache_root / scene_id / "sam_masks" / f"{numeric_stem}.pt",
            self.cache_root / scene_id / "sam_masks" / f"{numeric_stem}.npz",
            self.cache_root / "masks" / scene_id / f"{frame_stem}_masks.pt",
            self.cache_root / "masks" / scene_id / f"{frame_stem}_masks.npz",
            self.cache_root / "masks" / scene_id / f"{numeric_stem}.pt",
            self.cache_root / "masks" / scene_id / f"{numeric_stem}.npz",
        ]
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix == ".pt":
                return self._load_pt(path)
            if path.suffix == ".npz":
                return self._load_npz(path)
        return []

    def _load_pt(self, path: Path) -> list[Mask2D]:
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "masks" in payload:
                payload = payload["masks"]
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise TypeError(f"Unsupported .pt mask payload type: {type(payload)!r}")
        return [self._to_mask(item, idx) for idx, item in enumerate(payload)]

    def _load_npz(self, path: Path) -> list[Mask2D]:
        payload = np.load(path, allow_pickle=True)
        if "bitmaps" in payload:
            bitmaps = payload["bitmaps"]
        elif "masks" in payload:
            bitmaps = payload["masks"]
        else:
            raise KeyError(f"No 'bitmaps' or 'masks' array found in {path}")

        scores = payload["scores"] if "scores" in payload else np.ones(len(bitmaps), dtype=np.float32)
        feats = payload["feat2d"] if "feat2d" in payload else None
        boxes = payload["boxes"] if "boxes" in payload else None
        masks: list[Mask2D] = []
        for idx, bitmap in enumerate(bitmaps):
            bitmap = _as_numpy(bitmap).astype(bool)
            bbox = _as_numpy(boxes[idx]).astype(np.float32) if boxes is not None else _bbox_from_bitmap(bitmap)
            masks.append(
                Mask2D(
                    mask_id=idx,
                    bitmap=bitmap,
                    bbox_xyxy=bbox,
                    area=int(bitmap.sum()),
                    score=float(scores[idx]),
                    feat2d=_as_numpy(feats[idx]).astype(np.float32) if feats is not None else None,
                )
            )
        return masks

    def _to_mask(self, item: Any, idx: int) -> Mask2D:
        if isinstance(item, Mask2D):
            return item

        if hasattr(item, "__dict__") and not isinstance(item, dict):
            item = vars(item)
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported mask item type: {type(item)!r}")

        bitmap = _as_numpy(item.get("bitmap", item.get("mask"))).astype(bool)
        feat = item.get("feat2d", item.get("feat2d_raw"))
        feat_np = _as_numpy(feat).astype(np.float32) if feat is not None else None
        bbox = item.get("bbox_xyxy")
        bbox_np = _as_numpy(bbox).astype(np.float32) if bbox is not None else _bbox_from_bitmap(bitmap)
        return Mask2D(
            mask_id=int(item.get("mask_id", idx)),
            bitmap=bitmap,
            bbox_xyxy=bbox_np,
            area=int(item.get("area", bitmap.sum())),
            score=float(item.get("score", 1.0)),
            feat2d=feat_np,
        )


def build_mask_source(
    mode: str,
    cache_root: str | Path | None = None,
    gt_root: str | Path | None = None,
) -> BaseMaskSource:
    mode = mode.lower()
    if mode == "cache":
        if cache_root is None:
            raise ValueError("cache_root is required when mask mode is 'cache'.")
        return CacheMaskSource(cache_root)
    if mode == "empty":
        return EmptyMaskSource()
    if mode == "oracle":
        if gt_root is None:
            raise ValueError("gt_root is required when mask mode is 'oracle'.")
        return OracleMaskSource(gt_root)
    raise ValueError(f"Unsupported mask mode: {mode}")
