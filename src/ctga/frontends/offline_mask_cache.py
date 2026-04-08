"""Offline mask cache interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ctga.common.io import load_pt
from ctga.common.types import Mask2D


class OfflineMaskCache:
    def __init__(self, cache_root: str) -> None:
        self.cache_root = Path(cache_root)

    def load_masks(self, scene_id: str, frame_id: int) -> list[Mask2D]:
        frame_stem = f"frame_{frame_id:06d}"
        candidates = [
            self.cache_root / scene_id / f"{frame_stem}_masks.pt",
            self.cache_root / "masks" / scene_id / f"{frame_stem}_masks.pt",
        ]
        for path in candidates:
            if path.exists():
                payload = load_pt(path)
                return [self._to_mask(item, i) for i, item in enumerate(payload)]
        return []

    def _to_mask(self, item: Any, idx: int) -> Mask2D:
        if isinstance(item, Mask2D):
            return item
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported mask cache item: {type(item)!r}")
        bitmap = item["bitmap"].bool()
        bbox = item.get("bbox_xyxy", self._bbox_from_bitmap(bitmap))
        feat = item.get("feat2d_raw", torch.zeros(512, dtype=torch.float32))
        depth_minmax = item.get("depth_minmax", torch.zeros(2, dtype=torch.float32))
        return Mask2D(
            mask_id=int(item.get("mask_id", idx)),
            bitmap=bitmap,
            bbox_xyxy=bbox.float(),
            area=int(item.get("area", bitmap.sum().item())),
            score=float(item.get("score", 1.0)),
            feat2d_raw=feat.float(),
            depth_median=float(item.get("depth_median", 0.0)),
            depth_minmax=depth_minmax.float(),
        )

    def _bbox_from_bitmap(self, bitmap: torch.Tensor) -> torch.Tensor:
        coords = torch.nonzero(bitmap)
        if coords.numel() == 0:
            return torch.zeros(4, dtype=torch.float32)
        y0x0 = coords.min(dim=0).values
        y1x1 = coords.max(dim=0).values
        return torch.tensor(
            [y0x0[1], y0x0[0], y1x1[1] + 1, y1x1[0] + 1],
            dtype=torch.float32,
        )
