"""Mask provider abstractions."""

from __future__ import annotations

from ctga.common.types import Mask2D


class MaskProvider:
    def load_masks(self, scene_id: str, frame_id: int) -> list[Mask2D]:
        raise NotImplementedError
