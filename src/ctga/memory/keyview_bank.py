"""Keyview utilities for stable track visualization."""

from __future__ import annotations


class KeyviewBank:
    def add(self, track_id: int, frame_id: int, score: float) -> tuple[int, int, float]:
        return track_id, frame_id, score
