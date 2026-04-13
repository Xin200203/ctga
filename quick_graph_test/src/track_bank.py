"""Minimal track-bank interface for the quick graph test."""

from __future__ import annotations

from dataclasses import dataclass, field

from .common_types import CurrentObject, TrackState


@dataclass
class QuickTrackBank:
    active: dict[int, TrackState] = field(default_factory=dict)
    dormant: dict[int, TrackState] = field(default_factory=dict)
    next_track_id: int = 0

    def query_active(self):
        return sorted(self.active.values(), key=lambda item: item.track_id)

    def update_from_matches(self, *args, **kwargs):
        raise NotImplementedError("Task 4 / Task 5 will implement track updates.")
