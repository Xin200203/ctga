"""Track lifecycle transitions."""

from __future__ import annotations

from ctga.common.types import TrackState


class LifecycleManager:
    def __init__(self, dormant_after: int = 10, dead_after: int = 30) -> None:
        self.dormant_after = dormant_after
        self.dead_after = dead_after

    def step(self, track: TrackState, frame_id: int) -> TrackState:
        track.age += 1
        if track.last_seen < frame_id:
            track.miss_count += 1
            track.confidence *= 0.98
        if track.miss_count >= self.dead_after:
            track.status = "dead"
        elif track.miss_count >= self.dormant_after:
            track.status = "dormant"
        else:
            track.status = "active"
        return track
