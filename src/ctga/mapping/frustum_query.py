"""Convenience wrapper for active-frustum queries."""

from __future__ import annotations

from ctga.common.types import FramePacket
from ctga.mapping.active_map import ActiveMap


def query_active_frustum(active_map: ActiveMap, frame: FramePacket):
    return active_map.query_active_frustum(frame)
