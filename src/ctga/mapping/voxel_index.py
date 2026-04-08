"""Voxel indexing helpers."""

from __future__ import annotations


def coord_to_key(coord: tuple[int, int, int]) -> str:
    return f"{coord[0]}:{coord[1]}:{coord[2]}"
