"""Diagnostics and debug-packet helpers."""

from __future__ import annotations

from typing import Any

import torch

from ctga.common.io import save_pt


def no_nan(tensor: torch.Tensor) -> bool:
    return not torch.isnan(tensor).any().item() if tensor.numel() else True


def every_current_object_has_primitives(current_objects: list[Any]) -> bool:
    return all(len(obj.primitive_ids) > 0 for obj in current_objects)


def dump_debug_packet(path: str, packet: dict[str, Any]) -> None:
    save_pt(path, packet)
