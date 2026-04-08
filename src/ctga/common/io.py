"""IO helpers for datasets, cache, and debug packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_pt(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu")


def save_pt(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(data, target)


def list_sorted_files(path: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    root = Path(path)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in suffixes)
