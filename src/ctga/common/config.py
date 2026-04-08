"""Configuration entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load YAML configuration files.")

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file and recursively merge any `defaults` entries."""

    config_path = Path(path)
    config = load_yaml(config_path)

    defaults = config.pop("defaults", [])
    if not defaults:
        return config

    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = (config_path.parent / default).resolve()
        merged = _merge_dict(merged, load_config(default_path))
    return _merge_dict(merged, config)
