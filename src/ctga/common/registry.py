"""Simple string-to-object registry."""

from __future__ import annotations


class Registry(dict):
    def register(self, name: str, value) -> None:
        self[name] = value

    def get_or_raise(self, name: str):
        if name not in self:
            raise KeyError(f"{name!r} is not registered.")
        return self[name]
