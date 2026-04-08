"""Replay sequences and cache primitives."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder entry for primitive preprocessing.")
    parser.add_argument("--data-root", required=True)
    parser.parse_args()
    print("Primitive preprocessing should call ActiveMap and PrimitiveBuilder over a replayed sequence.")


if __name__ == "__main__":
    main()
