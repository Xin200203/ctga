"""Dump per-frame debug packets."""

from __future__ import annotations

import argparse

from ctga.common.io import load_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved debug packet file.")
    parser.add_argument("--packet", required=True)
    args = parser.parse_args()
    packet = load_pt(args.packet)
    print(sorted(packet.keys()))


if __name__ == "__main__":
    main()
