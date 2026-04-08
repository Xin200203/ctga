"""Replay a sequence through the online pipeline."""

from __future__ import annotations

import argparse

from ctga.common.config import load_config
from ctga.datasets.posed_rgbd_sequence import PosedRGBDSequence
from ctga.inference.online_engine import OnlineEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a sequence through the CTGA online engine.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = OnlineEngine(cfg)
    sequence = PosedRGBDSequence(args.data_root)
    for frame in sequence:
        engine.step(frame)
    print("Replay finished.")


if __name__ == "__main__":
    main()
