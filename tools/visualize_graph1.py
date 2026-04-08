"""Visualize layer-1 graph structure and scores."""

from __future__ import annotations

import argparse

from ctga.common.io import load_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved graph1 debug packet.")
    parser.add_argument("--packet", required=True)
    args = parser.parse_args()
    packet = load_pt(args.packet)
    print({
        "mp_edges": int(packet["graph1"].mp_edge_index.shape[1]) if "graph1" in packet else 0,
        "pt_edges": int(packet["graph1"].pt_edge_index.shape[1]) if "graph1" in packet else 0,
        "mt_edges": int(packet["graph1"].mt_edge_index.shape[1]) if "graph1" in packet else 0,
        "pp_edges": int(packet["graph1"].pp_edge_index.shape[1]) if "graph1" in packet else 0,
    })


if __name__ == "__main__":
    main()
