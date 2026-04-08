"""Visualize layer-2 association graph and compatibility."""

from __future__ import annotations

import argparse

from ctga.common.io import load_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved graph2 debug packet.")
    parser.add_argument("--packet", required=True)
    args = parser.parse_args()
    packet = load_pt(args.packet)
    assoc_graph = packet.get("assoc_graph")
    if assoc_graph is None:
        print("No assoc_graph found in packet.")
        return
    print(
        {
            "unary_edges": int(assoc_graph.unary_index.shape[1]),
            "obj_edges": int(assoc_graph.obj_edge_index.shape[1]),
            "trk_edges": int(assoc_graph.trk_edge_index.shape[1]),
        }
    )


if __name__ == "__main__":
    main()
