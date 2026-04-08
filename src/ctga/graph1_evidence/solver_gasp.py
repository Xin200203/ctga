"""Wrapper around GASP-style signed graph partition solvers."""

from __future__ import annotations

import torch

from ctga.graph1_evidence.signed_graph import SignedPrimitiveGraph


class GaspPartitionSolver:
    def __init__(self, merge_margin: float = 0.05, cannot_link_margin: float = 0.15) -> None:
        self.merge_margin = merge_margin
        self.cannot_link_margin = cannot_link_margin

    def solve(self, signed_graph: SignedPrimitiveGraph) -> torch.LongTensor:
        num_prims = signed_graph.prim_feat.shape[0]
        if num_prims == 0:
            return torch.zeros(0, dtype=torch.long)
        if signed_graph.prim_edge_index.numel() == 0:
            return torch.arange(num_prims, dtype=torch.long)

        parent = list(range(num_prims))
        cannot_link: set[tuple[int, int]] = set()

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        ranked_edges = []
        for edge_id in range(signed_graph.prim_edge_index.shape[1]):
            i = int(signed_graph.prim_edge_index[0, edge_id].item())
            j = int(signed_graph.prim_edge_index[1, edge_id].item())
            pos = float(signed_graph.weight_pos[edge_id].item())
            neg = float(signed_graph.weight_neg[edge_id].item())
            margin = pos - neg
            ranked_edges.append((margin, i, j, pos, neg))
            if neg - pos >= self.cannot_link_margin:
                cannot_link.add((min(i, j), max(i, j)))

        ranked_edges.sort(reverse=True, key=lambda item: item[0])
        for margin, i, j, _, _ in ranked_edges:
            if margin < self.merge_margin:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if (min(ri, rj), max(ri, rj)) in cannot_link or (min(i, j), max(i, j)) in cannot_link:
                continue
            union(i, j)

        root_to_cluster: dict[int, int] = {}
        cluster_ids = []
        for idx in range(num_prims):
            root = find(idx)
            if root not in root_to_cluster:
                root_to_cluster[root] = len(root_to_cluster)
            cluster_ids.append(root_to_cluster[root])
        return torch.tensor(cluster_ids, dtype=torch.long)
