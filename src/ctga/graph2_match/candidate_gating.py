"""Candidate pruning before graph matching."""

from __future__ import annotations

from collections import defaultdict

import torch

from ctga.graph2_match.graph_builder import AssociationGraphBatch


class CandidateGater:
    def prune(
        self,
        assoc_graph: AssociationGraphBatch,
        unary_logits: torch.FloatTensor,
        max_candidates_per_obj: int = 5,
        active_radius: float = 2.0,
    ) -> dict[int, list[int]]:
        per_object: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for edge_id in range(assoc_graph.unary_index.shape[1]):
            obj_idx = int(assoc_graph.unary_index[0, edge_id].item())
            trk_idx = int(assoc_graph.unary_index[1, edge_id].item())
            center_dist = float(assoc_graph.unary_feat[edge_id, 1].item())
            if center_dist > active_radius:
                continue
            score = float(torch.sigmoid(unary_logits[edge_id]).item())
            per_object[obj_idx].append((score, trk_idx))

        candidate_map: dict[int, list[int]] = {}
        num_objects = assoc_graph.obj_feat.shape[0]
        for obj_idx in range(num_objects):
            ranked = sorted(per_object.get(obj_idx, []), reverse=True)
            if not ranked:
                candidate_map[obj_idx] = []
                continue
            candidate_map[obj_idx] = [trk for _, trk in ranked[:max_candidates_per_obj]]
        return candidate_map
