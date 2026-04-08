"""Assembly of signed primitive graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctga.common.geometry import center_distance
from ctga.graph1_evidence.graph_builder import EvidenceGraphBatch

@dataclass
class SignedPrimitiveGraph:
    prim_edge_index: torch.LongTensor
    weight_pos: torch.FloatTensor
    weight_neg: torch.FloatTensor
    prim_feat: torch.FloatTensor


class SignedGraphAssembler:
    def __init__(
        self,
        lambda_pp: float = 1.0,
        lambda_mask: float = 1.0,
        lambda_track: float = 1.0,
        lambda_conflict_track: float = 1.0,
        lambda_conflict_mask: float = 0.5,
        far_mask_conflict_dist: float = 1.0,
    ) -> None:
        self.lambda_pp = lambda_pp
        self.lambda_mask = lambda_mask
        self.lambda_track = lambda_track
        self.lambda_conflict_track = lambda_conflict_track
        self.lambda_conflict_mask = lambda_conflict_mask
        self.far_mask_conflict_dist = far_mask_conflict_dist

    def assemble(
        self,
        graph: EvidenceGraphBatch,
        edge_logits: dict[str, torch.FloatTensor],
    ) -> SignedPrimitiveGraph:
        num_prims = graph.prim_nodes_feat.shape[0]
        pos = torch.zeros((num_prims, num_prims), dtype=torch.float32)
        neg = torch.zeros((num_prims, num_prims), dtype=torch.float32)

        pp_probs = torch.sigmoid(edge_logits.get("logit_pp", torch.zeros(0)))
        for edge_id in range(graph.pp_edge_index.shape[1]):
            i = int(graph.pp_edge_index[0, edge_id].item())
            j = int(graph.pp_edge_index[1, edge_id].item())
            pos[i, j] += self.lambda_pp * pp_probs[edge_id]
            pos[j, i] = pos[i, j]

        mp_support = self._edge_support_matrix(graph.mp_edge_index, edge_logits.get("logit_mp", torch.zeros(0)), graph.mask_nodes_feat.shape[0], num_prims)
        pt_support = self._edge_support_matrix(graph.pt_edge_index, edge_logits.get("logit_pt", torch.zeros(0)), graph.track_nodes_feat.shape[0], num_prims)

        if mp_support.numel():
            pos += self.lambda_mask * (mp_support.T @ mp_support)
        if pt_support.numel():
            pos += self.lambda_track * (pt_support.T @ pt_support)

        top_track_idx, top_track_score = self._top_support(pt_support)
        top_mask_idx, top_mask_score = self._top_support(mp_support)

        centers = graph.prim_nodes_feat[:, :3] if graph.prim_nodes_feat.numel() else torch.zeros((0, 3))
        for i in range(num_prims):
            for j in range(i + 1, num_prims):
                if top_track_score[i] > 0.5 and top_track_score[j] > 0.5 and top_track_idx[i] != top_track_idx[j]:
                    neg[i, j] += self.lambda_conflict_track * min(top_track_score[i], top_track_score[j])
                if top_mask_score[i] > 0.5 and top_mask_score[j] > 0.5 and top_mask_idx[i] == top_mask_idx[j]:
                    dist = center_distance(centers[i], centers[j])
                    if float(dist) > self.far_mask_conflict_dist:
                        neg[i, j] += self.lambda_conflict_mask
                neg[j, i] = neg[i, j]

        pos.fill_diagonal_(0.0)
        neg.fill_diagonal_(0.0)

        edge_pairs = []
        weight_pos = []
        weight_neg = []
        for i in range(num_prims):
            for j in range(i + 1, num_prims):
                if pos[i, j] > 0 or neg[i, j] > 0:
                    edge_pairs.append([i, j])
                    weight_pos.append(pos[i, j])
                    weight_neg.append(neg[i, j])

        edge_index = (
            torch.tensor(edge_pairs, dtype=torch.long).T.contiguous()
            if edge_pairs
            else torch.zeros((2, 0), dtype=torch.long)
        )
        return SignedPrimitiveGraph(
            prim_edge_index=edge_index,
            weight_pos=torch.stack(weight_pos).float() if weight_pos else torch.zeros(0, dtype=torch.float32),
            weight_neg=torch.stack(weight_neg).float() if weight_neg else torch.zeros(0, dtype=torch.float32),
            prim_feat=graph.prim_nodes_feat.float(),
        )

    def _edge_support_matrix(
        self,
        edge_index: torch.Tensor,
        logits: torch.Tensor,
        num_sources: int,
        num_prims: int,
    ) -> torch.Tensor:
        if edge_index.numel() == 0 or logits.numel() == 0 or num_sources == 0 or num_prims == 0:
            return torch.zeros((num_sources, num_prims), dtype=torch.float32)
        support = torch.zeros((num_sources, num_prims), dtype=torch.float32)
        probs = torch.sigmoid(logits)
        for edge_id in range(edge_index.shape[1]):
            src = int(edge_index[0, edge_id].item())
            dst = int(edge_index[1, edge_id].item())
            support[src, dst] = probs[edge_id]
        return support

    def _top_support(self, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if support.numel() == 0 or support.shape[0] == 0:
            num_prims = support.shape[1] if support.ndim == 2 else 0
            return torch.full((num_prims,), -1, dtype=torch.long), torch.zeros(num_prims, dtype=torch.float32)
        values, indices = support.max(dim=0)
        return indices.long(), values.float()
