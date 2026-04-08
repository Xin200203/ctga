"""Learned scorers for layer-1 evidence edges."""

from __future__ import annotations

import torch
import torch.nn as nn

from ctga.graph1_evidence.graph_builder import EvidenceGraphBatch


class _EdgeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=x.device)
        return self.net(x).squeeze(-1)


class EdgeScorerL1(nn.Module):
    def __init__(self, use_hetero_mp: bool = False) -> None:
        super().__init__()
        self.use_hetero_mp = use_hetero_mp
        self.edge_mlp_mp = _EdgeMLP(16)
        self.edge_mlp_pt = _EdgeMLP(16)
        self.edge_mlp_mt = _EdgeMLP(12)
        self.edge_mlp_pp = _EdgeMLP(12)

    def forward(self, graph: EvidenceGraphBatch) -> dict[str, torch.FloatTensor]:
        return {
            "logit_mp": self.edge_mlp_mp(graph.mp_edge_feat),
            "logit_pt": self.edge_mlp_pt(graph.pt_edge_feat),
            "logit_mt": self.edge_mlp_mt(graph.mt_edge_feat),
            "logit_pp": self.edge_mlp_pp(graph.pp_edge_feat),
        }
