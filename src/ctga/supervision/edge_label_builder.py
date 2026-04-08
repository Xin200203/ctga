"""Build supervision labels for layer-1 edges."""

from __future__ import annotations

import torch

from ctga.common.types import Mask2D, Primitive3D, TrackState
from ctga.graph1_evidence.graph_builder import EvidenceGraphBatch


class EdgeLabelBuilderL1:
    def build_labels(
        self,
        graph: EvidenceGraphBatch,
        masks: list[Mask2D],
        primitives: list[Primitive3D],
        active_tracks_teacher: list[TrackState],
        prim_gt: dict[int, int | None],
        mask_gt: dict[int, int | None],
        track_gt_ids: dict[int, int | None],
    ) -> dict[str, torch.Tensor]:
        return {
            "y_mp": self._build_mp(graph, masks, primitives, prim_gt, mask_gt),
            "y_pt": self._build_pt(graph, primitives, active_tracks_teacher, prim_gt, track_gt_ids),
            "y_mt": self._build_mt(graph, masks, active_tracks_teacher, mask_gt, track_gt_ids),
            "y_pp": self._build_pp(graph, primitives, prim_gt),
        }

    def _build_mp(
        self,
        graph: EvidenceGraphBatch,
        masks: list[Mask2D],
        primitives: list[Primitive3D],
        prim_gt: dict[int, int | None],
        mask_gt: dict[int, int | None],
    ) -> torch.Tensor:
        labels = torch.full((graph.mp_edge_index.shape[1],), -1, dtype=torch.long)
        for edge_id in range(graph.mp_edge_index.shape[1]):
            m_idx = int(graph.mp_edge_index[0, edge_id].item())
            p_idx = int(graph.mp_edge_index[1, edge_id].item())
            mg = mask_gt.get(masks[m_idx].mask_id)
            pg = prim_gt.get(primitives[p_idx].prim_id)
            if mg is None or pg is None:
                continue
            labels[edge_id] = 1 if mg == pg else 0
        return labels

    def _build_pt(
        self,
        graph: EvidenceGraphBatch,
        primitives: list[Primitive3D],
        tracks: list[TrackState],
        prim_gt: dict[int, int | None],
        track_gt_ids: dict[int, int | None],
    ) -> torch.Tensor:
        labels = torch.full((graph.pt_edge_index.shape[1],), -1, dtype=torch.long)
        for edge_id in range(graph.pt_edge_index.shape[1]):
            p_idx = int(graph.pt_edge_index[0, edge_id].item())
            t_idx = int(graph.pt_edge_index[1, edge_id].item())
            pg = prim_gt.get(primitives[p_idx].prim_id)
            tg = track_gt_ids.get(tracks[t_idx].track_id)
            if pg is None or tg is None:
                continue
            labels[edge_id] = 1 if pg == tg else 0
        return labels

    def _build_mt(
        self,
        graph: EvidenceGraphBatch,
        masks: list[Mask2D],
        tracks: list[TrackState],
        mask_gt: dict[int, int | None],
        track_gt_ids: dict[int, int | None],
    ) -> torch.Tensor:
        labels = torch.full((graph.mt_edge_index.shape[1],), -1, dtype=torch.long)
        for edge_id in range(graph.mt_edge_index.shape[1]):
            m_idx = int(graph.mt_edge_index[0, edge_id].item())
            t_idx = int(graph.mt_edge_index[1, edge_id].item())
            mg = mask_gt.get(masks[m_idx].mask_id)
            tg = track_gt_ids.get(tracks[t_idx].track_id)
            if mg is None or tg is None:
                continue
            labels[edge_id] = 1 if mg == tg else 0
        return labels

    def _build_pp(
        self,
        graph: EvidenceGraphBatch,
        primitives: list[Primitive3D],
        prim_gt: dict[int, int | None],
    ) -> torch.Tensor:
        labels = torch.full((graph.pp_edge_index.shape[1],), -1, dtype=torch.long)
        for edge_id in range(graph.pp_edge_index.shape[1]):
            p0 = primitives[int(graph.pp_edge_index[0, edge_id].item())].prim_id
            p1 = primitives[int(graph.pp_edge_index[1, edge_id].item())].prim_id
            g0 = prim_gt.get(p0)
            g1 = prim_gt.get(p1)
            if g0 is None or g1 is None:
                continue
            labels[edge_id] = 1 if g0 == g1 else 0
        return labels
