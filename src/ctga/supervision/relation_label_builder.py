"""Build unary and pairwise labels for layer-2 supervision."""

from __future__ import annotations

import torch

from ctga.common.types import CurrentObjectHypothesis, TrackState
from ctga.graph2_match.graph_builder import AssociationGraphBatch


class AssociationLabelBuilder:
    def build(
        self,
        assoc_graph: AssociationGraphBatch,
        current_objects_teacher: list[CurrentObjectHypothesis],
        active_tracks_teacher: list[TrackState],
        gt_map_obj: dict[int, int | None],
        gt_map_trk: dict[int, int | None],
    ) -> dict[str, object]:
        unary_label = torch.zeros((assoc_graph.unary_index.shape[1],), dtype=torch.float32)
        for edge_id in range(assoc_graph.unary_index.shape[1]):
            obj_idx = int(assoc_graph.unary_index[0, edge_id].item())
            trk_idx = int(assoc_graph.unary_index[1, edge_id].item())
            obj_id = current_objects_teacher[obj_idx].obj_id_local
            track_id = active_tracks_teacher[trk_idx].track_id
            unary_label[edge_id] = float(gt_map_obj.get(obj_id) is not None and gt_map_obj.get(obj_id) == gt_map_trk.get(track_id))

        pairwise_label: dict[tuple[int, int, int, int], int] = {}
        for obj_edge_id in range(assoc_graph.obj_edge_index.shape[1]):
            oa = int(assoc_graph.obj_edge_index[0, obj_edge_id].item())
            ob = int(assoc_graph.obj_edge_index[1, obj_edge_id].item())
            obj_a_gt = gt_map_obj.get(current_objects_teacher[oa].obj_id_local)
            obj_b_gt = gt_map_obj.get(current_objects_teacher[ob].obj_id_local)
            for trk_edge_id in range(assoc_graph.trk_edge_index.shape[1]):
                ta = int(assoc_graph.trk_edge_index[0, trk_edge_id].item())
                tb = int(assoc_graph.trk_edge_index[1, trk_edge_id].item())
                trk_a_gt = gt_map_trk.get(active_tracks_teacher[ta].track_id)
                trk_b_gt = gt_map_trk.get(active_tracks_teacher[tb].track_id)
                pairwise_label[(oa, ob, ta, tb)] = int(obj_a_gt == trk_a_gt and obj_b_gt == trk_b_gt and obj_a_gt is not None and obj_b_gt is not None)
        return {"unary_label": unary_label, "pairwise_label": pairwise_label}
