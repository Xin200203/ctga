"""Build current object hypotheses from primitive partitions."""

from __future__ import annotations

from collections import defaultdict

import torch

from ctga.common.geometry import bbox_from_points
from ctga.common.types import CurrentObjectHypothesis, Mask2D, Primitive3D, TrackState
from ctga.graph1_evidence.graph_builder import EvidenceGraphBatch


class CurrentObjectBuilder:
    def build(
        self,
        cluster_ids: torch.LongTensor,
        primitives: list[Primitive3D],
        masks: list[Mask2D],
        active_tracks: list[TrackState],
        edge_logits: dict[str, torch.FloatTensor],
        graph: EvidenceGraphBatch,
    ) -> list[CurrentObjectHypothesis]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for prim_idx, cluster_id in enumerate(cluster_ids.tolist()):
            grouped[int(cluster_id)].append(prim_idx)

        mp_support = self._build_support_map(graph.mp_edge_index, edge_logits["logit_mp"])
        pt_support = self._build_support_map(graph.pt_edge_index, edge_logits["logit_pt"])

        objects: list[CurrentObjectHypothesis] = []
        for obj_id_local, prim_indices in enumerate(grouped.values()):
            prim_subset = [primitives[idx] for idx in prim_indices]
            voxel_ids = torch.cat([p.voxel_ids for p in prim_subset], dim=0)
            xyz = torch.cat([p.world_xyz for p in prim_subset], dim=0)
            bbox = bbox_from_points(xyz)
            primitive_feats = torch.stack([p.feat3d_raw.float() for p in prim_subset], dim=0)
            feat_obj = torch.cat(
                [
                    primitive_feats.mean(dim=0),
                    primitive_feats.max(dim=0).values,
                    xyz.mean(dim=0),
                    (bbox[3:] - bbox[:3]),
                ],
                dim=0,
            )
            support_mask_ids = sorted(
                {
                    masks[mask_idx].mask_id
                    for prim_idx in prim_indices
                    for mask_idx in mp_support.get(prim_idx, [])
                }
            )
            support_track_ids = sorted(
                {
                    active_tracks[track_idx].track_id
                    for prim_idx in prim_indices
                    for track_idx in pt_support.get(prim_idx, [])
                }
            )
            objects.append(
                CurrentObjectHypothesis(
                    obj_id_local=obj_id_local,
                    primitive_ids=[primitives[idx].prim_id for idx in prim_indices],
                    support_mask_ids=support_mask_ids,
                    support_track_ids=support_track_ids,
                    voxel_ids=voxel_ids.long(),
                    center_xyz=xyz.mean(dim=0).float(),
                    bbox_xyzxyz=bbox.float(),
                    feat_obj=feat_obj.float(),
                )
            )
        return objects

    def _build_support_map(self, edge_index: torch.Tensor, edge_logits: torch.Tensor) -> dict[int, list[int]]:
        support: dict[int, list[int]] = defaultdict(list)
        if edge_index.numel() == 0 or edge_logits.numel() == 0:
            return support
        probs = torch.sigmoid(edge_logits)
        for edge_id in range(edge_index.shape[1]):
            src = int(edge_index[0, edge_id].item())
            dst = int(edge_index[1, edge_id].item())
            if probs[edge_id] >= 0.5:
                support[dst].append(src)
        return support
