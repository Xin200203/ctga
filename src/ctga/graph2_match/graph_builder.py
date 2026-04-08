"""Association graph structures and builders for layer-2 matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctga.common.types import CurrentObjectHypothesis, FramePacket, TrackState
from ctga.graph2_match.relation_features import RelationFeatureBuilder
from ctga.graph2_match.unary_features import UnaryFeatureBuilder


@dataclass
class AssociationGraphBatch:
    obj_feat: torch.FloatTensor
    trk_feat: torch.FloatTensor
    unary_index: torch.LongTensor
    unary_feat: torch.FloatTensor
    obj_edge_index: torch.LongTensor
    obj_edge_feat: torch.FloatTensor
    trk_edge_index: torch.LongTensor
    trk_edge_feat: torch.FloatTensor


class AssociationGraphBuilder:
    def __init__(
        self,
        unary_builder: UnaryFeatureBuilder | None = None,
        relation_builder: RelationFeatureBuilder | None = None,
    ) -> None:
        self.unary_builder = unary_builder or UnaryFeatureBuilder()
        self.relation_builder = relation_builder or RelationFeatureBuilder()

    def build(
        self,
        current_objects: list[CurrentObjectHypothesis],
        active_tracks: list[TrackState],
        frame: FramePacket,
    ) -> AssociationGraphBatch:
        obj_feat = self._object_features(current_objects)
        trk_feat = self._track_features(active_tracks)
        unary_index, unary_feat = self._unary_edges(current_objects, active_tracks, frame)
        obj_edge_index, obj_edge_feat = self._object_relations(current_objects, frame)
        trk_edge_index, trk_edge_feat = self._track_relations(active_tracks, frame)
        return AssociationGraphBatch(
            obj_feat=obj_feat,
            trk_feat=trk_feat,
            unary_index=unary_index,
            unary_feat=unary_feat,
            obj_edge_index=obj_edge_index,
            obj_edge_feat=obj_edge_feat,
            trk_edge_index=trk_edge_index,
            trk_edge_feat=trk_edge_feat,
        )

    def _object_features(self, objects: list[CurrentObjectHypothesis]) -> torch.Tensor:
        if not objects:
            return torch.zeros((0, 8), dtype=torch.float32)
        feats = []
        for obj in objects:
            feats.append(
                torch.cat(
                    [
                        obj.center_xyz.float(),
                        (obj.bbox_xyzxyz[3:] - obj.bbox_xyzxyz[:3]).float(),
                        torch.tensor(
                            [
                                float(len(obj.primitive_ids)),
                                float(len(obj.support_track_ids)),
                            ],
                            dtype=torch.float32,
                        ),
                    ]
                )
            )
        return torch.stack(feats, dim=0)

    def _track_features(self, tracks: list[TrackState]) -> torch.Tensor:
        if not tracks:
            return torch.zeros((0, 11), dtype=torch.float32)
        feats = []
        for track in tracks:
            feats.append(
                torch.cat(
                    [
                        track.center_xyz.float(),
                        (track.bbox_xyzxyz[3:] - track.bbox_xyzxyz[:3]).float(),
                        torch.tensor(
                            [
                                float(track.confidence),
                                float(track.age),
                                float(track.miss_count),
                                float(track.last_seen),
                                1.0,
                            ],
                            dtype=torch.float32,
                        ),
                    ]
                )
            )
        return torch.stack(feats, dim=0)

    def _unary_edges(
        self,
        objects: list[CurrentObjectHypothesis],
        tracks: list[TrackState],
        frame: FramePacket,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edges = []
        feats = []
        for obj_idx, obj in enumerate(objects):
            for trk_idx, track in enumerate(tracks):
                edges.append([obj_idx, trk_idx])
                feats.append(self.unary_builder.build(obj, track, frame))
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 20), dtype=torch.float32)
        return torch.tensor(edges, dtype=torch.long).T.contiguous(), torch.stack(feats, dim=0).float()

    def _object_relations(
        self, objects: list[CurrentObjectHypothesis], frame: FramePacket
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edges = []
        feats = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                edges.append([i, j])
                feats.append(self.relation_builder.build_object_relation(objects[i], objects[j], frame))
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 32), dtype=torch.float32)
        return torch.tensor(edges, dtype=torch.long).T.contiguous(), torch.stack(feats, dim=0).float()

    def _track_relations(
        self, tracks: list[TrackState], frame: FramePacket
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edges = []
        feats = []
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                edges.append([i, j])
                feats.append(self.relation_builder.build_track_relation(tracks[i], tracks[j], frame))
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 32), dtype=torch.float32)
        return torch.tensor(edges, dtype=torch.long).T.contiguous(), torch.stack(feats, dim=0).float()
