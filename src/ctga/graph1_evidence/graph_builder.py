"""Layer-1 evidence graph data structures and builder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctga.common.geometry import bbox_size
from ctga.common.types import FramePacket, Mask2D, Primitive3D, TrackState
from ctga.graph1_evidence.edge_features import EdgeFeatureBuilderL1


@dataclass
class EvidenceGraphBatch:
    mask_nodes_feat: torch.FloatTensor
    prim_nodes_feat: torch.FloatTensor
    track_nodes_feat: torch.FloatTensor
    mp_edge_index: torch.LongTensor
    mp_edge_feat: torch.FloatTensor
    pt_edge_index: torch.LongTensor
    pt_edge_feat: torch.FloatTensor
    mt_edge_index: torch.LongTensor
    mt_edge_feat: torch.FloatTensor
    pp_edge_index: torch.LongTensor
    pp_edge_feat: torch.FloatTensor


class EvidenceGraphBuilder:
    def __init__(self, feature_builder: EdgeFeatureBuilderL1 | None = None) -> None:
        self.feature_builder = feature_builder or EdgeFeatureBuilderL1()

    def build(
        self,
        frame: FramePacket,
        masks: list[Mask2D],
        primitives: list[Primitive3D],
        active_tracks: list[TrackState],
        rendered_tracks: dict[int, torch.BoolTensor],
    ) -> EvidenceGraphBatch:
        mask_nodes_feat = self._mask_node_features(masks, frame)
        prim_nodes_feat = self._primitive_node_features(primitives)
        track_nodes_feat = self._track_node_features(active_tracks)

        mp_index, mp_feat = self._build_mp_edges(masks, primitives, frame)
        pt_index, pt_feat = self._build_pt_edges(primitives, active_tracks)
        mt_index, mt_feat = self._build_mt_edges(masks, active_tracks, rendered_tracks)
        pp_index, pp_feat = self._build_pp_edges(primitives)

        return EvidenceGraphBatch(
            mask_nodes_feat=mask_nodes_feat,
            prim_nodes_feat=prim_nodes_feat,
            track_nodes_feat=track_nodes_feat,
            mp_edge_index=mp_index,
            mp_edge_feat=mp_feat,
            pt_edge_index=pt_index,
            pt_edge_feat=pt_feat,
            mt_edge_index=mt_index,
            mt_edge_feat=mt_feat,
            pp_edge_index=pp_index,
            pp_edge_feat=pp_feat,
        )

    def _mask_node_features(self, masks: list[Mask2D], frame: FramePacket) -> torch.Tensor:
        if not masks:
            return torch.zeros((0, 8), dtype=torch.float32)
        height, width = frame.depth.shape
        feats = []
        for mask in masks:
            bbox = mask.bbox_xyxy.float()
            feats.append(
                torch.tensor(
                    [
                        float(mask.area),
                        float(mask.score),
                        float(mask.depth_median),
                        float(mask.depth_minmax[0]),
                        float(mask.depth_minmax[1]),
                        float((bbox[2] - bbox[0]) / max(width, 1)),
                        float((bbox[3] - bbox[1]) / max(height, 1)),
                        1.0,
                    ],
                    dtype=torch.float32,
                )
            )
        return torch.stack(feats, dim=0)

    def _primitive_node_features(self, primitives: list[Primitive3D]) -> torch.Tensor:
        if not primitives:
            return torch.zeros((0, 16), dtype=torch.float32)
        feats = []
        for primitive in primitives:
            size = bbox_size(primitive.bbox_xyzxyz.float())
            raw = primitive.feat3d_raw.float()
            pad = torch.zeros(max(0, 6 - raw.shape[0]), dtype=torch.float32)
            raw_small = torch.cat([raw[:6], pad], dim=0)[:6]
            feats.append(
                torch.cat(
                    [
                        primitive.center_xyz.float(),
                        size.float(),
                        primitive.normal_mean.float(),
                        primitive.color_mean.float(),
                        raw_small,
                        torch.tensor([float(primitive.visible_pixel_count)], dtype=torch.float32),
                    ]
                )
            )
        return torch.stack(feats, dim=0)

    def _track_node_features(self, tracks: list[TrackState]) -> torch.Tensor:
        if not tracks:
            return torch.zeros((0, 14), dtype=torch.float32)
        feats = []
        for track in tracks:
            size = bbox_size(track.bbox_xyzxyz.float())
            feats.append(
                torch.cat(
                    [
                        track.center_xyz.float(),
                        size.float(),
                        track.feat3d_ema.float()[:3],
                        track.feat_track.float()[:3],
                        torch.tensor(
                            [
                                float(track.confidence),
                                float(track.age),
                                float(track.miss_count),
                                1.0,
                            ],
                            dtype=torch.float32,
                        ),
                    ]
                )
            )
        return torch.stack(feats, dim=0)

    def _build_mp_edges(
        self, masks: list[Mask2D], primitives: list[Primitive3D], frame: FramePacket
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = []
        feats = []
        for mi, mask in enumerate(masks):
            for pj, primitive in enumerate(primitives):
                feat = self.feature_builder.build_mp_features(mask, primitive, frame.K, frame.pose_c2w)
                if feat[0] > 0 or feat[1] > 0 or feat[2] > 0.05:
                    indices.append([mi, pj])
                    feats.append(feat)
        return self._edge_tensors(indices, feats, feat_dim=16)

    def _build_pt_edges(
        self, primitives: list[Primitive3D], tracks: list[TrackState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = []
        feats = []
        for pj, primitive in enumerate(primitives):
            for tk, track in enumerate(tracks):
                feat = self.feature_builder.build_pt_features(primitive, track)
                if feat[0] > 0 or feat[1] > 0 or feat[2] > 0.2:
                    indices.append([pj, tk])
                    feats.append(feat)
        return self._edge_tensors(indices, feats, feat_dim=16)

    def _build_mt_edges(
        self,
        masks: list[Mask2D],
        tracks: list[TrackState],
        rendered_tracks: dict[int, torch.BoolTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = []
        feats = []
        for mi, mask in enumerate(masks):
            for tk, track in enumerate(tracks):
                rendered = rendered_tracks.get(track.track_id, torch.zeros_like(mask.bitmap))
                feat = self.feature_builder.build_mt_features(mask, rendered, track)
                if feat[0] > 0 or feat[1] > 0 or feat[2] > 0.05:
                    indices.append([mi, tk])
                    feats.append(feat)
        return self._edge_tensors(indices, feats, feat_dim=12)

    def _build_pp_edges(self, primitives: list[Primitive3D]) -> tuple[torch.Tensor, torch.Tensor]:
        indices = []
        feats = []
        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives)):
                feat = self.feature_builder.build_pp_features(primitives[i], primitives[j])
                if feat[0] > 0 or feat[1] > 0.2:
                    indices.append([i, j])
                    feats.append(feat)
        return self._edge_tensors(indices, feats, feat_dim=12)

    def _edge_tensors(
        self, indices: list[list[int]], feats: list[torch.Tensor], feat_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not indices:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, feat_dim), dtype=torch.float32),
            )
        return torch.tensor(indices, dtype=torch.long).T.contiguous(), torch.stack(feats, dim=0).float()
