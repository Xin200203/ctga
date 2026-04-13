"""Layer-1 clustering and current-object construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common_types import CurrentObject, Mask2D, Primitive3D, TrackState
from .score_l1 import Layer1Graph


@dataclass
class Layer1ClusterConfig:
    min_positive_weight: float = 0.08
    merge_score_thresh: float = 0.12
    negative_veto_thresh: float = 0.70
    support_mask_thresh: float = 0.20
    support_track_thresh: float = 0.12


class Layer1Clusterer:
    def __init__(self, cfg: Layer1ClusterConfig | None = None) -> None:
        self.cfg = cfg or Layer1ClusterConfig()

    def cluster(
        self,
        graph: Layer1Graph,
        primitives: list[Primitive3D],
        masks: list[Mask2D],
        active_tracks: list[TrackState] | None = None,
    ) -> tuple[np.ndarray, list[CurrentObject]]:
        active_tracks = active_tracks or []
        num_prims = len(primitives)
        if num_prims == 0:
            return np.zeros((0,), dtype=np.int32), []

        parent = np.arange(num_prims, dtype=np.int32)
        rank = np.zeros(num_prims, dtype=np.int8)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        pairs: list[tuple[float, int, int]] = []
        for prim_idx_a in range(num_prims):
            for prim_idx_b in range(prim_idx_a + 1, num_prims):
                pairs.append((float(graph.merge_scores[prim_idx_a, prim_idx_b]), prim_idx_a, prim_idx_b))
        pairs.sort(key=lambda item: item[0], reverse=True)

        for merge_score, prim_idx_a, prim_idx_b in pairs:
            if merge_score < self.cfg.merge_score_thresh:
                break
            if float(graph.positive_weights[prim_idx_a, prim_idx_b]) < self.cfg.min_positive_weight:
                continue
            if float(graph.negative_weights[prim_idx_a, prim_idx_b]) > self.cfg.negative_veto_thresh:
                continue
            union(prim_idx_a, prim_idx_b)

        root_to_cluster: dict[int, int] = {}
        cluster_ids = np.full((num_prims,), -1, dtype=np.int32)
        for prim_idx in range(num_prims):
            root = find(prim_idx)
            if root not in root_to_cluster:
                root_to_cluster[root] = len(root_to_cluster)
            cluster_ids[prim_idx] = root_to_cluster[root]

        current_objects = self._build_current_objects(
            cluster_ids=cluster_ids,
            graph=graph,
            primitives=primitives,
            masks=masks,
            active_tracks=active_tracks,
        )
        return cluster_ids, current_objects

    def _build_current_objects(
        self,
        cluster_ids: np.ndarray,
        graph: Layer1Graph,
        primitives: list[Primitive3D],
        masks: list[Mask2D],
        active_tracks: list[TrackState],
    ) -> list[CurrentObject]:
        cluster_to_prims: dict[int, list[int]] = {}
        for prim_idx, cluster_id in enumerate(cluster_ids.tolist()):
            cluster_to_prims.setdefault(int(cluster_id), []).append(prim_idx)

        objects: list[CurrentObject] = []
        mask_id_lookup = [mask.mask_id for mask in masks]
        track_id_lookup = [track.track_id for track in active_tracks]
        for cluster_id, prim_indices in sorted(cluster_to_prims.items()):
            primitive_ids = [primitives[idx].prim_id for idx in prim_indices]
            xyz = np.concatenate([primitives[idx].xyz for idx in prim_indices if primitives[idx].xyz.size > 0], axis=0)
            voxel_ids = np.unique(
                np.concatenate([primitives[idx].voxel_ids for idx in prim_indices if primitives[idx].voxel_ids.size > 0], axis=0)
            )
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            bbox = np.concatenate([mins, maxs], axis=0).astype(np.float32)
            center = xyz.mean(axis=0).astype(np.float32)

            support_mask_ids = set()
            support_track_ids = set()
            for prim_idx in prim_indices:
                support_mask_ids.update(primitives[prim_idx].support_mask_ids)
                if graph.mp_scores.shape[0] > 0:
                    for mask_row, mask_id in enumerate(mask_id_lookup):
                        if float(graph.mp_scores[mask_row, prim_idx]) >= self.cfg.support_mask_thresh:
                            support_mask_ids.add(int(mask_id))
                if graph.pt_scores.shape[1] > 0:
                    for track_col, track_id in enumerate(track_id_lookup):
                        if float(graph.pt_scores[prim_idx, track_col]) >= self.cfg.support_track_thresh:
                            support_track_ids.add(int(track_id))

            objects.append(
                CurrentObject(
                    obj_id_local=int(cluster_id),
                    primitive_ids=primitive_ids,
                    support_mask_ids=sorted(support_mask_ids),
                    support_track_ids=sorted(support_track_ids),
                    voxel_ids=voxel_ids.astype(np.int64),
                    center_xyz=center,
                    bbox_xyzxyz=bbox,
                )
            )
        return objects
