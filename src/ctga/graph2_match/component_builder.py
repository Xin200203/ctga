"""Connected-component builder for association subproblems."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctga.graph2_match.graph_builder import AssociationGraphBatch


@dataclass
class AssociationComponent:
    obj_ids: list[int]
    track_ids: list[int]
    candidate_map: dict[int, list[int]]
    unary_score_lookup: dict[tuple[int, int], float]

    def is_unambiguous(self) -> bool:
        if len(self.obj_ids) <= 1:
            return True
        used = set()
        for obj_id in self.obj_ids:
            cand = set(self.candidate_map.get(obj_id, []))
            if used & cand:
                return False
            used |= cand
        return True

    def unary_cost(self, newborn_score: float = 0.0) -> torch.FloatTensor:
        matrix = torch.full((len(self.obj_ids), len(self.track_ids) + 1), -1e4, dtype=torch.float32)
        track_to_col = {track_id: idx for idx, track_id in enumerate(self.track_ids)}
        for row, obj_id in enumerate(self.obj_ids):
            for track_id in self.candidate_map.get(obj_id, []):
                col = track_to_col[track_id]
                matrix[row, col] = self.unary_score_lookup.get((obj_id, track_id), -1e4)
            matrix[row, -1] = newborn_score
        return matrix


class ComponentBuilder:
    def build_components(
        self,
        candidate_map: dict[int, list[int]],
        assoc_graph: AssociationGraphBatch | None = None,
        unary_logits: torch.FloatTensor | None = None,
    ) -> list[AssociationComponent]:
        obj_ids = sorted(candidate_map.keys())
        visited: set[int] = set()
        unary_lookup = self._build_unary_lookup(assoc_graph, unary_logits)
        components: list[AssociationComponent] = []

        for start in obj_ids:
            if start in visited:
                continue
            queue = [start]
            component_obj_ids = []
            component_tracks = set()
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_obj_ids.append(current)
                current_tracks = set(candidate_map.get(current, []))
                component_tracks |= current_tracks
                for other in obj_ids:
                    if other in visited or other == current:
                        continue
                    if current_tracks & set(candidate_map.get(other, [])):
                        queue.append(other)

            submap = {obj_id: candidate_map.get(obj_id, []) for obj_id in component_obj_ids}
            components.append(
                AssociationComponent(
                    obj_ids=sorted(component_obj_ids),
                    track_ids=sorted(component_tracks),
                    candidate_map=submap,
                    unary_score_lookup=unary_lookup,
                )
            )
        return components

    def _build_unary_lookup(
        self,
        assoc_graph: AssociationGraphBatch | None,
        unary_logits: torch.FloatTensor | None,
    ) -> dict[tuple[int, int], float]:
        lookup: dict[tuple[int, int], float] = {}
        if assoc_graph is None or unary_logits is None:
            return lookup
        for edge_id in range(assoc_graph.unary_index.shape[1]):
            obj_id = int(assoc_graph.unary_index[0, edge_id].item())
            trk_id = int(assoc_graph.unary_index[1, edge_id].item())
            lookup[(obj_id, trk_id)] = float(unary_logits[edge_id].item())
        return lookup
