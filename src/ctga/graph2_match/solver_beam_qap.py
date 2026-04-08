"""Beam-search QAP solver scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctga.graph2_match.component_builder import AssociationComponent


@dataclass
class _BeamState:
    assignment: dict[int, int]
    used_tracks: set[int]
    score: float


class BeamQAPSolver:
    def solve(
        self,
        component: AssociationComponent,
        unary_scores: torch.FloatTensor,
        pairwise_compat: dict[tuple[int, int], torch.FloatTensor],
        beam_size: int = 64,
    ) -> dict[int, int]:
        states = [_BeamState(assignment={}, used_tracks=set(), score=0.0)]
        ordered_obj_ids = component.obj_ids

        for obj_pos, obj_id in enumerate(ordered_obj_ids):
            next_states: list[_BeamState] = []
            candidates = list(component.candidate_map.get(obj_id, [])) + [-1]
            for state in states:
                for candidate in candidates:
                    if candidate != -1 and candidate in state.used_tracks:
                        continue
                    delta = component.unary_score_lookup.get((obj_id, candidate), 0.0) if candidate != -1 else 0.0
                    delta += self._pairwise_gain(state.assignment, obj_id, candidate, component, pairwise_compat)
                    used_tracks = set(state.used_tracks)
                    if candidate != -1:
                        used_tracks.add(candidate)
                    next_states.append(
                        _BeamState(
                            assignment={**state.assignment, obj_id: candidate},
                            used_tracks=used_tracks,
                            score=state.score + delta,
                        )
                    )
            next_states.sort(key=lambda item: item.score, reverse=True)
            states = next_states[:beam_size]

        if not states:
            return {}
        best = max(states, key=lambda item: item.score)
        return best.assignment

    def _pairwise_gain(
        self,
        current_assignment: dict[int, int],
        new_obj_id: int,
        new_track_id: int,
        component: AssociationComponent,
        pairwise_compat: dict[tuple[int, int], torch.FloatTensor],
    ) -> float:
        if new_track_id == -1:
            return 0.0
        gain = 0.0
        for old_obj_id, old_track_id in current_assignment.items():
            if old_track_id == -1:
                continue
            key = (min(old_obj_id, new_obj_id), max(old_obj_id, new_obj_id))
            matrix = pairwise_compat.get(key)
            if matrix is None:
                continue
            obj_a, obj_b = key
            cand_a = component.candidate_map.get(obj_a, [])
            cand_b = component.candidate_map.get(obj_b, [])
            trk_a = old_track_id if old_obj_id == obj_a else new_track_id
            trk_b = new_track_id if old_obj_id == obj_a else old_track_id
            if trk_a not in cand_a or trk_b not in cand_b:
                continue
            ia = cand_a.index(trk_a)
            ib = cand_b.index(trk_b)
            gain += float(matrix[ia, ib].item())
        return gain
