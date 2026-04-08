"""Online inference engine scaffold."""

from __future__ import annotations

import torch

from ctga.frontends.offline_mask_cache import OfflineMaskCache
from ctga.graph1_evidence.edge_scorers import EdgeScorerL1
from ctga.graph1_evidence.graph_builder import EvidenceGraphBuilder
from ctga.graph1_evidence.object_builder import CurrentObjectBuilder
from ctga.graph1_evidence.signed_graph import SignedGraphAssembler
from ctga.graph1_evidence.solver_gasp import GaspPartitionSolver
from ctga.graph2_match.candidate_gating import CandidateGater
from ctga.graph2_match.component_builder import ComponentBuilder
from ctga.graph2_match.graph_builder import AssociationGraphBuilder
from ctga.graph2_match.relation_scorer import RelationScorer
from ctga.graph2_match.solver_beam_qap import BeamQAPSolver
from ctga.graph2_match.solver_hungarian import HungarianFallbackSolver
from ctga.graph2_match.unary_scorer import UnaryScorer
from ctga.mapping.active_map import ActiveMap
from ctga.mapping.visible_track_renderer import VisibleTrackRenderer
from ctga.memory.track_bank import TrackBank
from ctga.primitives.primitive_builder import PrimitiveBuilder


class OnlineEngine:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.beam_size = int(self.cfg.get("beam_size", 64))
        self.map = ActiveMap(**self.cfg.get("mapping", {"voxel_size": 0.05}))
        self.mask_cache = OfflineMaskCache(self.cfg.get("mask_cache_root", "./cache"))
        self.primitive_builder = PrimitiveBuilder(self.cfg.get("primitive"))
        self.evidence_graph_builder = EvidenceGraphBuilder()
        self.edge_scorer_l1 = EdgeScorerL1(**self.cfg.get("graph1", {}))
        self.signed_graph_assembler = SignedGraphAssembler()
        self.layer1_solver = GaspPartitionSolver()
        self.current_object_builder = CurrentObjectBuilder()

        self.assoc_graph_builder = AssociationGraphBuilder()
        self.unary_scorer = UnaryScorer()
        self.relation_scorer = RelationScorer()
        self.candidate_gater = CandidateGater()
        self.component_builder = ComponentBuilder()
        self.graph_matcher = BeamQAPSolver()
        self.hungarian_fallback = HungarianFallbackSolver()

        self.track_bank = TrackBank(self.cfg.get("memory"))
        self.track_renderer = VisibleTrackRenderer()

    @torch.no_grad()
    def step(self, frame):
        self.map.integrate(frame)
        active = self.map.query_active_frustum(frame)

        masks = self.mask_cache.load_masks(frame.scene_id, frame.frame_id)
        primitives = self.primitive_builder.build(
            active["active_points_xyz"],
            active["active_points_rgb"],
            active["active_voxel_ids"],
        )

        active_tracks = self.track_bank.query_active(frame)
        local_to_track_id = {idx: track.track_id for idx, track in enumerate(active_tracks)}
        rendered_tracks = self.track_renderer.render(active_tracks, frame)

        graph1 = self.evidence_graph_builder.build(frame, masks, primitives, active_tracks, rendered_tracks)
        logits1 = self.edge_scorer_l1(graph1)
        signed_prim_graph = self.signed_graph_assembler.assemble(graph1, logits1)
        prim_cluster_ids = self.layer1_solver.solve(signed_prim_graph)
        current_objects = self.current_object_builder.build(
            prim_cluster_ids, primitives, masks, active_tracks, logits1, graph1
        )

        assoc_graph = self.assoc_graph_builder.build(current_objects, active_tracks, frame)
        unary_logits = self.unary_scorer(assoc_graph.unary_feat)
        candidate_map = self.candidate_gater.prune(assoc_graph, unary_logits)
        components = self.component_builder.build_components(candidate_map, assoc_graph, unary_logits)

        matched: dict[int, int] = {}
        for component in components:
            if component.is_unambiguous():
                assignment = self.hungarian_fallback.solve(component.unary_cost())
                for row_idx, col_idx in assignment.items():
                    obj_id = component.obj_ids[row_idx]
                    local_track_idx = component.track_ids[col_idx] if col_idx >= 0 and col_idx < len(component.track_ids) else -1
                    matched[obj_id] = local_to_track_id.get(local_track_idx, -1)
                continue

            pairwise_compat = self.relation_scorer(
                assoc_graph.obj_edge_feat,
                assoc_graph.trk_edge_feat,
                assoc_graph.obj_edge_index,
                assoc_graph.trk_edge_index,
                component.candidate_map,
            )
            local_assignment = self.graph_matcher.solve(
                component,
                unary_scores=component.unary_cost(),
                pairwise_compat=pairwise_compat,
                beam_size=self.beam_size,
            )
            for obj_id, local_track_idx in local_assignment.items():
                matched[obj_id] = local_to_track_id.get(local_track_idx, -1)

        self.track_bank.apply_assignments(frame.frame_id, matched, current_objects)
        self.track_bank.step_lifecycle(frame.frame_id)

        return {
            "current_objects": current_objects,
            "matched": matched,
            "tracks_active": list(self.track_bank.active.values()),
            "graph1": graph1,
            "graph1_logits": logits1,
            "assoc_graph": assoc_graph,
            "assoc_unary": unary_logits,
        }
