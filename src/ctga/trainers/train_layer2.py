"""Training entry point for layer-2."""

from __future__ import annotations

from ctga.graph1_evidence.object_builder import CurrentObjectBuilder
from ctga.graph1_evidence.signed_graph import SignedGraphAssembler
from ctga.graph2_match.candidate_gating import CandidateGater
from ctga.graph2_match.graph_builder import AssociationGraphBuilder
from ctga.graph2_match.relation_scorer import RelationScorer
from ctga.graph2_match.unary_scorer import UnaryScorer
from ctga.losses.association_losses import pairwise_loss, unary_loss
from ctga.supervision.relation_label_builder import AssociationLabelBuilder


def train_layer2_step(
    batch: dict,
    signed_graph_assembler: SignedGraphAssembler,
    layer1_solver,
    current_object_builder: CurrentObjectBuilder,
    association_graph_builder: AssociationGraphBuilder,
    unary_scorer: UnaryScorer,
    relation_scorer: RelationScorer,
    candidate_gater: CandidateGater,
    association_label_builder: AssociationLabelBuilder,
):
    signed_graph = signed_graph_assembler.assemble(batch["graph1"], batch["graph1_logits"])
    prim_clusters = layer1_solver.solve(signed_graph)
    current_objects = current_object_builder.build(
        prim_clusters,
        batch["primitives"],
        batch["masks"],
        batch["teacher_tracks"],
        batch["graph1_logits"],
        batch["graph1"],
    )
    assoc_graph = association_graph_builder.build(current_objects, batch["teacher_tracks"], batch["frame"])
    unary_logits = unary_scorer(assoc_graph.unary_feat)
    candidate_map = candidate_gater.prune(assoc_graph, unary_logits)
    pairwise_scores = relation_scorer(
        assoc_graph.obj_edge_feat,
        assoc_graph.trk_edge_feat,
        assoc_graph.obj_edge_index,
        assoc_graph.trk_edge_index,
        candidate_map,
    )
    labels = association_label_builder.build(
        assoc_graph,
        current_objects,
        batch["teacher_tracks"],
        batch["gt_map_obj"],
        batch["gt_map_trk"],
    )
    loss = unary_loss(unary_logits, labels["unary_label"]) + pairwise_loss(
        pairwise_scores, labels["pairwise_label"], candidate_map
    )
    return loss, unary_logits, pairwise_scores, labels
