"""Training entry point for layer-1."""

from __future__ import annotations

from ctga.graph1_evidence.edge_scorers import EdgeScorerL1
from ctga.graph1_evidence.graph_builder import EvidenceGraphBuilder
from ctga.losses.edge_losses import layer1_edge_loss
from ctga.supervision.edge_label_builder import EdgeLabelBuilderL1


def train_layer1_step(
    batch: dict,
    evidence_graph_builder: EvidenceGraphBuilder,
    edge_scorer_l1: EdgeScorerL1,
    edge_label_builder: EdgeLabelBuilderL1,
):
    graph = evidence_graph_builder.build(
        batch["frame"],
        batch["masks"],
        batch["primitives"],
        batch["teacher_tracks"],
        batch["rendered_tracks"],
    )
    logits = edge_scorer_l1(graph)
    labels = edge_label_builder.build_labels(
        graph=graph,
        masks=batch["masks"],
        primitives=batch["primitives"],
        active_tracks_teacher=batch["teacher_tracks"],
        prim_gt=batch["prim_gt"],
        mask_gt=batch["mask_gt"],
        track_gt_ids=batch["track_gt_ids"],
    )
    return layer1_edge_loss(logits, labels), logits, labels
