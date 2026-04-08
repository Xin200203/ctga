import pytest

torch = pytest.importorskip("torch")

from ctga.common.types import FramePacket, Mask2D, Primitive3D, TrackState
from ctga.graph1_evidence.edge_scorers import EdgeScorerL1
from ctga.graph1_evidence.graph_builder import EvidenceGraphBuilder
from ctga.graph1_evidence.object_builder import CurrentObjectBuilder
from ctga.graph1_evidence.signed_graph import SignedGraphAssembler
from ctga.graph1_evidence.solver_gasp import GaspPartitionSolver
from ctga.graph2_match.candidate_gating import CandidateGater
from ctga.graph2_match.component_builder import ComponentBuilder
from ctga.graph2_match.graph_builder import AssociationGraphBuilder
from ctga.graph2_match.unary_scorer import UnaryScorer


def _make_frame():
    return FramePacket(
        frame_id=0,
        rgb=torch.zeros((16, 16, 3), dtype=torch.uint8),
        depth=torch.ones((16, 16), dtype=torch.float32),
        pose_c2w=torch.eye(4, dtype=torch.float32),
        K=torch.tensor([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        scene_id="demo",
    )


def _make_mask():
    bitmap = torch.zeros((16, 16), dtype=torch.bool)
    bitmap[5:10, 5:10] = True
    return Mask2D(
        mask_id=0,
        bitmap=bitmap,
        bbox_xyxy=torch.tensor([5.0, 5.0, 10.0, 10.0]),
        area=int(bitmap.sum()),
        score=1.0,
        feat2d_raw=torch.zeros(512, dtype=torch.float32),
        depth_median=1.0,
        depth_minmax=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )


def _make_primitive():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=torch.float32)
    return Primitive3D(
        prim_id=0,
        voxel_coords=torch.tensor([[0, 0, 0]], dtype=torch.int32),
        voxel_ids=torch.tensor([0], dtype=torch.long),
        world_xyz=xyz,
        center_xyz=xyz.mean(dim=0),
        bbox_xyzxyz=torch.tensor([0.0, 0.0, 1.0, 0.05, 0.05, 1.0], dtype=torch.float32),
        normal_mean=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        color_mean=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        feat3d_raw=torch.zeros(16, dtype=torch.float32),
        visible_pixel_count=3,
    )


def _make_track():
    return TrackState(
        track_id=10,
        status="active",
        voxel_ids=torch.tensor([0], dtype=torch.long),
        center_xyz=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        bbox_xyzxyz=torch.tensor([0.0, 0.0, 1.0, 0.05, 0.05, 1.0], dtype=torch.float32),
        feat2d_ema=torch.zeros(128, dtype=torch.float32),
        feat3d_ema=torch.zeros(64, dtype=torch.float32),
        feat_track=torch.zeros(128, dtype=torch.float32),
        last_seen=0,
        age=1,
        miss_count=0,
        confidence=1.0,
    )


def test_graph1_to_graph2_smoke():
    frame = _make_frame()
    masks = [_make_mask()]
    primitives = [_make_primitive()]
    tracks = [_make_track()]
    rendered_tracks = {10: masks[0].bitmap.clone()}

    graph1_builder = EvidenceGraphBuilder()
    graph1 = graph1_builder.build(frame, masks, primitives, tracks, rendered_tracks)
    logits1 = EdgeScorerL1()(graph1)
    signed_graph = SignedGraphAssembler().assemble(graph1, logits1)
    cluster_ids = GaspPartitionSolver().solve(signed_graph)
    current_objects = CurrentObjectBuilder().build(cluster_ids, primitives, masks, tracks, logits1, graph1)

    assoc_graph = AssociationGraphBuilder().build(current_objects, tracks, frame)
    unary_logits = UnaryScorer()(assoc_graph.unary_feat)
    candidate_map = CandidateGater().prune(assoc_graph, unary_logits, max_candidates_per_obj=3)
    components = ComponentBuilder().build_components(candidate_map, assoc_graph, unary_logits)

    assert len(current_objects) == 1
    assert assoc_graph.unary_feat.shape[1] == 20
    assert len(components) == 1
