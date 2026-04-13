"""Main entry for the quick graph test prototype.

Currently covers:
- Task 1: single-frame RGB / depth / mask overlays
- Task 2: single-frame primitive over-segmentation and 2D/3D visualization
- Task 3/4: Layer-1 MP/PP/PT scoring, clustering, and minimal history support
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cluster_l1 import Layer1ClusterConfig, Layer1Clusterer
from .io_seq import PosedRGBDSequence
from .mask_source import build_mask_source
from .primitive_build import PrimitiveBuilder, PrimitiveConfig
from .score_l1 import Layer1Config, Layer1Scorer
from .track_bank import QuickTrackBank
from .viz import export_task1_overlays, export_task2_primitives, export_task34_layer1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight quick graph test prototype.")
    parser.add_argument("--scene-root", required=True, help="Path to a scene directory with color/depth/pose/intrinsic.txt")
    parser.add_argument("--cache-masks", default=None, help="Mask cache root used when --mask-mode cache")
    parser.add_argument("--mask-mode", default="cache", choices=["cache", "empty", "oracle"])
    parser.add_argument("--gt-root", default=None, help="GT root used when --mask-mode oracle")
    parser.add_argument("--intrinsic-path", default=None, help="Optional intrinsic file or ScanNet scene metadata .txt")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset index within the sampled sequence")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--history-frames", type=int, default=0, help="Number of previous sampled frames to warm up history support")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--out-dir", default=None, help="Output root; defaults to quick_graph_test/out")
    parser.add_argument("--tau-z", type=float, default=0.05)
    parser.add_argument("--tau-n-deg", type=float, default=30.0)
    parser.add_argument("--tau-c", type=float, default=35.0)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--min-primitive-pixels", type=int, default=32)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--layer1-merge-thresh", type=float, default=0.12)
    parser.add_argument("--layer1-negative-veto", type=float, default=0.70)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_quick_root = Path(__file__).resolve().parents[1]
    scene_root = Path(args.scene_root).resolve()
    out_root = Path(args.out_dir).resolve() if args.out_dir else repo_quick_root / "out"
    cache_root = Path(args.cache_masks).resolve() if args.cache_masks else repo_quick_root / "cache_masks"

    sequence = PosedRGBDSequence(
        scene_root=scene_root,
        interval=args.interval,
        depth_scale=args.depth_scale,
        intrinsic_path=args.intrinsic_path,
    )
    if args.frame_index < 0 or args.frame_index >= len(sequence):
        raise IndexError(f"--frame-index {args.frame_index} is out of range for sequence length {len(sequence)}")

    mask_source = build_mask_source(
        mode=args.mask_mode,
        cache_root=cache_root,
        gt_root=args.gt_root,
    )
    primitive_builder = PrimitiveBuilder(
        PrimitiveConfig(
            tau_z=args.tau_z,
            tau_n_deg=args.tau_n_deg,
            tau_c=args.tau_c,
            connectivity=args.connectivity,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_pixels=args.min_primitive_pixels,
            voxel_size=args.voxel_size,
        )
    )
    layer1_scorer = Layer1Scorer(Layer1Config())
    layer1_clusterer = Layer1Clusterer(
        Layer1ClusterConfig(
            merge_score_thresh=args.layer1_merge_thresh,
            negative_veto_thresh=args.layer1_negative_veto,
        )
    )
    track_bank = QuickTrackBank()

    start_index = max(0, args.frame_index - max(int(args.history_frames), 0))
    final_frame = None
    final_masks = []
    final_primitives = []
    final_cluster_ids = np.zeros((0,), dtype=np.int32)
    final_current_objects = []
    final_active_tracks = []
    final_outputs: dict[str, str] = {}
    final_track_update: dict[str, object] = {}

    for sample_index in range(start_index, args.frame_index + 1):
        frame = sequence[sample_index]
        masks = mask_source.load_masks(frame.scene_id, frame.frame_id, image_shape=frame.rgb.shape[:2])
        active_tracks = track_bank.query_active()
        primitives = primitive_builder.build(frame, masks=masks)
        graph = layer1_scorer.build_graph(frame=frame, masks=masks, primitives=primitives, active_tracks=active_tracks)
        cluster_ids, current_objects = layer1_clusterer.cluster(
            graph=graph,
            primitives=primitives,
            masks=masks,
            active_tracks=active_tracks,
        )

        if sample_index == args.frame_index:
            final_frame = frame
            final_masks = masks
            final_primitives = primitives
            final_cluster_ids = cluster_ids
            final_current_objects = current_objects
            final_active_tracks = active_tracks
            frame_out_dir = out_root / "single_frame" / frame.scene_id / f"frame_{frame.frame_id:06d}"
            final_outputs.update(export_task1_overlays(frame, masks, frame_out_dir))
            final_outputs.update(export_task2_primitives(frame, primitives, frame_out_dir))
            final_outputs.update(
                export_task34_layer1(
                    frame=frame,
                    primitives=primitives,
                    cluster_ids=cluster_ids,
                    current_objects=current_objects,
                    active_tracks=active_tracks,
                    out_dir=frame_out_dir,
                )
            )

        final_track_update = track_bank.update_from_current_objects(
            current_objects=current_objects,
            primitives=primitives,
            frame_id=frame.frame_id,
        )

    if final_frame is None:
        raise RuntimeError("No frame was processed.")

    print("Task 1 / Task 4 export complete:")
    print(f"  history_frames: {args.history_frames}")
    print(f"  num_masks: {len(final_masks)}")
    print(f"  num_primitives: {len(final_primitives)}")
    print(f"  num_clusters: {len(final_current_objects)}")
    print(f"  num_active_tracks_before_update: {len(final_active_tracks)}")
    print(f"  track_update_summary: {final_track_update}")
    for key, value in final_outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
