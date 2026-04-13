"""Main entry for the quick graph test prototype.

Currently covers:
- Task 1: single-frame RGB / depth / mask overlays
- Task 2: single-frame primitive over-segmentation and 2D/3D visualization
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_seq import PosedRGBDSequence
from .mask_source import build_mask_source
from .primitive_build import PrimitiveBuilder, PrimitiveConfig
from .viz import export_task1_overlays, export_task2_primitives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight quick graph test prototype.")
    parser.add_argument("--scene-root", required=True, help="Path to a scene directory with color/depth/pose/intrinsic.txt")
    parser.add_argument("--cache-masks", default=None, help="Mask cache root used when --mask-mode cache")
    parser.add_argument("--mask-mode", default="cache", choices=["cache", "empty", "oracle"])
    parser.add_argument("--gt-root", default=None, help="GT root used when --mask-mode oracle")
    parser.add_argument("--intrinsic-path", default=None, help="Optional intrinsic file or ScanNet scene metadata .txt")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset index within the sampled sequence")
    parser.add_argument("--interval", type=int, default=1)
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
    frame = sequence[args.frame_index]

    mask_source = build_mask_source(
        mode=args.mask_mode,
        cache_root=cache_root,
        gt_root=args.gt_root,
    )
    masks = mask_source.load_masks(frame.scene_id, frame.frame_id, image_shape=frame.rgb.shape[:2])

    frame_out_dir = out_root / "single_frame" / frame.scene_id / f"frame_{frame.frame_id:06d}"
    outputs = export_task1_overlays(frame, masks, frame_out_dir)
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
    primitives = primitive_builder.build(frame)
    outputs.update(export_task2_primitives(frame, primitives, frame_out_dir))

    print("Task 1 / Task 2 export complete:")
    print(f"  num_masks: {len(masks)}")
    print(f"  num_primitives: {len(primitives)}")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
