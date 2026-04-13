"""GT-based diagnostic analysis for quick graph test Layer-1 results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cluster_l1 import Layer1ClusterConfig, Layer1Clusterer
from .common_types import CurrentObject, FramePacket, Primitive3D
from .io_seq import PosedRGBDSequence
from .mask_source import build_mask_source
from .primitive_build import PrimitiveBuilder, PrimitiveConfig
from .score_l1 import Layer1Config, Layer1Scorer
from .track_bank import QuickTrackBank
from .viz import ensure_dir, save_rgb, stack_panels_h


@dataclass
class PipelineResult:
    frame: FramePacket
    primitives: list[Primitive3D]
    cluster_ids: np.ndarray
    current_objects: list[CurrentObject]
    history_frames: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze quick Layer-1 outputs against GT instances.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--cache-masks", required=True)
    parser.add_argument("--intrinsic-path", default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--history-before", type=int, default=0)
    parser.add_argument("--history-after", type=int, default=5)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--mask-mode", default="cache", choices=["cache", "empty", "oracle"])
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--gt-instance-dir", default=None)
    parser.add_argument("--out-dir", default=None)
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
    parser.add_argument("--min-gt-pixels", type=int, default=256)
    return parser


def _id_color(item_id: int) -> np.ndarray:
    base = (item_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = 50 + (base % 180)
    g = 50 + ((base // 181) % 180)
    b = 50 + ((base // (181 * 181)) % 180)
    return np.array([r, g, b], dtype=np.uint8)


def _load_gt_instance(gt_instance_dir: Path, frame_id: int) -> np.ndarray:
    candidates = [
        gt_instance_dir / f"{frame_id}.png",
        gt_instance_dir / f"{frame_id:06d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return np.array(Image.open(candidate))
    raise FileNotFoundError(f"Could not find GT instance png for frame {frame_id} under {gt_instance_dir}")


def _run_pipeline(args: argparse.Namespace, history_frames: int) -> PipelineResult:
    scene_root = Path(args.scene_root).resolve()
    cache_root = Path(args.cache_masks).resolve()
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

    start_index = max(0, args.frame_index - max(int(history_frames), 0))
    final_frame = None
    final_primitives: list[Primitive3D] = []
    final_cluster_ids = np.zeros((0,), dtype=np.int32)
    final_current_objects: list[CurrentObject] = []

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
            final_primitives = primitives
            final_cluster_ids = cluster_ids
            final_current_objects = current_objects
        track_bank.update_from_current_objects(
            current_objects=current_objects,
            primitives=primitives,
            frame_id=frame.frame_id,
        )

    if final_frame is None:
        raise RuntimeError("No frame processed.")
    return PipelineResult(
        frame=final_frame,
        primitives=final_primitives,
        cluster_ids=final_cluster_ids,
        current_objects=final_current_objects,
        history_frames=history_frames,
    )


def _label_map_from_primitives(primitives: list[Primitive3D], image_shape: tuple[int, int]) -> np.ndarray:
    label_map = np.full(image_shape, -1, dtype=np.int32)
    h, w = image_shape
    for primitive in primitives:
        coords = primitive.pixel_idx.astype(np.int32)
        if coords.size == 0:
            continue
        ys = np.clip(coords[:, 0], 0, h - 1)
        xs = np.clip(coords[:, 1], 0, w - 1)
        label_map[ys, xs] = int(primitive.prim_id)
    return label_map


def _label_map_from_clusters(primitives: list[Primitive3D], cluster_ids: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    label_map = np.full(image_shape, -1, dtype=np.int32)
    h, w = image_shape
    for primitive, cluster_id in zip(primitives, cluster_ids.tolist()):
        coords = primitive.pixel_idx.astype(np.int32)
        if coords.size == 0:
            continue
        ys = np.clip(coords[:, 0], 0, h - 1)
        xs = np.clip(coords[:, 1], 0, w - 1)
        label_map[ys, xs] = int(cluster_id)
    return label_map


def _boundary_map(label_map: np.ndarray, ignore_val: int | None = None) -> np.ndarray:
    boundary = np.zeros(label_map.shape, dtype=bool)
    for axis in (0, 1):
        shifted = np.roll(label_map, shift=-1, axis=axis)
        diff = label_map != shifted
        if ignore_val is not None:
            same_ignore = (label_map == ignore_val) & (shifted == ignore_val)
            diff &= ~same_ignore
        boundary |= diff
    boundary[-1, :] = False
    boundary[:, -1] = False
    return boundary


def _overlay_label_map(rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    canvas = rgb.astype(np.float32).copy()
    for label_id in sorted(int(v) for v in np.unique(label_map) if int(v) >= 0):
        bitmap = label_map == label_id
        if not np.any(bitmap):
            continue
        color = _id_color(label_id).astype(np.float32)
        canvas[bitmap] = (1.0 - alpha) * canvas[bitmap] + alpha * color
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _overlay_boundaries(rgb: np.ndarray, gt_boundary: np.ndarray, unit_boundary: np.ndarray) -> np.ndarray:
    canvas = rgb.astype(np.uint8).copy()
    both = gt_boundary & unit_boundary
    canvas[gt_boundary] = np.array([60, 220, 80], dtype=np.uint8)
    canvas[unit_boundary] = np.array([240, 80, 80], dtype=np.uint8)
    canvas[both] = np.array([255, 220, 80], dtype=np.uint8)
    return canvas


def _annotate_top_regions(image: np.ndarray, label_map: np.ndarray, prefix: str, topk: int = 12) -> np.ndarray:
    out = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(out)
    labels, counts = np.unique(label_map[label_map >= 0], return_counts=True)
    if labels.size == 0:
        return np.asarray(out, dtype=np.uint8)
    order = np.argsort(-counts)
    for idx in order[: min(topk, len(order))]:
        label_id = int(labels[idx])
        bitmap = label_map == label_id
        ys, xs = np.where(bitmap)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        color = tuple(int(v) for v in _id_color(label_id).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), f"{prefix}{label_id}", fill=color)
    return np.asarray(out, dtype=np.uint8)


def _summarize_units(label_map: np.ndarray, gt_map: np.ndarray) -> dict[str, object]:
    unit_ids = [int(v) for v in np.unique(label_map) if int(v) >= 0]
    if not unit_ids:
        return {
            "num_units": 0,
            "mean_purity": 0.0,
            "median_purity": 0.0,
            "num_overmerge_units": 0,
            "overmerge_ratio": 0.0,
            "units": [],
        }

    unit_rows = []
    purities = []
    overmerge_count = 0
    for unit_id in unit_ids:
        bitmap = label_map == unit_id
        unit_size = int(bitmap.sum())
        gt_vals = gt_map[bitmap]
        gt_vals = gt_vals[gt_vals > 0]
        if gt_vals.size == 0:
            dominant_gt = -1
            dominant_pixels = 0
            purity = 0.0
            significant_gt_ids: list[int] = []
            significant_gt_pixels: list[int] = []
        else:
            gt_ids, gt_counts = np.unique(gt_vals, return_counts=True)
            order = np.argsort(-gt_counts)
            gt_ids = gt_ids[order]
            gt_counts = gt_counts[order]
            dominant_gt = int(gt_ids[0])
            dominant_pixels = int(gt_counts[0])
            purity = dominant_pixels / max(unit_size, 1)
            sig_thresh = max(64, int(0.15 * unit_size))
            sig_mask = gt_counts >= sig_thresh
            significant_gt_ids = [int(v) for v in gt_ids[sig_mask].tolist()]
            significant_gt_pixels = [int(v) for v in gt_counts[sig_mask].tolist()]
        overmerge = len(significant_gt_ids) >= 2
        overmerge_count += int(overmerge)
        purities.append(float(purity))
        unit_rows.append(
            {
                "unit_id": int(unit_id),
                "pixel_count": unit_size,
                "dominant_gt_id": dominant_gt,
                "dominant_gt_pixels": dominant_pixels,
                "purity": float(purity),
                "significant_gt_ids": significant_gt_ids,
                "significant_gt_pixels": significant_gt_pixels,
                "overmerge": bool(overmerge),
            }
        )

    unit_rows.sort(key=lambda row: row["pixel_count"], reverse=True)
    return {
        "num_units": len(unit_rows),
        "mean_purity": float(np.mean(purities)),
        "median_purity": float(np.median(purities)),
        "num_overmerge_units": int(overmerge_count),
        "overmerge_ratio": float(overmerge_count / max(len(unit_rows), 1)),
        "units": unit_rows,
    }


def _summarize_gt_fragmentation(label_map: np.ndarray, gt_map: np.ndarray, min_gt_pixels: int) -> dict[str, object]:
    gt_ids, gt_counts = np.unique(gt_map[gt_map > 0], return_counts=True)
    valid = gt_counts >= int(min_gt_pixels)
    gt_ids = gt_ids[valid]
    gt_counts = gt_counts[valid]
    if gt_ids.size == 0:
        return {
            "num_gt_instances": 0,
            "mean_fragments_per_gt": 0.0,
            "median_fragments_per_gt": 0.0,
            "matched_gt_ratio": 0.0,
            "gt_instances": [],
        }

    rows = []
    fragment_counts = []
    matched = 0
    unit_ids = [int(v) for v in np.unique(label_map) if int(v) >= 0]

    for gt_id, gt_size in zip(gt_ids.tolist(), gt_counts.tolist()):
        gt_bitmap = gt_map == int(gt_id)
        overlaps = []
        best_iou = 0.0
        best_cover = 0.0
        for unit_id in unit_ids:
            unit_bitmap = label_map == int(unit_id)
            inter = int(np.logical_and(gt_bitmap, unit_bitmap).sum())
            if inter <= 0:
                continue
            unit_size = int(unit_bitmap.sum())
            union = int(gt_size) + unit_size - inter
            iou = inter / max(union, 1)
            cover = inter / max(int(gt_size), 1)
            if inter >= max(32, int(0.05 * gt_size)):
                overlaps.append(
                    {
                        "unit_id": int(unit_id),
                        "inter": inter,
                        "iou": float(iou),
                        "cover": float(cover),
                    }
                )
            best_iou = max(best_iou, float(iou))
            best_cover = max(best_cover, float(cover))
        overlaps.sort(key=lambda row: row["inter"], reverse=True)
        fragment_count = len(overlaps)
        fragment_counts.append(fragment_count)
        is_matched = best_cover >= 0.50 or best_iou >= 0.30
        matched += int(is_matched)
        rows.append(
            {
                "gt_id": int(gt_id),
                "pixel_count": int(gt_size),
                "fragment_count": int(fragment_count),
                "best_iou": float(best_iou),
                "best_cover": float(best_cover),
                "matched": bool(is_matched),
                "top_overlaps": overlaps[:5],
            }
        )

    rows.sort(key=lambda row: (-row["fragment_count"], -row["pixel_count"]))
    return {
        "num_gt_instances": len(rows),
        "mean_fragments_per_gt": float(np.mean(fragment_counts)),
        "median_fragments_per_gt": float(np.median(fragment_counts)),
        "matched_gt_ratio": float(matched / max(len(rows), 1)),
        "gt_instances": rows,
    }


def _build_summary(name: str, label_map: np.ndarray, gt_map: np.ndarray, min_gt_pixels: int) -> dict[str, object]:
    return {
        "name": name,
        "unit_summary": _summarize_units(label_map=label_map, gt_map=gt_map),
        "gt_fragmentation": _summarize_gt_fragmentation(
            label_map=label_map,
            gt_map=gt_map,
            min_gt_pixels=min_gt_pixels,
        ),
    }


def _write_text_summary(
    summary: dict[str, object],
    out_path: Path,
) -> None:
    lines = []
    lines.append(f"scene_id: {summary['scene_id']}")
    lines.append(f"frame_id: {summary['frame_id']}")
    lines.append(f"gt_num_instances: {summary['gt_num_instances']}")
    for key in ("before_primitives", "after_layer1_h0", "after_layer1_h5"):
        block = summary[key]
        unit_summary = block["unit_summary"]
        gt_summary = block["gt_fragmentation"]
        lines.append("")
        lines.append(f"[{key}]")
        lines.append(f"num_units: {unit_summary['num_units']}")
        lines.append(f"mean_purity: {unit_summary['mean_purity']:.4f}")
        lines.append(f"median_purity: {unit_summary['median_purity']:.4f}")
        lines.append(f"num_overmerge_units: {unit_summary['num_overmerge_units']}")
        lines.append(f"overmerge_ratio: {unit_summary['overmerge_ratio']:.4f}")
        lines.append(f"mean_fragments_per_gt: {gt_summary['mean_fragments_per_gt']:.4f}")
        lines.append(f"median_fragments_per_gt: {gt_summary['median_fragments_per_gt']:.4f}")
        lines.append(f"matched_gt_ratio: {gt_summary['matched_gt_ratio']:.4f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    repo_quick_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir).resolve() if args.out_dir else repo_quick_root / "out_gt_analysis"

    before = _run_pipeline(args, history_frames=args.history_before)
    after = _run_pipeline(args, history_frames=args.history_after)

    gt_instance_dir = Path(args.gt_instance_dir).resolve() if args.gt_instance_dir else Path(args.scene_root).resolve() / "instance"
    gt_map_raw = _load_gt_instance(gt_instance_dir=gt_instance_dir, frame_id=before.frame.frame_id)
    gt_ids, gt_counts = np.unique(gt_map_raw[gt_map_raw > 0], return_counts=True)
    valid_ids = {int(gt_id) for gt_id, count in zip(gt_ids.tolist(), gt_counts.tolist()) if int(count) >= int(args.min_gt_pixels)}
    gt_map = gt_map_raw.astype(np.int32).copy()
    invalid_mask = ~np.isin(gt_map, np.array(sorted(valid_ids | {0}), dtype=np.int32))
    gt_map[invalid_mask] = 0

    before_primitive_map = _label_map_from_primitives(before.primitives, before.frame.rgb.shape[:2])
    after_h0_cluster_map = _label_map_from_clusters(before.primitives, before.cluster_ids, before.frame.rgb.shape[:2])
    after_h5_cluster_map = _label_map_from_clusters(after.primitives, after.cluster_ids, after.frame.rgb.shape[:2])

    gt_overlay = _overlay_label_map(before.frame.rgb.astype(np.uint8), np.where(gt_map > 0, gt_map, -1))
    gt_overlay = _annotate_top_regions(gt_overlay, np.where(gt_map > 0, gt_map, -1), prefix="g", topk=12)

    primitive_overlay = _overlay_label_map(before.frame.rgb.astype(np.uint8), before_primitive_map)
    primitive_overlay = _annotate_top_regions(primitive_overlay, before_primitive_map, prefix="p", topk=12)

    h0_overlay = _overlay_label_map(before.frame.rgb.astype(np.uint8), after_h0_cluster_map)
    h0_overlay = _annotate_top_regions(h0_overlay, after_h0_cluster_map, prefix="c", topk=12)

    h5_overlay = _overlay_label_map(after.frame.rgb.astype(np.uint8), after_h5_cluster_map)
    h5_overlay = _annotate_top_regions(h5_overlay, after_h5_cluster_map, prefix="c", topk=12)

    gt_boundary = _boundary_map(gt_map, ignore_val=0)
    primitive_boundary = _boundary_map(before_primitive_map, ignore_val=-1)
    h0_boundary = _boundary_map(after_h0_cluster_map, ignore_val=-1)
    h5_boundary = _boundary_map(after_h5_cluster_map, ignore_val=-1)

    primitive_vs_gt = _overlay_boundaries(before.frame.rgb.astype(np.uint8), gt_boundary, primitive_boundary)
    h0_vs_gt = _overlay_boundaries(before.frame.rgb.astype(np.uint8), gt_boundary, h0_boundary)
    h5_vs_gt = _overlay_boundaries(after.frame.rgb.astype(np.uint8), gt_boundary, h5_boundary)

    compare_panel = stack_panels_h(
        [
            before.frame.rgb.astype(np.uint8),
            gt_overlay,
            primitive_vs_gt,
            h0_vs_gt,
            h5_vs_gt,
        ]
    )

    out_dir = ensure_dir(out_root / before.frame.scene_id / f"frame_{before.frame.frame_id:06d}")
    outputs = {
        "rgb": out_dir / "rgb.png",
        "gt_instance_overlay": out_dir / "gt_instance_overlay.png",
        "before_primitives_overlay": out_dir / "before_primitives_overlay.png",
        "after_layer1_h0_overlay": out_dir / "after_layer1_h0_overlay.png",
        "after_layer1_h5_overlay": out_dir / "after_layer1_h5_overlay.png",
        "before_primitives_vs_gt": out_dir / "before_primitives_vs_gt.png",
        "after_layer1_h0_vs_gt": out_dir / "after_layer1_h0_vs_gt.png",
        "after_layer1_h5_vs_gt": out_dir / "after_layer1_h5_vs_gt.png",
        "compare_panel": out_dir / "layer1_gt_compare_panel.png",
        "summary_json": out_dir / "layer1_gt_summary.json",
        "summary_txt": out_dir / "layer1_gt_summary.txt",
    }

    save_rgb(outputs["rgb"], before.frame.rgb.astype(np.uint8))
    save_rgb(outputs["gt_instance_overlay"], gt_overlay)
    save_rgb(outputs["before_primitives_overlay"], primitive_overlay)
    save_rgb(outputs["after_layer1_h0_overlay"], h0_overlay)
    save_rgb(outputs["after_layer1_h5_overlay"], h5_overlay)
    save_rgb(outputs["before_primitives_vs_gt"], primitive_vs_gt)
    save_rgb(outputs["after_layer1_h0_vs_gt"], h0_vs_gt)
    save_rgb(outputs["after_layer1_h5_vs_gt"], h5_vs_gt)
    save_rgb(outputs["compare_panel"], compare_panel)

    summary = {
        "scene_id": before.frame.scene_id,
        "frame_id": before.frame.frame_id,
        "gt_num_instances": int(len(valid_ids)),
        "history_before": int(args.history_before),
        "history_after": int(args.history_after),
        "before_primitives": _build_summary(
            name="before_primitives",
            label_map=before_primitive_map,
            gt_map=gt_map,
            min_gt_pixels=args.min_gt_pixels,
        ),
        "after_layer1_h0": _build_summary(
            name="after_layer1_h0",
            label_map=after_h0_cluster_map,
            gt_map=gt_map,
            min_gt_pixels=args.min_gt_pixels,
        ),
        "after_layer1_h5": _build_summary(
            name="after_layer1_h5",
            label_map=after_h5_cluster_map,
            gt_map=gt_map,
            min_gt_pixels=args.min_gt_pixels,
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }

    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_text_summary(summary, outputs["summary_txt"])

    print("GT Layer-1 analysis complete:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
