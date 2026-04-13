"""Render sample strips, score matrices, and galleries from mined samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .assoc_l2 import UnaryAssocConfig, UnaryAssociator
from .cluster_l1 import Layer1ClusterConfig, Layer1Clusterer
from .common_types import CurrentObject, Primitive3D, TrackState
from .io_seq import PosedRGBDSequence
from .mask_source import build_mask_source
from .primitive_build import PrimitiveBuilder, PrimitiveConfig
from .score_l1 import Layer1Config, Layer1Scorer
from .track_bank import QuickTrackBank
from .viz import (
    ensure_dir,
    overlay_layer1_clusters,
    overlay_layer2_assignments,
    overlay_masks,
    overlay_primitives,
    render_score_matrix,
    save_rgb,
    stack_panels_h,
)


@dataclass
class FrameRenderData:
    sample_index: int
    frame_id: int
    rgb: np.ndarray
    mask_overlay: np.ndarray
    primitive_overlay: np.ndarray
    layer1_overlay: np.ndarray
    layer2_overlay: np.ndarray
    score_matrix: np.ndarray
    gt_bbox_xyxy: np.ndarray
    num_primitives: int
    num_clusters: int
    num_matches: int
    assigned_track_ids: list[int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render strip/gallery assets for mined samples.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--cache-masks", required=True)
    parser.add_argument("--samples-index", required=True)
    parser.add_argument("--intrinsic-path", default=None)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--mask-mode", default="cache", choices=["cache", "empty", "oracle"])
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--gt-instance-dir", default=None)
    parser.add_argument("--out-dir", default=None, help="Defaults to samples-index parent directory")
    parser.add_argument("--sample-ids", default=None, help="Optional comma-separated sample indices to render")

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
    parser.add_argument("--assoc-top-k", type=int, default=5)
    parser.add_argument("--assoc-match-thresh", type=float, default=0.18)

    parser.add_argument("--crop-margin", type=int, default=24)
    parser.add_argument("--row-label-height", type=int, default=20)
    parser.add_argument("--header-height", type=int, default=34)
    parser.add_argument("--crop-target-height", type=int, default=132)
    parser.add_argument("--gallery-tile-width", type=int, default=360)
    parser.add_argument("--gallery-cols", type=int, default=3)
    return parser


def _load_gt_instance(gt_instance_dir: Path, frame_id: int) -> np.ndarray:
    candidates = [
        gt_instance_dir / f"{frame_id}.png",
        gt_instance_dir / f"{frame_id:06d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return np.array(Image.open(candidate))
    raise FileNotFoundError(f"Could not find GT instance png for frame {frame_id} under {gt_instance_dir}")


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], dtype=np.int32)


def _union_bbox(boxes: list[np.ndarray], image_shape: tuple[int, int], margin: int) -> np.ndarray:
    h, w = image_shape[:2]
    valid_boxes = [box for box in boxes if box is not None]
    if not valid_boxes:
        return np.array([0, 0, w, h], dtype=np.int32)
    x0 = max(min(int(box[0]) for box in valid_boxes) - margin, 0)
    y0 = max(min(int(box[1]) for box in valid_boxes) - margin, 0)
    x1 = min(max(int(box[2]) for box in valid_boxes) + margin, w)
    y1 = min(max(int(box[3]) for box in valid_boxes) + margin, h)
    return np.array([x0, y0, x1, y1], dtype=np.int32)


def _crop_image(image: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy.tolist()]
    return image[y0:y1, x0:x1].copy()


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    scale = target_height / max(image.shape[0], 1)
    width = max(1, int(round(image.shape[1] * scale)))
    return np.asarray(Image.fromarray(image, mode="RGB").resize((width, target_height), Image.BILINEAR), dtype=np.uint8)


def _add_top_bar(image: np.ndarray, text: str, height: int, bg: tuple[int, int, int], fg: tuple[int, int, int]) -> np.ndarray:
    panel = np.full((image.shape[0] + height, image.shape[1], 3), bg, dtype=np.uint8)
    panel[height:, :, :] = image
    pil = Image.fromarray(panel, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.text((6, 4), text, fill=fg)
    return np.asarray(pil, dtype=np.uint8)


def _stack_panels_v(panels: list[np.ndarray], pad: int = 4, bg: tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    heights = [panel.shape[0] for panel in panels]
    widths = [panel.shape[1] for panel in panels]
    out_h = sum(heights) + pad * (len(panels) - 1)
    out_w = max(widths)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = np.array(bg, dtype=np.uint8)
    cursor = 0
    for panel in panels:
        h, w = panel.shape[:2]
        x = (out_w - w) // 2
        canvas[cursor:cursor + h, x:x + w] = panel
        cursor += h + pad
    return canvas


def _make_frame_column(frame_data: FrameRenderData, crop_bbox: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    row_specs = [
        ("RGB", frame_data.rgb),
        ("Masks", frame_data.mask_overlay),
        ("Prims", frame_data.primitive_overlay),
        ("L1", frame_data.layer1_overlay),
        ("L2", frame_data.layer2_overlay),
    ]
    cropped_rows = []
    for label, image in row_specs:
        crop = _crop_image(image, crop_bbox)
        crop = _resize_to_height(crop, args.crop_target_height)
        crop = _add_top_bar(
            image=crop,
            text=label,
            height=args.row_label_height,
            bg=(34, 34, 38),
            fg=(245, 245, 245),
        )
        cropped_rows.append(crop)

    stats = f"f{frame_data.frame_id} | p{frame_data.num_primitives} c{frame_data.num_clusters} m{frame_data.num_matches}"
    if frame_data.assigned_track_ids:
        stats += " | " + ",".join(f"t{track_id}" for track_id in frame_data.assigned_track_ids[:4])
    header = np.full((args.header_height, cropped_rows[0].shape[1], 3), (24, 24, 28), dtype=np.uint8)
    header_pil = Image.fromarray(header, mode="RGB")
    header_draw = ImageDraw.Draw(header_pil)
    header_draw.text((6, 8), stats, fill=(250, 250, 250))
    column = _stack_panels_v([np.asarray(header_pil, dtype=np.uint8)] + cropped_rows, pad=4, bg=(18, 18, 18))
    representative_tile = _stack_panels_v([cropped_rows[0], cropped_rows[-1]], pad=4, bg=(20, 20, 20))
    return column, representative_tile


def _make_gallery_tile(
    sample_meta: dict[str, object],
    representative_tile: np.ndarray,
    target_width: int,
) -> np.ndarray:
    image = Image.fromarray(representative_tile, mode="RGB")
    scale = target_width / max(image.size[0], 1)
    target_height = max(1, int(round(image.size[1] * scale)))
    image = image.resize((target_width, target_height), Image.BILINEAR)

    footer_h = 72
    panel = np.full((target_height + footer_h, target_width, 3), 246, dtype=np.uint8)
    panel[:target_height] = np.asarray(image, dtype=np.uint8)
    pil = Image.fromarray(panel, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.text((8, target_height + 6), f"{sample_meta['sample_id']} | {sample_meta['sample_type']}", fill=(20, 20, 20))
    draw.text(
        (8, target_height + 28),
        f"GT {sample_meta['gt_instance_id']} | frag {sample_meta['avg_before_frag']:.1f}->{sample_meta['avg_after_frag']:.1f}",
        fill=(40, 40, 40),
    )
    draw.text(
        (8, target_height + 48),
        f"frames {sample_meta['frame_start']}-{sample_meta['frame_end']} | track {sample_meta['representative_track_id']}",
        fill=(70, 70, 70),
    )
    return np.asarray(pil, dtype=np.uint8)


def _stack_grid(tiles: list[np.ndarray], cols: int, bg: tuple[int, int, int] = (245, 245, 245), pad: int = 12) -> np.ndarray:
    if not tiles:
        placeholder = np.full((120, 360, 3), bg, dtype=np.uint8)
        pil = Image.fromarray(placeholder, mode="RGB")
        draw = ImageDraw.Draw(pil)
        draw.text((12, 45), "No samples", fill=(100, 100, 100))
        return np.asarray(pil, dtype=np.uint8)
    cols = max(int(cols), 1)
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(tile.shape[1] for tile in tiles)
    tile_h = max(tile.shape[0] for tile in tiles)
    canvas = np.full((rows * tile_h + pad * (rows - 1), cols * tile_w + pad * (cols - 1), 3), bg, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y = row * (tile_h + pad)
        x = col * (tile_w + pad)
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    return canvas


def _select_sample_indices(sample_ids_raw: str | None, total: int) -> list[int]:
    if not sample_ids_raw:
        return list(range(total))
    sample_ids = []
    for part in sample_ids_raw.split(","):
        part = part.strip()
        if not part:
            continue
        sample_ids.append(int(part))
    return [idx for idx in sample_ids if 0 <= idx < total]


def main() -> None:
    args = build_parser().parse_args()
    scene_root = Path(args.scene_root).resolve()
    samples_index_path = Path(args.samples_index).resolve()
    out_root = Path(args.out_dir).resolve() if args.out_dir else samples_index_path.parent
    gt_instance_dir = Path(args.gt_instance_dir).resolve() if args.gt_instance_dir else scene_root / "instance"

    sample_rows = json.loads(samples_index_path.read_text(encoding="utf-8"))
    sample_ids = _select_sample_indices(args.sample_ids, len(sample_rows))
    selected_samples = []
    for sample_idx in sample_ids:
        row = dict(sample_rows[sample_idx])
        row["sample_id"] = f"sample_{sample_idx:03d}"
        selected_samples.append(row)
    if not selected_samples:
        raise RuntimeError("No samples selected for rendering.")

    sample_by_index = {sample["sample_id"]: sample for sample in selected_samples}
    frame_to_samples: dict[int, list[str]] = {}
    for sample in selected_samples:
        for sample_index in range(int(sample["sample_index_start"]), int(sample["sample_index_end"]) + 1):
            frame_to_samples.setdefault(sample_index, []).append(sample["sample_id"])
    max_end = max(int(sample["sample_index_end"]) for sample in selected_samples)

    sequence = PosedRGBDSequence(
        scene_root=scene_root,
        interval=args.interval,
        depth_scale=args.depth_scale,
        intrinsic_path=args.intrinsic_path,
    )
    mask_source = build_mask_source(
        mode=args.mask_mode,
        cache_root=Path(args.cache_masks).resolve(),
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
    unary_associator = UnaryAssociator(
        UnaryAssocConfig(
            top_k=args.assoc_top_k,
            match_thresh=args.assoc_match_thresh,
        )
    )
    track_bank = QuickTrackBank()

    rendered_by_sample: dict[str, list[FrameRenderData]] = {sample["sample_id"]: [] for sample in selected_samples}

    for sample_index in range(max_end + 1):
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
        assoc_result = unary_associator.match(
            current_objects=current_objects,
            active_tracks=active_tracks,
            primitives=primitives,
        )

        interested_samples = frame_to_samples.get(sample_index, [])
        if interested_samples:
            rgb = frame.rgb.astype(np.uint8)
            mask_overlay = overlay_masks(rgb, masks)
            primitive_overlay = overlay_primitives(rgb, primitives)
            layer1_overlay = overlay_layer1_clusters(rgb, primitives, cluster_ids, current_objects)
            layer2_overlay = overlay_layer2_assignments(rgb, primitives, current_objects, active_tracks, assoc_result)
            score_matrix = render_score_matrix(current_objects, active_tracks, assoc_result)

            for sample_id in interested_samples:
                sample_meta = sample_by_index[sample_id]
                gt_map = _load_gt_instance(gt_instance_dir, frame.frame_id)
                gt_mask = gt_map == int(sample_meta["gt_instance_id"])
                gt_bbox = _bbox_from_mask(gt_mask)
                rendered_by_sample[sample_id].append(
                    FrameRenderData(
                        sample_index=int(sample_index),
                        frame_id=int(frame.frame_id),
                        rgb=rgb.copy(),
                        mask_overlay=mask_overlay.copy(),
                        primitive_overlay=primitive_overlay.copy(),
                        layer1_overlay=layer1_overlay.copy(),
                        layer2_overlay=layer2_overlay.copy(),
                        score_matrix=score_matrix.copy(),
                        gt_bbox_xyxy=np.array(gt_bbox if gt_bbox is not None else [0, 0, rgb.shape[1], rgb.shape[0]], dtype=np.int32),
                        num_primitives=len(primitives),
                        num_clusters=len(current_objects),
                        num_matches=len(assoc_result.assigned_track_ids),
                        assigned_track_ids=sorted(int(track_id) for track_id in assoc_result.assigned_track_ids.values()),
                    )
                )

        track_bank.update_from_current_objects(
            current_objects=current_objects,
            primitives=primitives,
            frame_id=frame.frame_id,
            assignments=assoc_result.assigned_track_ids,
            match_scores=assoc_result.match_scores,
        )

        if (sample_index + 1) % 20 == 0 or sample_index == max_end:
            print(f"[render_samples] processed {sample_index + 1}/{max_end + 1} sampled frames", flush=True)

    galleries_root = ensure_dir(out_root / "galleries")
    gallery_tiles = {
        "split_fixed": [],
        "still_fragmented": [],
        "wrong_merge": [],
    }

    for sample in selected_samples:
        sample_id = sample["sample_id"]
        frame_records = rendered_by_sample[sample_id]
        if not frame_records:
            continue
        sample_dir = ensure_dir(out_root / "samples" / sample_id)
        frames_dir = ensure_dir(sample_dir / "frames")
        crop_bbox = _union_bbox(
            boxes=[record.gt_bbox_xyxy for record in frame_records],
            image_shape=frame_records[0].rgb.shape[:2],
            margin=args.crop_margin,
        )

        frame_columns = []
        representative_tile = None
        representative_score = None
        representative_index = int(sample["representative_sample_index"])

        for record in frame_records:
            column, tile = _make_frame_column(record, crop_bbox, args)
            frame_columns.append(column)
            frame_panel_path = frames_dir / f"frame_{record.frame_id:06d}_panel.png"
            save_rgb(frame_panel_path, column)
            if record.sample_index == representative_index:
                representative_tile = tile
                representative_score = record.score_matrix

        strip = stack_panels_h(frame_columns, pad=8, bg=(18, 18, 18))
        strip_path = sample_dir / "strip.png"
        save_rgb(strip_path, strip)

        score_matrix_path = sample_dir / "score_matrix.png"
        if representative_score is None:
            representative_score = frame_records[len(frame_records) // 2].score_matrix
        save_rgb(score_matrix_path, representative_score)

        if representative_tile is None:
            _, representative_tile = _make_frame_column(frame_records[len(frame_records) // 2], crop_bbox, args)

        gallery_tile = _make_gallery_tile(sample, representative_tile, target_width=args.gallery_tile_width)
        gallery_tiles[str(sample["sample_type"])].append(gallery_tile)
        gallery_tile_path = sample_dir / "gallery_tile.png"
        save_rgb(gallery_tile_path, gallery_tile)

        meta_path = sample_dir / "meta.json"
        meta = dict(sample)
        meta["crop_bbox_xyxy"] = crop_bbox.tolist()
        meta["outputs"] = {
            "strip": str(strip_path),
            "score_matrix": str(score_matrix_path),
            "frames_dir": str(frames_dir),
            "gallery_tile": str(gallery_tile_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    success_grid = _stack_grid(gallery_tiles["split_fixed"], cols=args.gallery_cols)
    fail_grid = _stack_grid(gallery_tiles["still_fragmented"], cols=args.gallery_cols)
    merge_grid = _stack_grid(gallery_tiles["wrong_merge"], cols=args.gallery_cols)
    success_grid_path = galleries_root / "success_grid.png"
    fail_grid_path = galleries_root / "fail_grid.png"
    merge_grid_path = galleries_root / "merge_error_grid.png"
    save_rgb(success_grid_path, success_grid)
    save_rgb(fail_grid_path, fail_grid)
    save_rgb(merge_grid_path, merge_grid)

    summary = {
        "scene_id": scene_root.name,
        "num_rendered_samples": len(selected_samples),
        "outputs": {
            "success_grid": str(success_grid_path),
            "fail_grid": str(fail_grid_path),
            "merge_error_grid": str(merge_grid_path),
            "samples_root": str(out_root / "samples"),
        },
    }
    summary_path = out_root / "render_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Task 7 sample rendering complete:", flush=True)
    print(f"  scene_id: {scene_root.name}", flush=True)
    print(f"  num_rendered_samples: {len(selected_samples)}", flush=True)
    print(f"  success_grid: {success_grid_path}", flush=True)
    print(f"  fail_grid: {fail_grid_path}", flush=True)
    print(f"  merge_error_grid: {merge_grid_path}", flush=True)
    print(f"  render_summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
