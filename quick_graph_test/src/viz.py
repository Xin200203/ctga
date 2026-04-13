"""Visualization helpers for the quick graph test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common_types import CurrentObject, FramePacket, Mask2D, Primitive3D, TrackState
from .geometry import rotation_matrix_xyz


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d = depth.astype(np.float32).copy()
    lo = float(np.percentile(d[valid], 2))
    hi = float(np.percentile(d[valid], 98))
    hi = max(hi, lo + 1e-6)
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    r = (255.0 * norm).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(norm - 0.5) * 2.0)).astype(np.uint8)
    b = (255.0 * (1.0 - norm)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    return rgb


def _mask_color(mask_id: int) -> np.ndarray:
    base = (mask_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = 50 + (base % 180)
    g = 50 + ((base // 181) % 180)
    b = 50 + ((base // (181 * 181)) % 180)
    return np.array([r, g, b], dtype=np.uint8)


def _id_color(item_id: int) -> np.ndarray:
    return _mask_color(item_id)


def overlay_masks(rgb: np.ndarray, masks: list[Mask2D], alpha: float = 0.45) -> np.ndarray:
    canvas = rgb.astype(np.float32).copy()
    for mask in masks:
        color = _id_color(mask.mask_id).astype(np.float32)
        bitmap = mask.bitmap.astype(bool)
        if not np.any(bitmap):
            continue
        canvas[bitmap] = (1.0 - alpha) * canvas[bitmap] + alpha * color
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for mask in masks:
        x0, y0, x1, y1 = [int(v) for v in mask.bbox_xyxy.tolist()]
        if x1 <= x0 or y1 <= y0:
            continue
        color = tuple(int(v) for v in _id_color(mask.mask_id).tolist())
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), f"{mask.mask_id}", fill=color)
    return np.asarray(image, dtype=np.uint8)


def overlay_primitives(rgb: np.ndarray, primitives: list[Primitive3D], alpha: float = 0.45) -> np.ndarray:
    canvas = rgb.astype(np.float32).copy()
    h, w = rgb.shape[:2]
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    areas = np.array([primitive.pixel_idx.shape[0] for primitive in primitives], dtype=np.int32)
    topk_ids: set[int] = set()
    if areas.size > 0:
        order = np.argsort(-areas)
        topk_ids = {int(idx) for idx in order[: min(20, len(order))].tolist()}
    for primitive in primitives:
        color = _id_color(primitive.prim_id).astype(np.float32)
        coords = primitive.pixel_idx.astype(np.int32)
        if coords.size == 0:
            continue
        ys = np.clip(coords[:, 0], 0, h - 1)
        xs = np.clip(coords[:, 1], 0, w - 1)
        canvas_np = np.asarray(image, dtype=np.float32)
        canvas_np[ys, xs] = (1.0 - alpha) * canvas_np[ys, xs] + alpha * color
        image = Image.fromarray(np.clip(canvas_np, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        if primitive.prim_id not in topk_ids:
            continue
        cx = int(np.clip(coords[:, 1].mean(), 0, w - 1))
        cy = int(np.clip(coords[:, 0].mean(), 0, h - 1))
        bbox = np.array([coords[:, 1].min(), coords[:, 0].min(), coords[:, 1].max() + 1, coords[:, 0].max() + 1])
        draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]) - 1, int(bbox[3]) - 1], outline=tuple(color.astype(np.uint8).tolist()), width=1)
        draw.text((cx + 2, cy + 2), f"{primitive.prim_id}", fill=tuple(color.astype(np.uint8).tolist()))
    return np.asarray(image, dtype=np.uint8)


def render_primitive_cloud(primitives: list[Primitive3D], size: tuple[int, int] = (960, 720)) -> np.ndarray:
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = np.array([20, 20, 24], dtype=np.uint8)
    if not primitives:
        return canvas

    xyz_list = []
    color_list = []
    center_list = []
    center_color_list = []
    areas = np.array([primitive.pixel_idx.shape[0] for primitive in primitives], dtype=np.int32)
    label_ids: set[int] = set()
    if areas.size > 0:
        order = np.argsort(-areas)
        label_ids = {int(idx) for idx in order[: min(20, len(order))].tolist()}

    for primitive in primitives:
        xyz = primitive.xyz.astype(np.float32)
        if xyz.size == 0:
            continue
        xyz_list.append(xyz)
        color = np.repeat(_id_color(primitive.prim_id)[None, :], xyz.shape[0], axis=0)
        color_list.append(color)
        center_list.append(primitive.center_xyz.astype(np.float32))
        center_color_list.append(_id_color(primitive.prim_id))

    if not xyz_list:
        return canvas

    points = np.concatenate(xyz_list, axis=0)
    colors = np.concatenate(color_list, axis=0)
    centers = np.stack(center_list, axis=0)
    center_colors = np.stack(center_color_list, axis=0)

    pivot = points.mean(axis=0, keepdims=True)
    rot = rotation_matrix_xyz(rx_deg=18.0, ry_deg=-35.0, rz_deg=0.0)
    rotated = (points - pivot) @ rot.T
    rotated_centers = (centers - pivot) @ rot.T

    xy = rotated[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = 0.82 * min(width / span[0], height / span[1])
    offset = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    proj = (xy - (min_xy + max_xy) / 2.0) * scale + offset
    proj_centers = (rotated_centers[:, :2] - (min_xy + max_xy) / 2.0) * scale + offset
    proj[:, 1] = height - proj[:, 1]
    proj_centers[:, 1] = height - proj_centers[:, 1]

    order = np.argsort(rotated[:, 2])
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    for idx in order:
        x, y = proj[idx]
        if x < 1 or x >= width - 1 or y < 1 or y >= height - 1:
            continue
        color = tuple(int(v) for v in colors[idx].tolist())
        draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=color)

    for prim_idx, center_xy in enumerate(proj_centers):
        if prim_idx not in label_ids:
            continue
        x, y = center_xy
        color = tuple(int(v) for v in center_colors[prim_idx].tolist())
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255), outline=color, width=2)
        draw.text((x + 5, y + 5), f"{prim_idx}", fill=color)
    return np.asarray(image, dtype=np.uint8)


def stack_panels_h(panels: list[np.ndarray], pad: int = 6, bg: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    heights = [panel.shape[0] for panel in panels]
    widths = [panel.shape[1] for panel in panels]
    out_h = max(heights)
    out_w = sum(widths) + pad * (len(panels) - 1)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = np.array(bg, dtype=np.uint8)
    cursor = 0
    for panel in panels:
        h, w = panel.shape[:2]
        y = (out_h - h) // 2
        canvas[y : y + h, cursor : cursor + w] = panel
        cursor += w + pad
    return canvas


def export_task1_overlays(
    frame: FramePacket,
    masks: list[Mask2D],
    out_dir: str | Path,
) -> dict[str, str]:
    out_path = ensure_dir(out_dir)
    rgb_path = out_path / "rgb.png"
    depth_path = out_path / "depth.png"
    overlay_path = out_path / "mask_overlay.png"
    panel_path = out_path / "panel.png"
    meta_path = out_path / "meta.json"

    rgb = frame.rgb.astype(np.uint8)
    depth_rgb = colorize_depth(frame.depth)
    overlay = overlay_masks(rgb, masks)
    panel = stack_panels_h([rgb, depth_rgb, overlay])

    save_rgb(rgb_path, rgb)
    save_rgb(depth_path, depth_rgb)
    save_rgb(overlay_path, overlay)
    save_rgb(panel_path, panel)

    meta = {
        "scene_id": frame.scene_id,
        "frame_id": frame.frame_id,
        "num_masks": len(masks),
        "image_size": [int(frame.rgb.shape[1]), int(frame.rgb.shape[0])],
        "outputs": {
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "mask_overlay": str(overlay_path),
            "panel": str(panel_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "mask_overlay": str(overlay_path),
        "panel": str(panel_path),
        "meta": str(meta_path),
    }


def export_task2_primitives(
    frame: FramePacket,
    primitives: list[Primitive3D],
    out_dir: str | Path,
) -> dict[str, str]:
    out_path = ensure_dir(out_dir)
    overlay_path = out_path / "primitive_overlay_2d.png"
    cloud_path = out_path / "primitive_cloud_3d.png"
    meta_path = out_path / "primitive_meta.json"

    overlay = overlay_primitives(frame.rgb.astype(np.uint8), primitives)
    cloud = render_primitive_cloud(primitives)
    save_rgb(overlay_path, overlay)
    save_rgb(cloud_path, cloud)

    pixel_counts = [int(primitive.pixel_idx.shape[0]) for primitive in primitives]
    meta = {
        "scene_id": frame.scene_id,
        "frame_id": frame.frame_id,
        "num_primitives": len(primitives),
        "pixel_count_stats": {
            "min": int(min(pixel_counts)) if pixel_counts else 0,
            "max": int(max(pixel_counts)) if pixel_counts else 0,
            "mean": float(np.mean(pixel_counts)) if pixel_counts else 0.0,
        },
        "num_mask_guided_primitives": int(sum(1 for primitive in primitives if primitive.support_mask_ids)),
        "support_mask_histogram": {
            str(mask_id): int(sum(1 for primitive in primitives if mask_id in primitive.support_mask_ids))
            for mask_id in sorted({mask_id for primitive in primitives for mask_id in primitive.support_mask_ids})
        },
        "outputs": {
            "primitive_overlay_2d": str(overlay_path),
            "primitive_cloud_3d": str(cloud_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "primitive_overlay_2d": str(overlay_path),
        "primitive_cloud_3d": str(cloud_path),
        "primitive_meta": str(meta_path),
    }


def _cluster_color(cluster_id: int, support_track_ids: list[int]) -> np.ndarray:
    if support_track_ids:
        return _id_color(10000 + int(support_track_ids[0]))
    return _id_color(5000 + int(cluster_id))


def overlay_layer1_clusters(
    rgb: np.ndarray,
    primitives: list[Primitive3D],
    cluster_ids: np.ndarray,
    current_objects: list[CurrentObject],
) -> np.ndarray:
    canvas = rgb.astype(np.float32).copy()
    h, w = rgb.shape[:2]
    object_lookup = {int(obj.obj_id_local): obj for obj in current_objects}
    cluster_areas: dict[int, int] = {}
    for prim_idx, primitive in enumerate(primitives):
        cluster_id = int(cluster_ids[prim_idx]) if prim_idx < len(cluster_ids) else -1
        cluster_areas[cluster_id] = cluster_areas.get(cluster_id, 0) + int(primitive.pixel_idx.shape[0])
    topk_clusters = {cluster_id for cluster_id, _ in sorted(cluster_areas.items(), key=lambda item: -item[1])[:15]}

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for prim_idx, primitive in enumerate(primitives):
        coords = primitive.pixel_idx.astype(np.int32)
        if coords.size == 0:
            continue
        cluster_id = int(cluster_ids[prim_idx]) if prim_idx < len(cluster_ids) else -1
        current_object = object_lookup.get(cluster_id)
        support_track_ids = current_object.support_track_ids if current_object is not None else []
        color = _cluster_color(cluster_id, support_track_ids).astype(np.float32)
        ys = np.clip(coords[:, 0], 0, h - 1)
        xs = np.clip(coords[:, 1], 0, w - 1)
        canvas_np = np.asarray(image, dtype=np.float32)
        canvas_np[ys, xs] = 0.50 * canvas_np[ys, xs] + 0.50 * color
        image = Image.fromarray(np.clip(canvas_np, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)

    for cluster_id in topk_clusters:
        coords_list = [primitives[idx].pixel_idx for idx in range(len(primitives)) if int(cluster_ids[idx]) == cluster_id]
        if not coords_list:
            continue
        coords = np.concatenate(coords_list, axis=0)
        cx = int(np.clip(coords[:, 1].mean(), 0, w - 1))
        cy = int(np.clip(coords[:, 0].mean(), 0, h - 1))
        x0, y0 = int(coords[:, 1].min()), int(coords[:, 0].min())
        x1, y1 = int(coords[:, 1].max()) + 1, int(coords[:, 0].max()) + 1
        current_object = object_lookup.get(cluster_id)
        support_track_ids = current_object.support_track_ids if current_object is not None else []
        color = tuple(int(v) for v in _cluster_color(cluster_id, support_track_ids).tolist())
        label = f"c{cluster_id}"
        if support_track_ids:
            label += f"|t{support_track_ids[0]}"
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=2)
        draw.text((cx + 2, cy + 2), label, fill=color)
    return np.asarray(image, dtype=np.uint8)


def render_layer1_cloud(
    primitives: list[Primitive3D],
    cluster_ids: np.ndarray,
    current_objects: list[CurrentObject],
    size: tuple[int, int] = (960, 720),
) -> np.ndarray:
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = np.array([20, 20, 24], dtype=np.uint8)
    if not primitives:
        return canvas

    object_lookup = {int(obj.obj_id_local): obj for obj in current_objects}
    xyz_list = []
    color_list = []
    center_list = []
    center_color_list = []
    label_ids = []
    cluster_sizes: dict[int, int] = {}
    for prim_idx, primitive in enumerate(primitives):
        cluster_id = int(cluster_ids[prim_idx]) if prim_idx < len(cluster_ids) else -1
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + int(primitive.pixel_idx.shape[0])
        xyz = primitive.xyz.astype(np.float32)
        if xyz.size == 0:
            continue
        current_object = object_lookup.get(cluster_id)
        support_track_ids = current_object.support_track_ids if current_object is not None else []
        color = _cluster_color(cluster_id, support_track_ids)
        xyz_list.append(xyz)
        color_list.append(np.repeat(color[None, :], xyz.shape[0], axis=0))

    if not xyz_list:
        return canvas

    topk_clusters = {cluster_id for cluster_id, _ in sorted(cluster_sizes.items(), key=lambda item: -item[1])[:15]}
    for cluster_id in sorted(topk_clusters):
        cluster_xyz = np.concatenate(
            [primitives[idx].xyz for idx in range(len(primitives)) if int(cluster_ids[idx]) == cluster_id and primitives[idx].xyz.size > 0],
            axis=0,
        )
        if cluster_xyz.size == 0:
            continue
        current_object = object_lookup.get(cluster_id)
        support_track_ids = current_object.support_track_ids if current_object is not None else []
        center_list.append(cluster_xyz.mean(axis=0).astype(np.float32))
        center_color_list.append(_cluster_color(cluster_id, support_track_ids))
        label_ids.append(cluster_id)

    points = np.concatenate(xyz_list, axis=0)
    colors = np.concatenate(color_list, axis=0)
    pivot = points.mean(axis=0, keepdims=True)
    rot = rotation_matrix_xyz(rx_deg=18.0, ry_deg=-35.0, rz_deg=0.0)
    rotated = (points - pivot) @ rot.T

    xy = rotated[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = 0.82 * min(width / span[0], height / span[1])
    offset = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    proj = (xy - (min_xy + max_xy) / 2.0) * scale + offset
    proj[:, 1] = height - proj[:, 1]

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    for idx in np.argsort(rotated[:, 2]):
        x, y = proj[idx]
        if x < 1 or x >= width - 1 or y < 1 or y >= height - 1:
            continue
        draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=tuple(int(v) for v in colors[idx].tolist()))

    if center_list:
        centers = np.stack(center_list, axis=0)
        center_colors = np.stack(center_color_list, axis=0)
        rotated_centers = (centers - pivot) @ rot.T
        proj_centers = (rotated_centers[:, :2] - (min_xy + max_xy) / 2.0) * scale + offset
        proj_centers[:, 1] = height - proj_centers[:, 1]
        for label_idx, center_xy in enumerate(proj_centers):
            x, y = center_xy
            color = tuple(int(v) for v in center_colors[label_idx].tolist())
            cluster_id = label_ids[label_idx]
            current_object = object_lookup.get(cluster_id)
            support_track_ids = current_object.support_track_ids if current_object is not None else []
            label = f"c{cluster_id}"
            if support_track_ids:
                label += f"|t{support_track_ids[0]}"
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255), outline=color, width=2)
            draw.text((x + 5, y + 5), label, fill=color)
    return np.asarray(image, dtype=np.uint8)


def export_task34_layer1(
    frame: FramePacket,
    primitives: list[Primitive3D],
    cluster_ids: np.ndarray,
    current_objects: list[CurrentObject],
    active_tracks: list[TrackState],
    out_dir: str | Path,
) -> dict[str, str]:
    out_path = ensure_dir(out_dir)
    overlay_path = out_path / "layer1_clusters_overlay.png"
    cloud_path = out_path / "layer1_clusters_3d.png"
    meta_path = out_path / "layer1_meta.json"

    overlay = overlay_layer1_clusters(frame.rgb.astype(np.uint8), primitives, cluster_ids, current_objects)
    cloud = render_layer1_cloud(primitives, cluster_ids, current_objects)
    save_rgb(overlay_path, overlay)
    save_rgb(cloud_path, cloud)

    cluster_sizes = [len(obj.primitive_ids) for obj in current_objects]
    track_hist: dict[str, int] = {}
    for obj in current_objects:
        for track_id in obj.support_track_ids:
            key = str(track_id)
            track_hist[key] = track_hist.get(key, 0) + 1

    meta = {
        "scene_id": frame.scene_id,
        "frame_id": frame.frame_id,
        "num_primitives": len(primitives),
        "num_clusters": len(current_objects),
        "num_active_tracks": len(active_tracks),
        "cluster_size_stats": {
            "min": int(min(cluster_sizes)) if cluster_sizes else 0,
            "max": int(max(cluster_sizes)) if cluster_sizes else 0,
            "mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        },
        "support_track_histogram": track_hist,
        "outputs": {
            "layer1_clusters_overlay": str(overlay_path),
            "layer1_clusters_3d": str(cloud_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "layer1_clusters_overlay": str(overlay_path),
        "layer1_clusters_3d": str(cloud_path),
        "layer1_meta": str(meta_path),
    }


def _track_color(track_id: int) -> np.ndarray:
    return _id_color(20000 + int(track_id))


def overlay_layer2_assignments(
    rgb: np.ndarray,
    primitives: list[Primitive3D],
    current_objects: list[CurrentObject],
    active_tracks: list[TrackState],
    assoc_result,
) -> np.ndarray:
    canvas = rgb.astype(np.float32).copy()
    h, w = rgb.shape[:2]
    object_lookup = {int(obj.obj_id_local): obj for obj in current_objects}
    primitive_lookup = {primitive.prim_id: primitive for primitive in primitives}
    assigned_track_ids = getattr(assoc_result, "assigned_track_ids", {})
    track_lookup = {int(track.track_id): track for track in active_tracks}

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for obj_idx, current_object in enumerate(current_objects):
        track_id = assigned_track_ids.get(obj_idx)
        if track_id is None:
            color = np.array([250, 170, 40], dtype=np.uint8)
            label = f"c{current_object.obj_id_local}->new"
        else:
            color = _track_color(track_id)
            conf = track_lookup.get(track_id).confidence if track_id in track_lookup else 0.0
            label = f"c{current_object.obj_id_local}->t{track_id}"
            if conf > 0:
                label += f" ({conf:.2f})"

        coords_list = [
            primitive_lookup[prim_id].pixel_idx
            for prim_id in current_object.primitive_ids
            if prim_id in primitive_lookup and primitive_lookup[prim_id].pixel_idx.size > 0
        ]
        if not coords_list:
            continue
        coords = np.concatenate(coords_list, axis=0)
        ys = np.clip(coords[:, 0].astype(np.int32), 0, h - 1)
        xs = np.clip(coords[:, 1].astype(np.int32), 0, w - 1)
        canvas_np = np.asarray(image, dtype=np.float32)
        canvas_np[ys, xs] = 0.50 * canvas_np[ys, xs] + 0.50 * color.astype(np.float32)
        image = Image.fromarray(np.clip(canvas_np, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)

        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        cx, cy = int(xs.mean()), int(ys.mean())
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=tuple(int(v) for v in color.tolist()), width=2)
        draw.text((cx + 2, cy + 2), label, fill=tuple(int(v) for v in color.tolist()))
    return np.asarray(image, dtype=np.uint8)


def render_score_matrix(
    current_objects: list[CurrentObject],
    active_tracks: list[TrackState],
    assoc_result,
    cell_size: int = 44,
) -> np.ndarray:
    num_objects = len(current_objects)
    num_tracks = len(active_tracks)
    score_matrix = getattr(assoc_result, "score_matrix", np.zeros((num_objects, num_tracks), dtype=np.float32))
    candidate_mask = getattr(assoc_result, "candidate_mask", np.zeros((num_objects, num_tracks), dtype=bool))
    assignments = getattr(assoc_result, "assignments", {})

    header_h = 70
    label_w = 90
    width = max(label_w + max(num_tracks, 1) * cell_size + 20, 260)
    height = max(header_h + max(num_objects, 1) * cell_size + 20, 180)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)

    draw.text((12, 10), "Unary Score Matrix", fill=(20, 20, 20))
    if num_objects == 0:
        draw.text((12, 40), "No current objects.", fill=(100, 100, 100))
        return np.asarray(image, dtype=np.uint8)
    if num_tracks == 0:
        draw.text((12, 40), "No active tracks.", fill=(100, 100, 100))
        return np.asarray(image, dtype=np.uint8)

    for track_idx, track in enumerate(active_tracks):
        x = label_w + track_idx * cell_size
        draw.text((x + 6, 28), f"t{track.track_id}", fill=tuple(int(v) for v in _track_color(track.track_id).tolist()))

    for obj_idx, current_object in enumerate(current_objects):
        y = header_h + obj_idx * cell_size
        draw.text((10, y + 12), f"c{current_object.obj_id_local}", fill=(35, 35, 35))
        for track_idx, track in enumerate(active_tracks):
            x = label_w + track_idx * cell_size
            score = float(score_matrix[obj_idx, track_idx]) if obj_idx < score_matrix.shape[0] and track_idx < score_matrix.shape[1] else 0.0
            gated = bool(candidate_mask[obj_idx, track_idx]) if obj_idx < candidate_mask.shape[0] and track_idx < candidate_mask.shape[1] else False
            if gated:
                r = int(255 * (1.0 - score))
                g = int(220 * score + 35)
                b = int(80 * (1.0 - score))
                fill = (r, g, b)
            else:
                fill = (220, 220, 220)
            draw.rectangle([x, y, x + cell_size - 2, y + cell_size - 2], fill=fill, outline=(180, 180, 180), width=1)
            text_fill = (20, 20, 20) if score < 0.65 else (255, 255, 255)
            draw.text((x + 6, y + 12), f"{score:.2f}", fill=text_fill)
            if assignments.get(obj_idx) == track_idx:
                draw.rectangle([x + 1, y + 1, x + cell_size - 3, y + cell_size - 3], outline=(20, 20, 20), width=3)
    return np.asarray(image, dtype=np.uint8)


def export_task5_assoc(
    frame: FramePacket,
    primitives: list[Primitive3D],
    current_objects: list[CurrentObject],
    active_tracks: list[TrackState],
    assoc_result,
    out_dir: str | Path,
) -> dict[str, str]:
    out_path = ensure_dir(out_dir)
    overlay_path = out_path / "layer2_assignment_overlay.png"
    matrix_path = out_path / "score_matrix.png"
    meta_path = out_path / "layer2_meta.json"

    overlay = overlay_layer2_assignments(
        rgb=frame.rgb.astype(np.uint8),
        primitives=primitives,
        current_objects=current_objects,
        active_tracks=active_tracks,
        assoc_result=assoc_result,
    )
    matrix = render_score_matrix(
        current_objects=current_objects,
        active_tracks=active_tracks,
        assoc_result=assoc_result,
    )
    save_rgb(overlay_path, overlay)
    save_rgb(matrix_path, matrix)

    matched_pairs = []
    for obj_idx, track_id in getattr(assoc_result, "assigned_track_ids", {}).items():
        matched_pairs.append(
            {
                "obj_id_local": int(current_objects[obj_idx].obj_id_local),
                "track_id": int(track_id),
                "score": float(getattr(assoc_result, "match_scores", {}).get(obj_idx, 0.0)),
            }
        )
    meta = {
        "scene_id": frame.scene_id,
        "frame_id": frame.frame_id,
        "num_current_objects": len(current_objects),
        "num_active_tracks": len(active_tracks),
        "num_matches": len(matched_pairs),
        "solver": str(getattr(assoc_result, "solver", "none")),
        "matched_pairs": matched_pairs,
        "unmatched_objects": [
            int(current_objects[obj_idx].obj_id_local)
            for obj_idx in getattr(assoc_result, "unmatched_object_indices", [])
            if obj_idx < len(current_objects)
        ],
        "unmatched_track_ids": [
            int(active_tracks[track_idx].track_id)
            for track_idx in getattr(assoc_result, "unmatched_track_indices", lambda _n: [])(len(active_tracks))
            if track_idx < len(active_tracks)
        ],
        "outputs": {
            "layer2_assignment_overlay": str(overlay_path),
            "score_matrix": str(matrix_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "layer2_assignment_overlay": str(overlay_path),
        "score_matrix": str(matrix_path),
        "layer2_meta": str(meta_path),
    }
