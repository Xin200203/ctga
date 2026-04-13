"""Visualization helpers for the quick graph test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common_types import FramePacket, Mask2D, Primitive3D
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
