"""GT-based over-seg sample mining for the quick graph test."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .assoc_l2 import UnaryAssocConfig, UnaryAssociator
from .cluster_l1 import Layer1ClusterConfig, Layer1Clusterer
from .common_types import CurrentObject, Primitive3D
from .io_seq import PosedRGBDSequence
from .mask_source import build_mask_source
from .primitive_build import PrimitiveBuilder, PrimitiveConfig
from .score_l1 import Layer1Config, Layer1Scorer
from .track_bank import QuickTrackBank
from .viz import ensure_dir


@dataclass
class FrameGtRecord:
    scene_id: str
    sample_index: int
    frame_id: int
    gt_instance_id: int
    visible_pixels: int
    before_frag: int
    after_frag: int
    before_best_overlap_pixels: int
    after_best_overlap_pixels: int
    after_best_cluster_id: int
    after_best_track_id: int
    after_best_cluster_purity: float
    after_overmerge: bool
    sample_candidate: bool


@dataclass
class SampleCandidate:
    scene_id: str
    gt_instance_id: int
    sample_type: str
    sample_index_start: int
    sample_index_end: int
    frame_start: int
    frame_end: int
    num_frames: int
    representative_sample_index: int
    representative_frame_id: int
    avg_visible_pixels: float
    avg_before_frag: float
    avg_after_frag: float
    avg_improvement: float
    overmerge_ratio: float
    mean_best_purity: float
    representative_track_id: int
    score: float
    frame_ids: list[int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine GT-based over-seg samples from a short scene sequence.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--cache-masks", required=True)
    parser.add_argument("--intrinsic-path", default=None)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--mask-mode", default="cache", choices=["cache", "empty", "oracle"])
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--gt-instance-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on sampled frames for quick debugging.")

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

    parser.add_argument("--window-length", type=int, default=12)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--min-gt-pixels", type=int, default=256)
    parser.add_argument("--min-overlap-pixels", type=int, default=32)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.05)
    parser.add_argument("--before-frag-thresh", type=int, default=4)
    parser.add_argument("--split-fixed-max-after", type=float, default=2.0)
    parser.add_argument("--split-fixed-min-improvement", type=float, default=1.5)
    parser.add_argument("--still-frag-after-thresh", type=float, default=3.0)
    parser.add_argument("--wrong-merge-max-purity", type=float, default=0.75)
    parser.add_argument("--wrong-merge-min-ratio", type=float, default=0.20)

    parser.add_argument("--max-samples-split-fixed", type=int, default=10)
    parser.add_argument("--max-samples-still-fragmented", type=int, default=5)
    parser.add_argument("--max-samples-wrong-merge", type=int, default=5)
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


def _cluster_gt_stats(
    cluster_map: np.ndarray,
    gt_map: np.ndarray,
    min_gt_pixels: int,
    wrong_merge_min_ratio: float,
) -> dict[int, dict[str, object]]:
    stats: dict[int, dict[str, object]] = {}
    cluster_ids = [int(v) for v in np.unique(cluster_map) if int(v) >= 0]
    for cluster_id in cluster_ids:
        cluster_bitmap = cluster_map == cluster_id
        cluster_size = int(cluster_bitmap.sum())
        if cluster_size <= 0:
            continue
        gt_vals = gt_map[cluster_bitmap]
        gt_vals = gt_vals[gt_vals > 0]
        if gt_vals.size == 0:
            stats[cluster_id] = {
                "pixel_count": cluster_size,
                "dominant_gt_id": -1,
                "dominant_pixels": 0,
                "purity": 0.0,
                "significant_gt_ids": [],
                "overmerge": False,
            }
            continue
        gt_ids, gt_counts = np.unique(gt_vals, return_counts=True)
        order = np.argsort(-gt_counts)
        gt_ids = gt_ids[order]
        gt_counts = gt_counts[order]
        dominant_gt_id = int(gt_ids[0])
        dominant_pixels = int(gt_counts[0])
        purity = dominant_pixels / max(cluster_size, 1)
        significant_gt_ids = [
            int(gt_id)
            for gt_id, count in zip(gt_ids.tolist(), gt_counts.tolist())
            if count >= max(int(min_gt_pixels * wrong_merge_min_ratio), int(cluster_size * wrong_merge_min_ratio), 32)
        ]
        stats[cluster_id] = {
            "pixel_count": cluster_size,
            "dominant_gt_id": dominant_gt_id,
            "dominant_pixels": dominant_pixels,
            "purity": float(purity),
            "significant_gt_ids": significant_gt_ids,
            "overmerge": len(significant_gt_ids) >= 2,
        }
    return stats


def _fragment_count(
    unit_map: np.ndarray,
    gt_bitmap: np.ndarray,
    min_overlap_pixels: int,
    min_overlap_ratio: float,
) -> tuple[int, list[tuple[int, int]]]:
    gt_pixels = int(gt_bitmap.sum())
    if gt_pixels <= 0:
        return 0, []
    overlap_thresh = max(int(min_overlap_pixels), int(round(min_overlap_ratio * gt_pixels)))
    unit_vals = unit_map[gt_bitmap]
    unit_vals = unit_vals[unit_vals >= 0]
    if unit_vals.size == 0:
        return 0, []
    unit_ids, unit_counts = np.unique(unit_vals, return_counts=True)
    overlaps = [
        (int(unit_id), int(count))
        for unit_id, count in zip(unit_ids.tolist(), unit_counts.tolist())
        if int(count) >= overlap_thresh
    ]
    overlaps.sort(key=lambda item: item[1], reverse=True)
    return len(overlaps), overlaps


def _mode_track_id(track_ids: list[int]) -> int:
    if not track_ids:
        return -1
    counter = Counter(track_ids)
    return int(counter.most_common(1)[0][0])


def _find_runs(sample_indices: list[int], min_len: int) -> list[list[int]]:
    if not sample_indices:
        return []
    runs: list[list[int]] = []
    current = [int(sample_indices[0])]
    for idx in sample_indices[1:]:
        idx = int(idx)
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            if len(current) >= min_len:
                runs.append(current)
            current = [idx]
    if len(current) >= min_len:
        runs.append(current)
    return runs


def _slice_window(indices: list[int], records_by_sample: dict[int, FrameGtRecord], window_length: int) -> list[int]:
    if len(indices) <= window_length:
        return indices
    best_pos = max(range(len(indices)), key=lambda pos: records_by_sample[indices[pos]].before_frag)
    half = window_length // 2
    start = max(0, best_pos - half)
    end = start + window_length
    if end > len(indices):
        end = len(indices)
        start = end - window_length
    return indices[start:end]


def _classify_window(
    records: list[FrameGtRecord],
    args: argparse.Namespace,
) -> tuple[str | None, float]:
    avg_before = float(np.mean([record.before_frag for record in records]))
    avg_after = float(np.mean([record.after_frag for record in records]))
    avg_improvement = avg_before - avg_after
    overmerge_ratio = float(np.mean([1.0 if record.after_overmerge else 0.0 for record in records]))
    mean_best_purity = float(np.mean([record.after_best_cluster_purity for record in records]))

    if overmerge_ratio >= args.wrong_merge_min_ratio or mean_best_purity <= args.wrong_merge_max_purity:
        score = overmerge_ratio * 4.0 + max(0.0, args.wrong_merge_max_purity - mean_best_purity) + 0.15 * avg_before
        return "wrong_merge", float(score)

    if (
        avg_after <= args.split_fixed_max_after
        and avg_improvement >= args.split_fixed_min_improvement
        and overmerge_ratio == 0.0
    ):
        score = avg_improvement + 0.1 * avg_before - 0.1 * avg_after
        return "split_fixed", float(score)

    if avg_after >= args.still_frag_after_thresh:
        score = avg_after + 0.25 * avg_before
        return "still_fragmented", float(score)

    return None, 0.0


class SampleMiner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sequence = PosedRGBDSequence(
            scene_root=Path(args.scene_root).resolve(),
            interval=args.interval,
            depth_scale=args.depth_scale,
            intrinsic_path=args.intrinsic_path,
        )
        self.mask_source = build_mask_source(
            mode=args.mask_mode,
            cache_root=Path(args.cache_masks).resolve(),
            gt_root=args.gt_root,
        )
        self.primitive_builder = PrimitiveBuilder(
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
        self.layer1_scorer = Layer1Scorer(Layer1Config())
        self.layer1_clusterer = Layer1Clusterer(
            Layer1ClusterConfig(
                merge_score_thresh=args.layer1_merge_thresh,
                negative_veto_thresh=args.layer1_negative_veto,
            )
        )
        self.unary_associator = UnaryAssociator(
            UnaryAssocConfig(
                top_k=args.assoc_top_k,
                match_thresh=args.assoc_match_thresh,
            )
        )
        self.track_bank = QuickTrackBank()
        self.gt_instance_dir = (
            Path(args.gt_instance_dir).resolve()
            if args.gt_instance_dir
            else Path(args.scene_root).resolve() / "instance"
        )

    def mine(self) -> dict[str, object]:
        frame_records: list[FrameGtRecord] = []
        num_frames = len(self.sequence) if self.args.max_frames is None else min(len(self.sequence), int(self.args.max_frames))

        for sample_index in range(num_frames):
            frame = self.sequence[sample_index]
            masks = self.mask_source.load_masks(frame.scene_id, frame.frame_id, image_shape=frame.rgb.shape[:2])
            active_tracks = self.track_bank.query_active()
            primitives = self.primitive_builder.build(frame, masks=masks)
            graph = self.layer1_scorer.build_graph(frame=frame, masks=masks, primitives=primitives, active_tracks=active_tracks)
            cluster_ids, current_objects = self.layer1_clusterer.cluster(
                graph=graph,
                primitives=primitives,
                masks=masks,
                active_tracks=active_tracks,
            )
            assoc_result = self.unary_associator.match(
                current_objects=current_objects,
                active_tracks=active_tracks,
                primitives=primitives,
            )
            self.track_bank.update_from_current_objects(
                current_objects=current_objects,
                primitives=primitives,
                frame_id=frame.frame_id,
                assignments=assoc_result.assigned_track_ids,
                match_scores=assoc_result.match_scores,
            )

            gt_map_raw = _load_gt_instance(self.gt_instance_dir, frame.frame_id).astype(np.int32)
            gt_ids, gt_counts = np.unique(gt_map_raw[gt_map_raw > 0], return_counts=True)
            valid_gt_ids = {
                int(gt_id)
                for gt_id, count in zip(gt_ids.tolist(), gt_counts.tolist())
                if int(count) >= int(self.args.min_gt_pixels)
            }
            gt_map = gt_map_raw.copy()
            if valid_gt_ids:
                valid_mask = np.isin(gt_map, np.array(sorted(valid_gt_ids | {0}), dtype=np.int32))
                gt_map[~valid_mask] = 0
            else:
                gt_map[:] = 0

            primitive_map = _label_map_from_primitives(primitives, frame.rgb.shape[:2])
            cluster_map = _label_map_from_clusters(primitives, cluster_ids, frame.rgb.shape[:2])
            cluster_stats = _cluster_gt_stats(
                cluster_map=cluster_map,
                gt_map=gt_map,
                min_gt_pixels=self.args.min_gt_pixels,
                wrong_merge_min_ratio=self.args.wrong_merge_min_ratio,
            )
            cluster_to_track = {
                int(current_objects[obj_idx].obj_id_local): int(track_id)
                for obj_idx, track_id in assoc_result.assigned_track_ids.items()
                if obj_idx < len(current_objects)
            }

            for gt_id in sorted(valid_gt_ids):
                gt_bitmap = gt_map == int(gt_id)
                visible_pixels = int(gt_bitmap.sum())
                before_frag, before_overlaps = _fragment_count(
                    unit_map=primitive_map,
                    gt_bitmap=gt_bitmap,
                    min_overlap_pixels=self.args.min_overlap_pixels,
                    min_overlap_ratio=self.args.min_overlap_ratio,
                )
                after_frag, after_overlaps = _fragment_count(
                    unit_map=cluster_map,
                    gt_bitmap=gt_bitmap,
                    min_overlap_pixels=self.args.min_overlap_pixels,
                    min_overlap_ratio=self.args.min_overlap_ratio,
                )

                after_best_cluster_id = int(after_overlaps[0][0]) if after_overlaps else -1
                after_best_overlap_pixels = int(after_overlaps[0][1]) if after_overlaps else 0
                before_best_overlap_pixels = int(before_overlaps[0][1]) if before_overlaps else 0
                best_cluster_stat = cluster_stats.get(after_best_cluster_id, None)
                after_overmerge = False
                after_best_purity = 0.0
                if best_cluster_stat is not None:
                    after_overmerge = bool(best_cluster_stat["overmerge"])
                    after_best_purity = float(best_cluster_stat["purity"])
                if any(bool(cluster_stats.get(cluster_id, {}).get("overmerge", False)) for cluster_id, _ in after_overlaps):
                    after_overmerge = True

                frame_records.append(
                    FrameGtRecord(
                        scene_id=frame.scene_id,
                        sample_index=int(sample_index),
                        frame_id=int(frame.frame_id),
                        gt_instance_id=int(gt_id),
                        visible_pixels=visible_pixels,
                        before_frag=int(before_frag),
                        after_frag=int(after_frag),
                        before_best_overlap_pixels=before_best_overlap_pixels,
                        after_best_overlap_pixels=after_best_overlap_pixels,
                        after_best_cluster_id=after_best_cluster_id,
                        after_best_track_id=int(cluster_to_track.get(after_best_cluster_id, -1)),
                        after_best_cluster_purity=float(after_best_purity),
                        after_overmerge=bool(after_overmerge),
                        sample_candidate=bool(before_frag >= self.args.before_frag_thresh),
                    )
                )

            if (sample_index + 1) % 20 == 0 or sample_index + 1 == num_frames:
                print(f"[sample_mining] processed {sample_index + 1}/{num_frames} sampled frames")

        candidates = self._build_candidates(frame_records)
        return {
            "frame_records": frame_records,
            "candidates": candidates,
            "num_processed_frames": num_frames,
        }

    def _build_candidates(self, frame_records: list[FrameGtRecord]) -> list[SampleCandidate]:
        grouped: dict[int, list[FrameGtRecord]] = defaultdict(list)
        for record in frame_records:
            grouped[int(record.gt_instance_id)].append(record)
        for rows in grouped.values():
            rows.sort(key=lambda row: row.sample_index)

        candidates: list[SampleCandidate] = []
        for gt_id, rows in grouped.items():
            candidate_indices = [row.sample_index for row in rows if row.sample_candidate]
            runs = _find_runs(candidate_indices, min_len=int(self.args.min_consecutive))
            records_by_sample = {row.sample_index: row for row in rows}
            for run in runs:
                window_indices = _slice_window(run, records_by_sample, window_length=int(self.args.window_length))
                window_records = [records_by_sample[idx] for idx in window_indices]
                sample_type, score = _classify_window(window_records, self.args)
                if sample_type is None:
                    continue
                representative = max(window_records, key=lambda row: row.before_frag - row.after_frag)
                track_id = _mode_track_id([row.after_best_track_id for row in window_records if row.after_best_track_id >= 0])
                candidates.append(
                    SampleCandidate(
                        scene_id=window_records[0].scene_id,
                        gt_instance_id=int(gt_id),
                        sample_type=sample_type,
                        sample_index_start=int(window_records[0].sample_index),
                        sample_index_end=int(window_records[-1].sample_index),
                        frame_start=int(window_records[0].frame_id),
                        frame_end=int(window_records[-1].frame_id),
                        num_frames=len(window_records),
                        representative_sample_index=int(representative.sample_index),
                        representative_frame_id=int(representative.frame_id),
                        avg_visible_pixels=float(np.mean([row.visible_pixels for row in window_records])),
                        avg_before_frag=float(np.mean([row.before_frag for row in window_records])),
                        avg_after_frag=float(np.mean([row.after_frag for row in window_records])),
                        avg_improvement=float(np.mean([row.before_frag - row.after_frag for row in window_records])),
                        overmerge_ratio=float(np.mean([1.0 if row.after_overmerge else 0.0 for row in window_records])),
                        mean_best_purity=float(np.mean([row.after_best_cluster_purity for row in window_records])),
                        representative_track_id=int(track_id),
                        score=float(score),
                        frame_ids=[int(row.frame_id) for row in window_records],
                    )
                )

        candidates = self._filter_candidates(candidates)
        return candidates

    def _filter_candidates(self, candidates: list[SampleCandidate]) -> list[SampleCandidate]:
        by_type: dict[str, list[SampleCandidate]] = defaultdict(list)
        for candidate in candidates:
            by_type[candidate.sample_type].append(candidate)

        limits = {
            "split_fixed": int(self.args.max_samples_split_fixed),
            "still_fragmented": int(self.args.max_samples_still_fragmented),
            "wrong_merge": int(self.args.max_samples_wrong_merge),
        }

        selected: list[SampleCandidate] = []
        for sample_type, rows in by_type.items():
            rows.sort(key=lambda row: row.score, reverse=True)
            chosen: list[SampleCandidate] = []
            for candidate in rows:
                overlap = False
                for existing in chosen:
                    if existing.scene_id != candidate.scene_id or existing.gt_instance_id != candidate.gt_instance_id:
                        continue
                    if not (
                        candidate.sample_index_end < existing.sample_index_start
                        or candidate.sample_index_start > existing.sample_index_end
                    ):
                        overlap = True
                        break
                if overlap:
                    continue
                chosen.append(candidate)
                if len(chosen) >= limits.get(sample_type, len(rows)):
                    break
            selected.extend(chosen)

        selected.sort(key=lambda row: (row.sample_type, -row.score, row.scene_id, row.frame_start))
        return selected


def _dump_outputs(
    scene_id: str,
    out_root: Path,
    frame_records: list[FrameGtRecord],
    candidates: list[SampleCandidate],
) -> dict[str, str]:
    samples_root = ensure_dir(out_root / "samples")
    frame_records_path = out_root / "frame_records.jsonl"
    index_path = out_root / "samples_index.json"
    summary_path = out_root / "mining_summary.json"

    with frame_records_path.open("w", encoding="utf-8") as fp:
        for record in frame_records:
            fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    sample_rows = [asdict(candidate) for candidate in candidates]
    index_path.write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    type_counter = Counter(candidate.sample_type for candidate in candidates)
    summary = {
        "scene_id": scene_id,
        "num_frame_records": len(frame_records),
        "num_candidates": len(candidates),
        "num_split_fixed": int(type_counter.get("split_fixed", 0)),
        "num_still_fragmented": int(type_counter.get("still_fragmented", 0)),
        "num_wrong_merge": int(type_counter.get("wrong_merge", 0)),
        "outputs": {
            "frame_records": str(frame_records_path),
            "samples_index": str(index_path),
            "samples_root": str(samples_root),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for sample_idx, candidate in enumerate(candidates):
        sample_dir = ensure_dir(samples_root / f"sample_{sample_idx:03d}")
        meta = asdict(candidate)
        meta["sample_id"] = f"sample_{sample_idx:03d}"
        meta["scene_id"] = scene_id
        meta_path = sample_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "frame_records": str(frame_records_path),
        "samples_index": str(index_path),
        "summary": str(summary_path),
        "samples_root": str(samples_root),
    }


def main() -> None:
    args = build_parser().parse_args()
    scene_root = Path(args.scene_root).resolve()
    repo_quick_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir).resolve() if args.out_dir else (repo_quick_root / "out_sample_mining" / scene_root.name)
    out_root = ensure_dir(out_root)

    miner = SampleMiner(args)
    result = miner.mine()
    outputs = _dump_outputs(
        scene_id=scene_root.name,
        out_root=out_root,
        frame_records=result["frame_records"],
        candidates=result["candidates"],
    )

    print("Task 6 sample mining complete:")
    print(f"  scene_id: {scene_root.name}")
    print(f"  num_processed_frames: {result['num_processed_frames']}")
    print(f"  num_frame_records: {len(result['frame_records'])}")
    print(f"  num_candidates: {len(result['candidates'])}")
    print(f"  frame_records: {outputs['frame_records']}")
    print(f"  samples_index: {outputs['samples_index']}")
    print(f"  summary: {outputs['summary']}")
    print(f"  samples_root: {outputs['samples_root']}")


if __name__ == "__main__":
    main()
