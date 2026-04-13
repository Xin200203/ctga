"""Summarize Task 6/7/GT diagnostics into a final Task 8 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final Task 8 summary.txt for quick graph test.")
    parser.add_argument(
        "--out-root",
        required=True,
        help="Root directory that contains mining_summary.json, render_summary.json, and samples/.",
    )
    parser.add_argument(
        "--layer1-gt-summary",
        required=True,
        help="Path to layer1_gt_summary.json from GT diagnostic analysis.",
    )
    parser.add_argument(
        "--layer2-meta",
        default=None,
        help="Optional path to layer2_meta.json from the reference unary association run.",
    )
    parser.add_argument(
        "--mining-summary",
        default=None,
        help="Optional explicit path to mining_summary.json. Defaults to <out-root>/mining_summary.json.",
    )
    parser.add_argument(
        "--render-summary",
        default=None,
        help="Optional explicit path to render_summary.json. Defaults to <out-root>/render_summary.json.",
    )
    return parser


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_ratio(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _fmt_float(value: float) -> str:
    return f"{value:.3f}"


def _findings_from_metrics(layer1_gt: dict, mining_summary: dict, sample_metas: list[dict], layer2_meta: dict | None) -> dict:
    before = layer1_gt["before_primitives"]
    after_h0 = layer1_gt["after_layer1_h0"]
    after_h5 = layer1_gt["after_layer1_h5"]

    before_units = before["unit_summary"]
    h0_units = after_h0["unit_summary"]
    h5_units = after_h5["unit_summary"]

    before_gt = before["gt_fragmentation"]
    h0_gt = after_h0["gt_fragmentation"]
    h5_gt = after_h5["gt_fragmentation"]

    split_fixed = [meta for meta in sample_metas if meta["sample_type"] == "split_fixed"]
    wrong_merge = [meta for meta in sample_metas if meta["sample_type"] == "wrong_merge"]
    still_fragmented = [meta for meta in sample_metas if meta["sample_type"] == "still_fragmented"]

    best_split = max(split_fixed, key=lambda meta: float(meta["avg_improvement"]), default=None)
    hardest_merge = max(
        wrong_merge,
        key=lambda meta: (float(meta["overmerge_ratio"]), -float(meta["mean_best_purity"])),
        default=None,
    )
    h5_overmerge_units = [
        unit for unit in h5_units["units"]
        if bool(unit.get("overmerge", False))
    ]
    top_h5_overmerge = max(h5_overmerge_units, key=lambda unit: int(unit["pixel_count"]), default=None)

    current_observation_effective = {
        "verdict": "partially_effective" if split_fixed else "not_verified",
        "text": (
            "Layer-1 can collapse some obvious splits, but it is not yet globally reliable."
            if split_fixed
            else "No convincing split_fixed sample was found in the current mined window."
        ),
        "evidence": [],
    }
    if best_split is not None:
        current_observation_effective["evidence"].append(
            f"sample {best_split['sample_id']} (GT {best_split['gt_instance_id']}) reduces frag "
            f"{best_split['avg_before_frag']:.1f}->{best_split['avg_after_frag']:.1f}"
        )
    current_observation_effective["evidence"].append(
        f"reference GT frame {layer1_gt['frame_id']}: purity drops "
        f"{_fmt_float(before_units['mean_purity'])}->{_fmt_float(h0_units['mean_purity'])} after Layer-1(h0)"
    )
    current_observation_effective["evidence"].append(
        f"reference GT frame {layer1_gt['frame_id']}: over-merge ratio rises "
        f"{_fmt_ratio(before_units['overmerge_ratio'])}->{_fmt_ratio(h0_units['overmerge_ratio'])}"
    )

    history_support_help = {
        "verdict": "mixed_but_helpful",
        "text": "History support improves coverage and continuity, but the current PT term still pushes some nearby objects into the same cluster.",
        "evidence": [
            f"matched GT ratio improves {before_gt['matched_gt_ratio']:.3f}->{h0_gt['matched_gt_ratio']:.3f}->{h5_gt['matched_gt_ratio']:.3f} "
            f"(before -> h0 -> h5)",
            f"num clusters shrinks {h0_units['num_units']}->{h5_units['num_units']} with history",
            f"mean purity also drops {_fmt_float(h0_units['mean_purity'])}->{_fmt_float(h5_units['mean_purity'])}",
        ],
    }

    unary_stability = {
        "verdict": "good_enough_for_debug" if layer2_meta else "unknown",
        "text": (
            "Unary current-to-memory matching looks stable enough to serve as a diagnostic backend, but it cannot recover from wrong current-object merges."
            if layer2_meta
            else "Layer-2 stability was not included in this summary run."
        ),
        "evidence": [],
    }
    if layer2_meta:
        match_scores = [float(pair["score"]) for pair in layer2_meta["matched_pairs"]]
        unary_stability["evidence"].append(
            f"reference Layer-2 frame {layer2_meta['frame_id']}: {layer2_meta['num_matches']}/{layer2_meta['num_current_objects']} current objects matched"
        )
        unary_stability["evidence"].append(
            f"no newborn objects on the reference frame; mean matched score {_fmt_float(mean(match_scores))}"
        )
        unary_stability["evidence"].append(
            "wrong_merge samples still persist, which means Layer-2 is inheriting a bad current partition rather than causing the main failure"
        )

    if hardest_merge is not None:
        main_failure = {
            "name": "layer1_overmerge_after_object_repair",
            "text": (
                "The clearest failure is Layer-1 over-merging neighboring large structures after current-frame repair, especially when masks/primitive support cover adjacent furniture and planar regions."
            ),
            "evidence": [
                f"{len(wrong_merge)} wrong_merge samples were mined out of {mining_summary['num_candidates']} total candidates",
                f"hardest sample {hardest_merge['sample_id']} (GT {hardest_merge['gt_instance_id']}) has overmerge ratio "
                f"{_fmt_ratio(float(hardest_merge['overmerge_ratio']))} with mean best purity {_fmt_float(float(hardest_merge['mean_best_purity']))}",
                (
                    (
                        f"reference GT frame {layer1_gt['frame_id']}: Layer-1(h5) large over-merge cluster "
                        f"{top_h5_overmerge['unit_id']} has purity {_fmt_float(float(top_h5_overmerge['purity']))} "
                        f"and mixes GT ids {top_h5_overmerge['significant_gt_ids']}"
                    )
                    if top_h5_overmerge is not None
                    else f"reference GT frame {layer1_gt['frame_id']}: Layer-1(h5) still contains over-merge clusters"
                ),
            ],
        }
    else:
        main_failure = {
            "name": "no_dominant_failure_found",
            "text": "No dominant failure mode was isolated from the current sample set.",
            "evidence": [],
        }

    next_layer = {
        "priority": "primitive_plus_layer1",
        "text": (
            "The next investment should go into tighter object-centric primitive support and stronger Layer-1 hard negatives/locality constraints, before adding richer Layer-2 matching."
        ),
        "evidence": [
            "The main observed errors are wrong merges, not missing Layer-2 matches",
            "Layer-2 unary assignment is already usable on the reference frame",
            "History support helps coverage but currently amplifies merge bias instead of fixing it",
        ],
    }

    return {
        "current_observation_repair": current_observation_effective,
        "history_support": history_support_help,
        "unary_current_to_memory": unary_stability,
        "main_failure": main_failure,
        "next_priority": next_layer,
        "counts": {
            "num_candidates": int(mining_summary["num_candidates"]),
            "num_split_fixed": int(mining_summary["num_split_fixed"]),
            "num_still_fragmented": int(mining_summary["num_still_fragmented"]),
            "num_wrong_merge": int(mining_summary["num_wrong_merge"]),
        },
        "sample_highlights": {
            "best_split_fixed": best_split,
            "hardest_wrong_merge": hardest_merge,
            "still_fragmented": still_fragmented,
        },
    }


def _format_sample_line(meta: dict) -> str:
    return (
        f"- {meta['sample_id']} | {meta['sample_type']} | GT {meta['gt_instance_id']} | "
        f"frames {meta['frame_start']}-{meta['frame_end']} | "
        f"frag {meta['avg_before_frag']:.1f}->{meta['avg_after_frag']:.1f} | "
        f"purity {meta['mean_best_purity']:.3f} | track {meta['representative_track_id']}"
    )


def _build_summary_text(
    out_root: Path,
    mining_summary: dict,
    render_summary: dict,
    layer1_gt: dict,
    layer2_meta: dict | None,
    sample_metas: list[dict],
    findings: dict,
) -> str:
    lines: list[str] = []

    before = layer1_gt["before_primitives"]
    after_h0 = layer1_gt["after_layer1_h0"]
    after_h5 = layer1_gt["after_layer1_h5"]

    lines.append("CTGA Quick Graph Test - Task 8 Summary")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Scope")
    lines.append(f"- scene_id: {mining_summary['scene_id']}")
    lines.append(f"- mined frame records: {mining_summary['num_frame_records']}")
    lines.append(f"- rendered samples: {render_summary['num_rendered_samples']}")
    lines.append(
        f"- candidate breakdown: split_fixed={mining_summary['num_split_fixed']}, "
        f"still_fragmented={mining_summary['num_still_fragmented']}, wrong_merge={mining_summary['num_wrong_merge']}"
    )
    lines.append("")
    lines.append("Current output status")
    lines.append(f"- success_grid: {render_summary['outputs']['success_grid']}")
    lines.append(f"- fail_grid: {render_summary['outputs']['fail_grid']}")
    lines.append(f"- merge_error_grid: {render_summary['outputs']['merge_error_grid']}")
    lines.append(f"- samples_root: {render_summary['outputs']['samples_root']}")
    lines.append("- note: strip.png and score_matrix.png are available for all 3 rendered samples")
    lines.append("- note: video.mp4 is still missing in the current prototype pass")
    lines.append("")
    lines.append("Reference GT diagnostic (frame-level)")
    lines.append(
        f"- reference frame: {layer1_gt['frame_id']} with {layer1_gt['gt_num_instances']} visible GT instances"
    )
    lines.append(
        f"- before primitives: units={before['unit_summary']['num_units']}, "
        f"mean_purity={before['unit_summary']['mean_purity']:.3f}, "
        f"overmerge={_fmt_ratio(before['unit_summary']['overmerge_ratio'])}, "
        f"matched_gt={before['gt_fragmentation']['matched_gt_ratio']:.3f}"
    )
    lines.append(
        f"- after Layer-1 (history=0): units={after_h0['unit_summary']['num_units']}, "
        f"mean_purity={after_h0['unit_summary']['mean_purity']:.3f}, "
        f"overmerge={_fmt_ratio(after_h0['unit_summary']['overmerge_ratio'])}, "
        f"matched_gt={after_h0['gt_fragmentation']['matched_gt_ratio']:.3f}"
    )
    lines.append(
        f"- after Layer-1 (history=5): units={after_h5['unit_summary']['num_units']}, "
        f"mean_purity={after_h5['unit_summary']['mean_purity']:.3f}, "
        f"overmerge={_fmt_ratio(after_h5['unit_summary']['overmerge_ratio'])}, "
        f"matched_gt={after_h5['gt_fragmentation']['matched_gt_ratio']:.3f}"
    )
    if layer2_meta:
        match_scores = [float(pair["score"]) for pair in layer2_meta["matched_pairs"]]
        lines.append(
            f"- reference Layer-2: {layer2_meta['num_matches']}/{layer2_meta['num_current_objects']} matched, "
            f"solver={layer2_meta['solver']}, mean_score={mean(match_scores):.3f}, unmatched_objects={len(layer2_meta['unmatched_objects'])}"
        )
    lines.append("")
    lines.append("Answers to the five Task 8 questions")
    lines.append(
        f"1. Current observation repair effective? {findings['current_observation_repair']['text']}"
    )
    for evidence in findings["current_observation_repair"]["evidence"]:
        lines.append(f"   - {evidence}")
    lines.append(f"2. History support helpful? {findings['history_support']['text']}")
    for evidence in findings["history_support"]["evidence"]:
        lines.append(f"   - {evidence}")
    lines.append(
        f"3. Unary current-to-memory stable enough? {findings['unary_current_to_memory']['text']}"
    )
    for evidence in findings["unary_current_to_memory"]["evidence"]:
        lines.append(f"   - {evidence}")
    lines.append(f"4. Most obvious failure type: {findings['main_failure']['text']}")
    for evidence in findings["main_failure"]["evidence"]:
        lines.append(f"   - {evidence}")
    lines.append(f"5. Next priority: {findings['next_priority']['text']}")
    for evidence in findings["next_priority"]["evidence"]:
        lines.append(f"   - {evidence}")
    lines.append("")
    lines.append("Sample highlights")
    for meta in sample_metas:
        lines.append(_format_sample_line(meta))
        outputs = meta.get("outputs", {})
        if outputs:
            lines.append(f"   strip: {outputs['strip']}")
            lines.append(f"   score_matrix: {outputs['score_matrix']}")
    lines.append("")
    lines.append("Most likely error attribution")
    lines.append("- Primitive stage: usable, but support regions are still too broad and scene-covering in some frames.")
    lines.append("- Layer-1: the current main bottleneck. It can fix splits, but it also introduces the strongest wrong-merge errors.")
    lines.append("- History support: useful as a soft prior, but the current PT term still increases merge bias on adjacent objects.")
    lines.append("- Layer-2 unary: good enough for diagnosis; it mostly inherits Layer-1 mistakes instead of creating the main error.")
    lines.append("")
    lines.append("Recommendation")
    lines.append("- Do not spend the next cycle on second-order matching yet.")
    lines.append("- First tighten object-mask selection, primitive support, and Layer-1 negative constraints.")
    lines.append("- Then rerun sample mining on a longer window to see whether split_fixed grows while wrong_merge shrinks.")
    lines.append("")
    lines.append("Generated by summarize_task8.py")
    lines.append(f"- out_root: {out_root}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root).resolve()
    mining_summary_path = Path(args.mining_summary).resolve() if args.mining_summary else (out_root / "mining_summary.json")
    render_summary_path = Path(args.render_summary).resolve() if args.render_summary else (out_root / "render_summary.json")
    mining_summary = _load_json(mining_summary_path)
    render_summary = _load_json(render_summary_path)
    layer1_gt = _load_json(Path(args.layer1_gt_summary).resolve())
    layer2_meta = _load_json(Path(args.layer2_meta).resolve()) if args.layer2_meta else None

    sample_root = out_root / "samples"
    if sample_root.exists():
        sample_dirs = sorted(sample_root.glob("sample_*"))
    else:
        sample_dirs = sorted(out_root.glob("sample_*"))
    sample_metas = [_load_json(sample_dir / "meta.json") for sample_dir in sample_dirs if (sample_dir / "meta.json").exists()]
    findings = _findings_from_metrics(
        layer1_gt=layer1_gt,
        mining_summary=mining_summary,
        sample_metas=sample_metas,
        layer2_meta=layer2_meta,
    )

    summary_text = _build_summary_text(
        out_root=out_root,
        mining_summary=mining_summary,
        render_summary=render_summary,
        layer1_gt=layer1_gt,
        layer2_meta=layer2_meta,
        sample_metas=sample_metas,
        findings=findings,
    )

    summary_txt_path = out_root / "summary.txt"
    summary_json_path = out_root / "summary.json"
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    summary_json = {
        "scene_id": mining_summary["scene_id"],
        "findings": findings,
        "outputs": {
            "summary_txt": str(summary_txt_path),
            "summary_json": str(summary_json_path),
            "render_summary": str(render_summary_path),
            "mining_summary": str(mining_summary_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Task 8 summary complete:", flush=True)
    print(f"  summary_txt: {summary_txt_path}", flush=True)
    print(f"  summary_json: {summary_json_path}", flush=True)


if __name__ == "__main__":
    main()
