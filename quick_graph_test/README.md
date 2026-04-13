# Quick Graph Test

This directory contains the lightweight 12-hour diagnostic prototype described in
[docs/quick_12h_graph_test_plan.md](/Users/xin/Code/research/ctga/docs/quick_12h_graph_test_plan.md).

The goal is not to extend the full `src/ctga` system directly, but to build a
smaller, easier-to-debug prototype for:

- single-frame primitive over-segmentation
- Layer-1 observation repair
- Layer-2 unary current-to-memory association
- GT-based sample mining and strong visualization

## Layout

- `data/`: local placeholder for scene data roots or symlinks
- `cache_masks/`: local placeholder for cached 2D masks
- `out/`: exported overlays, debug packets, and visualization outputs
- `src/`: lightweight prototype source files

## Current Status

Task 1 is implemented:

- sequence reading
- mask cache reading
- RGB / depth / mask overlay export for a single frame

Task 2 is implemented:

- single-frame primitive over-segmentation on the image grid
- primitive lifting into 3D points / voxels
- `primitive_overlay_2d.png`
- `primitive_cloud_3d.png`
- primitive metadata export

Current default primitive settings are intentionally milder than the first
prototype pass:

- `connectivity=8`
- `tau_z=0.05`
- `tau_n_deg=30`
- `tau_c=35`
- `min_pixels=32`

This keeps the result in the "still over-segmented, but not visually exploded"
range for the first Layer-1 debugging pass.

Task 3 is implemented:

- hand-crafted Layer-1 `MP + PP` scoring
- primitive clustering into current objects
- `layer1_clusters_overlay.png`
- `layer1_clusters_3d.png`
- `layer1_meta.json`

Task 4 is implemented:

- lightweight history-aware track bank
- `PT` support added as a local geometric prior
- history rollout via `--history-frames N`
- per-frame track update summary in stdout

The current history term is intentionally conservative:

- tracks only help merge primitives with the same strong top-track support
- track support is gated by current-frame locality
- distant fragments are not allowed to merge only because they resemble the same track

Tasks 5+ remain stubbed with explicit interfaces so development can continue in
the order defined by the 12h plan.

## Current Entry Point

```bash
python -m quick_graph_test.src.run_quick_test \
  --scene-root /path/to/scene \
  --intrinsic-path /path/to/intrinsic.txt \
  --mask-mode cache \
  --cache-masks /path/to/mask_cache \
  --frame-index 5 \
  --history-frames 0
```

Outputs are written to:

```text
quick_graph_test/out/single_frame/<scene_id>/frame_xxxxxx/
```

Useful comparison runs:

```bash
# Task 3: no history support
python -m quick_graph_test.src.run_quick_test \
  --scene-root /path/to/scene \
  --intrinsic-path /path/to/intrinsic.txt \
  --mask-mode cache \
  --cache-masks /path/to/mask_cache \
  --frame-index 5 \
  --history-frames 0
```

```bash
# Task 4: add short history support
python -m quick_graph_test.src.run_quick_test \
  --scene-root /path/to/scene \
  --intrinsic-path /path/to/intrinsic.txt \
  --mask-mode cache \
  --cache-masks /path/to/mask_cache \
  --frame-index 5 \
  --history-frames 5
```
