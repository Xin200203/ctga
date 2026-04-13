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

Later tasks remain stubbed with explicit interfaces so development can continue
in the order defined by the 12h plan.

## Current Entry Point

```bash
python -m quick_graph_test.src.run_quick_test \
  --scene-root /path/to/scene \
  --intrinsic-path /path/to/intrinsic.txt \
  --mask-mode cache \
  --cache-masks /path/to/mask_cache
```

Outputs are written to:

```text
quick_graph_test/out/single_frame/<scene_id>/frame_xxxxxx/
```
