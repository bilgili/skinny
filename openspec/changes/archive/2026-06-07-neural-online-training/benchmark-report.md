# neural-online-training - Benchmark Report (task 7.3)

**Date:** 2026-06-07
**Proposal:** `neural-online-training` (Stage 2 - online training of the neural
directional proposal)
**Harness:** `tests/bench_neural_online.py`

## Environment

| Field | Value |
|---|---|
| Vulkan device | NVIDIA GeForce RTX 4090 (driver 596.21) |
| Vulkan API | 1.3 (timeline semaphore + external memory/semaphore enabled) |
| Python | 3.13.3 (renderer venv at repo root) |
| Backend | Vulkan, `execution_mode="wavefront"` |
| Scene | `assets/cornell_box_emissive.usda` (8 instances) |
| Proposal preset | `bsdf,neural` |
| torch | **2.8.0+cu126** (real CUDA trainer active) |
| cuda-python | 13.3.1 (`[interop]` extra; interop handoff active) |
| spline_flow | sibling repo on path (+ matplotlib, a top-level import there) |

This is the complete task-7.3 run: both weight-handoff backends, the real CUDA
trainer interleaved with the Vulkan render in one process, a literally-moving
object, and frames-to-recover. The interop CUDA-write seam (task 5.2) and torch
in the renderer venv - the two things that were NVIDIA-box-pending in the earlier
file-handoff-only report - are now in place.

## Results (RTX 4090, 96x96, medians)

### 1. Handoff cost, isolated to `publish()` (file vs interop)

`publish()` only (no training), so the timing is the handoff machinery alone.

| backend | publish (med) | publish (p90) | render+swap (med) |
|---|---:|---:|---:|
| file (NFW1 write + reload) | 30.31 ms | 32.02 ms | 42.38 ms |
| interop (CUDA memcpy + timeline signal) | **0.46 ms** | 0.60 ms | 34.41 ms |

**interop publish is ~65x faster** than the file backend's NFW1 CPU round-trip
(0.46 ms vs 30.31 ms) - the no-CPU-round-trip win, quantified. `final_version ==
frames` on both, so the double-buffer lifecycle promotes a new version every frame
with zero dropped swaps. (Earlier file-only runs put publish at ~25-30 ms, flat
across resolution since the weight file is fixed-size; render+swap scales with
resolution as pixel-bound work.)

### 2. CUDA training concurrent with the Vulkan render (one process, interop)

Each frame: a real warm-started trainer cycle, then interop publish, then render.

| stage | median |
|---|---:|
| trainer cycle (64 Adam steps, autocast-fp16) | 2959.9 ms |
| interop publish | 0.24 ms |
| render + swap | 61.5 ms |

Weighted-NLL drops **-0.21 -> -0.41 over 12 frames** while rendering - the net
adapts live, CUDA training and Vulkan rendering interleaved on the one GPU.

### 3. Moving-object render (geometry animated per frame, interop)

One instance (the sphere, index 7) is translated each frame via a real TLAS
re-upload (`_reupload_instance_transforms`), online loop live:

| field | value |
|---|---:|
| object world displacement | 0.399 units |
| trainer cycle | 2856.8 ms |
| interop publish | 0.23 ms |
| render + swap (med / p90) | 58.7 ms / 66.3 ms |

render+swap holds steady under geometry motion + concurrent CUDA training - the
moving-object scene runs with no instability.

### 4. Frames-to-recover after a move (recency-weighted replay)

The real flow converges on radiance lobe A, then the object "moves" (mirror lobe
B at the **same** condition, fed each cycle into the recency-weighted replay), and
we count cycles until the NLL returns to the A-converged level.

| field | value |
|---|---:|
| converged NLL on A | -2.501 |
| NLL spike right after the move | -1.649 |
| **frames-to-recover** (eps 0.03) | **~12 cycles** |

Trajectory: -1.65, -1.80, -1.88, -2.17, -2.26, -2.35, -2.37, -2.41, -2.43, -2.45,
-2.45, -2.51, ... A clean spike then recovery as recency demotes the stale A
records. This is the live online-adaptation metric.

**Honesty note:** the records for (4) are synthesized-but-pose-grounded (the `wi`
lobe re-points at the moved pose). The live GPU record drain device-losts under
the 2 s Windows TDR on this box (task 1.2, a separate hardware seam; a follow-up
`wavefront-native-path-records` moves emission off the megakernel), so real live
records are not available here - but the trainer + recency-replay under test are
fully real.

### 5. CUDA trainer cycle, isolated

| field | value |
|---|---:|
| cycle (64 Adam steps) | 2661.6 ms |
| NLL first -> last | -0.190 -> -0.257 |

## How to reproduce

```bash
PYTHONUTF8=1 PYTHONPATH=src ./Scripts/python tests/bench_neural_online.py \
  --frames 64 --concurrent-frames 12 --moving-frames 12 --max-recover-cycles 30
```

The real trainer needs torch (CUDA build, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu126`),
`matplotlib` (a top-level import in `spline_flow/train.py`), and the `spline_flow`
sibling repo on the path. Without them the loop falls back to the placeholder
trainer and the handoff numbers still run.
