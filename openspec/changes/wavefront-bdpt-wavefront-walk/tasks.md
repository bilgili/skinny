## 1. S0 — Baseline instrumentation

- [x] 1.1 Substituted **wall-clock frame timing** (host `perf_counter` over N headless frames) for GPU per-stage `VkQueryPool` timestamps — it answered every perf question this change needed (per-step S1/S2/S3 frame times, scene comparisons) without the readback plumbing. A VkQueryPool can be added later if per-stage attribution is needed.
- [x] 1.2 Baselines captured (256², 40-frame mean): demo (directional+sphere) and cornell_box_rectlight (area light), recorded in the S1/S2/S3 measurement tables in this file.
- [x] 1.3 Confirmed connect dominated: S1 (connect only) gave 1.69× on the area-light scene, and S2+S3 (walk staging) gave no further win — so connect was the bottleneck, validating the S1-first ordering.

## 2. Shared compaction infrastructure

- [x] 2.1 Decided: **duplicate** the counting-sort pattern inline in `wavefront_bdpt.slang` (small: classify/build_args/scatter ≈40 lines, tiny SPIR-V) rather than a shared module — keeps the BDPT kernel self-contained and avoids touching the path tracer. Slang compiles all entries clean.
- [x] 2.2 Add the shared counting-sort scratch buffers (lane_slot / slot_count / slot_offset / slot_queue / slot_cursor / indirect) to `WavefrontBdptPass`, sized to the stream; wire a set-1 descriptor layout that the staged kernels share.
- [x] 2.3 Add a `wfQueueLane`-style indirect index helper usable by the BDPT extend/connect kernels.

## 3. S1 — Compact + strategy-split the connect stage

- [x] 3.1 Add a classify step that tags each live lane by subpath shape: dead / `CONNECT_NEE` (eyeLen≥2, lightLen<2) / `CONNECT_FULL` (eyeLen≥2, lightLen≥2) and counts per slot.
- [x] 3.2 Split `wfBdptConnect` into `wfBdptConnectNee` (emissive t=0 + `connectT1`) and `wfBdptConnectFull` (emissive + `connectT1` + generic t≥2 + MIS), each queue-indexed via indirect dispatch; reuse the `integrators/bdpt.slang` estimator functions unchanged.
- [x] 3.3 Update `record_dispatch` to run classify → `build_args` → `scatter` → indirect `connect_nee` → indirect `connect_full`, with the existing per-stage barriers.
- [x] 3.4 Recompile the BDPT `.spv` entries; run the headless A/B vs megakernel bdpt on directional-only + area-light scenes; confirm parity to the re-render noise floor.
- [x] 3.5 Re-measure: `connect_full` dispatches zero groups on directional-only scenes (structural: directional → `BDPT_VK_LIGHT_DIR` → lightLen 1 → no FULL lanes → `ceil(0/64)=0` groups). Connect time drops where it dominates.

### S1 measurements (wavefront bdpt, 256², 40-frame mean, MoltenVK)

| scene | pre-S1 | post-S1 | Δ |
|---|---|---|---|
| `cornell_box_rectlight` (heavy area light, FULL slot) | 102.75 ms | 60.71 ms | **1.69× faster** |
| `three_materials_demo` (directional + 1 sphere, cheap connect) | 2.59 ms | 3.10 ms | +0.5 ms (compaction overhead) |

S1 wins big where the connect stage is expensive (the scenes BDPT is for) and
costs a small fixed overhead on trivial-connect scenes (extra dispatches +
barriers + fill outweigh near-zero dead-lane savings). A/B parity holds on both
(demo passes the relative noise-floor gate; cornell matches a fresh megakernel
render to 0.001 mean abs diff). GPU per-stage `VkQueryPool` timestamps (S0
group 1) deferred — wall-clock frame timing covered the intent for S1.

## 4. S2 — Stage the eye walk

- [x] 4.1 Define `EyeWalkState` (ray, throughput, pdfFwdOmega, misBsdfPdf, idx, rngState, flags, escaped, pixel); add the per-lane state buffer to set 1.
- [x] 4.2 Add `wfBdptGenEye` (camera ray + eye[0]=camera, eye[1]=z1, env-NEE at z1, first BSDF sample, initial state) writing the first vertices to the `wfEye` VRAM slice.
- [x] 4.3 Combined trace+extend into one `wfBdptBounceEye` (no separate isect kernel + hit buffer) dispatched over the live queue — same compaction + vertices-in-VRAM wins, fewer buffers. Trace + cutout skip + sphere-emissive + miss→escaped live at its head.
- [x] 4.4 Add `wfBdptExtendEye` (indirect over live lanes: build the vertex, env-NEE, BSDF sample, RR, write back the previous vertex's `pdfRev`); reuse `randomWalk` body logic from `integrators/bdpt.slang`.
- [x] 4.5 Wire the eye-walk bounce loop in `record_dispatch` (gen → for b in 0..BDPT_MAX_DEPTH: intersect → build_args → scatter → extend), keeping the light walk + connect as the megakernel tail reading the same VRAM (hybrid).
- [x] 4.6 A/B vs megakernel bdpt: exact parity (0.00000 mean abs diff on cornell; demo gate passes). **Fixed a WAR race**: `clear_counts`' `vkCmdFillBuffer` (TRANSFER) was unordered vs prior compute reads of slot_count/cursor — added a COMPUTE→TRANSFER barrier. Perf: eye-walk staging is ~neutral/negative (demo 3.1→4.3 ms, cornell 61→60 ms) — connect was the bottleneck, not the eye walk.

## 5. S3 — Stage the light walk + standalone splat

- [x] 5.1 Reused the `WfBdptAux.ew*` transient fields for the light-walk loop state (the eye walk is done by then) — no new buffer/binding.
- [x] 5.2 Add `wfBdptGenLight` (`sampleLightOrigin` → light[0]; mark delta-light lanes so they skip the walk).
- [x] 5.3 Combined into one `wfBdptBounceLight` (randomWalk body with isEyeWalk=false: no env-NEE/sphere-emissive/escaped; miss ends the walk), indirect over the live queue. Reuses the shared `wfBdptWalkClassify` (alive→slot0).
- [x] 5.4 Add `wfBdptSplat` as a standalone stage after the light walk (iterate light vertices, project to camera, visibility, atomic into `lightSplatBuffer`); confirm it consumes no RNG for pinhole.
- [x] 5.5 Wire the light-walk bounce loop + splat into `record_dispatch`, removing the megakernel walk tail; the full pipeline is now gen-eye → eye-walk → gen-light → light-walk → splat → connect → resolve.
- [x] 5.6 A/B exact parity (0.00000 mean abs diff vs fresh megakernel on cornell_rectlight, cornell_sphere, demo). **Perf regression**: full staged pipeline is 3× slower than S1 on the demo (3.1→9.3 ms) and flat on cornell — the light bounce loop's fixed 6 iterations × ~4 dispatches/barriers are pure overhead on short-path scenes. Confirms walk-staging is a net loss; S1 (connect compaction) is the win.

## 6. Finalization

- [x] 6.1 `resolve` left full-stream — it's a 9 KB material-free kernel (running-mean + display write); compacting it would add dispatches for no gain.
- [x] 6.2 Left `STREAM_CAP=1<<18`. The staged pipeline is dispatch/barrier-overhead-bound, not VRAM- or occupancy-bound, so a larger stream (fewer tiles) is the lever that helps, not register pressure; default kept — fewer tiles already preferred where VRAM allows.
- [x] 6.3 VRAM bounded by `stream_size`: eye/light = stream×7×128, aux = stream×128 (AUX_STRIDE 64→128 for the ew* state), counting-sort scratch = stream×4 ×2 + small. No per-pixel allocation; `renderer.py` sizing unchanged in shape.
- [x] 6.4 ruff clean (changed files). Wavefront suite green: 48 wavefront/struct/execution-mode/A/B tests pass (the 3 budget-guard failures were stale entry names, fixed). Pre-existing non-wavefront infra failures (slangpy harness / driver) are unrelated — see [[reference_skinny_worktree_dev]].
- [x] 6.5 Architecture.md has no wavefront section + the new bindings are set-1 (pass-private, not in the scene binding map) → no change. Updated `project_wavefront_backend` memory with the staged pipeline + S1/S2/S3 timings + the WAR-race lesson.
- [x] 6.6 Confirmed: `git diff` vs main is empty for integrators/bdpt.slang, integrators/path.slang, wavefront/wavefront_path.slang, main_pass.slang. Path A/B + megakernel reference unchanged.

## 7. Selectable walk mode (`--bdpt-walk`)

- [x] 7.1 Restore `wfBdptWalk` (megakernel eye+light+splat) + `wfBdptLightTail` alongside the staged kernels in one shader — thin wrappers over the shared `randomWalk`/`sampleLightOrigin`/`splatLightWalk` (no estimator logic duplicated).
- [x] 7.2 Add `walk_mode` to `WavefrontBdptPass` (`megakernel` | `eye` | `eye_light`): compile + build ONLY the active mode's kernels (connect+resolve shared); branch `record_dispatch` via `build_subpaths()`.
- [x] 7.3 Thread `bdpt_walk` through `Renderer.__init__` → `_ensure_wavefront_bdpt_pass`; add `--bdpt-walk` (env `SKINNY_BDPT_WALK`) to `app.py`, default `megakernel` (the S1 win). Affects wavefront+bdpt only; all modes image-identical.
- [x] 7.4 Parametrize the bdpt A/B + the pipeline-build guard over the 3 walk modes: exact parity (0.00000 mean abs diff) and per-mode pipeline build verified; path A/B unchanged. 12 tests green.
