## 1. S0 — Baseline instrumentation

- [ ] 1.1 Add a `VkQueryPool` (timestamp) to `WavefrontBdptPass`; write timestamps around each stage in `record_dispatch` and expose a readback of per-stage milliseconds.
- [ ] 1.2 Capture a baseline: per-stage ms for a directional-only scene and an area-light scene at a fixed resolution/sample count (record in the change notes).
- [ ] 1.3 Confirm which stage(s) dominate (expected: `connect`, then `walk`) to validate the ordering of S1–S3.

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

- [ ] 6.1 Decide whether `resolve` needs compaction from the S0/S3 timestamps; implement only if it shows up.
- [ ] 6.2 Re-tune `STREAM_CAP` / tile count for the staged walk now that the walk no longer holds 14 vertices in registers; pick the value that minimizes total frame time without exceeding the VRAM budget.
- [ ] 6.3 Verify VRAM stays bounded by stream size (not resolution) with the new walk-state buffers; update buffer-sizing in `renderer.py`.
- [ ] 6.4 Run `.venv/bin/ruff check src/` and the full `.venv/bin/pytest` suite; fix regressions.
- [ ] 6.5 Update `Architecture.md` (descriptor binding map + module map) and the `project_wavefront_backend` memory with the staged BDPT pipeline; note the before/after timings.
- [ ] 6.6 Confirm the megakernel BDPT and wavefront path tracer are unchanged (no edits to their entry points); spot-check their A/B still green.
