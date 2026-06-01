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
- [ ] 3.5 Re-measure (S0 timestamps): confirm `connect_full` dispatches zero groups on directional-only scenes and the connect stage time drops.

## 4. S2 — Stage the eye walk

- [ ] 4.1 Define `EyeWalkState` (ray, throughput, pdfFwdOmega, misBsdfPdf, idx, rngState, flags, escaped, pixel); add the per-lane state buffer to set 1.
- [ ] 4.2 Add `wfBdptGenEye` (camera ray + eye[0]=camera, eye[1]=z1, env-NEE at z1, first BSDF sample, initial state) writing the first vertices to the `wfEye` VRAM slice.
- [ ] 4.3 Add `wfBdptIsectEye` (material-free trace + cutout skip + sphere-light emissive deposit + miss→env into `escaped`; classify alive vs dead and count).
- [ ] 4.4 Add `wfBdptExtendEye` (indirect over live lanes: build the vertex, env-NEE, BSDF sample, RR, write back the previous vertex's `pdfRev`); reuse `randomWalk` body logic from `integrators/bdpt.slang`.
- [ ] 4.5 Wire the eye-walk bounce loop in `record_dispatch` (gen → for b in 0..BDPT_MAX_DEPTH: intersect → build_args → scatter → extend), keeping the light walk + connect as the megakernel tail reading the same VRAM (hybrid).
- [ ] 4.6 Headless A/B vs megakernel bdpt (both scenes); confirm parity and identical RNG draw order; re-measure walk stage time.

## 5. S3 — Stage the light walk + standalone splat

- [ ] 5.1 Define `LightWalkState`; add its per-lane buffer to set 1.
- [ ] 5.2 Add `wfBdptGenLight` (`sampleLightOrigin` → light[0]; mark delta-light lanes so they skip the walk).
- [ ] 5.3 Add `wfBdptIsectLight` (trace + miss/non-flat break + classify/count) and `wfBdptExtendLight` (indirect: build vertex + BSDF sample + RR + `pdfRev`).
- [ ] 5.4 Add `wfBdptSplat` as a standalone stage after the light walk (iterate light vertices, project to camera, visibility, atomic into `lightSplatBuffer`); confirm it consumes no RNG for pinhole.
- [ ] 5.5 Wire the light-walk bounce loop + splat into `record_dispatch`, removing the megakernel walk tail; the full pipeline is now gen-eye → eye-walk → gen-light → light-walk → splat → connect → resolve.
- [ ] 5.6 Headless A/B vs megakernel bdpt (both scenes); confirm parity and RNG draw order; re-measure all stages.

## 6. Finalization

- [ ] 6.1 Decide whether `resolve` needs compaction from the S0/S3 timestamps; implement only if it shows up.
- [ ] 6.2 Re-tune `STREAM_CAP` / tile count for the staged walk now that the walk no longer holds 14 vertices in registers; pick the value that minimizes total frame time without exceeding the VRAM budget.
- [ ] 6.3 Verify VRAM stays bounded by stream size (not resolution) with the new walk-state buffers; update buffer-sizing in `renderer.py`.
- [ ] 6.4 Run `.venv/bin/ruff check src/` and the full `.venv/bin/pytest` suite; fix regressions.
- [ ] 6.5 Update `Architecture.md` (descriptor binding map + module map) and the `project_wavefront_backend` memory with the staged BDPT pipeline; note the before/after timings.
- [ ] 6.6 Confirm the megakernel BDPT and wavefront path tracer are unchanged (no edits to their entry points); spot-check their A/B still green.
