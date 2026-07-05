## Why

`wavefront-nonflat-path-fallback` fixed the wavefront BDPT/SPPM black-out for the
**terminal** non-flat types (`SUBSURFACE`, `SKIN`) — whose `estimateRadiance`
evaluates one vertex and stops. It deliberately left the **non-terminal** types
(`VOLUME`, `PYTHON`) bailing to black, because their `estimateRadiance` continues
bouncing (up to `MAX_BOUNCES`), and the wavefront BDPT/SPPM Metal path records the
whole frame into a single `MetalFrameEncoder` and submits **once** — an unbounded
command buffer under the macOS GPU watchdog (metal-dispatch-hygiene) at stream caps
of `1<<18` (BDPT) / `1<<20` (SPPM) lanes.

The megakernel renders these same non-flat pixels (`main_pass.slang` routes every
non-flat first hit to the path tracer) and stays watchdog-safe by committing the
frame in **row bands** (`metal-megakernel-watchdog-tiling`). The wavefront path
already streams in tiles; the only missing piece is committing the heavy eye stage
per tile so no single command buffer runs an unbounded multi-bounce path over the
full frame.

Confirmed: `assets/cornell_box_python_material.usda` (a `PYTHON` material) renders
under wavefront **path** (center-mean 0.72) but is **black** under wavefront BDPT
and SPPM (center-mean 0.00) on the shipped terminal-only fallback.

## What Changes

- **Un-gate the fallback** in the three eye-stage sites (`wfBdptWalk`,
  `wfBdptGenEye`, `wfSppmEye`): a non-flat first hit of *any* type now
  path-integrates via `PathTracer.estimateRadiance` (removing the
  `SUBSURFACE || SKIN` gate), matching the megakernel's non-flat→path routing for
  every type.
- **Bound the heavy eye stage per tile on Metal.** Add `flush()` to the wavefront
  recorder protocol (Metal → `MetalFrameEncoder.flush()`, which submits + drains +
  reopens a fresh encoder — the primitive already used by the indirect
  CPU-readback fallback; Vulkan → no-op, no watchdog). `record_bdpt_loop` and
  `record_sppm_loop` call it per eye tile when a new `bound_heavy_eye` flag is set,
  so each committed command buffer covers at most one `stream_size` tile of the
  multi-bounce fallback — the same bounded shape as a megakernel row band.
- **Host flag.** The renderer sets `bound_heavy_eye` when the scene contains a
  non-terminal non-flat material (`MATERIAL_TYPE_VOLUME` or `MATERIAL_TYPE_PYTHON`)
  under wavefront BDPT/SPPM. Scenes without one keep the single-submit path
  (byte-identical, no per-tile flush overhead); terminal-only scenes (subsurface /
  skin) are unaffected.
- **Tile-size cap (codex review).** Per-tile flush bounds *accumulation*, but one
  full-frame tile is itself up to `1<<20` (SPPM) lanes — a 1024² volume eye tile of
  6-bounce walks could trip the watchdog even flushed. So for heavy scenes the
  Metal BDPT/SPPM eye `stream_size` is also capped to the megakernel BDPT band
  (`_METAL_WAVEFRONT_HEAVY_EYE_BAND_LANES = 200_000`), so each committed command
  buffer is ≤ the proven-safe band size. Validated: 1024² clouds-SPPM completes,
  GPU usable after.
- **Metal dispatch hygiene.** The change is validated by the kill harness
  (`tests/test_metal_cleanup.py -m gpu`: SIGKILL-mid-render → GPU-usable) — the
  authoritative watchdog check — plus a render/parity gate.

Notes: `VOLUME` under BDPT/SPPM still has no *bidirectional/photon* medium
transport — the eye-visible pixels shade via the path fallback (as the megakernel
does), the connection/photon strategies are still volume-blind (recorded
exclusion). Vulkan output and flat/terminal scenes stay byte-identical.
