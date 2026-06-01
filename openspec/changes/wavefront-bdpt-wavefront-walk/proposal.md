## Why

Wavefront `path` is fast but wavefront `bdpt` is slow. The path tracer earned
its speed from per-bounce compaction (counting-sort live lanes, indirect-dispatch
shade over only live, material-coherent lanes) and an intersect/shade kernel
split. The wavefront BDPT pass got none of that: `walk` / `connect` / `resolve`
each dispatch over the **full** stream every frame, dead lanes (background, non-flat
hits, short paths) keep occupying warp slots, the heavy `connectGeneric`+MIS
double-loop runs even when no area lights exist, and both the `walk` and `connect`
megakernels hold `eye[7]`+`lightPath[7]` (~1.7 KB/thread) of `BDPTVertex` in
registers for their whole duration — collapsing occupancy.

## What Changes

- Stage the BDPT **eye and light random walks** into per-bounce wavefront stages
  (`gen` → for each bounce: `intersect` → `build_args` → `scatter` →
  `extend`-indirect), mirroring the path tracer. Subpath vertices live in VRAM and
  are touched one at a time, eliminating the per-thread `eye[7]`+`lightPath[7]`
  register arrays during the walk.
- Keep the s=1 light-tracer **splat** as its own stage ordered after the light
  walk (RNG-order-safe: pinhole `sampleWi` draws no RNG).
- Compact and **strategy-split the connect stage**: route live lanes by subpath
  shape into `connect_nee` (emissive + `connectT1`) and `connect_full`
  (emissive + `connectT1` + generic + MIS), each indirect-dispatched over its own
  queue. Dead lanes are skipped; the heavy generic+MIS kernel dispatches zero
  groups on directional-only scenes.
- **Reuse** the path tracer's counting-sort machinery — `wfBuildArgs`,
  `wfScatter`, and the lane-slot/count/offset/queue/cursor/indirect buffers — at
  every compaction point.
- Add per-stage **GPU timestamp** instrumentation (VkQueryPool) to baseline and
  verify the win.
- Build **incrementally**, A/B-verifying the accumulated image against the
  megakernel BDPT (and the current wavefront BDPT) at each step, to the existing
  re-render noise floor.

No change to the rendered image, the integrator math, the BDPT scope (flat
first-hit, pinhole camera), or the public execution-mode/integrator selection.

**Outcome (measured):** the **connect compaction (S1) is the win** — 1.69× on
heavy area-light BDPT — and is always on. The **walk staging (S2/S3) is correct
(exact parity) but overhead-bound** (per-bounce dispatch/barrier cost outweighs
the occupancy gain on the tested scenes; ~3× slower on a trivial scene). It is
therefore shipped as a **selectable `--bdpt-walk` mode** (`megakernel` default =
connect compaction only; `eye` / `eye_light` opt into the staged eye / eye+light
walks) rather than the default — the win ships clean while the staged paths stay
available for hardware/scenes where occupancy, not dispatch count, is the limiter.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `wavefront-execution`: add a work-efficiency requirement for the wavefront
  `bdpt` integrator — the eye and light subpath walks SHALL be built through
  per-bounce staged dispatches with live-lane compaction, and the connection
  stage SHALL be compacted and split by strategy so dead lanes and the
  unused generic+MIS double-loop are not dispatched. Parity (equivalent
  accumulated image within tolerance) is preserved.

## Impact

- **Shaders**: `shaders/wavefront/wavefront_bdpt.slang` (split into staged
  entry points: `wfBdptGenEye`, `wfBdptIsectEye`, `wfBdptExtendEye`,
  `wfBdptGenLight`, `wfBdptIsectLight`, `wfBdptExtendLight`, `wfBdptSplat`,
  `wfBdptConnectNee`, `wfBdptConnectFull`, `wfBdptResolve`); reuses
  `shaders/wavefront/build_args.slang` / `compaction.slang` / `scatter.slang`.
  Estimator functions in `shaders/integrators/bdpt.slang` are reused unchanged.
- **Python**: `vk_wavefront.py` (`WavefrontBdptPass` — new per-lane walk-state
  buffers, shared counting-sort scratch, the staged bounce-loop `record_dispatch`,
  GPU-timestamp query pool); `renderer.py` (pass wiring, buffer sizing).
- **Tests**: extend the headless BDPT A/B harness for the staged pipeline and
  per-step parity.
- **Docs**: `Architecture.md` (descriptor binding map / module map),
  `project_wavefront_backend` memory.
- No new dependencies. Vulkan-only (consistent with the wavefront feature).
