# metal-dispatch-hygiene Specification

## Purpose
TBD - created by archiving change metal-dispatch-hygiene. Update Purpose after archive.
## Requirements
### Requirement: Metal command buffers are watchdog-bounded

On the Metal backend, no single committed command buffer SHALL contain unbounded per-pixel loop
work. Long marches (volume delta tracking, subsurface walks) SHALL be capped per dispatch
(`SKINNY_METAL`-gated step/bounce caps, continued across accumulation frames) so each command
buffer completes within a watchdog-safe budget. Caps SHALL be tuned so the standing parity gates
still pass (a cap that shifts a gated metric is re-measured, not hidden).

#### Scenario: volume render never trips the watchdog

- **WHEN** `disney_cloud` renders on Metal (megakernel and wavefront) for a full gate-length
  accumulation
- **THEN** every command buffer completes without a GPU fault / watchdog kill, and the run
  produces the gated image

### Requirement: Metal context teardown runs on every exit path

`MetalContext` SHALL guarantee idempotent teardown (`wait_for_idle` + device close) on: normal
exit, exception propagation, interpreter shutdown (`atexit`), and SIGINT/SIGTERM. It SHALL support
the context-manager protocol, and the headless render entry points and app SHALL acquire it through
a scope that triggers teardown. Repeated `destroy()` calls SHALL be safe no-ops.

#### Scenario: exception between dispatches still tears down

- **WHEN** a headless render raises after dispatch N but before dispatch N+1
- **THEN** the context's teardown drains the queue and closes the device before the process exits
  (observable via the teardown hook having run)

#### Scenario: SIGINT tears down

- **WHEN** a rendering process receives SIGINT mid-accumulation
- **THEN** the signal path drains and closes the device before exit, and a subsequent process can
  construct a Metal device and dispatch normally

### Requirement: abandoned GPU work does not wedge the device

A kill-harness regression test SHALL prove GPU recovery: a subprocess that is SIGKILLed
mid-render (no chance to run teardown) SHALL leave the GPU usable — a fresh probe subprocess
SHALL construct a Metal device and complete a trivial dispatch within a fixed time budget. The
harness SHALL be gpu-marked and respect the one-guarded-Metal-process rule.

#### Scenario: SIGKILL mid-render leaves GPU usable

- **WHEN** a gpu-marked test SIGKILLs a child mid volume render and then runs a probe dispatch in
  a new subprocess
- **THEN** the probe completes within its time budget, proving queued work died with the child and
  no kernel kept running

### Requirement: Metal megakernel dispatch is watchdog-bounded by tiling

On the Metal backend, the megakernel frame SHALL be committed as a sequence of
screen-space row bands — one command buffer per band — so that each command
buffer's total work is bounded to `width × bandHeight` pixels, independent of the
active integrator's per-pixel breadth (path, BDPT connection matrix, SPPM eye
walk) or per-material shader cost (including inlined MaterialX graph materials).
No single committed megakernel command buffer SHALL cover the full frame when the
estimated per-pixel cost would risk the macOS GPU watchdog budget.

The band count SHALL be chosen from an integrator-aware per-pixel cost estimate
that scales with resolution (BDPT, the widest per-pixel work, using more and
smaller bands than the path tracer), overridable via
`SKINNY_METAL_MEGAKERNEL_BANDS` for tuning. Tiling SHALL be purely a commit-time
subdivision: the accumulation image persists across the bands of a frame and each
pixel is written exactly once, so the tiled output is bit-identical to a single
full-frame dispatch. The Vulkan backend SHALL be unaffected — it continues to
dispatch the full frame in one command buffer, and the shader addition SHALL be
`#if defined(SKINNY_METAL)`-gated so the Vulkan SPIR-V is byte-unchanged.

#### Scenario: BDPT megakernel on a graph-material scene completes without wedging

- **WHEN** the BDPT megakernel renders a scene bound to image-textured MaterialX
  graph materials on Metal (e.g. the regenerated `bathroom.usda` with 22 graph
  materials) at 1280×720
- **THEN** every per-band command buffer completes without a GPU fault or watchdog
  kill, the frame is produced, and a subsequent Metal dispatch on the same device
  still succeeds (the GPU is not wedged)

#### Scenario: tiling does not change the image

- **WHEN** the same megakernel frame is rendered with 1 band and with N bands on
  Metal
- **THEN** the accumulation images are bit-identical, and the parity-matrix gates
  and `megakernel ≡ wavefront` self-consistency anchor are unchanged

### Requirement: SPPM photon dispatch is watchdog-bounded by tiling

On the Metal backend, the SPPM phase-3 photon pass (`wfSppmPhotonTrace`) SHALL be
committed as a sequence of photon sub-batches — one command buffer per batch,
flushed at each boundary — so that each command buffer's work is bounded to
`batch × (visible points gathered per photon)`, independent of how many visible
points cluster into a grid cell (caustic focus). The per-pass photon count SHALL
NOT be reduced to bound the command buffer; tiling SHALL bound it by breadth while
the full `width × height` photon budget is emitted per pass. The batch size SHALL
be a `SKINNY_METAL`-gated, env-overridable calibration knob
(`SKINNY_SPPM_METAL_PHOTON_BATCH`).

The tiling SHALL be unbiased: photon flux is deposited by additive atomics into a
per-pass accumulator that is cleared once before the batch loop, and the radiance
resolve SHALL divide by the unchanged per-pass total `sppmPhotonsEmitted`, so the
tiled pass produces the same estimator as a single full-photon dispatch (modulo
integer-atomic ordering, which is exact). Off Metal (Vulkan, no watchdog) the pass
SHALL run as a single full-photon dispatch with base offset zero, behaviorally
unchanged.

#### Scenario: caustic SPPM scene never trips the watchdog

- **WHEN** `assets/glass_caustics_test.usda` renders on Metal under
  `--integrator sppm` with the full `width × height` per-pass photon budget for a
  gate-length accumulation
- **THEN** every command buffer completes without a GPU fault / watchdog kill, and
  a fresh probe subprocess afterward constructs a Metal device and completes a
  trivial dispatch within its time budget

#### Scenario: tiled photon pass is unbiased vs a single dispatch

- **WHEN** the same SPPM pass is rendered with the photon batch set larger than the
  photon count (single dispatch) and again with a smaller batch (multiple flushed
  dispatches), all else equal
- **THEN** the resolved radiance images match to within integer-atomic ordering,
  and neither exhibits the dark-starvation bias produced by reducing the per-pass
  photon count

#### Scenario: photon count is no longer starved to avoid the wedge

- **WHEN** SPPM renders on Metal after this change with
  `SKINNY_SPPM_METAL_PHOTON_CAP` unset
- **THEN** the per-pass photon count is the full `width × height` (not the prior
  262144 cap), and the mean radiance of a lit diffuse region converges toward the
  `path` anchor as samples accumulate rather than plateauing dark

