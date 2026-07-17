# Design — MLT integrator (PSSMLT over BDPT)

## Context

skinny has three integrators — `path` (0), `bdpt` (1), `sppm` (2) — registered in
`cli_common.INTEGRATOR_INDEX` / `renderer.integrator_modes`, with SPPM as the
wavefront-only precedent (auto mode derivation, startup refusal, per-frame
multi-phase staged sequence, persistent progressive state, Metal breadth-tiled
dispatch). The wavefront BDPT stack (`shaders/wavefront/wavefront_bdpt.slang`
wrapping `integrators/bdpt.slang`'s `randomWalk` / `sampleLightOrigin` /
`misWeight` / `atomicSplatRadiance`) already evaluates every (s,t) strategy and
splats into `lightSplatBuffer` (binding 21, uint fixed-point). The RNG is a
single scalar PCG-32 (`common.slang createRNG`); **no primary-sample-space
abstraction exists** — that is the main net-new machinery.

Reference implementation: pbrt-v4 at `~/projects/pbrt-v4` —
`src/pbrt/cpu/integrators.{h,cpp}` (`MLTIntegrator`, lines 2478–2750) and
`src/pbrt/samplers.{h,cpp}` (`MLTSampler`, lazy Kelemen mutations). pbrt's
`MLTSampler::EnsureReady` explicitly `LOG_FATAL`s on GPU (dynamic vector
resize), so the GPU port must fix the primary-sample dimension budget at
compile/pipeline-build time.

## Goals / Non-Goals

**Goals:**
- `mlt` as a fourth integrator: PSSMLT driving the existing BDPT strategy set,
  matching pbrt-v4's estimator (bootstrap b-normalization, per-mutation dual
  splat, depth-stratified chains) so pbrt-truth gating is meaningful.
- Wavefront-only, both backends, Metal dispatch hygiene, full registration
  (CLI × 4 front-ends, GUI cycle, state hash, pbrt import, parity matrix).
- Land harness-first behind an `MLT_IMPLEMENTED` capability gate (mirror
  `spectral_capability.SPECTRAL_IMPLEMENTED`) so registration/refusals/tests can
  merge before or with the transport, and nothing silently renders `path`.

**Non-Goals (v1):**
- No spectral MLT, no neural proposal, no ReSTIR reuse, no online training.
- No megakernel variant.
- No skin/subsurface/volume chain targets (flat-material envelope; scenes
  dominated by non-flat transport are recorded parity skips).
- No advanced mutations (manifold/MEMLT, delayed rejection) — PSSMLT only.

## Decisions

### D1. Chain-parallel mapping: many short GPU chains, fused evaluation kernel

One GPU lane = one Markov chain. Default `nChains = 16384` (CLI/import
override), each advancing one mutation per dispatch iteration; a frame runs
`ceil(pixels × mutations_per_pixel_per_frame / nChains)` iterations so the
per-frame mutation budget is proportional to pixel count (pbrt semantics,
progressive).

Proposal evaluation (`MLTIntegrator::L`) runs in a **fused per-chain kernel**
(`wfMltMutate`): strategy selection, eye subpath of exactly `t` vertices, light
subpath of exactly `s` vertices, single-strategy connection ×`nStrategies`,
calling `bdpt.slang`'s `randomWalk` / `sampleLightOrigin` / `misWeight`
directly — the same code the fused `wfBdptWalk` walk mode already runs per
pixel. *Alternative considered:* staging chains through the counting-sorted
BDPT queue kernels (`wfBdptGenEye/BounceEye/...`). Rejected for v1: per-chain
variable `(s,t)` makes the queue bookkeeping the hard part, and the fused BDPT
walk is already the default production path, so the fused kernel is both
simpler and precedented. Staged split is a recorded follow-up if register
pressure or divergence measurably hurts.

Chains are **assigned to depth strata contiguously** (sorted by their bootstrap
depth `k`) so warps mutate mostly-equal path lengths — cheap divergence win, no
estimator change.

### D2. Primary-sample-space sampler: fixed-size per-chain X vector

New `MLTSampler` struct in `integrators/mlt_sampler.slang` mirroring pbrt
exactly, minus dynamic resize:

- `PrimarySample = { value: f32, valueBackup: f32, lastMod: u32, modBackup: u32 }`
  (16 B). Iteration counters are u32 (a chain never exceeds 2³² iterations).
- Fixed dimension budget computed at pipeline build: 3 streams
  (camera/light/connection, pbrt's `GetNextIndex = streamIndex + 3·sampleIndex`)
  × `DIMS_PER_STREAM` derived from `maxDepth` (per-vertex dimension worst case ×
  `(maxDepth + 2)` + fixed header dims). Overflow dimensions (should not occur
  for the computed budget; defensive) fall back to decorrelated hash RNG —
  per-spec, never an out-of-bounds read.
- Lazy Kelemen mutation in `ensureReady`: large-step reset via
  `lastLargeStepIteration`, aggregated small step `σ·√nSmall` Gaussian sampled
  by inverse-CDF (1 uniform per dimension — Box-Muller would consume 2 and
  break the PSS mapping; an `erfInv`-based `sampleNormal` is **net-new shader
  code**, no such helper exists today), wrap to `[0,1)`; `accept()` /
  `reject()` backup-restore semantics verbatim from pbrt.
- Per-chain scalar PCG (`createRNG(chainId, seed)`) drives mutation randomness
  and the accept decision. This is a **deliberate two-stream deviation from
  pbrt** (which reuses the bootstrap-index-seeded rng for mutations): the
  bootstrap replay stream (D3) must stay bit-reproducible from
  `bootstrapIndex`, while mutation randomness only needs per-chain
  independence. Do not unify them — unifying desyncs the replay.

The **flat-material envelope is a hard prerequisite of the fixed budget, not a
scope preference**: pbrt's `RandomWalk` medium branch consumes one dimension
per null-collision (Woodcock) — unbounded — which is exactly why pbrt's
`MLTSampler::X` resizes dynamically and `LOG_FATAL`s on GPU. Excluding
volumes/media/subsurface is what makes a compile-time budget possible.
Counting pbrt's actual per-stream draws at `maxDepth = 5` (camera ≈ 25, light
≈ 20, connection ≈ 3 Get1D-equivalents) the worst-case X index is ≈ 72; the
192-dim budget is a proven over-estimate, so the overflow path is
**unreachable in-envelope** and is a hard invariant (debug assert /
NaN-poison), not a statistical fallback — a hash-RNG dimension is fresh every
evaluation, is not restored on reject, and would break detailed balance
(biased). It exists solely as memory-safety.

Memory: `maxDepth = 5` → ~192 dims × 16 B ≈ 3 KB/chain ≈ 48 MB at 16384 chains.
Sized by `nChains` (not `stream_size`) and MSL-stride-aware via a new
`mlt_buffer_sizes` in `wavefront_layout.py` (SPPM `sppm_buffer_sizes`
precedent). Chain metadata buffer holds `{depth, cCurrent, pCurrent, LCurrent
(rgb), rngState, iteration counters}`.

*Alternative considered:* pbrt-GPU-style re-derivation of samples from seeds
each iteration (no stored X). Rejected: loses the aggregated-small-step
semantics and makes reject-restore expensive; stored X is the standard GPU
PSSMLT design.

### D3. Bootstrap: GPU evaluation, host resample, RNG-reconstructible seeds

Bootstrap phase at every accumulation reset (state-hash change):

1. GPU: `nBootstrap × (maxDepth + 1)` evaluations of the fused L kernel in
   "fresh sample" mode (X filled from `createRNG(bootstrapIndex, seed)`), each
   writing luminance to a weights buffer. Breadth-tiled like the SPPM photon
   phase.
2. Host readback (once per reset): numpy CDF over weights, `b = (maxDepth+1)/N ×
   Σw`; all-zero weights → loud "no light-carrying paths" error. Sample
   `nChains` bootstrap indices proportional to weight; chain depth
   `k = index mod (maxDepth+1)`; upload `{bootstrapIndex, depth}` per chain.
3. First mutation iteration reconstructs each chain's current state by
   re-evaluating L with the seed-derived X (identical to pbrt constructing
   `MLTSampler(rngSequenceIndex = bootstrapIndex)`), storing `cCurrent`,
   `LCurrent`, `pCurrent`. **Initial bookkeeping is pinned to pbrt exactly**:
   the reconstruction evaluates L at `currentIteration = 0`, `largeStep =
   true`, `lastLargeStepIteration = 0`, with every dimension getting
   `lastModificationIteration = 0` and **no** `startIteration()` before the
   initial evaluation; the first real mutation then advances to iteration 1.
   Any deviation makes `cCurrent` inconsistent with the stored X and corrupts
   the first acceptance ratio.

Host readback is acceptable: bootstrap is one round-trip per accumulation
reset. But a full `nBootstrap × (maxDepth+1) = 600k`-evaluation bootstrap on
**every** state-hash change (each camera nudge) would make interactive MLT
unusable. The bootstrap budget is therefore a first-class knob:
- **Headless / parity**: full `bootstrapsamples` (pbrt default 100000) — one
  reset per render, cost immaterial, truth-comparable.
- **Interactive**: a reduced budget (pbrt's own `quickRender` divides by 16)
  plus a settle debounce — during rapid consecutive resets (camera drag) the
  renderer reuses a small quick bootstrap and schedules the full bootstrap
  when the state stops changing (the displacement-rebake 300 ms debounce
  precedent). Transient startup quality during a drag is an accepted
  interactive trade; the post-settle image is fully seeded.

Defaults from pbrt: `bootstrapsamples 100000`, `maxdepth 5`, `sigma 0.01`,
`largestepprobability 0.3`, `mutationsperpixel 100` (import-visible).

### D4. Splat + resolve: per-frame-cleared fixed-point splats, film-averaged

Each mutation atomically splats **both** states (proposal weighted `a/c_p`,
current weighted `(1−a)/c_c`) as uint fixed-point RGB — the
`atomicSplatRadiance` pattern (no portable fp atomics). A dedicated per-frame
MLT splat buffer is preferred over reusing `lightSplatBuffer` only if BDPT's
buffer lifecycle conflicts with MLT's per-frame clear; resolved at
implementation — the binding-count cost of a new buffer must be checked
against the Metal argument-table budget first (`BINDLESS_TEXTURE_CAPACITY`
squeeze precedent), reuse preferred.

**Overflow analysis (per-splat magnitude is NOT the driver).** Because
`c = luminance(L)`, each splatted color `L·w/c` is a unit-luminance color:
per-channel it is bounded by `w / Y_channel ≤ 1/0.0722 ≈ 13.9` (saturated
blue), independent of `b` (which enters only at resolve). The overflow driver
is the **per-pixel splat count**: MLT concentrates chains onto exactly the hot
pixels (caustic foci) this integrator exists for, and `atomicSplatRadiance`'s
uint32 accumulate **silently wraps** past its integer headroom. Discipline:

- The splat buffer is **cleared every frame** and resolved before the next, so
  the count bound is the per-frame mutation budget `M_frame`, not the whole
  accumulation. (A pbrt-style persistent splat plane was considered and
  rejected: unbounded splat counts break uint32 fixed-point.)
- The MLT splat scale is chosen from the inequality
  `14 · 2 · M_frame · scale < 2^32` (both states can land on one pixel in the
  worst case), traded against precision; if the resulting scale is too coarse
  at the default `M_frame` (≈ one mutation per pixel per frame), the fallback
  is uint64 fixed-point atomics (portability check: Vulkan
  `shaderBufferInt64Atomics` / Metal `atomic_ulong` — gate at pipeline build,
  not assumed).
- **No upper clamp is permitted** on the splat sum — clamping truncates energy
  on the brightest features and biases exactly the caustics MLT targets (pbrt
  splats unclamped floats). Overflow must be prevented by construction, never
  masked.

`wfMltResolve` folds the frame's splats into `accumBuffer` scaled by
`b / mpp_actual_this_frame`, film-averaged across accumulation frames — the
SPPM `wfSppmUpdate` "film-average of per-pass estimates" structure. Each
frame's fold is an unbiased estimate (`E = b·μ`), so the running mean is
unbiased and, with a constant per-frame budget, collapses exactly to pbrt's
`WriteImage(…, b / mutationsPerPixel)`. `mpp_actual` is the **actually
executed** mutation count `iterations × nChains / pixels` — when
`pixels × mpp_target < nChains`, one iteration over-delivers mutations, and
dividing by the requested target instead of the actual count would drift the
magnitude. The per-frame budget is constant across frames of one accumulation
run.

### D5. Capability gate + registration order

`src/skinny/mlt_capability.py` with `MLT_IMPLEMENTED` (monkeypatchable),
referenced by `cli_common.reject_mlt_unsupported` and `parity.combo_is_valid`
— exactly the `SPECTRAL_IMPLEMENTED` pattern. Registration lands with the gate:

- `INTEGRATOR_INDEX["mlt"] = 3`, `integrator_modes += ["MLT"]`,
  `DEFAULT_EXECUTION_FOR_INTEGRATOR["mlt"] = "wavefront"`.
- `--integrator` choices + `reject_mlt_without_wavefront` (SPPM template at
  `cli_common.py:179`) + mlt × spectral/neural/ReSTIR/online-training refusals
  in `validate_render_flags`; persisted-integrator case in `app.py` /
  `headless.py` / `web_app.py` beside the existing SPPM calls.
- **Megakernel-session fallback (matches the SPPM wart exactly):**
  `_active_integrator_index()` passes the index verbatim and
  `main_pass.slang` branches only on `INTEGRATOR_BDPT` — any other index
  (SPPM today, MLT tomorrow) hits the else-branch and renders the megakernel
  **path tracer**. So cycling to MLT at runtime in a session whose fixed
  execution mode is `megakernel` shows the path tracer, not MLT — same
  behavior SPPM has today, safe (no crash), documented. Skipping wavefront-only
  integrators in the runtime cycle for megakernel sessions is a recorded
  follow-up (it would change SPPM's shipped behavior too, out of scope here).
  MLT renders as MLT only in a wavefront session.
- Import: `pbrt/metadata.py` emits `{"integrator": "mlt", params…}` into
  `customLayerData["pbrt"]["skinny"]`; `pbrt/api.py` reports it mapped and a
  generalized selection reader replaces the SPPM-only `sppm_selection`.

### D6. Parity matrix: own tolerance class, measured harness-first

- `parity.INTEGRATORS += ("mlt",)`; `combo_is_valid`: wavefront-only, RGB-only,
  layer-free, and an **explicit `material_class == "flat"` gate** — no such
  general gate exists today (only the neural axis and the volume guard filter
  by material class), so without it an MLT combo on a subsurface/skin scene
  would be wrongly admitted. The v1 boundary is concrete: **MLT is valid only
  on flat-material scenes; non-flat scenes are recorded skips** — the
  wavefront non-flat first-hit path-fallback is NOT extended to MLT chains in
  v1 (a fallback inside a Markov chain would mix estimators). `render_linear`
  gets the force-wavefront shim like SPPM's.
- **Deterministic budget mapping for gates**: the harness maps the manifest
  `spp` to MLT as `spp` total mutations per pixel, executed as `spp`
  accumulation frames at a fixed 1 mutation/pixel/frame — independent of the
  interactive tuning knob, so gate numbers are reproducible.
- New `"mlt"` axis class in `combo_axis_class` + `_DEFAULT_SELF_CONSISTENCY`
  row: Markov correlation ⇒ different noise structure at equal spp; tolerance
  measured harness-first (SPPM row 0.15/0.12 is the starting shape),
  tighten-only.
- pbrt-truth EXRs regenerated with the pinned pbrt binary running `Integrator
  "mlt"` for the gated scenes (`regen_refs.py`); pbrt constraint: perspective
  camera only — matrix skips non-perspective scenes with a recorded reason.

### D7. Metal watchdog

- Mutation iterations dispatch `nChains` lanes; per-frame iteration loop is
  flushed into bounded sub-batches under `SKINNY_METAL` (SPPM photon
  breadth-tiling at `wavefront_driver.py:332`, 64-aligned, `65535·64` ceiling —
  the Vulkan dispatch hard-cap applies too).
- Bootstrap phase breadth-tiled the same way; `flush()` at every phase boundary
  (bootstrap / mutate / resolve).
- Fused per-chain evaluation bounds work per lane by `maxDepth` (≤ 5 default) —
  far below the heavy-eye cases; `flush_heavy_eye` not needed for flat-only v1.
- Kill harness (`tests/test_metal_cleanup.py`) runs before merge — new dispatch
  shape rule.

## Risks / Trade-offs

- [Fixed-point splat overflow on caustic-hot pixels] → prevented by
  construction (D4): per-frame clear bounds the splat count, scale chosen from
  the worst-case inequality, uint64 atomics as the gated fallback; **no clamp
  ever** — clamping biases the exact features MLT targets.
- [Fused kernel register pressure / divergence on Metal] → depth-stratified
  chain layout; if occupancy is measurably poor, the staged-queue split is the
  recorded follow-up (D1 alternative), not a v1 blocker.
- [48 MB chain state at 16384 chains] → sized by `nChains`, CLI-tunable;
  documented; far below the SPPM grid + visible-point footprint at 1080p.
- [Startup bias if chains are seeded non-proportionally] → seeding strictly by
  bootstrap-weight resampling (D3), which is the textbook startup-bias
  elimination; hostless test asserts the resampling distribution.
- [pbrt-truth flakiness: MLT truth EXRs are themselves noisy/correlated] →
  generate truth at high mutation counts; baselines recorded per combo
  harness-first like every prior integrator landing.
- [Interactive feel: MLT images "swim" early (chains exploring)] → expected MCMC
  behavior; document in README; progressive film average stabilizes like SPPM.

## Migration Plan

Pure addition behind `MLT_IMPLEMENTED`; no existing shader path changes (BDPT
functions are imported, not modified — RGB `.spv` for existing kernels must stay
byte-identical, asserted during implementation). Rollback = flip the capability
gate off; matrix records "not yet wired" skips and CLI refuses cleanly.

## Open Questions

- Splat buffer reuse (`lightSplatBuffer`) vs dedicated MLT buffer — decided at
  implementation against the Metal argument-table budget (D4).
- Interactive default for per-frame mutation budget — tune for ~60 ms frames
  on the M-series reference machine during GPU validation (the **parity-gate**
  budget mapping is already pinned in D6 and does not depend on this knob).
- Interactive quick-bootstrap size and settle-debounce interval (D3) — tune
  during GPU validation.
- Whether suite needs a new SDS discriminator scene (glass-enclosed caustic)
  where MLT should beat path/BDPT at equal budget — decide when measuring
  gates; if added, follows the `tests/assets/suite/_gen/` recipe.
