# metropolis-light-transport Specification

## Purpose
TBD - created by archiving change mlt-integrator. Update Purpose after archive.
## Requirements
### Requirement: MLT integrator selectable, wavefront-only

The renderer SHALL expose a Metropolis Light Transport integrator (`mlt`) as a
fourth integrator alongside `path`, `bdpt`, and `sppm`. It SHALL implement
primary-sample-space Metropolis (PSSMLT, Kelemen-style) whose target function is
the existing bidirectional path-tracer contribution — matching pbrt-v4's
`MLTIntegrator` (`~/projects/pbrt-v4/src/pbrt/cpu/integrators.{h,cpp}`) as the
reference implementation. MLT SHALL run only under the wavefront execution mode;
it has no megakernel variant. It SHALL be registered in
`renderer.integrator_modes` with a stable integer index that participates in the
accumulation state hash, be runtime-cycleable on interactive front-ends like the
other integrators, and be selectable via `--integrator mlt` on all four
front-ends.

#### Scenario: MLT is selectable and resets accumulation
- **WHEN** the integrator is switched to `mlt` on an interactive front-end
  whose session execution mode is `wavefront`
- **THEN** the accumulation counter resets (state hash change) and subsequent
  frames are rendered by the MLT wavefront sequence

#### Scenario: Cycling to MLT in a megakernel session falls back like SPPM
- **WHEN** the integrator is cycled to `mlt` at runtime in a session whose
  fixed execution mode is `megakernel`
- **THEN** the megakernel renders its fallback path-tracer branch (identical to
  the existing SPPM-in-megakernel behavior) with no crash, and the behavior is
  documented

#### Scenario: MLT has no megakernel path
- **WHEN** MLT is requested under the megakernel execution mode
- **THEN** the combination is refused at startup with a clear error (see
  `render-cli`), and the megakernel shader set is unchanged by this capability

### Requirement: Primary-sample-space chain state with Kelemen mutations

Each Markov chain SHALL own a fixed-size vector of primary samples
`X[i] ∈ [0,1)` in GPU storage, sized at pipeline build time to the worst-case
dimension count for the configured maximum depth (pbrt's dynamically-resized
`MLTSampler::X` cannot be reproduced on the GPU; overflow dimensions SHALL fall
back to decorrelated hash RNG rather than reading out of bounds). Samples SHALL
be served through pbrt's three-stream interleaving (camera / light / connection
streams, `GetNextIndex = streamIndex + 3 * sampleIndex`) so mutation locality is
preserved per stream. Mutations SHALL follow pbrt's lazy per-dimension scheme:
per iteration a chain is either a **large step** (probability
`largeStepProbability`, fresh uniform values) or a **small step** (per-dimension
Gaussian perturbation with effective sigma `σ·√nSmall` aggregated over the
iterations since that dimension was last touched, wrapped to `[0,1)`); each
dimension SHALL carry `lastModificationIteration` metadata plus a one-deep
backup so a rejected proposal restores the prior value exactly
(`Accept`/`Reject` semantics identical to pbrt's `MLTSampler`).

#### Scenario: Rejected mutation restores chain state
- **WHEN** a proposal is rejected
- **THEN** every dimension modified in that iteration is restored from its
  backup and the chain's iteration counter rewinds, so the next proposal is
  generated from the pre-mutation state

#### Scenario: Large step refreshes lazily
- **WHEN** a dimension is next read after one or more intervening large steps
- **THEN** it is re-seeded with a fresh uniform value before any small-step
  perturbation is applied, matching pbrt's `EnsureReady`

### Requirement: Bootstrap phase estimates b and seeds chains

Before the first MLT frame, the renderer SHALL run a bootstrap phase: `N_boot`
ordinary full samples of the target function evaluated with independent
primary-sample vectors, one per bootstrap index. The scalar contribution `c`
SHALL be the luminance of the complete sample (eye contribution plus captured
light-tracer splats). The renderer SHALL build a discrete distribution over
bootstrap weights, compute the normalization `b = (1/N) × Σ c_i`, and seed each
chain by resampling a bootstrap entry proportional to its weight — the chain
inherits that entry's RNG-reconstructible primary-sample state. (Full-sample
Kelemen chains — no per-depth strata; see design D1: skinny's env transport is
not strategy-partitioned, so per-depth chains would drop env light.) A
bootstrap whose weights sum to zero SHALL fail loudly (black-image error), not
render silently.

#### Scenario: Chains are seeded proportional to luminance
- **WHEN** the bootstrap completes on a scene with bright and dark regions
- **THEN** chain initial states are drawn from the bootstrap distribution
  proportional to path luminance

#### Scenario: Black bootstrap fails loudly
- **WHEN** every bootstrap sample carries zero luminance
- **THEN** the renderer reports a clear "no light-carrying paths found" error
  instead of accumulating a black image

### Requirement: Mutation step evaluates the full BDPT sample and splats both states

Each chain iteration SHALL evaluate the target function as one **complete**
sample of the existing BDPT integrator (`BDPTIntegrator.estimateRadiance`,
unmodified): camera ray from the chain's primary-sample raster position, all
depths and eye-side strategies including environment NEE/escape, plus the
light-tracer splats — captured into a bounded per-chain record stack instead of
written to the film during evaluation. The Metropolis acceptance SHALL be
`a = min(1, c_proposed / c_current)` on the scalar luminance of the complete
sample, and each iteration SHALL splat **both** states into the splat buffer:
every captured contribution of the proposal (eye value at its raster position,
each light-splat record at its own) weighted `a / c_proposed`, and the current
state's contributions weighted `(1 − a) / c_current`. Acceptance SHALL update
chain state; rejection SHALL restore it exactly.

#### Scenario: Both states contribute every iteration
- **WHEN** a proposal with nonzero luminance is evaluated and rejected
- **THEN** the frame's splat buffer still received the proposal's contribution
  weighted `a / c_proposed` and the current state's weighted `(1 − a) / c_current`

#### Scenario: Zero-luminance proposal splats only the current state
- **WHEN** a proposal carries zero luminance (`a = 0`)
- **THEN** only the current state is splatted, with full weight `1 / c_current`,
  and the chain state is unchanged

### Requirement: Progressive splat resolve is b-normalized and unbiased

The splat buffer SHALL be cleared every frame, and frame accumulation SHALL
fold each frame's splats into the accumulation image scaled by
`b / mpp_actual_this_frame` — where `mpp_actual` is the mutation count **actually
executed** that frame per pixel (`iterations × nChains / pixels`), never the
requested target — film-averaged across accumulation frames. Each per-frame
fold is an unbiased estimate, so the running mean is unbiased, and with a
constant per-frame mutation budget (which SHALL be constant across the frames
of one accumulation run) the accumulated image reproduces pbrt's
`WriteImage(…, b / mutationsPerPixel)` estimator in expectation. The splat
accumulator SHALL NOT apply an upper clamp (clamping biases the brightest
features); fixed-point overflow SHALL be prevented by construction (per-frame
count bound × scale within integer headroom). Chain state (primary samples,
current contribution, raster position, depth) SHALL persist across
accumulation frames and SHALL reset — along with the bootstrap — whenever the
accumulation state hash changes.

#### Scenario: MLT converges to the path-tracer image
- **WHEN** a flat-material scene is accumulated to convergence under `mlt` and
  under the `(path, wavefront)` anchor
- **THEN** the exposure-aligned mean images agree within the recorded
  MLT-equivalence tolerance (correlated Markov samples: not bit-identical, means
  agree)

#### Scenario: Scene change restarts the chains
- **WHEN** any state-hash field changes mid-session (camera, light, integrator
  parameter)
- **THEN** accumulation, bootstrap, and all chain state restart cleanly

### Requirement: MLT runs on both backends under dispatch hygiene

MLT SHALL run on native Metal and Vulkan at parity. Chain-mutation dispatches
SHALL respect the `metal-dispatch-hygiene` capability: under `SKINNY_METAL` the
per-frame chain work SHALL be committed as bounded sub-batches (breadth-tiled
like the SPPM photon phase) so no single command buffer can exceed the watchdog
budget, with the tiling bit-identical to an untiled dispatch. Any change to
dispatch shape SHALL pass the GPU kill harness (`tests/test_metal_cleanup.py`).

#### Scenario: Metal chain dispatch is watchdog-bounded
- **WHEN** a frame's mutation budget exceeds the per-dispatch cap under
  `SKINNY_METAL`
- **THEN** the work is split across multiple committed command buffers and the
  accumulated result is identical to a single dispatch

### Requirement: pbrt importer maps the mlt integrator

The pbrt importer SHALL map `Integrator "mlt"` to skinny's `mlt` integrator,
carrying `maxdepth`, `mutationsperpixel`, `largestepprobability`, `sigma`,
`chains`, and `bootstrapsamples` (pbrt defaults 5 / 100 / 0.3 / 0.01 / 1000 /
100000) into the render configuration the same way the `sppm` mapping does,
instead of silently falling back to another integrator.

#### Scenario: Imported mlt scene renders under MLT
- **WHEN** a `.pbrt` scene declaring `Integrator "mlt" "integer mutationsperpixel" [200]`
  is imported and rendered headlessly
- **THEN** the render uses the `mlt` integrator with `mutationsperpixel = 200`
  and the remaining parameters at pbrt defaults

### Requirement: MLT is dual-gated in the parity matrix

MLT SHALL enter the parity matrix (see `render-parity-matrix`) with both gates:
pbrt-truth against EXRs regenerated by the pinned pbrt v4 binary running
`Integrator "mlt"`, and self-consistency against the `(Path, wavefront)` anchor
with a recorded MLT-equivalence tolerance measured harness-first. The spectral
combo `(mlt, wavefront, spectral, flat)` SHALL be admitted by the validity
rules and gated the same way: spectral pbrt-truth EXRs from the same pinned
pbrt v4 (whose MLT is spectral natively), and self-consistency against the
**spectral** `(Path, wavefront)` anchor with its own recorded tolerance —
recorded from measurement, never by loosening an existing RGB tolerance. The
confirming-suite caustic/SDS discriminator scenes SHALL be part of the gated
set in both modes, including the dispersion prism scene under spectral — the
scenes MLT exists to improve must not regress.

#### Scenario: MLT gates run in the suite matrix

- **WHEN** the GPU parity sweep runs on suite scenes
- **THEN** every valid `(mlt, wavefront)` combo — RGB and spectral — is
  rendered and asserted against both the pbrt-mlt truth EXR and the matching
  Path anchor within recorded tolerances

#### Scenario: spectral MLT gated on the dispersion prism

- **WHEN** the spectral parity sweep renders the BK7 dispersion prism suite
  scene under `(mlt, wavefront, spectral)`
- **THEN** the render is asserted against the spectral pbrt-mlt truth EXR and
  the spectral Path anchor within the recorded per-scene tolerances

### Requirement: MLT scope is flat-material; other axes refused

MLT SHALL target flat materials (`UsdPreviewSurface` / `standard_surface` /
OpenPBR / Python materials) under RGB rendering or, when the session is started
with `--spectral`, under the hero-wavelength spectral mode (see the spectral
chain-transport requirement below). Flat materials remain a hard prerequisite,
not a preference: the fixed primary-sample dimension budget is only boundable
without volumetric/subsurface transport (per-null-collision sample draws are
unbounded). The combination of `mlt` with a neural directional proposal, ReSTIR
DI reuse, or online training SHALL be refused at startup with a clear error
(same pattern as the existing neural gates); `--spectral --integrator mlt`
SHALL NOT be refused. The wavefront non-flat first-hit path-fallback SHALL NOT
be extended to MLT chains (a fallback inside a Markov chain would mix
estimators): skin / subsurface / volume scenes are outside the envelope and are
recorded parity-matrix skips via an explicit flat-material validity rule.

#### Scenario: mlt + spectral accepted

- **WHEN** any front-end is launched with `--integrator mlt --spectral` on a
  flat-material scene
- **THEN** the session starts and renders under the spectral MLT wavefront
  sequence, with no unsupported-combination error

#### Scenario: mlt + neural or ReSTIR refused

- **WHEN** launched with `--integrator mlt --proposals bsdf,neural` (or ReSTIR
  reuse, or `--online-training`), with or without `--spectral`
- **THEN** it exits with a clear error naming the incompatible option

#### Scenario: spectral mlt on a non-flat scene is a recorded skip

- **WHEN** the parity matrix evaluates `(mlt, wavefront, spectral)` against a
  skin/subsurface/volume scene
- **THEN** the combo is excluded with an explicit flat-material reason, never
  rendered via a fallback estimator

### Requirement: Spectral chain transport composes the spectral BDPT estimator

Under `--spectral`, the MLT kernel set SHALL be compiled with both variant
defines (`SKINNY_MLT` + `SKINNY_SPECTRAL`) so each chain mutation evaluates one
complete sample of the existing **spectral** wavefront BDPT estimator,
unmodified — all strategy families, per-λ NEE, exact spectral sources, and
hero-λ dispersion with secondary termination included — preserving
`E[spectral MLT] = E[spectral BDPT]` by construction. The hero-wavelength set
SHALL be a primary-sample dimension: the estimator's existing
`sampleWavelengths(rng.next())` draw (its first camera-stream consumption in
`estimateRadiance`, after ray generation) is served by the primary-sample
override, with no estimator or mutation-logic change — a large step redraws it
uniformly and a small step perturbs it wrapped to `[0,1)`, so the chain
explores the spectrum ergodically. The spectral estimator draws **one** extra
uniform than RGB (the wavelength), but because primary samples are indexed on
the strided `streamIndex + MLT_NUM_STREAMS * sampleIndex` layout, the realised
worst-case *index* is far higher: **measured ~160 of `MLT_MAX_DIMS` = 192,
against ~88 for RGB** — only ~17 % headroom. Any draw past the budget falls into
`RNG.ensureReady`'s escape hatch, which returns a fresh hash that is **not
restored on reject**, breaking detailed balance and silently biasing the chain
(no crash). The measured configurations sit inside the budget, so this is a
recorded limitation rather than a defect; the worst-case index SHALL be pinned
in the numpy-mirror dimension model and its hostless test so a future spectral
draw cannot silently cross it, and a deeper `maxdepth` or a scene with more
lights SHALL be treated as out of the validated envelope until the budget is
raised or the layout packed densely. A pbrt-style per-stream split is NOT a
viable remedy here — it was attempted and reverted (it hangs `wfMltBootstrap`
on Metal). The scalar contribution `c` used for bootstrap
weights, the normalization `b`, and Metropolis acceptance SHALL be the CIE-Y
luminance of the **gamut-clamped** resolved spectral sample. Captured
contributions SHALL resolve at record-capture time through the existing
clamped film resolve (`Spectrum → XYZ → linear sRGB` including the
non-negative gamut clamp), before the scalar Metropolis weights are applied;
the clamp SHALL NOT be bypassed on the MLT capture path (it keeps the unsigned
fixed-point splat buffer valid and guarantees `c = 0 ⇔ contribution = 0`, the
target-support condition), and no negative value SHALL ever reach the splat
buffer. The chain record layout, fixed-point splat buffer, resolve pass, and
`b / mpp_actual` fold remain the RGB code paths unchanged. The RGB (non-
spectral) wavefront MLT kernels and the RGB megakernel SPIR-V SHALL remain
byte-identical.

#### Scenario: spectral MLT converges to the spectral path image

- **WHEN** a flat-material scene is accumulated to convergence under
  `(mlt, wavefront, spectral)` and under the spectral `(path, wavefront)`
  anchor
- **THEN** the exposure-aligned mean images agree within the recorded spectral
  MLT-equivalence tolerance (Markov-correlated: not bit-identical, means agree)

#### Scenario: wavelength dimension mutates with the chain

- **WHEN** a chain iteration executes a large step
- **THEN** the sample's hero-wavelength set is redrawn from the
  visible-wavelength importance distribution via its primary-sample dimension,
  and a subsequent small step perturbs — not redraws — it

#### Scenario: out-of-gamut resolve cannot corrupt the splat buffer

- **WHEN** a chain sample on a dispersion scene resolves to an out-of-gamut
  color (negative pre-clamp sRGB component)
- **THEN** the capture-time gamut clamp bounds it at zero before the Metropolis
  weights apply, the record's contribution `c` is zero if and only if its
  clamped RGB is zero, and no negative value reaches the unsigned fixed-point
  splat accumulator

#### Scenario: dispersion is reachable under spectral MLT

- **WHEN** the dispersion prism scene renders under `(mlt, wavefront,
  spectral)` to the suite accumulation budget
- **THEN** the transmitted light spatially separates by wavelength, matching
  the spectral pbrt-mlt truth within the recorded gate

#### Scenario: RGB builds unchanged

- **WHEN** the wavefront MLT kernels and megakernel are compiled without
  `SKINNY_SPECTRAL` after the change lands
- **THEN** the produced SPIR-V is byte-identical to the pre-change binaries

### Requirement: Spectral MLT uniform tail packs against the compiled layout

The host-side MLT uniform tail SHALL be packed against the target kernel set's
reflected MSL layout (the layout-keyed mechanism, not session flags), so the
spectral kernel set's differing uniform stride is honored automatically, and a
hostless test SHALL pin that the spectral-MLT layout round-trips the tail
offsets on both backends' packing paths.

#### Scenario: spectral MLT tail offsets round-trip

- **WHEN** the hostless uniform-packing test packs the MLT tail against the
  spectral kernel set's reflected layout
- **THEN** every tail field lands at the reflected offset, and a preview or
  non-MLT pack in the same session is unaffected

### Requirement: Spectral chain mutation respects Metal dispatch hygiene

Spectral MLT chain-mutation dispatches SHALL run under the existing
breadth-tiled chain-batch mechanism, with the per-sub-batch budget re-measured
for the spectral estimator's longer per-mutation cost under `SKINNY_METAL`
(lowering the default chain batch for spectral sessions if the measurement
requires), the tiling bit-identical to an untiled dispatch, and the GPU kill
harness (`tests/test_metal_cleanup.py`) passing since kernel length changes.

#### Scenario: spectral mutation dispatch is watchdog-bounded

- **WHEN** a spectral MLT frame's mutation budget exceeds the per-dispatch cap
  under `SKINNY_METAL`
- **THEN** the work is split across multiple committed command buffers and the
  accumulated result is identical to a single dispatch

