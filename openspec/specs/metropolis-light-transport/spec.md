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

### Requirement: MLT scope is flat-material RGB; other axes refused

MLT v1 SHALL target flat materials (`UsdPreviewSurface` / `standard_surface` /
OpenPBR / Python materials) under RGB rendering only — a hard prerequisite, not
a preference: the fixed primary-sample dimension budget is only boundable
without volumetric/subsurface transport (per-null-collision sample draws are
unbounded). The combination of `mlt` with `--spectral`, a neural directional
proposal, ReSTIR DI reuse, or online training SHALL be refused at startup with
a clear error (same pattern as the existing spectral/neural gates). The
wavefront non-flat first-hit path-fallback SHALL NOT be extended to MLT chains
in v1 (a fallback inside a Markov chain would mix estimators): skin /
subsurface / volume scenes are outside the envelope and are recorded
parity-matrix skips via an explicit flat-material validity rule.

#### Scenario: mlt + spectral refused
- **WHEN** any front-end is launched with `--integrator mlt --spectral`
- **THEN** it exits with a clear unsupported-combination error before GPU init

#### Scenario: mlt + neural or ReSTIR refused
- **WHEN** launched with `--integrator mlt --proposals bsdf,neural` (or ReSTIR
  reuse, or `--online-training`)
- **THEN** it exits with a clear error naming the incompatible option

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
with a recorded MLT-equivalence tolerance measured harness-first. The
confirming-suite caustic/SDS discriminator scenes SHALL be part of the gated
set — the scenes MLT exists to improve must not regress.

#### Scenario: MLT gates run in the suite matrix
- **WHEN** the GPU parity sweep runs on suite scenes
- **THEN** every valid `(mlt, wavefront)` combo is rendered and asserted against
  both the pbrt-mlt truth EXR and the Path anchor within recorded tolerances

