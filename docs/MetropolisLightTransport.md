# Skinny — Metropolis Light Transport (PSSMLT)

This document is the implementation reference for skinny's fourth integrator:
**Metropolis Light Transport** (`--integrator mlt`). The implementation is
Kelemen **primary-sample-space MLT** (PSSMLT) over skinny's existing
bidirectional path tracer. Each GPU lane owns a persistent Markov chain whose
state is a vector of primary random numbers; mutating that vector changes a
complete BDPT sample without changing the transport code itself.

MLT is intended for difficult indirect-lighting and caustic scenes where
independent path samples rarely discover the important paths. Once a chain finds
one of those paths, small mutations explore its neighbourhood while occasional
large steps keep the estimator ergodic.

The generic staged execution machinery is documented in
[Wavefront.md](Wavefront.md), the BDPT target estimator and descriptor layout in
[Architecture.md](Architecture.md), and the hero-wavelength target variant in
[Spectral.md](Spectral.md#spectral-mlt). This page owns the full MLT algorithm,
state, host orchestration, controls, limitations, and verification map.

## What skinny's MLT is

Skinny uses **full-sample chains**. One chain state produces one complete BDPT
sample: camera position, eye and light subpaths, every applicable connection,
environment transport, and light-tracer splats. This differs from pbrt-v4's
per-depth strategy decomposition. Skinny's environment estimator is not
partitioned into fixed `(s,t)` strata, so full-sample chains preserve the
existing estimator verbatim and give the useful invariant:

`expected MLT image = expected skinny BDPT image`.

The PSS sampler is a compile-time `RNG` replacement in `common.slang` under
`SKINNY_MLT`. It keeps the same `next()` and `next2()` interface consumed by
BDPT, cameras, lights, and materials. The transport modules therefore do not
know whether their random numbers came from an independent sampler or a Markov
chain.

### Scope and limits

| Property | Value |
| --- | --- |
| Integrator | `INTEGRATOR_MLT = 3`; select with `--integrator mlt`. |
| Execution | **Wavefront only.** `auto` selects wavefront; an explicit megakernel session is refused. |
| Backends | Vulkan `WavefrontMltPass` and native Metal `MetalWavefrontMltPass`. |
| Color | RGB and hero-wavelength spectral (`--spectral`). |
| Materials | **Flat materials only.** No skin, subsurface, heterogeneous volume, or path fallback inside a chain. |
| Target | Full skinny BDPT sample, including its MIS weights, environment terms, and light-tracer splats. |
| Sampling layers | BSDF proposal only; neural/environment directional mixtures, ReSTIR reuse, and online neural training are refused. |
| Camera | Pinhole camera in the MLT target. |

The flat-only rule is structural. Switching to a different estimator when a
Markov chain reaches a non-flat material would change the chain's target
distribution and invalidate its normalization.

## Running it

The execution mode follows the integrator, so the short form is enough:

```bash
skinny-render scene.usda --integrator mlt --samples 256 --output mlt.png
```

An explicit equivalent is:

```bash
skinny-render scene.usda --integrator mlt \
  --execution-mode wavefront --proposals bsdf --reuse none \
  --samples 256 --output mlt.png
```

Early frames can appear to **swim** as the chains move through primary-sample
space. That is expected MCMC correlation rather than an accumulation reset
problem; the progressive film average stabilizes as more mutations are folded
in.

### Parameters

The imported pbrt metadata and the corresponding `Renderer` attributes use the
same meanings:

| Parameter | Renderer field | Interactive default | Effect |
| --- | --- | --- | --- |
| chains | `mlt_num_chains` | 16,384 | persistent GPU-parallel Markov chains |
| bootstrapsamples | `mlt_bootstrap_samples` | 8,192 | fresh samples used to measure `b` and seed chains; `SKINNY_MLT_BOOTSTRAP` overrides the default |
| sigma | `mlt_sigma` | 0.01 | small-step mutation scale |
| largestepprobability | `mlt_large_step_prob` | 0.3 | probability of an independent restart |
| maxdepth | `mlt_max_depth` | 5 | bound on the complete BDPT target evaluation |
| mutationsperpixel | derived frame budget | about 1/frame interactively | desired mutation budget; resolve always divides by the actually executed `mpp_actual` |

`SKINNY_MLT_METAL_CHAIN_BATCH` controls only watchdog-safe dispatch breadth on
Metal; it does not change the estimator or its total mutation budget.

## Step-by-step transport sketch

![Introductory MLT derivation: the rendering equation becomes a path-space integral and primary-sample-space target, followed by bootstrap normalization, parallel Markov chains, large and small mutations, Metropolis acceptance, dual splats, and normalized film resolve.](diagrams/sketches/mlt-integrator-step-by-step.png)

1. **Bootstrap.** Evaluate many independent full BDPT samples and measure each
   sample's non-negative scalar contribution.
2. **Seed chains.** Build a contribution-weighted CDF on the host, compute the
   normalization constant `b`, and replay sampled bootstrap indices into the
   persistent chain buffers.
3. **Mutate.** Each chain proposes either a large independent restart or a small
   perturbation of its primary-sample vector, evaluates the complete BDPT
   target, splats both states with the Metropolis weights, and accepts or
   rejects the proposal.
4. **Resolve.** Scale the frame's fixed-point splats by `b / mpp_actual`, fold
   them into the progressive HDR film, display the result, and clear the
   per-frame splat buffer.

## Phase 1 — bootstrap and normalization

At an accumulation reset, `wfMltBootstrap` evaluates `nBootstrap` fresh samples.
Each invocation writes only its scalar contribution `c` to
`mltBootstrapWeights`; its captured RGB records are scratch.

The host then calls `mlt_bootstrap.resample_chain_seeds`:

1. Replace negative and non-finite weights with zero.
2. Compute `b` as the mean bootstrap contribution.
3. Build the cumulative weight distribution.
4. Draw `nChains` bootstrap indices proportional to contribution.
5. Upload those indices to `mltChainSeeds`.

An all-zero bootstrap raises a clear error instead of starting chains from a
degenerate distribution. `wfMltInit` replays each selected bootstrap index,
reconstructs its primary-sample vector, captures the accepted contribution
records, and initializes its independent mutation RNG stream.

Bootstrap is repeated whenever a scene, camera, integrator parameter, or other
accumulation-defining state changes.

## Phase 2 — primary-sample state

One GPU lane maps to one persistent Markov chain. The default is 16,384 chains.
Each chain owns up to 192 primary dimensions. A dimension stores its current and
backup values plus the iteration numbers needed for lazy mutation and rollback.

The sampler divides dimensions into interleaved streams while preserving the
ordinary `RNG` surface used by the renderer. `startIteration` chooses a large or
small step:

- A **large step** replaces touched dimensions with fresh independent values.
  Its probability defaults to `0.3`.
- A **small step** uses the pbrt-v4 wrapped Gaussian mutation implemented by
  `mltSampleNormal`: for a dimension last touched `n` small steps ago,
  `u′ = fract(u + Normal(0, sigma²·n))`. The `√n` scale preserves the lazy
  mutation result without replaying every skipped iteration; `sigma` defaults
  to `0.01`.

Only dimensions touched by the proposed BDPT sample are mutated lazily. On
rejection, `RNG.reject()` restores only the dimensions up to `maxDim`, not all
192 slots. That bound is essential to the Metal live-state budget and is
output-neutral.

## Phase 3 — evaluate, splat, accept or reject

`wfMltMutate` performs one complete Metropolis iteration per chain:

1. Start a large or small primary-sample mutation.
2. `mltEvaluate` uses the first two dimensions to choose a film pixel, generates
   the camera ray, and calls `BDPTIntegrator.estimateRadiance` (or
   `SpectralBDPTIntegrator.estimateRadiance`).
3. Capture the eye contribution plus each valid light-tracer splat in
   `mltProposalRecords`.
4. Reduce the captured records to scalar luminance `cProposal`.
5. Compute acceptance `a = min(1, cProposal / cCurrent)`, with zero guards.
6. Splat the proposal records with weight `a / cProposal` and the current
   records with weight `(1 - a) / cCurrent`.
7. Accept by committing the primary samples and replacing
   `mltCurrentRecords`, or reject by restoring the backed-up dimensions.

The **dual splat happens before the random accept/reject decision**. It is the
Kelemen estimator: both states contribute according to the acceptance
probability even though only one becomes the next chain state.

The film splat buffer uses unsigned Q24.8 fixed point because portable floating
point atomics are not available on every target. Contributions are sanitized for
NaN, infinity, and negative values, but **never upper-clamped**; clamping would
remove energy precisely at the caustic foci MLT is meant to discover. The buffer
is cleared after every frame, so its overflow bound is the frame's mutation
budget rather than the lifetime accumulation.

## Phase 4 — resolve and progressive accumulation

The host executes enough chain iterations to cover the desired per-frame
mutation budget. Because the chain count may not divide the number of pixels,
the shader receives the actual budget:

`mpp_actual = iterations × nChains / numPixels`.

`wfMltResolve` converts fixed-point splats back to RGB, scales by
`b / mpp_actual`, and writes a per-frame unbiased estimate. It then uses the
same running film average as the other progressive integrators and calls the
shared wavefront display tail for exposure, tonemapping, overlays, and output.

## GPU state and bindings

MLT's chain buffers are compiled only into the `SKINNY_MLT` wavefront variant
and are absent from the normal megakernel and wavefront layouts.

| Binding | Buffer | Per-element state | Size basis |
| --- | --- | --- | --- |
| 52 | `mltPrimarySamples` | value, backup, last-modified iteration, backup iteration | `nChains × 192` |
| 53 | `mltChainMeta` | mutation RNG, iteration counters, seed, current `c`, record count | `nChains` |
| 54 | `mltCurrentRecords` | accepted eye and light-splat RGB records | `nChains × 8` |
| 55 | `mltBootstrapWeights` | scalar bootstrap contributions | `nBootstrap` |
| 56 | `mltChainSeeds` | contribution-resampled bootstrap index | `nChains` |
| 57 | `mltProposalRecords` | proposed eye and light-splat RGB records | `nChains × 8` |

All structs use scalar fields so their Vulkan scalar-layout and Metal MSL
strides match: 16 bytes for a primary sample, 32 bytes for chain metadata, and
16 bytes for a captured record. `wavefront_layout.mlt_buffer_sizes` is the host
source of truth and sizes persistent state by **chain count**, never by the
ordinary wavefront stream tile.

## Host and backend orchestration

The backend-neutral ordering lives in `wavefront_driver.py`:

```text
accumulation reset:
  wfMltBootstrap batches
  host weight readback → CDF, b, chain seeds
  wfMltInit batches

every frame:
  wfMltMutate batches × iterations
  wfMltResolve over pixels
```

Vulkan records this through `WavefrontMltPass`; Metal uses
`MetalWavefrontMltPass`. Both compile the same four Slang entry points and bind
the same logical state.

One mutation evaluates a complete BDPT sample, so the heavy kernels are
**breadth-tiled** into 64-aligned chain windows. Metal flushes every sub-batch
and every phase boundary to stay inside the macOS GPU watchdog budget. The
per-chain target itself remains bounded by `mltMaxDepth`.

## Spectral MLT

`--spectral --integrator mlt` changes only the target evaluator:
`mltEvaluate` instantiates `SpectralBDPTIntegrator` instead of the RGB
`BDPTIntegrator`. Hero wavelengths are primary-sample dimensions, and the
spectral estimator resolves its result to linear sRGB before capture, so
bootstrap, mutation, acceptance, splatting, and resolve remain unchanged.

Three implementation rules keep spectral MLT within Metal's live-state budget:

- proposed records live in device memory at binding 57, never in a thread-local
  record array;
- spectral MIS reads the spectral vertex arrays directly rather than creating
  RGB mirror arrays;
- rejection scans only the dimensions touched by the current iteration.

The detailed spectral transport and verification results live in
[Spectral.md § Spectral MLT](Spectral.md#spectral-mlt).

## pbrt import

`Integrator "mlt"` is normalized into stage metadata with
`mutationsperpixel`, `largestepprobability`, `sigma`, `chains`,
`bootstrapsamples`, and `maxdepth`. Authored values override pbrt's defaults:
100, 0.3, 0.01, 1,000, 100,000, and 5 respectively. The importer mapping is
documented in [PbrtImport.md § Integrator mapping](PbrtImport.md#integrator-mapping).

Skinny's interactive defaults intentionally use more parallel chains (16,384)
and fewer bootstrap samples (8,192) for responsive GPU startup; imported pbrt
metadata can override them for parity runs.

## Verification

The MLT stack has hostless guards at every seam:

- `tests/test_mlt_sampler.py` — lazy primary-sample mutation, stream indexing,
  wraparound, accept, and reject semantics;
- `tests/test_mlt_driver.py` — bootstrap/init/mutate/resolve ordering, batching,
  and cross-backend struct strides;
- `tests/test_mlt_host.py` — bootstrap resampling, uniform packing, pass
  lifecycle, descriptor bindings, and Vulkan/Metal adapters;
- `tests/test_mlt_selection.py` — CLI registration, auto-wavefront selection,
  and rejection of incompatible layers;
- `tests/pbrt/` — recorded MLT self-consistency and pbrt-truth tolerances.

MLT is unbiased in expectation but Markov-correlated, so an equal-sample render
is not expected to be pixel-identical to path tracing or BDPT. Verification uses
converged radiance and recorded stochastic tolerances rather than the
bit-identical execution-mode parity used by independent-sample path tracing.

## Key files

| File | Role |
| --- | --- |
| `shaders/wavefront/wavefront_mlt.slang` | bootstrap, init, mutate, dual splat, resolve |
| `shaders/common.slang` | `SKINNY_MLT` primary-sample `RNG` override |
| `shaders/integrators/bdpt.slang` | RGB target estimator |
| `shaders/integrators/bdpt_spectral.slang` | spectral target estimator |
| `mlt_bootstrap.py` | host CDF, normalization `b`, chain-seed resampling |
| `wavefront_driver.py` | backend-neutral stage order and batching |
| `wavefront_layout.py` | chain structs, strides, and buffer sizes |
| `vk_wavefront.py` | Vulkan pass and recorder |
| `metal_wavefront.py` | native Metal pass and recorder |
| `renderer.py` | lifecycle, reset/bootstrap, dispatch, uniforms, defaults |

## References

1. E. Veach and L. Guibas. *Metropolis Light Transport.* SIGGRAPH, 1997.
2. C. Kelemen, L. Szirmay-Kalos, G. Antal, and F. Csonka.
   *A Simple and Robust Mutation Strategy for the Metropolis Light Transport
   Algorithm.* Computer Graphics Forum, 2002.
3. M. Pharr, W. Jakob, and G. Humphreys. *Physically Based Rendering: From
   Theory to Implementation*, 4th ed. — MLT integrator and MLTSampler.
