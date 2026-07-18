# MLT Integrator (PSSMLT over BDPT)

## Why

Hard light-transport scenes — caustics seen through glass (SDS chains), strongly occluded interiors, small bright sources — converge poorly under uncorrelated path/BDPT sampling: most samples land where the integrand is small. Metropolis Light Transport concentrates samples where the image contribution actually is by mutating a Markov chain over paths. pbrt-v4 ships an `mlt` integrator (PSSMLT: Kelemen-style primary-sample-space Metropolis driving all BDPT strategies); skinny imports pbrt scenes and gates renders against pbrt truth, so the same integrator closes a real coverage gap — today `Integrator "mlt"` in an imported `.pbrt` silently falls back to a different integrator, and the parity matrix has no Metropolis column.

## What Changes

- New `mlt` integrator: Kelemen primary-sample-space Metropolis (PSSMLT) whose target function is the existing wavefront BDPT path contribution (all strategy families, existing MIS weights). Per-chain state = a vector of primary-space samples `u ∈ [0,1)^n`; small (perturbation) + large-step (independent restart) mutations; Metropolis acceptance on scalar luminance; splat accumulation of both current and proposed states weighted by acceptance.
- **Wavefront-only, GPU-parallel chains**: thousands of independent chains advance one mutation per frame through the staged BDPT kernels; splats resolve into the existing accumulation image. No megakernel variant (mirrors SPPM). `--execution-mode auto` with `--integrator mlt` resolves to `wavefront`; explicit `megakernel` + `mlt` refused at startup.
- **b-normalization bootstrap**: startup phase estimates the average image luminance `b` from ordinary BDPT samples (pbrt's bootstrap), used to scale splat contributions to an unbiased estimate; bootstrap also seeds initial chain states by resampling proportional to luminance.
- Registered end-to-end: `renderer.integrator_modes` + integer index + state hash, `--integrator mlt` on all four front-ends, GUI selector, pbrt importer maps `Integrator "mlt"` (with `mutationsperpixel`, `largestepprobability`, `bootstrapsamples`, `chains` parameters) to the new integrator.
- **Out-of-envelope combos refused at startup** (same pattern as SPPM/spectral gates): no spectral (v1), no neural proposal, no ReSTIR reuse, flat materials only for the chain target — the wavefront non-flat path-fallback is NOT extended to MLT chains (mixing estimators inside a Markov chain); non-flat scenes are recorded parity skips.
- Parity matrix wired: `parity.INTEGRATORS` gains `mlt`, `combo_is_valid` admits `(mlt, wavefront, rgb)` only, coverage meta-test satisfied, pbrt-truth + self-consistency gates measured harness-first (MLT is unbiased but correlated — expect a recorded self-consistency tolerance like spectral's, not bit-identity).
- Metal dispatch hygiene: chain-mutation dispatches breadth-tiled under `SKINNY_METAL` like the SPPM photon phase; kill harness run.

## Capabilities

### New Capabilities

- `metropolis-light-transport`: the PSSMLT integrator — primary-sample-space chain state and mutations, Metropolis acceptance over the BDPT contribution function, bootstrap b-normalization and chain seeding, splat-based progressive accumulation, envelope restrictions (wavefront-only, RGB-only, no neural/ReSTIR), and pbrt `Integrator "mlt"` import mapping.

### Modified Capabilities

- `render-parity-matrix`: matrix gains the `mlt` integrator axis value; validity rules admit `(mlt, wavefront)` RGB combos; dual gates (pbrt-truth + self-consistency vs the `(Path, wavefront)` anchor) extended with an MLT-appropriate recorded tolerance.
- `wavefront-execution`: a fourth staged integrator sequence (chain mutation → BDPT walks → acceptance/splat resolve) with persistent cross-frame chain state, alongside path/BDPT/SPPM.
- `render-cli`: `--integrator` accepts `mlt`; `auto` execution mode resolves `mlt → wavefront`; explicit `megakernel` + `mlt` and out-of-envelope axis combos refused at startup.

## Impact

- **Shaders**: new `integrators/mlt.slang` (chain state, mutation, acceptance, splat) reusing `bdpt.slang` walk/connection/MIS code paths; possible small hooks in the shared sampler seam so BDPT walks consume primary-space samples from a chain buffer instead of hash RNG.
- **Host**: `wavefront_driver.py` (+ `metal_wavefront.py`) new staged sequence + persistent chain/bootstrap buffers; `renderer.py` registration, state hash, splat resolve reuse; `cli_common.py` refusals + execution-mode resolution; `src/skinny/pbrt/` importer integrator mapping.
- **Tests**: hostless registration/refusal/import tests; parity matrix + suite wiring; GPU gates (caustic/SDS discriminator scenes are where MLT must win); `tests/test_metal_cleanup.py` kill harness (new dispatch shape).
- **Docs**: README (flag + matrix), CLAUDE.md/`docs/Architecture.md` compatibility matrix, new or extended integrator doc section, CHANGELOG.
- **No new Python/C++ dependencies**; no descriptor-binding growth expected beyond chain-state storage buffers (design confirms against the Metal argument-table budget).
- **Reference implementation**: the pinned pbrt-v4 checkout at `~/projects/pbrt-v4` (`src/pbrt/cpu/integrators.{h,cpp}` `MLTIntegrator`, `src/pbrt/samplers.{h,cpp}` `MLTSampler`) is the algorithmic ground truth — mutation kernel, bootstrap/b-normalization, splat weighting, and parameter defaults are ported from it, and its binary regenerates truth EXRs.
