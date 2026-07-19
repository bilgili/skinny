# Proposal: spectral-mlt

## Why

MLT (PSSMLT over the wavefront BDPT estimator, change `mlt-integrator`) shipped
RGB-only, while spectral rendering (`--spectral`, hero-wavelength) now covers
path/bdpt/sppm under the wavefront mode. MLT is the one wavefront integrator
excluded from the spectral envelope, so scenes that need both Metropolis
exploration (caustics, SDS paths) and spectral effects (dispersion, named
conductors, blackbody/illuminant SPDs) have no valid combo. pbrt-v4's
`MLTIntegrator` is spectral natively — its wavelength sample lives in the
primary sample space — so a faithful spectral MLT also tightens pbrt-truth
parity.

## What Changes

- Compose the two existing compile-time variants: build the wavefront MLT
  kernel set with `-DSKINNY_MLT -DSKINNY_SPECTRAL` so the PSS sampler drives the
  existing **spectral** BDPT estimator (change `spectral-wavefront`) unmodified.
- Wavelengths join the primary sample space (pbrt-v4 style): the spectral
  estimator's existing `sampleWavelengths(rng.next())` draw becomes a PSS
  camera-stream dimension automatically under the override, so large steps
  redraw λ and small steps perturb it — the chain explores the spectrum
  ergodically with zero estimator edits.
- Scalar target function becomes CIE-Y luminance of the **gamut-clamped**
  resolved spectral sample (bootstrap `c`, acceptance `a`, and
  `b`-normalization all key off it).
- Chain splats resolve through the existing clamped film resolve
  (`Spectrum → XYZ → linear sRGB`, gamut clamp included, like the SPPM
  per-pass resolve), so the unsigned fixed-point splat buffer, resolve pass,
  and film pipeline stay RGB and unchanged.
- Lift the `mlt × spectral` refusals: `reject_mlt_unsupported` /
  `reject_spectral_unsupported` (cli_common, all four front-ends) and
  `parity.combo_is_valid` admit `(mlt, wavefront, spectral, flat)`.
- Parity wiring: spectral-MLT pbrt-truth baselines (pinned pbrt v4 running
  spectral `Integrator "mlt"`) + recorded self-consistency tolerance vs the
  spectral `(path, wavefront)` anchor, harness-first, on the confirming-suite
  discriminator scenes (incl. the BK7 dispersion prism).
- Both backends (Vulkan `WavefrontMltPass`, Metal `MetalWavefrontMltPass`)
  under existing Metal watchdog chain tiling; RGB MLT kernels and the RGB
  megakernel `.spv` stay byte-identical.

Out of scope (unchanged refusals): neural proposal, ReSTIR reuse, online
training, non-flat materials, megakernel MLT.

**Archive-order dependency:** the `spectral-rendering` and
`render-parity-matrix` deltas here are written against the post-
`spectral-wavefront` spec text. The in-flight `spectral-rendering` /
`spectral-bdpt-megakernel` / `spectral-wavefront` changes MUST archive before
this change; re-validate the MODIFIED bases at archive time (a later archive of
`spectral-wavefront` would otherwise revert the mlt admissions).

## Capabilities

### New Capabilities

None — this composes two existing capabilities.

### Modified Capabilities

- `metropolis-light-transport`: the "MLT scope is flat-material RGB" requirement
  changes — spectral becomes an in-envelope mode (PSS-resident wavelength
  dimension, luminance target function, splat-time spectral resolve); the
  parity-matrix requirement gains the spectral combos.
- `spectral-rendering`: the spectral envelope requirement changes — `mlt` joins
  the admitted integrator set (wavefront-only), and startup no longer refuses
  `--spectral --integrator mlt`.
- `render-cli`: the impossible-combination requirement changes — `--spectral`
  drops out of the `mlt` refusal bullet (neural / ReSTIR / online-training
  refusals stand).
- `render-parity-matrix`: the validity-table requirement changes — the "every
  `(MLT, spectral)` combo is skipped" wording is replaced; `(mlt, wavefront,
  spectral, flat)` becomes a rendered, dual-gated combo against the spectral
  anchors.

## Impact

- Shaders: `shaders/wavefront/mlt*.slang` (or the MLT sections of the wavefront
  kernel set) compiled under `SKINNY_SPECTRAL`; PSS `RNG` override in
  `common.slang`; spectral resolve shared from `spectral_flat_common.slang` /
  `spectrum.slang`. No RGB kernel bytes change.
- Host: `wavefront_driver.py` (`WavefrontMltPass`), `metal_wavefront.py`
  (`MetalWavefrontMltPass`), `renderer.py` (MLT uniform tail × spectral defines
  interplay — the MSL layout-keyed pack from c9985b1), `cli_common.py` gates,
  `pbrt/parity.py` validity rules.
- Tests: hostless gate/matrix tests (`tests/test_mlt_host.py`,
  `tests/pbrt/test_matrix.py`), GPU suite gates (`tests/pbrt/test_suite.py`)
  with new spectral-mlt baselines; `regen_refs.py` spectral MLT truth EXRs;
  Metal kill harness re-run (kernel length changes).
- Docs: CLAUDE.md + README compatibility matrices, `docs/Spectral.md`,
  `docs/Wavefront.md`, CHANGELOG.
