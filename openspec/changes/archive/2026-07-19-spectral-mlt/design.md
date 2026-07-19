# Design: spectral-mlt

## Context

Two shipped compile-time variants meet here:

- **MLT** (`mlt-integrator`): Kelemen full-sample PSSMLT driving the wavefront
  BDPT estimator via a compile-time `RNG` override in `common.slang` under
  `-DSKINNY_MLT`. Per-frame sequence `wfMltBootstrap → wfMltInit →
  wfMltMutate× → wfMltResolve`; chain state in bindings 52–56; dual splat as
  uint fixed-point RGB, no clamp; `b/mpp_actual` fold. RGB-only, wavefront-only,
  flat-only. Backends: `WavefrontMltPass` (Vulkan) / `MetalWavefrontMltPass`
  (Metal, breadth-tiled chain batches under the watchdog).
- **Spectral wavefront** (`spectral-wavefront`): hero-wavelength transport
  threaded through the staged path/bdpt/sppm integrators behind
  `#if defined(SKINNY_SPECTRAL)` — records carry `Spectrum` +
  `SampledWavelengths`, per-λ NEE, sigmoid upsampling, exact conductor/SPD/
  blackbody sources, hero-λ dispersion with secondary termination, CIE film
  resolve. Not bit-identical to megakernel spectral (unbiased MC agreement,
  `spectral_self_consistency` tolerance field).

MLT is now the only wavefront integrator refused under `--spectral`
(`reject_mlt_unsupported`, `reject_spectral_unsupported`,
`parity.combo_is_valid`). pbrt-v4's `MLTIntegrator` is spectral natively: the
wavelength sample is a primary-sample dimension, mutated with the chain.

## Goals / Non-Goals

**Goals:**

- `--spectral --integrator mlt` renders on both backends: PSSMLT chains over
  the **spectral** BDPT estimator, unbiased, converging to the spectral
  `(path, wavefront)` image.
- Wavelength exploration is part of the Markov chain (pbrt-parity), so
  dispersion caustics are reachable and ergodic.
- Film/splat pipeline stays RGB; RGB MLT kernels and the RGB megakernel `.spv`
  stay byte-identical.
- Dual-gated in the parity matrix with spectral pbrt-truth + recorded
  self-consistency, harness-first.

**Non-Goals:**

- No megakernel MLT, no neural/ReSTIR/online-training composition, no non-flat
  chains — all existing refusals stand.
- No change to the spectral estimator itself (upsampling, dispersion, resolve
  math are reused as-is).
- No new spectral accumulation format (no per-λ film).

## Decisions

### D1: Compose the two compile-time variants; estimator unmodified

Build the wavefront MLT kernel set with `-DSKINNY_MLT -DSKINNY_SPECTRAL`. The
PSS `RNG` override is wavelength-agnostic (it only serves uniforms), and the
spectral BDPT estimator consumes uniforms through the same `RNG` seam — so
`E[spectral MLT] = E[spectral BDPT]` by construction, exactly the argument that
justified full-sample chains in RGB (spec D1 of `mlt-integrator`).

*Alternative rejected:* a separate spectral-MLT integrator or per-depth
decomposition — same env-transport-dropping problem as in RGB, plus double
maintenance.

### D2: Wavelength lives in the primary sample space — via the estimator's existing draw

The spectral estimator already draws its hero-λ set as its **first
camera-stream RNG consumption**: `SpectralBDPTIntegrator.estimateRadiance`
opens with `sw = sampleWavelengths(rng.next())` (`bdpt_spectral.slang`), after
`mltEvaluate`'s raster dims and `generateRay`'s jitter/lens dims. Under
`-DSKINNY_MLT` that draw is served by the PSS override automatically — the
wavelength dimension is a chain dimension with **zero estimator edits**, and
the standard per-dimension Kelemen machinery covers it: large steps redraw λ
uniformly, small steps perturb it (wrapped), giving mutation locality in the
spectrum. Any fixed deterministic PSS position is unbiased (pbrt-v4 draws λ at
a different but equally fixed stream position), so we keep the shipped
consumption order rather than reordering to match pbrt. The worst-case PSS
dimension budget grows by exactly one; enforcement is pinned in the
`sampling/mlt_sampler.py` numpy-mirror worst-case accounting + its hostless
test (so a future spectral draw can't silently push a dimension into the
non-restorable overflow-hash branch); the overflow fallback itself is
unchanged. Task 1.2 is therefore *verify + pin*, not *add*.

*Alternatives rejected:*
- λ from an independent (non-PSS) RNG per evaluation: the target function
  becomes stochastic — acceptance compares c-values drawn at different λ,
  which is not a valid Metropolis target.
- λ fixed per chain for a whole frame: valid but non-ergodic across the
  spectrum within a chain; dispersion features then rely purely on chain
  seeding. pbrt's choice (mutate λ) is strictly better and no harder.
- Reordering λ "immediately after raster" to a bespoke position: forces an
  estimator edit, violating D1 and the spectral non-MLT kernel byte-identity
  guard, for no bias or pbrt-parity benefit.

### D3: Scalar target function is CIE-Y luminance of the clamped resolved sample

`c = Y(clamped resolved sample)`: the complete spectral sample (eye
contribution + captured light splats) is resolved through the existing clamped
film resolve (`spectrumResolveToLinearSRGB` — CMF ÷ λ-pdf, ÷ CIE-Y integral,
XYZ→sRGB, then the `max(·, 0)` gamut clamp) and its luminance taken as the
scalar contribution — for bootstrap weights, `b`, and acceptance
`a = min(1, c_p/c_c)`. The λ-pdf and Y-integral normalizations live inside the
resolve, so `c`, `b`, and the splat values are automatically in consistent
units. The shipped `mltLuminance` (Rec.709 dot on linear sRGB) *is* CIE Y, so
the RGB code path computes this target unchanged when `SKINNY_SPECTRAL` is off.

### D4: Records resolve Spectrum → linear sRGB through the CLAMPED film resolve at capture time

During estimator evaluation, records carry `Spectrum + SampledWavelengths` as
spectral-wavefront already defines. At capture into the per-chain record stack,
each contribution is resolved to linear sRGB via the existing **clamped** film
resolve (`spectrumResolveToLinearSRGB`, gamut clamp included). The resolve is
not linear — the gamut clamp is a `max(·, 0)` — and the clamp is load-bearing
twice over, so bypassing it as an "exactness optimization" is forbidden:

1. **Splat-buffer validity.** The splat accumulator is unsigned Q-format
   fixed-point; an unclamped out-of-gamut resolve would feed negative values
   into the uint atomics → wraparound garbage.
2. **Target-support validity.** Unclamped, an out-of-gamut sample like
   `(r, −g, b)` can have `F ≠ 0` with `Y = 0` — a nonzero-contribution state
   with zero target density, i.e. unreachable-state bias. Post-clamp,
   `c = 0 ⇔ F = 0` per record, so `support(F) ⊆ support(c)` holds.

Given the clamped, deterministic per-state `F(X)`, the dual splat with scalar
Metropolis weights `a/c_p` and `(1−a)/c_c` is the standard Rao-Blackwellized
estimator — exact regardless of resolve nonlinearity. Self-consistency holds
because the spectral `(path, wavefront)` anchor applies the *same* clamped
resolve: both estimate the same clamped-gamut image. (Recorded pbrt divergence:
pbrt's `RGBFilm::AddSplat` keeps negative sensor RGB; already absorbed in the
spectral baselines.) The current state's stored RGB records stay valid across λ
mutations: λ is a PSS dimension, restored by the one-deep backup on reject and
replaced wholesale with the record stack on accept — no stale-λ record is ever
re-splatted. Consequences: `mltCurrentRecords` (binding 54) keeps its RGB
layout, the fixed-point splat buffer, `wfMltResolve`, and the `b/mpp_actual`
fold are byte-for-byte the RGB code paths. Mirrors the SPPM decision to resolve
per-pass φ to sRGB before the progressive fold. A test asserts no negative
value ever reaches the splat buffer.

*Alternative rejected:* spectral record stack + resolve in `wfMltResolve` —
larger chain-state buffers, a second resolve implementation, zero estimator
benefit (film is RGB either way).

### D5: Dispersion composes for free

Hero-λ Cauchy dispersion with secondary termination is inside the spectral BDPT
estimator; the chain mutating λ (D2) explores the dispersed image. No MLT-side
dispersion logic.

### D6: Uniform tail keys off the compiled layout (already structural)

The MLT uniform tail is packed against the **target MSL layout** (fix c9985b1),
not session flags — the spectral kernel set's layout (spectral stride changes)
is picked up automatically. A hostless test pins that the spectral-MLT MSL
layout round-trips the tail offsets.

### D7: Parity gates, harness-first

- **pbrt-truth:** regenerate suite EXRs with the pinned pbrt v4 running
  spectral `Integrator "mlt"` (it is spectral natively) for the gated scenes —
  emissive / caustic / BK7-prism / OpenPBR discriminators. Per-combo
  `baselines` recorded from measurement.
- **Self-consistency:** vs the **spectral** `(path, wavefront)` anchor with a
  recorded tolerance (`spectral_self_consistency`-style). Expect strictly worse
  than the RGB MLT 0.15 — spectral MLT stacks Markov correlation on the
  recorded spectral estimator/gamut-clamp divergence (0.06–0.08 at 256 spp) —
  and pre-commit to a **per-scene baseline on the prism** (dispersion is the
  hardest case) rather than one matrix-wide floor. Record what is measured,
  never loosen an existing RGB tolerance.
- `combo_is_valid` admits `(mlt, wavefront, spectral, flat)`; everything else
  about the MLT rule block stands.
- Spec surface: `render-cli` (drop `--spectral` from the mlt refusal bullet)
  and `render-parity-matrix` (validity-table MLT scenario + spectral-MLT
  rendered combo) get their own deltas alongside the two named in the
  proposal.
- Scenes authoring `maxcomponentvalue` stay out of the spectral-MLT gated set
  (pre-existing asymmetry: the path anchor clamps per sample, the MLT splat
  path is spec-forbidden from clamping — matching pbrt's unclamped
  `AddSplat`); record the asymmetry if such a scene is ever added.

### D8: Metal watchdog budget re-measured

One mutation = one complete **spectral** BDPT sample per chain — longer than
RGB. Keep the existing breadth-tiled chain-batch mechanism; re-measure the
per-sub-batch time under `SKINNY_METAL` and lower
`_MLT_METAL_CHAIN_BATCH_DEFAULT` for spectral sessions if needed (env override
already exists). Kill harness re-runs because kernel length changes.

## Risks / Trade-offs

- [λ-mutation correlation: a chain camped on a dispersion caustic mutates λ
  slowly, biasing short runs' color noise] → large-step probability guarantees
  ergodicity; film-averaged fold is unbiased per frame; tolerance measured at
  the suite's accumulation budget.
- [Spectral bootstrap `c` is noisier (4-λ estimate of Y), worsening chain
  seeding] → same estimator pbrt uses; bootstrap count already configurable
  (`bootstrapsamples`); black-bootstrap loud-failure unchanged.
- [Tolerance inflation could mask a real spectral-MLT bug] → dual gate: the
  pbrt-truth baseline is an absolute anchor per scene; both must hold. Fold
  in the prism scene where λ errors are maximally visible.
- [Spectral mutation kernel exceeds watchdog per sub-batch on Metal] → D8
  re-measure + smaller default batch; tiling already bit-identical.
- [Uniform-layout drift between spectral kernel set and host tail pack] → D6
  layout-keyed pack + pinned hostless test; this is the exact failure class
  c9985b1 fixed.
- [RGB regression risk] → byte-identity guards: RGB wavefront kernel `.spv`
  set and megakernel `.spv` asserted unchanged (same guard style as
  spectral-wavefront's 28-kernel check).

## Migration Plan

Additive, flag-gated. Land order: shader composition → host pass wiring
(Vulkan, then Metal) → gate flips (`cli_common`, `combo_is_valid`) → truth
regen + measured baselines → docs. Rollback = re-refuse the combo (flip the
two gate sites back); no data or format migration.

**Archive order:** the `spectral-rendering` / `render-parity-matrix` deltas
are written against the post-`spectral-wavefront` spec text, and OpenSpec
MODIFIED replaces wholesale by title — so the in-flight `spectral-rendering`,
`spectral-bdpt-megakernel`, and `spectral-wavefront` changes must archive
**before** this one (a later `spectral-wavefront` archive would revert the mlt
admissions). Re-validate the MODIFIED bases at archive time.

## Open Questions

None blocking.
