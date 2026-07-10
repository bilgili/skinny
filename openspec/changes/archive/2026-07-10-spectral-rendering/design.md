# Spectral Rendering — Design

## Context

skinny transports radiance as `float3` RGB everywhere: `BSDFSample`/`LightSample`/
`BounceResult` in `interfaces.slang`, `throughput`/`radiance` in `integrators/path.slang` and
`wavefront_state.slang`, `float3` env lookups in `environment.slang`, and an RGBA32F
accumulation buffer fed by `main_pass.slang` (megakernel) / `wavefront_env.slang` (wavefront).
The pbrt importer collapses every authored spectrum to linear sRGB at import
(`pbrt/spectra.py`, analytic Wyman–Sloan–Shirley CMFs, XYZ→sRGB matrix); no spectral data
reaches the GPU. pbrt v4 — the parity ground truth — renders spectrally, so the RGB reduction
is a standing, recorded divergence.

Constraints that shape the design:

- **Session-fixed modes are the house pattern** (`--execution-mode`, `--backend`): a spectral
  mode fixed at startup is acceptable and simpler than a runtime toggle.
- **Vulkan SPIR-V byte-stability**: the `SKINNY_METAL` precedent shows compile-time `#if`
  gating that leaves the default build byte-unchanged is the accepted way to add a variant.
- **Metal compute argument table is full** (128 textures: 119-slot bindless pool + discrete
  maps). A new *texture* is expensive; a new *storage buffer* is cheap.
- **Parity harness discipline**: every new renderer axis must land in `combo_is_valid` +
  coverage meta-tests, and every image number comes from `compute_metrics`.
- **Metal dispatch hygiene**: no change may lengthen a committed command buffer past the
  watchdog budget; the megakernel band tiling and wavefront per-tile flushes must keep working.

## Goals / Non-Goals

**Goals:**

- Hero-wavelength spectral transport in the **path integrator, megakernel execution mode**,
  Vulkan + Metal, producing images that converge closer to pbrt v4 spectral ground truth than
  the RGB pipeline on spectrum-authored scenes.
- Spectral-vs-RGB is decided **once, at startup**: kernels compile as the spectral variant or
  the RGB variant, never both, never switched mid-session.
- All existing RGB assets (material params, textures, HDR envs) render under spectral mode with
  no re-authoring, via RGB→spectrum upsampling.
- Exact spectral evaluation where the source data is spectral: blackbody emitters, named
  conductor eta/k, authored illuminant SPDs, wavelength-dependent dielectric IOR (dispersion).
- The RGB pipeline stays bit-identical when spectral mode is off.
- Spectral mode is a first-class parity-matrix axis with recorded validity.

**Non-Goals:**

- Spectral **wavefront** execution mode (per-path wavelength in `WavefrontPathState`, staged
  kernels, `wavefront_env.slang` resolve) — the designated **first follow-up**; the `Spectrum`
  typealias and variant-define plumbing land ready for it.
- Spectral BDPT / SPPM / ReSTIR DI reuse / neural proposal (float3 carriers in
  `BDPTVertex`, photon/splat buffers, reservoirs, and training records stay RGB; follow-ups).
- Spectral heterogeneous volumes (spectral majorants for delta tracking are a separate design).
- Spectral skin BSSRDF (the skin estimator chain keeps its documented RGB approximation).
- Spectral reflectance SPDs on materials (authored *reflectance* spectra still reduce to RGB
  then re-upsample in v1; only illuminants/conductors/blackbody get exact spectral paths).
- Runtime toggling mid-session; GUI control (CLI/env only in v1).
- Polarization, fluorescence.

## Decisions

### D1. Compile-time spectral variant, not a runtime branch

Spectral mode selects a **separately compiled shader variant** (`-DSKINNY_SPECTRAL`), chosen at
startup like the execution mode. A `Spectrum` typealias in `common.slang` resolves to `float3`
(RGB build) or `float4` (spectral build); carriers (`BSDFSample.weight/response/emission`,
`LightSample.radiance`, path `throughput`/`radiance`) use it. The default
build's SPIR-V stays **byte-identical** (same guarantee style as `SKINNY_METAL`), so all
existing parity baselines remain valid without re-measurement.

- *Alternative — runtime `fc.spectralMode` branch*: rejected. Doubles register pressure and
  divergence in the megakernel for every user, contaminates the RGB codepath, and cannot change
  struct layouts (wavefront state needs a per-path wavelength).
- *Alternative — Slang generics over a spectrum interface*: rejected; would force a large
  refactor of concrete `float3` code and both backends' pipeline caching for no runtime
  benefit over a define.
- Consequence: a second megakernel binary (`main_pass_spectral.spv`) for Vulkan; the Metal
  variant compiles through the existing in-process SlangPy path with the define folded in.
  The define/typealias plumbing is deliberately shared-header (`common.slang` /
  `spectrum.slang`) so the wavefront follow-up only touches wavefront files.

### D2. Hero-wavelength sampling, N = 4, pbrt-matching distribution

Each camera path draws one hero wavelength at path start (one extra `rng.next()`; the PCG
stream needs no dimension bookkeeping) from pbrt's **visible-wavelength importance
distribution** over [360, 830] nm, with 3 secondary wavelengths by equal-spaced rotation
(Wilkie et al. 2014). `SampledWavelengths` carries `float4 lambda`, `float4 pdf`, and a
secondary-terminated flag — a megakernel local for the whole path lifetime (no persistent
state; the wavefront follow-up will pack hero λ + terminated bit into `WavefrontPathState`).

- *Why pbrt's distribution*: the parity gates compare against pbrt v4; matching its wavelength
  pdf removes one source of variance/bias mismatch and lets furnace tests close tightly.
- *Alternative — uniform sampling*: simpler but measurably noisier near the photopic peak;
  rejected since the visible-pdf is a few lines.

### D3. RGB→spectrum upsampling: Jakob–Hanika 2019 via storage buffer

RGB reflectances (material params, textures, tattoo/albedo maps) upsample on the GPU with the
Jakob–Hanika sigmoid-polynomial model: fetch (c0, c1, c2) from a precomputed sRGB coefficient
table, evaluate `sigmoid(polynomial(λ))` per sampled wavelength. RGB **illuminants** (lights,
env map texels) use the same coefficients scaled by a tabulated CIE **D65** SPD, matching
pbrt's `RGBIlluminantSpectrum`. The table ships as a checked-in compressed `.npz` under
`src/skinny/pbrt/data/` (generated by a checked-in port of pbrt's `rgb2spec` optimizer;
regeneration script committed, table validated against pbrt-published coefficients) and
uploads to a **StorageBuffer** with manual trilinear interpolation in-shader.

- *Why a buffer, not a `Texture3D`*: the Metal argument table has zero texture slack (the
  nanovdb change already trimmed the bindless pool 120→119); pbrt interpolates this table in
  code anyway, so hardware filtering buys nothing.
- *Alternatives considered*: Smits 1999 basis (fast, but non-smooth and inaccurate — rejected);
  Meng et al. 2015 (needs an iterative solve per lookup — rejected); on-the-fly sigmoid fit
  (cheap but diverges from pbrt — rejected because parity is the point).
- Env map stays RGBA32F; upsampling happens per radiance lookup (no pre-decomposed coefficient
  env texture in v1 — revisit if profiling shows the fetch dominating).

### D4. Film resolve before the accumulation buffer; everything downstream unchanged

At path termination the spectral estimate resolves to color **per sample**:
`L(λ_i)/pdf(λ_i)` averaged over the 4 samples → XYZ via the **analytic Wyman–Sloan–Shirley
CMF fit already used by `spectra.py`** (no CMF table; GPU and importer share one definition,
normalized by the CIE Y integral) → linear sRGB via the existing matrix → existing
`clampSampleRadiance` → RGBA32F accumulation. Progressive accumulation, exposure, tonemap,
sRGB display, readback, and the parity harness's linear-HDR sampling are untouched.

- *Alternative — accumulate XYZ and convert at display*: rejected; would fork the accumulation
  buffer semantics, `render_linear`, and every readback consumer for no accuracy gain at 4
  samples/pixel-sample.
- Negative sRGB components from out-of-gamut spectra are clamped at the existing clamp site,
  matching current importer behavior.

### D5. Exact spectral sources: blackbody, conductors, authored illuminant SPDs, dispersion

- **Blackbody**: importer preserves the authored temperature (via `skinnyOverrides`, the
  established side-channel); the spectral build evaluates Planck's law at the sampled
  wavelengths directly, normalized as pbrt does. RGB build keeps using the imported RGB.
- **Named conductor eta/k**: full spectral curves for the pbrt named metals join the vendored
  data (currently only 3-primary RGB values); uploaded densely resampled on the 5 nm grid into
  a shared **spectral-asset storage buffer**; conductor Fresnel evaluates at sampled λ.
- **Authored illuminant SPDs** (`spectrum L` on lights): densely resampled at import onto the
  360–830/5 nm grid, preserved through USD, uploaded into the same spectral-asset buffer, and
  looked up per sampled λ. (Reflectance SPDs stay RGB-reduced in v1 — recorded limitation.)
- **Dispersion**: dielectrics with wavelength-dependent IOR (Cauchy fits for pbrt named
  glasses, preserved at import) evaluate IOR at the hero wavelength; on the first dispersive
  refraction the path **terminates secondary wavelengths** (pbrt's strategy: hero pdf ×= 1/N),
  yielding correct dispersion without 4-way ray splitting.

### D6. Scope enforcement mirrors existing refusal/validity precedents

`--spectral` (env `SKINNY_SPECTRAL`, session-fixed, resolved in `cli_common.py`) is refused at
startup when combined with: integrator ≠ path, execution mode `wavefront` (explicit flag or
env; `auto` + path already resolves to megakernel via `resolve_execution_mode`), a scene
requiring subsurface/skin or heterogeneous-volume transport, ReSTIR DI reuse, or the neural
proposal — same pattern as `sppm + explicit megakernel` refusal. `parity.py` gains a
`spectral` axis; `combo_is_valid` records the same rules; the coverage meta-test enforces that
every integrator × spectral pair has a validity entry. Spectral mode participates in
`_current_state_hash()` indirectly (it is session-fixed, so accumulation-reset wiring is not
needed beyond startup).

### D7. Parity gating: spectral must tighten, never loosen

Spectral combos gate against the **same** checked-in pbrt EXRs (which are spectral renders):
- Self-consistency: with a single execution mode in scope there is no mega≡wave assertion for
  spectral combos — the mode-equivalence gate records the skip reason "spectral is
  megakernel-only" (cleared by the wavefront follow-up). Correctness is instead pinned by
  GPU≡numpy kernel mirrors (wavelength sampling, upsampling, film resolve) and a
  Vulkan-vs-Metal spectral A/B on one corpus scene.
- pbrt-truth: on spectrum-discriminating scenes the spectral combo's relMSE/FLIP must be **at
  or below** the RGB combo's recorded measurement; the confirming suite gains at least one
  discriminating scene (dispersive dielectric and/or blackbody-lit) where RGB visibly
  mismatches and spectral closes the gap.
- Furnace closure: the white furnace (constant env → upsampled flat spectrum → lossless
  material) must still close under spectral mode within the existing uniformity gate.
- Spectral-vs-RGB anchor comparisons are *reported*, not asserted tight — they legitimately
  differ; that difference is the feature.

## Risks / Trade-offs

- [Sigmoid-table divergence from pbrt's exact table] → port pbrt's `rgb2spec` generator
  verbatim, validate generated coefficients against pbrt's published table values in a hostless
  unit test before any GPU work.
- [Second megakernel SPIR-V doubles the stale-spv failure surface] → spectral variant compiles
  only when `--spectral` is active; the compile check covers both builds; byte-identity test
  pins the default binary.
- [Extra ALU + table fetches per bounce regress megakernel throughput] → measure equal-time
  spectral-vs-RGB on a corpus scene; spectral is opt-in, RGB perf unaffected by construction
  (byte-identical build).
- [Metal watchdog: spectral megakernel is heavier per pixel] → existing row-band tiling
  (`_metal_megakernel_bands`) already bounds breadth; re-run the kill harness
  (`tests/test_metal_cleanup.py -m gpu`) since kernel length changes.
- [Table size in-repo (~MB-scale .npz)] → acceptable precedent (vendored data dir); compressed;
  regeneration script checked in so it can be regenerated rather than edited.
- [Out-of-gamut/negative sRGB after resolve] → clamp at the existing `clampSampleRadiance`
  site; assert non-negativity in the furnace suite.
- [Importer payload preservation bloats USD] → payloads ride `skinnyOverrides` only when a
  spectrum was actually authored; RGB-authored scenes carry nothing new.

## Migration Plan

1. Land data + CPU groundwork first (tables, importer preservation, hostless tests) — no
   renderer behavior change.
2. Land the spectral megakernel variant behind `--spectral` (default off; RGB SPIR-V
   byte-checked in a test).
3. Wire parity axis + confirming scenes; record baselines.
4. Rollback = don't pass `--spectral`; the flag can be removed cleanly since no persisted
   settings or file formats change (mode is intentionally not persisted, like execution mode).
5. Follow-up change (separate proposal): spectral wavefront — extends `WavefrontPathState`,
   stage kernels, and clears the recorded mode-equivalence skip.

## Open Questions

- Whether the checked-in `main_pass_spectral.spv` should be deferred (compile-on-demand via
  `spv_cache` on Vulkan too) to avoid a second stale-spv failure mode — decide during
  implementation based on Vulkan startup-latency measurement.
- Whether env-map radiance upsampling per lookup is fast enough at 1024×512 IBL-heavy scenes,
  or a pre-decomposed coefficient environment (3×float32 per texel storage buffer) is needed —
  profile in task order before optimizing.
- Exact placement of the two new storage buffers in the descriptor map (next free bindings
  after 32) — assign during implementation and update the Architecture.md binding map.
