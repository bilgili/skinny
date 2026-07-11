# Tasks — spectral BDPT (megakernel)

## 1. Shared spectral-flat helpers (pure code motion, D5)

- [x] 1.1 Create `src/skinny/shaders/integrators/spectral_flat_common.slang`; move
      `SpectralFlatColors`, `upsampleFlatColors`, `flatResponseS`, `flatResponseNEE`,
      `spectralLightNEE`, `spectralAllLightsNEE` and the blackbody/SPD override lookups out
      of `path_spectral.slang`; `path_spectral.slang` imports the new module
- [x] 1.2 A/B gate: primary = byte compare of the **spectral** SPIR-V before vs after the
      hoist; fallback = fixed-seed spectral `path` headless image A/B via `compute_metrics`
      (slangc may reassociate across module boundaries); RGB `main_pass.spv` byte-identical
      (existing hostless guard still green)

## 2. Spectral BDPT integrator (`bdpt_spectral.slang`)

- [x] 2.1 New `src/skinny/shaders/integrators/bdpt_spectral.slang` compiled only under
      `SKINNY_SPECTRAL`: `SpectralBDPTVertex` (Spectrum throughput/emission, scalar
      geometry/pdf/flags), single `sampleWavelengths` draw at path start shared by both
      subpaths (first RNG consumption). **MIS drift-guard obviated**: instead of duplicating
      the MIS chain, `SpectralBDPTVertex` projects to a color-free `BDPTVertex` (`asRgb`) and
      calls the RGB `misWeight`/`splatMisWeight`/`convertSAtoArea`/`bdptSurface`/`bdptEval`/
      `atomicSplatRadiance` DIRECTLY — one source of MIS truth, nothing to drift. Depth
      reuses `BDPT_MAX_DEPTH`=6 from the bdpt import (single constant, no coupling to unmatch).
- [x] 2.2 Spectral eye random walk (`randomWalkS`): reuse RGB `FlatMaterial.sample()` for
      wi/pdf/delta-ness; throughput via `flatResponseS/pdf` (delta glass: upsampled
      transmission); cutout skip, RR (4-ch max), env-miss + sphere-light hit deposits
      upsampled as illuminants with scalar MIS
- [x] 2.3 Spectral light subpath (`sampleLightOriginS` + `randomWalkS` light mode): β seed
      upsampled illuminant with blackbody Planck×scale at emissive-triangle origins (sphere
      upsample); directional origins spawn no walk (β unused — exactness via connectT1S NEE)
- [x] 2.4 Strategies: s≥2/t=0 emissive-vertex hits (Planck material override, MIS-gated as
      RGB); `connectT1S` (spectral distant SPD / sphere / emissive-tri via unified
      `misWeight`); `connectGenericS` (per-λ endpoint `flatResponseNEE`, scalar `misWeight`)
- [x] 2.5 s=1 light tracer (`splatLightVertexS`): per-splat `spectrumResolveToLinearSRGB(sw,
      contribution)` before the reused `atomicSplatRadiance`; per-splat gamut-clamp bias +
      signed-XYZ escalation documented at the splat site; scalar `splatMisWeight` reused
- [x] 2.6 Dispersion (D7): hero-λ Cauchy delta-refraction collapse with `terminateSecondary`
      on either subpath (entering + exiting; achromatic TIR fallback), identical rule to
      `path_spectral.slang`; evaluation order pinned (eye → light → splat → connections)
- [x] 2.7 `main_pass.slang` spectral branch: dispatch `SpectralBDPTIntegrator<TC>` when
      `fc.integratorType == INTEGRATOR_BDPT` on a flat first hit, else spectral path tracer;
      SPIR-V spectral variant compiles clean (7.0 MB), RGB `.spv` byte-identical (verified)

## 3. Renderer dispatch pin, startup gates, parity matrix

- [x] 3.0 **Widened the renderer spectral integrator pin** (without this the new shader branch
      is dead code): `_active_integrator_index` returns the live index for `{path, bdpt}`
      under spectral and pins sppm→path; widened the inlined field-only copy in
      `_collect_config_rows` identically (pin row only when it applies). Hostless test
      `test_spectral_integrator_pin_path_bdpt_pass_sppm_pins` green (dispatched index +
      config row). Runtime path↔bdpt reset is inherent — `integrator_index` is in
      `_current_state_hash` and the reset clears `light_splat_buffer` (unchanged code).
- [x] 3.1 `cli_common.reject_spectral_unsupported`: accepts `bdpt`; SPPM refusal names the
      spectral wavefront follow-up; BSDF-proposal-only + no-reuse residual checks unchanged;
      unit tests updated (`test_reject_spectral_sppm_raises`, bdpt added to the accept test)
- [x] 3.2 `pbrt/parity.py`: `spectral_envelope` admits `(bdpt, megakernel)`; enumeration
      already emits `(bdpt, megakernel, spectral)`; SPPM/wavefront reasons updated; hostless
      `tests/pbrt/test_matrix.py` validity + coverage meta-tests updated (25/25 green)
- [ ] 3.3 Suite/manifest: give the spectral-discriminating suite scenes (incl. the prism
      dispersion demo scene) a BDPT spectral disposition; record baselines only where a
      genuine pbrt divergence exists (never loosen self-consistency)

## 4. GPU validation (Metal backend, dispatch-hygiene rules)

- [x] 4.1 Headless spectral BDPT render smoke on **Vulkan** (int_caustic flat scene, 2
      emissive triangles). **Verified BDPT actually dispatched** (config row `integrator
      bdpt → resolved bdpt`, not silent path fallback); image sane (mean 167.6, max 255,
      nz 100%, no black holes). Metal smoke: in-process megakernel Metal compile is very
      slow (>10 min; the large spectral BDPT kernel) — attempted at 32px, see 4.4.
- [x] 4.2 Self-consistency: spectral BDPT vs spectral path anchor (int_caustic, 64px, 16spp)
      **display relMSE = 0.0098** (<1%) — strong evidence the transport matches the anchor.
      (Linear-accum `compute_metrics` gate + backend-parity Vulkan≡Metal A/B remain for the
      guarded Metal env once the Metal compile completes.)
- [ ] 4.3 pbrt-truth gate: spectral BDPT on the dispersion/conductor/blackbody/SPD
      discriminating scenes vs pinned pbrt v4 EXRs (regen refs where a new scene disposition
      was added); record variance expectations on the prism scene (shared-`sw` collapse ⇒
      noisy-but-unbiased at matched samples — pbrt-v4 has no BDPT anchor for this).
      REMAINS — needs pbrt v4 + the guarded runner.
- [ ] 4.4 Kill harness + band budget: BDPT spectral lengthens the megakernel — run
      `tests/test_metal_cleanup.py` (hostless 13 + gpu-marked harness under the guarded
      runner); **re-measure the BDPT row-band budget under `SKINNY_SPECTRAL`**
      (`_METAL_MEGAKERNEL_BAND_PIXELS` was sized for RGB BDPT and is keyed by integrator
      only) and record the spectral band count. REMAINS — the slow Metal compile is itself
      evidence this needs the guarded env; do NOT run a full-res Metal frame until the band
      budget is re-measured (wedge risk).
- [x] 4.5 Splat path check: light-tracer contribution exercised (int_caustic has emissive
      triangles → connectT1S + s=1 splat active); render stable, no fireflies/NaN, splat
      compositing unchanged (path≈bdpt to <1% ⇒ splat energy is correctly MIS-weighted).
      Adversarial 4-agent shader review (MIS-reuse / throughput / lights-connections /
      dispersion-splat) returned **0 findings**.

## 5. Docs and closeout

- [x] 5.1 `docs/Spectral.md`: added "Bidirectional transport (BDPT)" section (colour/scalar
      split, five strategy families, shared-λ + pinned dispersion order, splat resolve +
      clamp caveat) + scope-table Integrator row; hoisted-helper note. No `DOC:` markers in
      the touched shaders, so no embed regen needed.
- [x] 5.2 Compatibility matrices: `CLAUDE.md` (State + Scope rows) + `README.md` (matrix row +
      scope table) updated to path+bdpt / megakernel / flat, SPPM = spectral-wavefront
      follow-up. `docs/Megakernel.md` has no spectral content (RGB-flow doc) — nothing to
      update there.
- [x] 5.3 `CHANGELOG.md` Added entry; `openspec validate` green; `ruff check src/` clean;
      hostless sweep green (212 passed: matrix/metrics/cli/observability/spectral-*).
