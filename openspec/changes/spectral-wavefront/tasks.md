# Tasks — spectral wavefront integrators

## 1. Widen the wavefront carriers to `Spectrum` (D1)

- [~] 1.1 Retype color roles `float3` → `Spectrum` (no `#if` on the field — RGB keeps
      `float3`). **DONE + RGB byte-identity verified** for `WavefrontPathState.throughput`/
      `radiance` (`wavefront_state.slang`) — recompiled RGB byte-identical vs fresh unedited
      control (same slangc). **`WfBdptAux`/`BDPTVertex` DONE (group 3):** color roles
      (`escaped`/`radiance`/`ewThroughput` on aux, `throughput`/`emission` on `BDPTVertex`) →
      `Spectrum`, `+sw` on aux; retype ripples into the merged spectral-megakernel `bdpt.slang`
      verified 0-effect — all 13 wavefront bdpt entries + megakernel `mainImage` RGB `.spv`
      byte-identical vs pristine `main`, spectral variant compiles clean. GPU co-verify pending.
      **`VisiblePoint`/`SppmAccum` DONE (group 4, [~]):** `beta`/`ld` → `Spectrum`;
      `tau` stays `float3` (D5 invariant); `+conductorMetalId` (spectral) on VP;
      `SppmAccum` `phiR/G/B` `+phiW` (spectral) — RGB names & `.spv` byte-identical.
      All 8 SPPM entries RGB byte-identical + spectral-compile-clean; GPU pending.
- [x] 1.2 Add `#if defined(SKINNY_SPECTRAL) SampledWavelengths sw;` to `WavefrontPathState`
      (imports `common` for `Spectrum`, `spectrum` for `SampledWavelengths` under the define).
      RGB layout byte-identical. BDPT/SPPM eye-carrier `sw` lands with groups 3/4.
- [x] 1.3 Keep the RGB arithmetic textually unchanged; add the `.w` term ONLY under
      `#if defined(SKINNY_SPECTRAL)`. Specifically the `wfFinishShade` RR max-component
      (`wf_shade_common.slang:135`) keeps `max(max(x,y),z)` verbatim and adds `max(·,w)` under
      the define (a rewritten `SPECTRUM_N` loop risks a non-bit-identical N=3 lowering → breaks
      the RGB `.spv`). RR survival `p` must match `path_spectral`/`bdpt_spectral`'s `max(…,.w)`.
- [~] 1.4 Mirror the widened strides in `wavefront_layout.py`. **DONE for path-state:**
      `_path_state_fields(spectral)` + `path_state_size(spectral=)` +
      `PATH_STATE_STRIDE_SPECTRAL`(=108 scalar)/`_MSL`(=128); `tests/test_wavefront_state.py`
      locks RGB (unchanged, 68/96) AND the spectral variant (parser now resolves
      `#if SKINNY_SPECTRAL` + `Spectrum`→float3/4) — 15/15 pass. **SPPM DONE (group 4, [~]):**
      `_visible_point_fields(spectral)` (192 B scalar / 240 B MSL) + `_sppm_accum_fields(spectral)`
      (20 B) + spectral strides + `sppm_buffer_sizes(spectral=)`; `tests/test_sppm_state.py`
      parser upgraded to resolve `#if SKINNY_SPECTRAL`+`Spectrum` and locks both variants — 22/22
      pass, RGB widths (180/240, 16) unchanged.
- [~] 1.5 Path-state spectral MSL stride computed (128, `float3`→16 B pad holds for `float4` so
      color retype is 0-byte on MSL; only `sw` adds +32). **The runtime Metal-reflection assert
      (`metal_wavefront.py:515`) needs a Metal device → GPU-interactive verification (deferred).**

## 2. Spectral wavefront path integrator (Group 2)

- [~] 2.1 `wfPathGenerate` (`wavefront_path.slang`) draws
      `sw = sampleWavelengths(rng.next())` **after** `cam.generateRay` and stores it in
      `s.sw`; `s.throughput`/`s.radiance` inits retyped `float3(x)` → `Spectrum(x)` (byte-id in
      RGB). **VERIFIED CPU:** `wfPathGenerate` RGB `.spv` byte-identical vs pre-Phase1 control +
      spectral-compile-clean. (GPU-render correctness pending user.)
- [~] 2.2 Spectral bounce body in `wf_shade_common.slang::wfFinishShade` (reachable via
      `wfPathShadeFlat`) + env-miss in `wfPathIntersect`. Reuses RGB `FlatMaterial.sample()`
      geometry (`br.bsdfSample.wi/pdf/delta/transmitted`); recolors NEE via
      `spectralAllLightsNEE`, throughput via `flatResponseS`/`cols.transmission`, emission via
      `cols.emission`/`planckSpectrum`, env-miss + sphere-hit via `upsampleIlluminantBound`,
      all with the scalar RGB MIS reused; hero-λ Cauchy dispersion collapse (`terminateSecondary`,
      `s.sw` persisted after) mirrors `path_spectral.slang` verbatim. RGB path is the `#else`
      branch (unchanged text). **VERIFIED CPU:** `wfPathShadeFlat` RGB byte-identical + spectral
      compiles. (GPU-render correctness pending user.)
- [~] 2.3 `wfPathResolve` resolves lane radiance via `spectrumResolveToLinearSRGB(sw, radiance)`
      before the running-mean write (`#if`-guarded; RGB `float3 sample = radiance` unchanged).
      **VERIFIED CPU:** `wfPathResolve` RGB byte-identical + spectral compiles.
- [~] 2.4 Only `wfPathShadeFlat` is the spectral path; the catch-all `wfPathShade`
      (skin/subsurface/volume/python) stays RGB-only and unreachable under spectral (scene
      refusal blocks those materials) — it still compiles the spectral variant (the shared
      `wfFinishShade` reloads the flat material for a non-flat hit, harmless-but-unreachable).
      **VERIFIED CPU:** `wfPathShade` RGB byte-identical + spectral compiles clean.

  NOTE (Group 2): `wfTerminate` param retyped `float3`→`Spectrum` in BOTH defs
  (`wavefront_path.slang` + `wf_shade_common.slang`); path-record emitters (`wfEmitRecords`/
  `wfPushRecord`, RGB-only float3 — neural guiding is refused under spectral) are compiled out
  under `#if !defined(SKINNY_SPECTRAL)` rather than widened. No new helper added to
  `spectral_flat_common.slang` — Group 2 consumes its existing `upsampleFlatColors`/
  `flatResponseS`/`spectralAllLightsNEE` unchanged. CPU verify: all 7 touched
  `wavefront_path.slang` entries (`wfPathGenerate`/`wfPathIntersect`/`wfBuildArgs`/`wfScatter`/
  `wfPathShadeFlat`/`wfPathShade`/`wfPathResolve`) RGB byte-identical vs the pre-Phase1 control
  (same slangc) AND spectral-compile-clean. GPU-render/Metal validation NOT possible here.

## 3. Spectral wavefront BDPT (Group 3)

- [~] 3.1 Thread `sw` (drawn in `wfBdptGenEye`) through eye + light subpath kernels; both
      subpaths of one sample share the 4 wavelengths. **DONE:** `aux.sw = sampleWavelengths(...)`
      in `wfBdptGenEye`; every eye/light/connect/splat/resolve kernel reads `aux.sw`.
- [~] 3.2 Spectral eye/light random walks: reuse the RGB sample/pdf machinery; recolor
      throughput via `flatResponseS`; light-origin β upsampled illuminant + blackbody/SPD
      overrides (mirror `bdpt_spectral.slang` §2.2-2.3). **DONE** (prior agent): eye/light-tail
      walk kernels + `wfSampleLightOriginS` wired.
- [~] 3.3 Spectral connect kernels (`wfBdptConnectNee`/`wfBdptConnectFull`): per-λ endpoint
      `flatResponseNEE`, scalar `misWeight` reused via the `asRgb` color-free projection
      (single MIS source, as the megakernel does). **DONE:** `wfBdptConnectLane` now has a
      `#if SKINNY_SPECTRAL` branch — `Spectrum L`; emissive via `z.throughput*z.emission`
      (both Spectrum, verbatim); `L += wfConnectT1S(eye,s,aux.sw,rng)`; generic via
      `WfSpectralConnectResult cr = wfConnectGenericS(eye,s,lightPath,t,aux.sw)` feeding the
      SAME scalar `misWeight(...,cr.pdf_*)`. RGB `#else` textually verbatim.
- [~] 3.4 `wfBdptSplat` (D4): `spectrumResolveToLinearSRGB(sw, contribution)` before the atomic
      add into the splat buffer; per-splat gamut-clamp bias note at the site. Because the clamp
      is nonlinear and splat granularity differs mega-vs-wave, spectral BDPT mega≡wave is
      asserted tight only on in-gamut scenes; the out-of-gamut dispersion-splat scene is a
      recorded self-consistency skip (see 6.4). Signed-XYZ linear fix deferred, out of scope.
      **DONE** (prior agent): `wfSplatLightWalkS` resolves via `spectrumResolveToLinearSRGB`
      before `atomicSplatRadiance`.
- [~] 3.5 `wfBdptResolve` spectral env/emissive escaped-radiance resolve. **DONE:**
      `#if SKINNY_SPECTRAL float3 sample = spectrumResolveToLinearSRGB(wfAux[slot].sw,
      wfAux[slot].radiance)` before the NaN-guard/clamp/running-mean (RGB `#else` verbatim).
      `wfBdptClassify` escaped-only finalise (`aux.radiance = aux.escaped * lensWeight`) works
      verbatim (both Spectrum).

  **CPU VERIFY (Group 3):** all 13 wavefront_bdpt entries (`wfBdptGenEye` `wfBdptBounceEye`
  `wfBdptWalk` `wfBdptLightTail` `wfBdptClassify` `wfBdptConnectNee` `wfBdptConnectFull`
  `wfBdptSplat` `wfBdptResolve` `wfBdptGenLight` `wfBdptScatter` `wfBdptBuildArgs`
  `wfBdptWalkClassify`) spectral-compile-clean (`-D SKINNY_SPECTRAL=1`) AND RGB `.spv`
  byte-identical vs the pristine `main` checkout (same slangc). Megakernel `mainImage`
  (`main_pass.slang`, no `-D SKINNY_WAVEFRONT`) RGB byte-identical vs main + spectral compiles
  (BDPTVertex retype in `bdpt.slang` is 0-effect on RGB `.spv`). No TODO. GPU-render/Metal
  correctness NOT verified here (CPU-only).

## 4. Spectral wavefront SPPM (Group 4, the hard one — D5, redesigned per review CRIT)

- [~] 4.1 **One shared hero-λ set per SPPM *pass***: `sppmPassWavelengths()` (`wavefront_sppm.slang`)
      seeds `sw = sampleWavelengths(pcgHash(fc.frameIndex)/2³²)` — used by BOTH the photon stage
      AND the eye visible-point stage of that pass, so photons and VPs agree on λ (a photon
      deposits onto many VPs → `phi = beta ⊗ f_r(VP)` is a coherent per-λ product). NOT per-path.
      **TODO(spectral-wavefront 4.1):** assumes `fc.frameIndex` is the per-pass counter (it already
      seeds the eye + photon RNGs via `createRNG(..., fc.frameIndex)`, so both stages observe the
      same value within a pass — no new renderer plumbing). If a dedicated `sppmPassIndex` uniform
      is ever added, seed from it instead. **VERIFIED CPU**, GPU-render pending.
- [~] 4.2 `sppmEmitPhoton` seeds `Spectrum beta` at the pass λ: emissive triangle → exact
      `planckSpectrum` (blackbody) else `upsampleIlluminantBound`; sphere → `upsampleIlluminantBound`;
      distant → `distantLightSpd`(slot≥0) else upsampled. `sw` threaded in `#if`-guarded param.
- [~] 4.3 **Photon-deposit BSDF recolor** (`sppmDepositPhoton`): `f = flatResponseNEE(mat, cols,
      woVP, wiVP)/wiVP.z` at the pass `sw` via `upsampleFlatColors(mat, sw, vp.conductorMetalId)`
      — exact conductor Fresnel from the VP-stored metal id — so `phi = beta ⊗ f_r` is coherent
      (`flatResponseNEE` includes `·opacity`, matching the RGB `evaluate().response = f·op`).
- [~] 4.4 `SppmAccum` grows `phiR/G/B` `+phiW` under `#if SKINNY_SPECTRAL` (**not** renamed to
      `phi0..3` — that changed OpMemberName strings and broke RGB `.spv` byte-identity; `phiW` is
      the RGBA hero-λ slot); atomically summed at deposit; `sppmDecodeFlux` returns `Spectrum`.
      `SPPM_FLUX_FIXED_SCALE` kept 2²⁰ — no dispersion collapse in SPPM (see 4.5) spreads flux
      across 4 λ so per-λ range ≤ RGB; numpy-mirror re-measure is a GPU-render task (pending).
- [~] 4.5 **Resolve λ→linear sRGB at pass end BEFORE the progressive fold** (`wfSppmUpdate`):
      `spectrumResolveToLinearSRGB(gSw, phiS/denom)` → 3-wide `lIndirect`; `vp.ld` resolved into
      the pass sample; `sppmUpdate` runs on the resolved 3-wide `phi` so `tau`/estimator stay
      spectral-invariant (the `(r'/r)²` ratio is scalar geometry). Rotating `sw` across passes
      integrates the spectrum via the film mean. `tau` is NOT persisted per-λ. NOTE: secondary-λ
      dispersion collapse is deliberately NOT applied in the SPPM eye/photon carriers (would break
      photon/VP λ-coherence) → SPPM caustics are non-dispersive (documented v1 limit; mega
      path/bdpt carry dispersion). **VERIFIED CPU.**
- [~] 4.6 `wavefront_layout.py`: `_visible_point_fields(spectral)` (192 B / 240 B MSL, `+conductorMetalId`,
      `tau` float3) + `_sppm_accum_fields(spectral)` (`+phiW`, 20 B) + spectral strides +
      `sppm_buffer_sizes(spectral=)`; `tests/test_sppm_state.py` locks both variants (parser now
      resolves `#if SKINNY_SPECTRAL`+`Spectrum`; store/load regex tolerates the guarded param).
      RGB strides unchanged (180/240, 16). 22/22 pass. Runtime Metal-reflection assert deferred (GPU).

## 5. Startup gate + renderer pin + capability (Group 5, D6)

- [x] 5.1 `reject_spectral_unsupported` (`cli_common.py:197-265`): drop the `sppm`-refused
      (`:221-227`) and `wavefront`-refused (`:228-233`) branches; still refuse non-BSDF
      proposals (`:240-249`), `--reuse` (`:250-254`), neural, and skin/volume scenes. Update
      refusal wording.
- [x] 5.2 Thread `self._spectral` into the wavefront pipeline build (today only megakernel
      `ComputePipeline(..., spectral=...)` at `renderer.py:3719/3738`) — the wavefront
      `vk_wavefront`/`metal_wavefront` compiles inject `-DSKINNY_SPECTRAL`.
- [x] 5.3 Widen the renderer spectral pin so `{path, bdpt, sppm}` under wavefront are reachable
      (`integrator_index`, `effective_execution_mode_index`, `_collect_config_rows`); confirm
      `resolve_execution_mode` sends `sppm` → wavefront for a spectral session.
- [x] 5.4 `spectral_capability`: add a wavefront-scope note (envelope now includes wavefront +
      sppm). Keep `SPECTRAL_IMPLEMENTED = True`.

## 6. Parity matrix (Group 6, D7)

- [x] 6.1 `pbrt/parity.py`: `spectral_envelope`/`combo_is_valid` admit `(path|bdpt|sppm,
      wavefront, spectral)`.
- [x] 6.2 **Spectral-aware anchor** (review, major — new): make `ANCHOR`/`combo_axis_class`/
      anchor-image selection pick the **megakernel spectral path** image as the anchor for the
      spectral axis, so a spectral combo is never compared against the RGB anchor. Lift the
      mega≡wave "spectral-only skip" for spectral path/bdpt (now assertable against the spectral
      anchor). SPPM spectral anchors to the **spectral path** anchor at the `sppm` tolerance
      class (NOT the RGB golden); fallback = pbrt-truth-only if spectral-path-anchoring is
      infeasible.
- [x] 6.3 Coverage meta-test: the `sppm × spectral` validity entry exists so the build does not
      fail; neural/reuse/skin spectral stay recorded exclusions.
- [x] 6.4 Register a dispersion / blackbody-lit discriminating suite scene under wavefront if
      not already covered; record the spectral-BDPT dispersion-splat mega≡wave skip on it.

## 7. Metal specifics (Group 7)

- [ ] 7.1 The widening grows element **strides within existing buffers**, not the count of
      bound buffers, so the 31-slot argument-table cap is low-risk (review, minor). Real Metal
      watch-item is the **MSL stride asserts** (7.2/1.5): color retype is 0-byte on MSL
      (`float3` already 16 B), only `sw` (+32 B) grows the MSL stride; the 128-texture cap is
      untouched (spectral adds no textures). Cheap slot-count sanity check only.
- [ ] 7.2 Metal indirect-dispatch CPU-readback fallback (`metal_wavefront.py:233-249`) carries
      the widened record stride correctly (latent stride bug shows only on Metal).
- [ ] 7.3 Metal dispatch hygiene: spectral shade/connect/photon kernels are longer — the
      naturally-short staged kernels stay under the watchdog; kill harness
      (`tests/test_metal_cleanup.py -m gpu`) passes with the spectral variant compiled.

## 8. Tests / gates (Group 8)

- [ ] 8.1 Hostless: CLI-gate tests (wavefront + sppm now accepted; neural/reuse/skin still
      refused); `wavefront_layout` stride tests both variants (expect scalar≠MSL divergence).
- [ ] 8.1b **Build the byte-identity guard** (review, major — does not exist today;
      `test_spectrum_compile.py` only checks *compilation*). Per touched wavefront kernel, a
      before/after **compile-and-compare**: compile the RGB variant from the pre-edit and
      post-edit source and assert byte-equal. NOT "equals the checked-in `.spv`" (won't survive
      a slangc bump).
- [ ] 8.2 GPU self-consistency A/B (Metal + Vulkan): spectral wavefront path/bdpt vs the
      megakernel spectral path anchor; spectral wavefront sppm vs its RGB golden — via
      `compute_metrics` (no hand-rolled error formulas).
- [ ] 8.3 Dispersion demo (BK7 prism) under spectral wavefront BDPT renders caustics;
      compare to megakernel spectral BDPT.
- [ ] 8.4 Furnace closure holds under spectral wavefront for each integrator.

## 9. Docs (Group 9)

- [x] 9.1 `docs/Spectral.md` scope: wavefront + sppm now in the envelope.
- [x] 9.2 `docs/Wavefront.md`: spectral carrier (`Spectrum` retype + `SampledWavelengths`),
      per-λ SPPM deposit accumulator.
- [x] 9.3 `CLAUDE.md` + `README.md` compatibility matrices: spectral row spans wavefront and
      all three integrators.
- [x] 9.4 `CHANGELOG.md` entry.
