# Tasks — spectral wavefront integrators

## 1. Widen the wavefront carriers to `Spectrum` (D1)

- [~] 1.1 Retype color roles `float3` → `Spectrum` (no `#if` on the field — RGB keeps
      `float3`). **DONE + RGB byte-identity verified** for `WavefrontPathState.throughput`/
      `radiance` (`wavefront_state.slang`) — recompiled RGB byte-identical vs fresh unedited
      control (same slangc). **DEFERRED to groups 3/4:** `WfBdptAux`/`BDPTVertex` (retype ripples
      into the merged spectral-megakernel `bdpt.slang` `asRgb` — GPU co-verify) and
      `VisiblePoint`/`SppmAccum` (SPPM redesign D5 — 4-wide accum, co-verify with group 4).
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
      `#if SKINNY_SPECTRAL` + `Spectrum`→float3/4) — 15/15 pass. `VISIBLE_POINT_FIELDS`/
      `SPPM_ACCUM_FIELDS` spectral width lands with group 4.
- [~] 1.5 Path-state spectral MSL stride computed (128, `float3`→16 B pad holds for `float4` so
      color retype is 0-byte on MSL; only `sw` adds +32). **The runtime Metal-reflection assert
      (`metal_wavefront.py:515`) needs a Metal device → GPU-interactive verification (deferred).**

## 2. Spectral wavefront path integrator (Group 2)

- [ ] 2.1 `wfPathGenerate` (`wavefront_path.slang:61`) draws
      `sw = sampleWavelengths(rng.next())` **after** `cam.generateRay` (generateRay consumes the
      RNG first — matches the megakernel/`path_spectral` ordering) and stores it in the record.
      The draw is `#if`-guarded → RGB stream byte-identical.
- [ ] 2.2 Spectral bounce body: reuse RGB `FlatMaterial.sample()` for wi/pdf/delta; recolor
      throughput/NEE via `spectral_flat_common.slang` (`flatResponseS`/`flatResponseNEE`/
      `spectralAllLightsNEE`); env-miss + emissive/sphere hits deposit upsampled illuminant
      with scalar MIS. Hero-λ Cauchy dispersion collapse (`terminateSecondary`) identical to
      `path_spectral.slang`.
- [ ] 2.3 `wfPathResolve` (`:266`) resolves lane radiance via `spectrumResolveToLinearSRGB(sw,
      radiance)` before the accumulation running-mean write.
- [ ] 2.4 Guard: only the flat shade path (`wfPathShadeFlat`) is reachable under spectral; the
      catch-all `wfPathShade` (skin/subsurface/volume/python) is unreachable (scene-level
      refusal already blocks those materials).

## 3. Spectral wavefront BDPT (Group 3)

- [ ] 3.1 Thread `sw` (drawn in `wfBdptGenEye`) through eye + light subpath kernels; both
      subpaths of one sample share the 4 wavelengths.
- [ ] 3.2 Spectral eye/light random walks: reuse the RGB sample/pdf machinery; recolor
      throughput via `flatResponseS`; light-origin β upsampled illuminant + blackbody/SPD
      overrides (mirror `bdpt_spectral.slang` §2.2-2.3).
- [ ] 3.3 Spectral connect kernels (`wfBdptConnectNee`/`wfBdptConnectFull`): per-λ endpoint
      `flatResponseNEE`, scalar `misWeight` reused via the `asRgb` color-free projection
      (single MIS source, as the megakernel does).
- [ ] 3.4 `wfBdptSplat` (D4): `spectrumResolveToLinearSRGB(sw, contribution)` before the atomic
      add into the splat buffer; per-splat gamut-clamp bias note at the site. Because the clamp
      is nonlinear and splat granularity differs mega-vs-wave, spectral BDPT mega≡wave is
      asserted tight only on in-gamut scenes; the out-of-gamut dispersion-splat scene is a
      recorded self-consistency skip (see 6.4). Signed-XYZ linear fix deferred, out of scope.
- [ ] 3.5 `wfBdptResolve` spectral env/emissive escaped-radiance resolve.

## 4. Spectral wavefront SPPM (Group 4, the hard one — D5, redesigned per review CRIT)

- [ ] 4.1 **One shared hero-λ set per SPPM *pass*** (seeded from the pass index), used by BOTH
      the photon stage and the eye visible-point stage that pass — photons and VPs must agree on
      λ so `phi = beta ⊗ f_r(VP)` is a coherent per-λ product (a photon deposits onto many VPs).
      NOT a per-path/per-photon draw.
- [ ] 4.2 `sppmEmitPhoton` (`wavefront_sppm.slang:413`) seeds `Spectrum beta` (light SPD
      upsampled as illuminant / exact source) at the pass λ, transported per-λ through the
      photon walk.
- [ ] 4.3 **Photon-deposit BSDF recolor** (`wavefront_sppm.slang:525`,
      `mat.evaluate(...).response`): recolor per-λ via `spectral_flat_common` at the pass λ so
      `phi = beta ⊗ f_r` is coherent. (Review: previously unowned.)
- [ ] 4.4 Grow `SppmAccum` (`sppm_state.slang:79`) from `phiR/G/B` → 4 fixed-point channels
      `phi0..phi3` (one per hero λ); atomically summed at deposit; `sppmDecodeFlux` returns
      `Spectrum`. Re-measure `SPPM_FLUX_FIXED_SCALE` for per-λ range (hero-collapsed dispersion
      spikes one λ) against the numpy mirror.
- [ ] 4.5 **Resolve λ→linear sRGB/XYZ at pass end, BEFORE the progressive fold.** Keep
      `VisiblePoint.tau` (and the estimator) a spectral-**invariant** 3-wide quantity: resolve
      the per-pass per-λ `phi` at the pass λ, then run `sppmUpdate`'s progressive reduction on
      the resolved value (the `(r'/r)²` ratio is scalar geometry — unchanged). Rotating λ across
      passes integrates the spectrum. `VisiblePoint.beta`/`ld` stay per-pass spectral (resolved
      into the pass direct term). Do NOT persist per-λ `tau` across passes.
- [ ] 4.6 Update `wavefront_layout.py` SPPM accum width (4 channels) + `VisiblePoint` field
      types + host + MSL stride asserts.

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

- [ ] 9.1 `docs/Spectral.md` scope: wavefront + sppm now in the envelope.
- [ ] 9.2 `docs/Wavefront.md`: spectral carrier (`Spectrum` retype + `SampledWavelengths`),
      per-λ SPPM deposit accumulator.
- [ ] 9.3 `CLAUDE.md` + `README.md` compatibility matrices: spectral row spans wavefront and
      all three integrators.
- [ ] 9.4 `CHANGELOG.md` entry.
