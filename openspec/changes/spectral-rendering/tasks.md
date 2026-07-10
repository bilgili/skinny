# Spectral Rendering — Tasks

## 1. Spectral data groundwork (hostless)

- [x] 1.1 Port pbrt's `rgb2spec` sigmoid-coefficient optimizer to a checked-in numpy script
      under `src/skinny/pbrt/data/`; generate the sRGB coefficient table and check it in as
      compressed `.npz` (`git add -f`; repo `.gitignore` is `*`)
      — DONE: pbrt-v4 source present on disk, so vendored pbrt-EXACT table via
      `_extract_pbrt_spectra.py` (RES=64 `rgb2spec_srgb.npz`, 8.8 MB) instead of re-optimizing;
      `spectral_tables.rgb_to_sigmoid_coeffs`/`sigmoid_poly` mirror pbrt's exact lookup.
      Vendored `.npz` declared in `pyproject.toml` `[tool.setuptools.package-data]`
      (`pbrt/data/*.npz`) — verified a real `build_py` copies both into the build tree and
      they load from a non-editable install (codex stop-review fix).
- [x] 1.2 Vendor spectral tables under `src/skinny/pbrt/data/`: CIE D65 SPD (5 nm grid),
      full spectral eta/k for the pbrt named metals, Cauchy/Sellmeier IOR fits for pbrt
      named glasses
      — DONE: `spectral_curves.npz` (pbrt D65 + Ag/Al/Au/Cu eta/k resampled to 5 nm);
      glass = documented BK7 Cauchy fit in `spectral_tables.named_glass_ior`.
- [x] 1.3 Hostless validation tests: coefficient spot-checks against pbrt-published values;
      upsample→CIE-integrate round-trip error tolerance for reflectance and D65-illuminant
      modes; D65/eta-k grid sanity
      — DONE: `tests/pbrt/test_spectral_data.py` (D65-illuminant reflectance round-trip tol
      0.06 — bias is Wyman-vs-pbrt CMF convention, documented; D65 chromaticity, metal/glass
      sanity). Exact-vs-pbrt-coefficient check deferred: table IS pbrt's file verbatim.
- [x] 1.4 Add a shared CPU reference implementation (numpy) of `SampledWavelengths` (visible
      pdf, rotation, secondary termination), sigmoid evaluation, and the Wyman CMF film
      resolve — the mirror the GPU kernel tests will compare against
      — DONE: `src/skinny/pbrt/spectral.py` + `tests/pbrt/test_spectral_mirror.py` (17 tests
      total green).

## 2. Importer spectral payload preservation

- [x] 2.1 Preserve `blackbody` temperature, resampled illuminant SPDs (360–830/5 nm), named
      conductor/glass spectrum identities, and glass IOR fits on `skinnyOverrides` alongside
      the existing RGB reduction (`pbrt/spectra.py`, `materials.py`, light import paths)
      — DONE for emitters: `spectra.param_spectral_payload` + wired into all light branches
      (`lights.py` distant/point/infinite `spectral` key) and area lights (`api.py`
      `emissive_spectral` key). Named-conductor/glass MATERIAL identity preservation DEFERRED
      to Group 6 (only consumed by the GPU conductor/dispersion binding; threading it through
      `_conductor_basecolor`/`_author_material` belongs with that consumer).
- [x] 2.2 Unit tests: each payload type round-trips through import→USD; RGB-only scenes
      author no new payload (byte-identical import); existing RGB reductions unchanged
      — DONE: `tests/pbrt/test_spectral_payload.py` (8 tests); importer regression green
      (46 tests across metadata/materials/media/cloud).
- [ ] 2.3 Renderer-side scene intake: parse preserved payloads into light/material records
      (CPU structs only; no GPU consumption yet), with unit tests
      — DEFERRED into Group 3/6: the payload is only read at GPU-upload time, so intake is
      built with its consumer (renderer resources / exact-source binding) rather than as a
      standalone reader.

## 3. Session wiring and renderer resources

- [x] 3.1 `--spectral` flag + `SKINNY_SPECTRAL` env in `cli_common.py` and all front-ends;
      startup refusals (non-path integrator, explicit wavefront execution mode, ReSTIR
      reuse, neural proposal, skin/subsurface or heterogeneous-volume scene) mirroring the
      `reject_sppm_without_wavefront` pattern; hostless tests for every refusal and for
      `--spectral` + `--execution-mode auto` resolving to megakernel
      — DONE: `--spectral` in `add_render_flags` (store_true, `_env_flag("SKINNY_SPECTRAL")`,
      not persisted); `reject_spectral_unsupported(spectral, integrator, execution_mode,
      proposals, reuse)` wired after execution-mode resolution in app.py/headless.py/web_app.py;
      13 new tests in `tests/test_cli_common.py` (91 total green). SCENE-level skin/subsurface/
      volume refusal DEFERRED to renderer setup (Group 3.2/5) where the material set is known —
      the CLI validator covers only flag-level combos; documented in the function docstring.
      — FOLLOW-UP (branch spectral-cli-gate): `reject_spectral_unsupported` now also refuses an
      in-envelope `--spectral` while the transport is unwired ("not yet implemented"), so it can
      no longer silently render RGB. Single source of truth `skinny.spectral_capability.
      SPECTRAL_IMPLEMENTED` (False) now drives BOTH this CLI gate and `parity.combo_is_valid`
      (referenced live); flip it once with the Group 5 transport to enable both at once.
- [x] 3.2 Upload the sigmoid coefficient table and a spectral-asset buffer (D65, conductor
      eta/k curves, authored illuminant SPDs) as storage buffers; assign the next free
      descriptor bindings; extend the Vulkan descriptor layout AND the layout test
      (`test_vk_binding_layout.py` precedent) plus Metal bind-by-name
      — DONE (D65 + coeff table; conductor eta/k deferred to 6.2): `spectralScale`/`spectralData`/
      `spectralD65` at vk bindings 45/46/47, gated `#if defined(SKINNY_SPECTRAL)` so the RGB
      layout + SPIR-V are byte-unchanged (table res=64 / d65Count=95 are compile constants —
      no FrameConstants growth). `ComputePipeline` gained a `spectral` kwarg on both backends:
      Vulkan adds the 3 STORAGE_BUFFER layout entries + descriptor writes (pool +3); Metal
      binds by name. Buffers built + uploaded once (C-order f32) in the `renderer._spectral`
      block; `--spectral` threaded through all four front-ends → `Renderer(spectral=...)`.
      D65 uploaded UNIT-LUMINANCE-normalized (`spectral.d65_normalized()`) — see 4.1 note.
      `test_vk_binding_layout.py` unchanged (spectral bindings are invisible to its
      all-macros-off Vulkan-branch walker; recorded rationale).
- [x] 3.3 Shader-variant plumbing: when `--spectral` is active the megakernel compiles with
      `-DSKINNY_SPECTRAL` (Vulkan `main_pass_spectral.spv` checked-in vs compiled-on-demand —
      resolve the design open question and record the decision; Metal folds the define into
      the in-process SlangPy compile)
      — DONE. DECISION: compiled-on-demand on BOTH backends; NO checked-in `main_pass_spectral.spv`.
      `main_pass.spv` isn't checked in at all (compiled per-construction), and the Vulkan
      `spv_cache` key already hashes the flag tuple, so `-D SKINNY_SPECTRAL=1` gets a distinct
      cache slot for free. Vulkan: append the define to the slangc flags. Metal: add it to
      `opts.defines`. BUG FIXED (Metal): `opts.defines[k]=v` silently no-ops —
      `SlangCompilerOptions.defines` getter returns a COPY, so the define must be assembled in
      a dict and assigned ONCE. (Symptom: `pipeline.spectral=True` but spectral globals never
      reflected and the render came out byte-identical to RGB.)

## 4. Spectral shader core

- [x] 4.1 New `spectrum.slang`: `Spectrum` typealias (float3 RGB / float4 spectral),
      `SampledWavelengths` (visible-pdf sampling, rotation, secondary termination), Wyman
      CMF fit, XYZ→sRGB resolve, sigmoid-polynomial evaluation with manual trilinear
      coefficient fetch (reflectance + D65-illuminant modes)
      — DONE: `src/skinny/shaders/spectrum.slang` mirrors `pbrt/spectral.py` +
      `spectral_tables.py` bit-for-bit (baked `CIE_Y_INTEGRAL=106.9229674725` + XYZ→sRGB
      matrix to match the numpy mirror; table lookup takes the coeff/scale/D65 buffers the
      Group-3.2 descriptors will bind). Hostless slangc compile gate
      `tests/test_spectrum_compile.py` (2 tests) proves BOTH variants (`-DSKINNY_SPECTRAL`
      and RGB) typecheck to SPIR-V (48516 B each); harness
      `tests/harnesses/test_spectrum_harness.slang`.
      — AMENDED (Group 5 GPU bring-up): the illuminant path was corrected to pbrt's
      `RGBIlluminantSpectrum` — `scale·sigmoid(rgb/scale)·D65` with `scale = 2·max(rgb)` (keeps
      HDR emitters in the sigmoid gamut instead of clamping a bright light to white and losing
      its intensity) and D65 normalized to unit luminance (`spectral.d65_normalized()`, GPU
      reads the pre-normalized buffer). Fixed in BOTH `spectrum.slang::upsampleIlluminant` and
      `spectral.py::upsample_illuminant`; numpy verified white→1, [16,16,16]→16, chromatic
      preserved. (Before this a white light resolved ~49× too bright.)
- [x] 4.2 GPU≡numpy kernel tests (`tests/kernels/` harness pattern) for wavelength sampling,
      upsample evaluation, and film resolve against the task-1.4 CPU mirror
      — DONE: `tests/kernels/test_spectrum_kernels.py` (5 gpu-marked tests) compares the
      `spectrum.slang` harness wrappers to `pbrt/spectral.py` — sampleWavelengths (λ+pdf),
      terminateSecondary, upsampleReflectance, upsampleIlluminant (incl. HDR + chromatic),
      spectrumResolveToLinearSRGB. Max rel err ~1e-6 (one atol-covered ~1e-4); the visible-λ
      pdf's hard range-edge step is masked within 0.05 nm of 360/830 (fp32 vs fp64 boundary).
      The harness's upsample wrappers now take the sRGB table + D65 as explicit
      `StructuredBuffer` params (slangpy `NDBuffer.from_numpy(...).storage`); the computeMain
      compile gate stays 2/2. GPU feeds `spectral.d65_normalized()` to match the upload.
- [ ] 4.3 Byte-identity guard: test that the default (no-define) `main_pass.spv` compile is
      byte-identical to the checked-in binary after all shader edits
      — PARTIAL: reframed as "the RGB (no-define) build is unchanged by the spectral `#if`
      blocks" (no checked-in spv; slangc drift). Empirically confirmed: the RGB render of
      int_bleed produced the IDENTICAL mean (0.31306) across every A/B run before and after all
      spectral edits. A formal compile-diff test is TODO.

## 5. Megakernel spectral path

- [x] 5.1 Convert carriers to `Spectrum` under `SKINNY_SPECTRAL`: `BSDFSample`,
      `LightSample`, `BounceResult`, path `throughput`/`radiance` in
      `integrators/path.slang`; draw hero wavelength at path start from the existing RNG and
      keep it a path-lifetime local
      — DONE via a DIFFERENT (lower-risk) design than "widen the shared carriers": widening
      `BSDFSample`/`LightSample`/`BounceResult` to float4 would break every `float3` assignment
      across the flat/skin/python/bdpt/record tree and force an IMaterial signature change. So
      the spectral transport lives in a SEPARATE integrator `integrators/path_spectral.slang`
      (`SpectralPathTracer`, compiled only under `SKINNY_SPECTRAL`) that carries `Spectrum`
      throughput/radiance itself and reuses the RGB flat `sample()` for the λ-INDEPENDENT
      geometry (wi, pdf, delta-ness), recoloring per wavelength. Hero λ drawn as the first
      `rng.next()` in `estimateRadiance` (path-local); the RGB build has no such draw ⇒ its
      stream + SPIR-V are unchanged. All shared RGB structs/files are UNTOUCHED.
- [x] 5.2 Spectral evaluation in `materials/flat/*` (upsampled params/textures),
      `lights/*` (upsampled RGB emitters), `environment.slang` (upsampled env radiance,
      importance sampling unchanged), MIS scalar pdfs unchanged
      — DONE: `flat_lobes.slang::flatBsdfResponseSpectral` (per-λ mirror of `flatBsdfResponse`,
      colored reflectances passed pre-upsampled + per-λ Schlick on F0; every scalar factor
      identical to RGB so `response/pdf` reproduces the RGB sample weight). Materials, lights,
      env, and emission all upsample inside `path_spectral.slang` (reflectances via
      `upsampleReflectanceBound`, illuminants via `upsampleIlluminantBound`); NEE forms the
      `response(λ)·L(λ)` product per wavelength (`spectralAllLightsNEE`). MIS companion pdfs
      reuse the scalar RGB `mixtureProposalPdf` (λ-independent). `environment.slang` untouched —
      upsampling happens at the consumer.
- [x] 5.3 Film resolve in `main_pass.slang` before `clampSampleRadiance` + accumulation;
      BDPT/SPPM/wavefront code paths excluded from the spectral build (compile-time)
      — DONE: `SpectralPathTracer.estimateRadiance` resolves `spectrumResolveToLinearSRGB(sw,
      radiance)` → float3 internally at return, so the `IIntegrator` interface + main_pass tail
      (NaN sanitize / `clampSampleRadiance` / accumulation) stay `float3` and byte-identical.
      main_pass routes the hit case to `SpectralPathTracer` and the primary env-miss through a
      spectral upsample+resolve, all under `#if defined(SKINNY_SPECTRAL)`. BDPT/SPPM/wavefront
      are excluded by construction (separate integrator; the spectral branch never calls them).
- [x] 5.4 GPU smoke test: flat corpus scene renders under `--spectral` megakernel on Metal;
      furnace scene closes within the existing uniformity gate
      — DONE: `int_bleed` (flat Cornell box) renders correctly under `--spectral` megakernel on
      native Metal (M5 Pro) — energy-conserving (spectral mean 0.31343 vs RGB 0.31306) with
      plausible metameric red/green color-bleed (relMSE 0.0195); labelled side-by-side captured.
      `furnace_lambert` (diffuseColor (1,1,1)) now CLOSES under `--spectral`: mean luminance
      0.8797 vs RGB 0.8789 (both to the furnace constant 0.879), near-neutral — enabled by the
      sigmoid-endpoint fix (uniform RGB r∈{0,1} was reflecting 0.5, so white furnace never
      closed; now rides pbrt's IEEE-inf limit → reflectance {1,0}). TODO (harness wiring):
      register the spectral furnace disposition in the confirming suite's meta-test.
- [ ] 5.5 Backend A/B: one corpus scene rendered spectrally on Vulkan and native Metal
      agrees within the recorded backend-parity tolerance (render the labelled side-by-side)

## 6. Exact spectral sources and dispersion

- [~] 6.1 Blackbody: Planck evaluation at sampled wavelengths from the preserved
      temperature; GPU≡numpy test + chromaticity-moves-toward-pbrt check on a
      blackbody-lit scene
      — NUMPY PREP DONE (0b2219f): `spectral.blackbody_emission(sw,T)` (raw Planck at hero λ) +
      `blackbody_scale(T, emission_rgb)` = Y_target·_Y_INTEGRAL/Y_planck (6 tests, MC round-trip
      luminance+chromaticity within 2%, hotter→bluer). GPU CONSUMER DEFERRED — DISCOVERY: pbrt
      blackbody emitters import as AREA LIGHTS → emissive TRIANGLES (light buffer), NOT flat-
      material emission (the emissive quad's material has emissiveColor 0). So the exact-Planck
      hook belongs on the emissive-triangle emission path (spectralAllLightsNEE's ls.radiance
      upsample + the direct light-hit emission), not the flat-material `emissive_spectral`
      consumer I first built (reverted — it had no producer). Reusable GPU machinery drafted
      (planckSpectrum + a per-emitter spectral buffer) but must be re-hooked to emissive
      triangles; that needs per-emissive-triangle blackbody metadata + the light-side consumer.
- [x] 6.2 Spectral conductor Fresnel from the eta/k asset buffer for named metals; test vs
      CPU mirror
      — DONE: importer preserves the metal identity on `skinnyOverrides["conductor_metal"]`
      (au/ag/al/cu; `materials.material_spectral_overrides` + `api._author_material[_mtlx]`).
      GPU: `spectralMetals` StorageBuffer (vk binding 48, spectral-only) with au/ag/al/cu eta/k
      on the 5nm grid; `namedMetalEtaK` samples eta(λ)/k(λ) at the 4 hero λ; `fresnelConductor`
      (pbrt FrComplex, float2 complex) in flat_shading.slang; `flatBsdfResponseSpectral`'s spec
      lobe uses it when the hit is a named conductor (id in `FlatMaterialParams.conductorMetalId`
      = spare `_specularColorPad.w`, 0 = RGB Schlick). numpy mirror `spectral.fresnel_conductor`
      + `named_metal_eta_k` (gold R(580)=0.899, R(650)=0.940, ≡ fresnel_conductor_rgb at normal
      incidence). VALIDATED: conductor_infinite.pbrt (gold) renders gold under --spectral, exact
      Au Fresnel warmer/more-saturated than RGB (R/G 1.177 vs 1.125). GPU≡numpy conductor-Fresnel
      harness test = follow-up (4.2 harness could add it).
- [ ] 6.3 Authored illuminant SPD lookup from the spectral-asset buffer on lights that
      preserved one; test that a non-constant `spectrum L` renders from the SPD, not the RGB
      upsample
- [~] 6.4 Dispersion: wavelength-dependent IOR at hero λ for preserved glass fits;
      secondary termination with pdf adjustment on first dispersive refraction; unbiasedness
      test (constant-IOR dielectric keeps all 4 samples) and a visible prism-dispersion
      render
      — IMPORTER PREP DONE: `skinnyOverrides["glass_dispersion"]` = bk7/default preserved for a
      named-glass eta (`materials.material_spectral_overrides` + a pre-existing scalar("eta")
      crash fix). GPU CONSUMER TODO: the delta-glass branch in path_spectral.slang (:283-287) +
      flat_material.slang (:37-65) must re-evaluate `eta(λ)` (Cauchy `named_glass_ior`, upload a
      glass-fit buffer or a Cauchy A/B pair in FlatMaterialParams) per hero λ + `terminateSecondary`
      on first dispersive refraction (λ-dependent direction).
      — GPU DESIGN READY (in memory): store Cauchy B in the spare `_normalBiasPad.w` (offset 124,
      `glassCauchyB` property), A = the existing `ior` lane — NO new buffer. Dispersion block in
      path_spectral.slang after `if (!bs.valid) break;`: gate `bs.pdf==0 && bs.transmitted &&
      glassCauchyB>0`, recompute refraction at `n(λ0)=ior+B/(λ0_um²)`, override `bs.wi`, `sw =
      terminateSecondary(sw)`. Constant-IOR (B=0) never terminates ⇒ unbiased. packer:
      `glass_dispersion`→`named_glass_cauchy`→ ior=A, B in pad. Glass IS flat (opacity<1) so flows
      through path_spectral cleanly — end-to-end testable (unlike 6.1). Needs a spectral-specific refraction
      path (the RGB `bs.wi` is λ-independent).
- [ ] 6.5 Build the spectral-discriminating confirming-suite scene(s) (dispersive dielectric
      and/or blackbody-lit) via `tests/assets/suite/_gen/`, regen pbrt refs
      (`regen_refs.py`), register dispositions

## 7. Parity matrix integration

- [x] 7.1 Add the `spectral` axis to `parity.py` (`combo_is_valid` rules per the delta
      spec); machine-readable skip reasons for BDPT/SPPM/wavefront/proposal/reuse/volume/
      skin × spectral, including the mode-equivalence skip "spectral is megakernel-only"
      — DONE: `RenderCombo.spectral` field + label tag; `spectral_envelope` helper (the v1
      path+megakernel+flat rules) + `combo_is_valid` spectral block. GATED behind
      `SPECTRAL_IMPLEMENTED=False` (capability flag): the megakernel transport is unwired, so
      an in-envelope spectral combo is a recorded "not yet wired" SKIP and is ABSENT from
      `enumerate_combos` (the rendered set) — it is never rendered as RGB and gated as if
      spectral (codex stop-review fix). Flip `SPECTRAL_IMPLEMENTED=True` with Group 5 to admit
      it; the delta-spec "(path, megakernel, spectral) present" scenario holds from that point.
      Mode-equivalence skip is implicit (only megakernel spectral is envelope-eligible, no
      wavefront pair). Render pass-through of the flag is Group 7.3 (GPU).
- [x] 7.2 Extend the coverage meta-tests: integrator × spectral validity completeness;
      suite spectral-discriminating disposition presence (hostless)
      — DONE (validity completeness): `test_coverage_meta_spectral_axis_covered` + 8 spectral
      validity tests in `tests/pbrt/test_matrix.py` (23 pass, 141 hostless total green). Suite
      spectral-discriminating disposition coverage DEFERRED to Group 6.5 (the discriminating
      scene must exist first); the meta-test will assert its disposition once 6.5 lands.
- [ ] 7.3 GPU sweep: run `(Path, megakernel, spectral)` across the corpus; record spectral
      pbrt-truth measurements; assert spectral ≤ RGB on spectrum-authored scenes; report
      (not assert) spectral-vs-RGB anchor deltas
- [ ] 7.4 Verify no RGB baseline changes anywhere (byte-identical default build ⇒ recorded
      measurements stand)

## 8. Metal hygiene and performance

- [ ] 8.1 Run the Metal kill harness (hostless 13 + gpu-marked under the guarded runner)
      with spectral mode exercised; confirm megakernel row-band tiling stays within
      watchdog budget with the heavier spectral kernel
- [ ] 8.2 Equal-time throughput measurement spectral-vs-RGB megakernel on a corpus scene;
      record the overhead factor in the change notes

## 9. Documentation

- [x] 9.1 `docs/Architecture.md`: new bindings in the descriptor map, spectrum module in
      the module map, spectral-variant compile note
      — DONE: bindings 45/46/47 (spectralScale/Data/D65, spectral-build-only) added to the
      Descriptor Binding Map with the byte-unchanged-RGB-layout note; `spectrum.slang` +
      `integrators/path_spectral.slang` in the shader module map (+ `flatBsdfResponseSpectral`);
      a "spectral compile variant" paragraph (-DSKINNY_SPECTRAL both backends, gated `Spectrum`
      typealias in common.slang, spv_cache flag hashing).
- [x] 9.2 README + CLAUDE.md compatibility matrices: `--spectral` flag, scope guards
      (path + megakernel only, flat-only, no volumes/skin/reuse/neural; wavefront =
      designated follow-up), backend support; CHANGELOG entry
      — DONE: README compatibility matrix (spectral row ⏳ WIP + a `--spectral` scope
      subsection), matching CLAUDE.md matrix row + WIP constraint block (kept in sync per the
      "keep the two in sync" note), and a CHANGELOG `[Unreleased] Added` entry. All honestly
      flag the transport as unwired / `--spectral` refused until `SPECTRAL_IMPLEMENTED` flips.
- [ ] 9.3 New `docs/` section (or doc) for spectral rendering: hero-wavelength estimator,
      upsampling model, film resolve — equations as LaTeX-rendered SVG per repo convention;
      run `node docs/diagrams/embed_code.cjs --check` if marked shader regions were touched
- [~] 9.4 `docs/PythonAPI.md` if any public Python symbol was added; `ruff check src/` and
      full hostless pytest sweep green
      — DONE (docs): `spectral.d65_normalized()` + `upsample_illuminant` RGBIlluminantSpectrum
      semantics added to `docs/PythonAPI.md` §1.1; `ruff check src/` green. Full hostless sweep
      re-run pending the remaining Group-6 work.
