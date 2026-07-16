# Changelog

All notable changes to Skinny are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **SPPM environment photon emission — env-INDIRECT transport** (change
  `sppm-env-indirect-transport`). The environment light is now a fourth photon
  group in `sppmEmitPhoton` (present iff `furnaceMode == 0 && envIntensity > 0`):
  a sky direction is importance-sampled with `sampleEnvDir` (the same
  distribution env NEE uses), emitted inward from the scene-bounding disc, with
  pbrt `ImageInfiniteLight::SampleLe` flux `beta = L_env·πR²/(gsel·p_dir)` and a
  pole-pdf validity guard. Previously SPPM had **no** env-indirect transport —
  env light reached the film only via the eye stage's direct terms — leaving
  env-lit scenes dim (recorded 0.78 shadow-box vs the path anchor on the fair
  null-sun glass-caustics scene). Post-change probe: totals vs path ≈ 0.99–1.05
  on the same regions (constant-env closure exactly 1.00; residual excess in
  glass-shadow regions is env-through-glass SDS caustic transport only photons
  can carry). Spectral branch upsamples with `upsampleIlluminantBound` at the
  shared per-pass wavelengths. Env DIRECT stays owned by the eye stage; photons
  deposit only at `depth ≥ 1`, so the partition is disjoint. The prior
  env-photon attempt's "8× over-bright" was diagnosed as probe methodology
  (forced selection kept `gsel = 1/G`; mean-not-median; indirect-vs-total
  mismatch), not a flux-formula error.

### Fixed

- **SPPM photon-indirect term now scaled by the eye throughput `vp.beta` at
  resolve** (change `sppm-vp-beta-resolve`). `wfSppmUpdate` resolved the
  per-pass photon flux as `Φ/(N_emitted·π·r²)` without the camera→visible-point
  throughput the eye stage stores (`VisiblePoint.beta`, previously write-only) —
  the only radiance component that skipped it (`ld` already folds `throughput`
  per add). pbrt-v4 folds `vp.beta` into the flux at pass end. Invisible for
  directly-viewed visible points and clear Fresnel-sampled delta glass
  (weight ≡ 1; bit-identical A/B on `glass_caustics_test.usda`), but the photon
  term through a *tinted/lossy* eye chain was over-bright and un-tinted:
  through-tinted-glass region mean 0.00102 → 0.00067 vs BDPT reference 0.00074.
  Applied per-λ before the spectral λ→sRGB resolve; the radius/N advance is
  flux-independent and unchanged.

- **Parity gates `disney_cloud` and `subsurface_infinite` repaired — scene-data
  integrity, not renderer regressions** (change `parity-scene-asset-integrity`).
  `disney_cloud` (relMSE 0.075 → 0.584) was a *deleted untracked side-file*: the
  usda's DomeLight references the baked constant blue sky
  `assets/light_infinite_f620_const.hdr`, whose silent-fallback loss swapped the
  illuminant for the default gray env (src at the Jul-3 baseline and current
  main render bit-identically). The baked env (and the outright-deleted
  `bunny_cloud.usda`) are restored; the 173-byte const `.hdr` is now
  git-tracked (the usda stays untracked — it bakes a machine-absolute `.nvdb`
  path); `usd_loader` warns loudly when a DomeLight's
  `texture:file` is missing instead of silently falling back.
  `subsurface_infinite` died on a spectral combo the scene can't render: the
  manifest entry predated `material_class`, defaulted to `flat`, and the
  spectral envelope admitted combos the renderer refuses at scene build
  (`SystemExit`). It now declares `material_class: "subsurface"` and records
  the by-design `path|megakernel`-vs-anchor delta (wavefront true 3D interior
  walk vs megakernel watchdog-safe 1D slab, change `pbrt-subsurface-3d-walk`)
  as a measured, tighten-only `self_consistency` mode override (relMSE
  0.0362 / FLIP 0.0554 at 512 spp, tol 0.05/0.07). Two hostless integrity
  meta-tests (`tests/pbrt/test_matrix.py`) now catch both failure classes
  without a GPU: a `texture:file` dangling-reference sweep over on-disk
  manifest usd assets (plus `OpenVDBAsset` ⇒ `material_class: "volume"`), and
  a non-flat `.pbrt`-source vs declared-`material_class` cross-check.

### Changed

- **BREAKING: the built-in default DistantLight is no longer injected into
  scenes that author their own lights** (change `distant-light-caustic-parity`).
  A USD scene with any powered light (DistantLight, SphereLight, emissive-
  material mesh, or an authored DomeLight) now renders with exactly its
  authored lights on every front-end; only a truly light-less scene (or the
  default no-USD session) keeps the slider-driven synth sun. The phantom sun
  changed authored scenes' lighting, and its glass caustic was renderable only
  by SPPM (a delta light through delta glass is unsampleable by the path
  tracer, and BDPT skipped the distant light walk) — the source of the
  long-standing "SPPM caustics don't match bdpt" speckle, which is gone at the
  default SPPM radius with the phantom suppressed.

### Added

- **BDPT walks light subpaths from distant lights** (change
  `distant-light-caustic-parity`). The distant light-origin sample is now a
  real emission ray from a scene-bounds-covering disk (mirroring the SPPM
  photon emitter and pbrt's `DistantLight::Sample_Le`), so distant-light
  specular caustics — which unidirectional path tracing cannot sample — are
  carried by the s = 1 camera splat and the s ≥ 2 connections, agreeing with
  the SPPM photon estimate (Gate B: bdpt/sppm full 1.0095, mega ≡ wave
  1.0000). The eye-side distant NEE joined the `misWeight` partition (it was
  an unconditional full-weight term — a guaranteed 2× double-count once the
  walk exists), the first walked vertex uses the parallel-projection area
  density (no cos/d² — the disk distance is a placement artifact), and
  distant **direct** stays owned by NEE (the t = 2 splat is skipped; the
  repo's `z1.pdfFwd = 1` camera convention cannot partition it at grazing
  camera pdfs). RGB + spectral (per-λ SPD recolor), megakernel + wavefront.

### Fixed

- **SPPM env-lit scenes no longer render ~15–25% dim vs path/bdpt** (change
  `sppm-caustic-dimness`). The wavefront SPPM eye terminates at the first diffuse
  visible point (photons carry the indirect) and computed that vertex's direct via
  the MIS-weighted `allLightsNEE`; its env NEE expected a BSDF-sampled env-miss
  companion that a continuing path would add next bounce, but SPPM never fires it,
  so env NEE was down-weighted with no counterpart → env **direct** under-counted
  under a broad environment. The eye now adds that env-miss companion at the
  terminal visible point (one BSDF sample, MIS-weighted), so env direct = NEE +
  BSDF-miss exactly as the path tracer. GPU-verified: env-only flat ground
  0.735→0.998; `glass_caustics_test` all regions 0.75–0.87 → 0.95–1.04. Small
  analytic lights unaffected.


- **SPPM no longer wedges the Metal GPU on caustic scenes, without starving
  photons** (change `sppm-photon-dispatch-tiling`). The wavefront phase-3 photon
  pass (`wfSppmPhotonTrace`) was committed as one command buffer whose work is
  `photons × visible-points-in-cell` — unbounded where visible points cluster in
  a caustic focus cell, tripping the macOS GPU watchdog. The prior cap on
  photons-per-pass avoided the wedge but **starved** the estimator (dark bias).
  The photon dispatch is now **tiled by breadth** into flushed sub-batches of
  `SKINNY_SPPM_METAL_PHOTON_BATCH` photons (default 65536, 64-aligned), like the
  megakernel row bands, so the full `width×height` photon budget renders under the
  watchdog. Unbiased: additive fixed-point deposits + a single pre-loop
  `clear_accum` make the tiled pass bit-identical to one dispatch (GPU-verified
  |diff| 0.0000). `SKINNY_SPPM_METAL_PHOTON_CAP` default `262144→0` (unlimited;
  kept as an optional ceiling). Also repairs 5 SPPM `test_wavefront_driver.py`
  tests left red by the earlier phase-boundary flush commit.

### Added

- **Spectral rendering in the wavefront mode** (change `spectral-wavefront`) —
  `--spectral --execution-mode wavefront` now carries hero-wavelength transport
  through all three staged wavefront integrators — **path, BDPT, and SPPM** —
  widening the envelope beyond megakernel-only. Everything is gated behind
  `#if defined(SKINNY_SPECTRAL)`: the wavefront records
  (`WavefrontPathState`/`WfBdptAux`/`BDPTVertex`/`VisiblePoint`/`SppmAccum`) carry
  a `Spectrum` bundle + `SampledWavelengths` under the define, so the RGB build is
  **byte-identical** across all 28 wavefront kernels + the megakernel; the host
  allocators size each buffer by the spectral stride
  (`path_state_size`/`wf_bdpt_aux_size`/`sppm_buffer_sizes`). Per-λ transport
  mirrors the megakernel spectral integrators (`spectralAllLightsNEE`, per-λ
  emission/Planck, `spectrumResolveToLinearSRGB` at the film; BDPT reuses the
  scalar `misWeight` via the colour-free projection). **SPPM (design D5):** one
  shared hero-wavelength set per pass (photons and eye visible points agree),
  per-pass φ resolved λ→linear-sRGB before the progressive fold, `VisiblePoint.tau`
  a spectral-invariant 3-wide quantity; **v1 limit — no dispersion in the SPPM
  photon/eye carriers** (path + BDPT do carry hero-λ Cauchy dispersion). The
  startup gate (`reject_spectral_unsupported`) and parity matrix admit
  `(path|bdpt|sppm, wavefront)`. **Status: wired + CPU-verified (179+ hostless
  tests, codex pre-merge review clean after a host-stride fix) + merged; the
  GPU-render gates (self-consistency / prism-BDPT / white-furnace) and the
  `SPPM_FLUX_FIXED_SCALE` numpy re-measure are a pending interactive-Metal
  follow-up — not yet render-validated.** See [Spectral.md](docs/Spectral.md) and
  [Wavefront.md](docs/Wavefront.md).

- **Spectral BDPT in the megakernel** (change `spectral-bdpt-megakernel`) —
  `--spectral --integrator bdpt` now renders on both backends, widening the
  hero-wavelength envelope from path-only to **path + BDPT** (megakernel, flat
  materials). A separate `integrators/bdpt_spectral.slang` (`SpectralBDPTIntegrator`,
  compiled only under `-DSKINNY_SPECTRAL`) carries `Spectrum` throughput/emission
  and transports all five strategy families spectrally — eye/light random walks,
  s≥2/t=0 emissive hits, t=1 NEE, t≥2 connections, and the s=1 camera splat —
  while **reusing the RGB `bdpt.slang` MIS chain verbatim** (a colour-free vertex
  projection feeds `misWeight`/`splatMisWeight`/`convertSAtoArea`, so spectral and
  RGB BDPT can never disagree on weighting). Wavelengths are drawn once per pixel
  path and shared across both subpaths; hero-λ glass dispersion collapses on either
  walk; the light-tracer splat resolves λ→linear-sRGB before the atomic add. The
  per-λ flat NEE machinery is hoisted into `integrators/spectral_flat_common.slang`,
  shared with the path integrator (spectral path output byte-unchanged). SPPM stays
  excluded (no megakernel path — spectral SPPM awaits spectral wavefront). Startup
  gate (`reject_spectral_unsupported`) and the parity matrix envelope
  (`spectral_envelope`) admit `(bdpt, megakernel)`; the renderer integrator pin
  (`_active_integrator_index`) dispatches path/bdpt live and pins sppm→path. The
  RGB `main_pass.spv` is byte-identical. See [Spectral.md](docs/Spectral.md).

### Changed

- **Execution mode follows the integrator** (change
  `integrator-default-execution-mode`) — `--execution-mode` gains an `auto`
  value, now the default (env `SKINNY_EXECUTION_MODE`, mirroring `--backend
  auto`). `auto` derives the mode from the startup integrator — `path`/`bdpt`
  → `megakernel`, `sppm` → `wavefront` — so `--integrator sppm` alone runs
  under wavefront instead of erroring, and a persisted `sppm` integrator no
  longer relaunches into the "SPPM requires wavefront" error. An explicit
  `--execution-mode megakernel`/`wavefront` (flag or env) still overrides the
  derived default and pins the mode for the session; the only impossible combo,
  `sppm` + explicit `megakernel`, is still refused at startup. Resolution is
  shared across all four front-ends (`skinny`, `skinny-gui`, `skinny-web`,
  `skinny-render`) via `cli_common.resolve_execution_mode` /
  `startup_integrator_name`. The mode remains fixed for the session (not
  runtime-switchable or persisted); no shader, binding, or accumulation-hash
  change.

### Fixed

- **SPPM renders a polished metal under an environment** (change
  `sppm-glossy-env-escape-mis`) — the `sppm|wavefront` parity gate on
  `conductor_infinite` (a pbrt-roughness-`0.1` gold sphere lit only by a constant
  infinite light) failed at relMSE ≈ 0.45 vs the path anchor. Two coupled eye-walk
  defects: (1) the glossy-continuation threshold `sppmGlossyContinueRoughness`
  (perceptual/USD roughness) did not reach pbrt-imported polished metals — pbrt
  roughness `r` imports as `usd = r**0.25`, so a polished `0.1` conductor lands at
  usd `≈ 0.562`, just above the old `0.5` default, and was stored as a
  photon-gather visible point that never receives a deposit (env photons hit the
  sole metal at depth 0 and escape); (2) a non-delta glossy-continued vertex that
  escapes to a distant/env light added the env at full weight while also running
  env NEE at that vertex, double-counting the environment. The default is raised to
  `0.6` (an alpha `≲ 0.36` polished-metal cutoff that still leaves a
  pbrt-roughness-`0.3` metal on the gather side), and the eye walk now
  MIS-weights every light a glossy carrier reaches — the escaped env by
  `powerHeuristic(bsdfPdf, envPdf(dir))` and an emissive-triangle hit by
  `powerHeuristic(bsdfPdf, pdfLightSA)` (`spawnedBySpecular` is now delta-only,
  correcting a pre-existing full-weight double-count the higher threshold would
  expose) — mirroring the path tracer's env-miss and emissive-hit MIS. Delta
  carriers / transmitted / furnace keep full weight. SPPM now converges to the
  path reference; no binding, ABI, or photon-stage change. `mat_conductor` SPPM
  baselines are unchanged.
- **MaterialX imagemap `base_color` textures now render** (change
  `confirming-test-scenes` follow-up) — a `standard_surface` whose `base_color`
  is driven by an `<image>` node (what `import_pbrt --mtlx` emits for a pbrt
  `"texture reflectance"` imagemap) rendered as a flat colour instead of the
  texture. Two independent defects, one per intake path:
  1. **Graph-param offset skew** (`materialx_runtime.generate_for_compute`) —
     when the emitted `GraphParams_*` struct dropped an unused leading gen
     uniform, `pack_uniform_block` kept the gen's original offsets (with the
     hole) while Slang's `Load<GraphParams_*>` reads the emitted struct
     contiguously. Every field skewed by the gap; an `<image>` graph misread
     `uv_scale` as `(0,1)`, collapsing the U coordinate so the texture sampled a
     single column (a flat smear). The kept uniforms' offsets are now
     re-compacted dense-from-0 to match the struct Slang emits. Marble/wood
     graphs were unaffected (their kept uniforms already started at 0). This is
     the path `import_pbrt --mtlx` scenes take (via the `_load_mtlx_materials`
     fallback → graph render). Regression: `test_graph_uniform_offsets_are_dense`
     in `tests/test_materialx_graph.py`.
  2. **Texture-key not remapped** (`usd_loader._extract_material`) — on the
     usdMtlx-plugin-present path a composed `standard_surface` names its
     texture-bound input `base_color`, but the renderer's flat texture binder
     only looks up UsdPreviewSurface keys (`diffuseColor`/…). The texture was
     stored under `base_color` and dropped, so the input fell back to the
     `standard_surface` default grey. `_extract_material` now folds the input
     name through `_OPENPBR_TO_STD_SURFACE`/`_STD_SURFACE_TO_FLAT` before storing
     the texture (mirroring `_store_shader_override` for constants and the
     `_load_mtlx_materials` fallback). Regressions in
     `tests/pbrt/test_mtlx_roundtrip.py`. Both paths now match the plain
     UsdPreviewSurface authoring (A/B relMSE 2.3/2.1 → ~0.00 on Metal). The suite
  scene `mat_textured_mtlx` (`tests/pbrt/test_suite.py`) is now un-`xfail`ed
  (`known_divergent: false`) — its strict pbrt-truth + authoring-equivalence
  gates pass on every integrator. (Also repaired the two `mat_textured` assets,
  whose imagemap `file` was an absolute path into the now-removed
  `confirming-test-scenes` worktree — the dead path broke BOTH the plain and
  mtlx renders; both now use a layer-relative `texture_uv.png`.)

- **SPPM photon term restored — bathroom walls no longer render black**
  (change `fix-sppm-bathroom-black-walls`) — the SPPM photon deposit rebuilt
  its `FlatMaterial` from `VisiblePoint` fields that predated the Stage-2 rich
  inputs (`transmissionColor` / `specularColor` / `diffuseRoughness`, added to
  `FlatHitMat` by `flat-lobes-rich-inputs` after PM-1 shipped), feeding
  undefined values into `evaluate()` at every deposit: deposited flux was
  exactly zero scene-wide (`τ == 0` at 100% of visible points on bathroom AND
  cornell_box_sphere) while the photon count kept shrinking the search radius,
  so SPPM rendered its eye-pass direct term only and indirect-lit surfaces
  (bathroom walls/ceiling/tub) went black. The `VisiblePoint` now stores the
  three fields (scalar 152→180 B, MSL 192→240 B; `wavefront_layout` mirror in
  lockstep) and `sppmLoadMaterial` rebuilds them, so the deposit evaluates the
  exact eye-pass BSDF. New hostless parse-locks in `tests/test_sppm_state.py`
  fail the build if `FlatHitMat` ever grows a field without a VP slot + store +
  rebuild. The re-armed cornell energy gate passes on both backends (its
  `xfail(strict)` marker is removed); bathroom `sppm_vs_path` measured relMSE
  64.97 → 2.59 (MSE 10.42 → 0.297, linear-mean ratio 1.007 at 128 spp).
- **`skinny-render --execution-mode wavefront` no longer fails on every
  invocation** (change `headless-wavefront-readiness-gate`) —
  `HeadlessRenderer.render_to_array`/`render_scene` gated readiness on
  `renderer.pipeline is None`, but wavefront intentionally never builds the
  megakernel pipeline (`scene_bindings_only`), so every wavefront headless
  render was rejected with the misleading
  `render pipeline failed to build — scene has no usable materials` even though
  the scene built fine. Both entry points now gate on
  `renderer._backend_render_ready`, the backend- and execution-mode-aware
  readiness signal the interactive front-ends already use; a genuinely unready
  scene still raises the same error. Hostless regression tests in
  `tests/test_headless_api.py` (`TestHeadlessReadinessGate`).

### Added

- **Spectral rendering** (`--spectral`, change `spectral-rendering`) — opt-in
  hero-wavelength transport instead of RGB, **v1 live**
  (`SPECTRAL_IMPLEMENTED = True`). The megakernel spectral integrator
  (`integrators/path_spectral.slang`) renders the **path + megakernel + flat**
  envelope on Vulkan and native Metal: per-wavelength NEE, pbrt's exact
  sRGB→spectrum sigmoid upsampling + CIE D65 illuminant model, exact
  named-conductor complex-index Fresnel, authored `spectrum L` illuminant SPDs +
  blackbody Planck emission (area lights and distant lights), and hero-λ Cauchy
  glass dispersion, all resolved through the Wyman CMF to the existing RGBA32F
  accumulation (exposure/tonemap/readback unchanged). Vendors pbrt-v4's exact
  upsampling table + CIE/eta-k curves, a numpy CPU estimator mirror
  (`skinny.pbrt.spectral`), pbrt-import payload preservation (`skinnyOverrides`),
  the `--spectral` CLI flag (`SKINNY_SPECTRAL`), and a parity-matrix spectral
  axis. An in-envelope `--spectral` run is accepted on every front-end;
  out-of-envelope combos (non-path integrator, wavefront, ReSTIR reuse, neural
  proposal, skin/subsurface/volume scene) are refused at startup. The RGB
  (no-`--spectral`) build is byte-unchanged — every spectral binding/kernel is
  compiled only under `-DSKINNY_SPECTRAL`. Wavefront spectral, BDPT/SPPM, and
  volume/skin spectral are designated follow-ups.

- **pbrt procedural `cloud` medium** (change `pbrt-cloud-procedural-medium`) —
  `MakeNamedMedium "cloud"` (pbrt's built-in analytic cloud, `clouds.pbrt`) now
  imports and renders: a new `MEDIUM_CLOUD` kind fills the existing
  `densityAt`/`mediumMajorant` seam with pbrt's exact `CloudMedium::Density`
  (256-entry `NoisePerm` classic Perlin, 5-octave fBm, 2-iteration wispiness
  domain warp, altitude falloff — new `materials/subsurface/cloud_noise.slang`,
  validated against a numpy mirror of the identical constants on the GPU). No
  texture, no new binding: density is analytic in medium-local `[0,1]³` (zero
  outside — pbrt's bounds clip), the packed σ_t is the exact global majorant,
  and `FlatMaterialParams` grows 240→256 B (one appended float4:
  density/wispiness/frequency). `Material ""` on a `MediumInterface` shape now
  routes to the interface null boundary (per the living `pbrt-volume-import`
  spec) instead of grey diffuse. Corpus gains `clouds` with a pbrt v4 reference:
  dual gate CLEAN — megakernel ≡ wavefront EXACT (relMSE 0.0), pbrt-truth
  relMSE 0.094 / FLIP 0.091 / mean ratio 1.012 (constant-σ spectrum tint +
  RGB-vs-spectral + scatter-cap floor, recorded); zero-σ cloud ≡ no-sphere at
  MC-noise level (relMSE 7e-7). Note: pbrt `density 0` is NOT an empty medium
  (the altitude floor term survives) — the port matches pbrt exactly. The
  segment clip (`clipSegmentToGrid`, `volume_walk.slang`) now bounds
  `MEDIUM_CLOUD` to the same `[0,1]³` support box as the grid kind, so an
  escape/shadow ray toward the open sky no longer spends the whole
  null-collision budget marching vacuum past the cube (result unchanged,
  faster + watchdog-safe). `Material ""` is treated as a null boundary for a
  shape with *either* an inside or an outside `MediumInterface` (cavity
  boundaries too), matching pbrt's empty-material semantics.

- **NanoVDB heterogeneous volume rendering** (change `nanovdb-volume-rendering`) —
  pbrt `MakeNamedMedium "nanovdb"` + `Material "interface"` scenes (the WDAS
  `disney-cloud` and `bunny-cloud`) import to `.usda` as a `UsdVol.Volume` with an
  `OpenVDBAsset` field and render as free-standing heterogeneous participating
  media on both backends and both execution modes (path integrator). A pure-Python
  `.nvdb` reader (`pbrt/nanovdb.py`, FloatGrid/FogVolume, NONE/ZIP codecs — no new
  native dependency) decodes the density grid to one R16F `Texture3D` (binding 26);
  the `MEDIUM_NANOVDB` kind slots into the existing `densityAt`/`mediumMajorant`
  seam (`materials/subsurface/{medium,volume_walk}.slang`) as majorant/null-
  collision (Woodcock) delta-tracking with an index-matched pass-through boundary,
  HG phase, distant + environment NEE (delta-tracked shadow transmittance), and an
  escape-ray continuation so geometry behind the volume still shades. BDPT and SPPM
  have no volume transport (recorded parity-matrix exclusions). On Metal the
  bindless flat-material texture pool is trimmed 120→119 to fit the 3D grid under
  Apple's 128-texture compute-argument limit. Parity: `disney_cloud` clean dual
  gate (megakernel ≡ wavefront, relMSE 0.077 vs pbrt); `bunny_cloud` relMSE 0.117
  (megakernel) with megakernel ≡ wavefront recorded known-divergent (pre-existing
  HDR-env execution-mode brightness difference + env-NEE firefly variance; follow-up
  `nanovdb-volume-wavefront-parity`).
- **Metal dispatch hygiene** (change `metal-dispatch-hygiene`) — `MetalContext`
  guarantees GPU teardown on every process-exit path (idempotent `destroy()`,
  context-manager, one weakref-registry `atexit` + chained SIGINT/SIGTERM handler
  set) so abandoned compute work can no longer wedge the GPU until reboot; a
  gpu-marked kill harness (`tests/test_metal_cleanup.py`) proves a SIGKILLed render
  leaves the device usable.

- **Integrator × execution-mode parity matrix** (change `integrator-parity-matrix`)
  — a data-driven regression harness that sweeps `{Path, BDPT, SPPM}` ×
  `{megakernel, wavefront}` plus the **ReSTIR DI** and **neural directional
  proposal** axes, pruned by one validity table that mirrors the compatibility
  matrix (SPPM is wavefront-only; the neural proposal is wavefront + path +
  flat-material only; ReSTIR DI is path + wavefront). Each valid combo is gated
  two ways: **pbrt-truth** (relMSE/FLIP vs the checked-in pbrt v4 reference EXR,
  honouring per-combo recorded baselines for known mismatches) and
  **self-consistency** (every combo must match the `(Path, wavefront)` anchor
  image within a per-axis tolerance, so `megakernel ≡ wavefront` and a one-combo
  divergence are both caught). `bathroom.usda` (pbrt `contemporary-bathroom`) and
  the SSS dragon join the corpus as heavy reference scenes; a coverage meta-test
  fails if a renderer combo the app exposes has no matrix entry. `tests/pbrt/
  regen_refs.py` regenerates the reference EXRs offline from the pinned pbrt v4.
- **Standardized image-metric battery** (`skinny.pbrt.metrics.ImageMetrics` +
  `compute_metrics`) — one canonical entry point for every reported number: error
  vs a reference (MSE, RMSE, MAE, relMSE, PSNR, FLIP) plus single-image quality
  stats (variance, an Immerkær noise-σ estimate, and a firefly outlier fraction),
  all pure-numpy. The parity gates report and log the full battery.

### Fixed

- **Every Vulkan pipeline failed to build on MoltenVK** (change
  `fix-vulkan-volume-density-binding`) — since `nanovdb-volume-rendering` added
  the heterogeneous-medium density field `volumeDensity` (`[[vk::binding(26)]]
  Sampler3D<float>`), the megakernel SPIR-V referenced set-0 binding 26
  unconditionally, but the binding was never added to the Vulkan descriptor-set
  layout (`ComputePipeline._create_descriptor_set_layout`). The shader
  declaration, the descriptor write, and the pool sizing all landed; only the
  layout entry was missing. On MoltenVK — which derives its SPIR-V→MSL resource
  map from the pipeline layout — the undeclared binding made SPIRV-Cross return a
  null mapping and every `VulkanContext` render died at pipeline create with
  `SPIR-V to MSL conversion error: nullptr`
  (`VUID-VkComputePipelineCreateInfo-layout-07988`). Declaring binding 26 as a
  combined image sampler in the shared set-0 layout fixes the megakernel and all
  wavefront stage pipelines at once (no shader/Metal/stride change). A hostless
  audit (`tests/test_vk_binding_layout.py`) now asserts every Vulkan-branch
  `[[vk::binding(N)]]` in `bindings.slang` has a layout entry, so a new shared
  binding cannot repeat this. This un-silences the raw-Vulkan SPPM GPU gate
  (`tests/test_sppm_gpu.py`), which had failed to build for months;
  `test_sppm_builds_and_renders_finite` passes again, and the re-armed energy
  gate immediately caught a genuine cross-backend SPPM diffuse-indirect deficit
  (ratio ≈0.71, both backends) — xfail-tracked to `fix-sppm-bathroom-black-walls`,
  not masked by loosening the band. (The reported "big-kernel SPIRV-Cross limit
  in a `wfSppm*` entry" was a red herring: all eight convert to MSL cleanly; the
  failure was the megakernel's undeclared binding.)
- **Coated diffuse rendered with a dark region** (change
  `fix-flat-coat-fresnel-eta`) — `assets/dragon_removed.usda` (a pbrt
  `coateddiffuse` floor exported as a `UsdPreviewSurface` with `clearcoat = 1`)
  rendered ~2.4× too dark with a large dark region instead of a near-uniform
  diffuse. The flat coat lobe's selection probability
  `pCoat = coat · fresnelDielectric(NdotV, coatIOR)` passed `coatIOR` raw, but
  `fresnelDielectric`'s convention is `eta = η_incident / η_transmitted` — for a
  view ray entering the coat from air the correct ratio is `1/coatIOR`. The raw
  IOR computes the *exiting* (coat→air) direction, which triggers spurious total
  internal reflection past ~42° from normal: `pCoat` saturates to 1 over most of
  the surface, zeroing the base diffuse/spec lobes (attenuated by `1 − pCoat`)
  while only the correct (small, `F0 = 0.04`) coat reflection survives — a large
  energy loss. Fixed at the three coat-selection sites
  (`FlatMaterial.sample()`/`evaluate()`, `flatBsdfResponse()`) to use
  `1/coatIOR`, matching the glass-refraction branch and the subsurface boundary.
  The bug was latent until `pbrt-mtlx-roundtrip-fix` (`eb1a0f2`) folded
  UsdPreviewSurface `clearcoat → coat`, turning the coat lobe on for such
  materials. Coat is gated `coat > 0`, so non-coated flat materials are
  byte-unchanged; no coated scene is in the parity corpus, so the dual gate is
  unperturbed. New Vulkan harness gate `tests/test_flat_coat_fresnel.py` pins the
  entering-eta convention.
- **MaterialX `standard_surface` glass rendered opaque** (change
  `glass-transmission-opacity-gate`) — the right sphere of
  `assets/glass_caustics_test.usda` (a MaterialX `standard_surface` with
  `transmission = 1`) rendered as a solid opaque ball while the left
  UsdPreviewSurface glass (`opacity = 0`) refracted correctly. skinny's flat path
  only refracts surfaces whose `opacity < 1` (`flat_material.slang`), so the
  loader bridges `transmission → opacity = 1 − transmission`
  (`_derive_opacity_from_transmission`). That bridge skipped whenever **any**
  `opacity` was authored, but a `standard_surface` shader always authors `opacity`
  (its MaterialX default is the fully-opaque `(1, 1, 1)`) — so on the
  usdMtlx-plugin intake path the default blocked the bridge and the glass stayed
  opaque. The bridge now skips only for a **genuine cutout** (`opacity < 1`, via
  the new `_opacity_is_fully_opaque`); a default-opaque opacity is overwritten by
  `1 − transmission`, so the two material-intake paths agree and the glass
  refracts. Host-side material-loading fix — no shader change, so every backend /
  execution mode / integrator inherits it. New
  `test_transmission_opacity_gate_ignores_default_opaque`
  (`tests/test_struct_layout.py`) and scene-level
  `test_glass_caustics_both_spheres_transparent`
  (`tests/test_glass_caustics_test.py`); GPU-verified on Metal + path tracer
  (megakernel) — both spheres now render as glass.
- **Megakernel SPIR-V had an invalid negative array index in the BDPT path**
  (`fix-bdpt-negative-index`) — when a BDPT MIS helper
  (`misWeight`/`splatMisWeight` in `integrators/bdpt.slang`) is inlined with a
  compile-time `t == 1` (e.g. the `t = 1` NEE/splat strategies), index
  expressions like `litC[t - 2]` and the guarded `litC[i - 1]` fold to a constant
  `litC[-1]`, emitting `OpAccessChain ... %int_n1` into `main_pass.spv`.
  spirv-val rejects it (`VUID-VkShaderModuleCreateInfo-pCode-08737`, "Index … may
  not have a negative value"), so the megakernel module printed a validation error
  on every load — even under `--integrator path` (the megakernel compiles all
  integrators). The guarded neighbour indices are now clamped with `max(…, 0)` so
  the folded index is `0`, never `-1` — behaviour-preserving, since the value is
  always discarded by the surrounding `i > 0` / `t >= 2` guard. New gated
  regression test compiles the megakernel and asserts spirv-val is clean
  (`tests/test_megakernel_spirv_valid.py`).
- **Area lights rendered dim — emissive-triangle MIS under-count** (change
  `emissive-triangle-bsdf-mis`) — the path tracer weighted its emissive-triangle
  NEE sample with the power heuristic (`wNEE < 1`) but **dropped** the
  complementary BSDF-sampled hit on the same light, so it lost a `(1 − wNEE)`
  fraction of every area light — a dim bias that grows with the light's solid
  angle (a large window lost far more than a small bulb). The fix adds the
  BSDF-hit **MIS complement** in both the megakernel (`path.slang`) and wavefront
  (`wf_shade_common.slang`) path tracers, reconstructing the NEE solid-angle pdf
  at the hit without the triangle index (`pdfLightSA = lum·d² /
  (emissiveTotalPower·cosLight)`; the per-triangle area cancels under
  power-weighted selection). New `FrameConstants.emissiveTotalPower` reuses the
  retired `irisZ` slot (no UBO layout change). On the contemporary-bathroom corpus
  scene the path render's mean-vs-pbrt ratio goes 1.36 → 0.97 and FLIP 0.222 →
  0.155, with `megakernel ≡ wavefront` (self-consistency relMSE 0.0000); the path
  bathroom FLIP baselines are lowered accordingly. The investigation also
  **disproved** the original hypothesis that the dimness was a `blackbody`
  emitter-normalisation error: pbrt's peak normalisation cancels in
  `scale /= SpectrumToPhotometric`, and skinny's emitted luminance already matches
  pbrt's `s·imagingRatio` exactly (direct-view ratio 1.0000). Scope: the Path
  tracer on Metal; BDPT, SPPM and Vulkan are follow-ups.
- **Marble in the three-materials demo rendered as broken clear glass** (change
  `marble-subsurface-opacity-gate`) — the demo marble is a plain
  `standard_surface` with a `subsurface = 0.4` *weight* (no interior medium). The
  loader's subsurface→opacity bridge (added for pbrt `Material "subsurface"`)
  fired on any `subsurface > 0` and forced `opacity = 0`, so the flat path
  refracted the marble as a clear dielectric — a dark, near-black, blue-tinted,
  speckled ball that ignored the lights, on every backend / execution mode /
  integrator. `_derive_opacity_from_subsurface` now opens the refraction gate
  **only** when a genuine interior medium is present (`subsurface_sigma_a/σ_s`,
  via the new `_has_subsurface_medium`, mirroring `_material_is_subsurface`); a
  bare subsurface weight stays opaque diffuse. Genuine pbrt subsurface materials
  (which carry σ_a/σ_s) are unchanged. Host-side material-loading fix — no shader
  change.
- **Film per-sample radiance clamp (`maxcomponentvalue`)** (change
  `film-maxcomponent-clamp`) — skinny imported the pbrt film `iso` exposure but
  not its `maxcomponentvalue` firefly clamp, so scenes with tiny ultra-bright
  emitters threw fireflies the pbrt reference never has. On `bathroom.usda` (a
  window area light + four `scale 7000` bulb filaments) this was the entire
  parity divergence: **9 firefly pixels carried 99.7 %** of the path-vs-pbrt
  relMSE (232.8) and **100 %** of the BDPT-vs-path relMSE (6612, FLIP only
  0.085 — structurally identical). The renderer now reads `maxcomponentvalue`
  from the imported film and clamps each sample's radiance proportionally
  (hue-preserving, `max(r,g,b) ≤ C`) before accumulation — matching pbrt
  `RGBFilm::AddSample` — across the megakernel and the wavefront path / BDPT /
  SPPM accumulation sites (`FrameConstants.filmMaxComponent`; 0 = disabled, so
  scenes without the clamp render byte-identically). Result: bathroom
  path-vs-pbrt **232.8 → 0.36**, BDPT-vs-path **6612 → 0.36** (MSE 0.017 —
  structurally identical, the documented BDPT divergence gone), skinny max
  **7985 → 50**. The bathroom reference is regenerated with pbrt's `path`
  integrator (apples-to-apples with skinny's path anchor; `regen_refs.py
  --integrator path`) and the recorded bathroom baselines drop ~500×. The scene
  **stays `known_divergent`** for two residuals left as separate follow-ups: the
  RGB-vs-spectral / blackbody-emitter-normalization pbrt-truth mismatch (~0.34),
  and the sppm-vs-path / dark-region-`/b²`-amplified self-consistency relMSE on
  this noisy caustic scene (not loosened here). `megakernel ≡ wavefront` is
  reconfirmed exactly (path relMSE `0.0000`).

- **MaterialX graph codegen: rewrite the default `<texcoord>` UV input** (change
  `mtlx-graph-texcoord-uv`) — a graph driven by a bare `<image>` node on the
  default UV set (e.g. every `base_color` image in `bathroom.usda`) made
  `MaterialXGenSlang` emit `vd.texcoord_0`, which `_emit_graph_fragment` did not
  rewrite (it handled only the `<geompropvalue geomprop="UVMap">` form). The
  leftover `vd` is undefined in the per-material fragment, so the generated
  module failed to compile and took down every pipeline that imports it —
  surfacing as `skinny --integrator sppm assets/bathroom.usda --execution-mode
  wavefront` aborting with `undefined identifier 'vd'`. The emitter now maps both
  UV forms to `UV_in`, and falls back to the flat / std_surface path (instead of
  emitting an uncompilable module) for any `vd.*` vertex input it does not pipe.
- **pbrt importer: reproject equal-area env maps** (change
  `pbrt-env-equiarea-projection`) — pbrt v4 `infinite` light images use the
  equal-area octahedral parameterization, but the importer copied the `.exr`/`.pfm`
  pixels verbatim into the `.hdr` that skinny samples equirectangularly, scrambling
  every incoming direction. Square image maps are now reprojected equal-area →
  equirectangular (`src/skinny/pbrt/equiarea.py`, ports pbrt's
  `EqualAreaSquareToSphere`/`SphereToSquare`, `Rx(+90)` axis map for pbrt Z-up →
  skinny +y-up). This is the dominant cause of `sss_dragon_small.pbrt` looking
  wrong as `dragon_sss.usda` — the subsurface dragon is lit almost entirely by the
  environment. Non-square maps are assumed lat-long and passed through; constant
  and uniform infinite lights are byte-unchanged.

- **pbrt importer: honor authored camera up/roll** (change `pbrt-camera-up-axis`)
  — an imported camera whose up vector is not ≈ +Y (the pbrt Z-up convention,
  e.g. `sssdragon`) previously rendered ~90° rolled because `_extract_camera`
  dropped the up vector and the renderer hardcoded `world_up = (0,1,0)`. The
  loader now carries the authored up on `CameraOverride.up`, and `_look_at` /
  `OrbitCamera` build the view basis from `(position, forward, up)` (with a
  degenerate up∥forward fallback), so Z-up scenes orient correctly. Y-up cameras
  default to `(0,1,0)` ⇒ byte-identical; the pbrt parity corpus is unchanged.
  Composes with the existing mirrored-camera (`Scale -1`) ndc.x flip.
  (Scope: orientation only — env-light intensity and env-map rotation for Z-up
  scenes remain follow-ups.)

### Added

- **pbrt subsurface materials now render as a volumetric interior random walk**
  (was clear glass) (change `pbrt-subsurface-volumetric`, Stage-2 Ch5 of the
  pbrt-mtlx roadmap) — a pbrt `Material "subsurface"` imports to a new
  `MATERIAL_TYPE_SUBSURFACE` instead of being lowered to the flat material with
  `opacity = 0` (which rendered as clear glass). It is a smooth dielectric
  boundary (`eta`) wrapping a homogeneous interior medium (`σ_a`, `σ_s`,
  Henyey-Greenstein `g`), transported by a delta-tracked (Woodcock /
  null-collision) 1D-slab volumetric random walk: refract in, march the interior
  with HG scattering, NEE a single analytic distant light + the environment on
  escape, refract out. Coefficients follow pbrt precedence — explicit
  `sigma_a`/`sigma_s` (× `scale`) → named preset (`Skin1`, …, the measured
  table) → `reflectance` + `mfp` via the Jensen diffuse-albedo inversion — and
  the `-mtlx` `standard_surface` (`subsurface_color`/`radius`/`scale`/
  `anisotropy`) maps to identical coefficients. Runs in **both execution modes**
  (megakernel + wavefront) and **both backends** (Vulkan + native Metal): one
  dispatch case in `integrators/path.slang` `evaluateBounce()` serves the
  megakernel and the wavefront catch-all kernel (BDPT excludes it, flat-only eye
  walk, like skin). The medium is packed **inline** into the existing
  `FlatMaterialParams` buffer (binding 13) — no new SSBO, to respect Metal's
  31-buffer cap. Energy-conserving: a furnace (`σ_a → 0`) returns ~unity
  (measured 0.996); PT≡BDPT relMSE 0.0058, Metal↔Vulkan relMSE 0.0175,
  true-glass back-compat unchanged, pbrt-v4 corpus `subsurface_infinite` relMSE
  0.079 (dipole-vs-random-walk ⇒ milky, not bit-parity). Follow-ups: the walk
  samples only a single distant light + the environment, so area/emissive lights
  inside the medium, heterogeneous / NanoVDB grids, and free-standing
  `MediumInterface` media remain future work (the majorant / null-collision
  transport and handle-referenced medium let them slot in additively).

- **Flat BSDF: colored glass, tinted speculars, Oren-Nayar diffuse** (change
  `flat-lobes-rich-inputs`, Stage-2 Tier A of the pbrt-mtlx roadmap) — the
  unified flat / `standard_surface` lobe BSDF now consumes three previously-dead
  `standard_surface` inputs, filling the existing `{coat, spec, diffuse,
  delta-transmission}` lobe set with **no** new lobe and **without** calling the
  preview-only `evalStdSurfaceBSDF`: `transmission_color` tints the
  delta-transmission branch (smooth colored glass; still a delta event, no
  MIS/Jacobian change), `specular_color` scales the GGX spec response
  (response-only, pdf unchanged), and `diffuse_roughness` drives an Oren-Nayar
  diffuse response while keeping cosine sampling (`0` ⇒ exact Lambert; diffuse
  pdf and `sample().pdf == evaluate().pdf` preserved). `FlatMaterialParams`
  (binding 13) grows 128 → 160 B for the appended `transmissionColor` /
  `diffuseRoughness` / `specularColor`. Back-compat: `pack_flat_material` falls
  back to `transmission_color ← diffuseColor`, `specular_color ← white`,
  `diffuse_roughness ← 0`, so absent inputs reproduce prior behavior exactly —
  the pbrt parity corpus and existing UsdPreviewSurface renders are unchanged.

- **pbrt importer: MaterialX sidecar export (`-mtlx`)** (change
  `pbrt-mtlx-export`) — `skinny-import-pbrt -mtlx` additionally writes a portable
  MaterialX `.mtlx` of `standard_surface` materials, referenced from the exported
  stage, capturing pbrt parameters UsdPreviewSurface cannot express
  (`transmission`/`transmission_color`, separate `coat`/`coat_IOR`,
  `subsurface_radius`, `specular_anisotropy` from `uroughness`/`vroughness`,
  `thin_walled`). Interop + Stage-2-enabling: the production integrators consume
  the `FlatMaterial` subset of either export, so for **diffuse / conductor /
  dielectric** the `-mtlx` and UsdPreviewSurface renders are pixel-identical
  (`glass_arealight` relMSE 0.0215 vs pbrt for both, 0.000000 between them).

- **pbrt importer: `subsurface` + `coated*` `-mtlx` round-trip** (change
  `pbrt-mtlx-roundtrip-fix`) — closes the limitation flagged when `-mtlx` shipped.
  The `.mtlx` fallback loader (`_load_mtlx_materials`) now bridges
  `subsurface → opacity = 0` (the transmissive boundary, alongside the existing
  `transmission`/`emission` bridges); `_resolve_material_binding` merges the bound
  prim's `skinnyOverrides` customData (the homogeneous SSS interior) into the
  loaded `Material`; the loader canonicalizes UsdPreviewSurface
  `clearcoat`/`clearcoatRoughness` onto the `coat`/`coat_roughness` slots the
  `FlatMaterial` packer reads (fixing a silent coat drop on the UsdPreviewSurface
  path too); and `map_material_mtlx` `coateddiffuse` now reads the coat roughness
  from pbrt `roughness` (mirroring `map_material`) instead of a non-existent
  `interface.roughness`. The `-mtlx` and UsdPreviewSurface exports of
  `sssdragon/dragon_10` (subsurface dragon + coateddiffuse floor) now render
  equivalently on Metal wavefront (relMSE 0.0001 / FLIP 0.0002 between them, was
  0.88 / FLIP 0.46 opaque-white). **Note:** `coateddiffuse`/`coatedconductor`
  renders change
  on **both** export paths — the coat lobe now actually contributes.

- **pbrt importer: Loop subdivision surfaces** (change `pbrt-loopsubdiv-shape`) —
  `Shape "loopsubdiv"` is no longer skipped. The control cage (`P`, `indices`,
  `levels`) is tessellated to triangles at import time exactly as pbrt does it:
  Loop refinement applied `levels` times, then vertices pushed to the Loop limit
  surface with per-vertex limit normals (tangent masks), routed through the
  existing `trianglemesh` emit path. Unblocks the killeroo scenes, whose bodies
  are `loopsubdiv` (`killeroo-simple` import goes from "2 skipped" to "0 skipped").

- **pbrt importer: texture-valued material parameters** (change
  `pbrt-float-texture-params`) — a `FloatTexture`/`SpectrumTexture` parameter
  bound to a named texture (e.g. `"texture roughness" ["…"]`) is now resolved
  through a single promoting accessor that mirrors pbrt's
  `GetFloatTexture`/`GetSpectrumTexture` (constants promoted, named textures
  resolved), so material mapping never calls `float()` on a texture name. Each
  textured parameter maps to its **own** USD input via the `_TEXTURABLE` table
  (`roughness → roughness` `.r`, `reflectance → diffuseColor` `.rgb`) — the map's
  `value_type` is the single source of truth (the duplicate `_SCALAR_TEX_INPUTS`
  set is gone). Unsupported/unresolvable textures and inputs with no USD texture
  slot degrade to the scalar/rgb default with an `approx` note instead of raising.
  Fixes the `ValueError: could not convert string to float` crash importing the
  `crown` scene (texture-valued roughness, nested `scale`/`imagemap`). See
  [docs/PbrtImport.md](docs/PbrtImport.md).
- **Render-area resolution flags** (change `cli-render-resolution`) — `--width`
  and `--height` (env `SKINNY_WIDTH` / `SKINNY_HEIGHT`) set the render-area pixel
  size on the interactive front-ends `skinny` and `skinny-gui`, from one shared
  definition. Both default to **640×480** (precedence flag > env > default); a
  non-positive value is rejected at startup. On `skinny` they size the GLFW
  window and the GPU render target; on `skinny-gui` they size the offscreen
  render area while the Qt window and docks keep their own size. The headless
  `skinny-render` keeps its own `--width` / `--height` (1024² offline-output
  default); `skinny-web` does not expose the flags. **Note:** this changes the
  interactive default render area from 1280×720 to 640×480 — pass
  `--width 1280 --height 720` (or set the env vars) to restore the old size.

- **SPPM glossy / near-specular reflector reconstruction** (change
  `sppm-glossy-final-gather`) — a metallic-gated, roughness-thresholded eye-walk
  continuation for the SPPM integrator. A metallic sample
  (`metalness ≥ 0.5`) whose roughness is below the new
  `--sppm-glossy-roughness` threshold (env `SKINNY_SPPM_GLOSSY_ROUGHNESS`;
  default ≈ 0.5) is continued **one bounce** like the delta caustic carrier
  instead of storing a visible point, so the visible point lands on the next
  non-glossy surface and the sharp reflection is reconstructed there and averaged
  across passes. The photon stage treats a glossy-continued vertex as specular
  (no deposit), preserving the disjoint NEE-direct / photon-indirect split and the
  energy ratio. PM-1 stored a VP at the first non-delta hit, so glossy metals
  (e.g. the brass sphere in `three_materials_demo`) failed to reflect their
  neighbours; a **threshold of 0 reproduces PM-1 exactly** (delta-only). The
  metallic guard is required because `BSDFSample` carries no sampled-lobe id and
  the shared flat BSDF stays byte-frozen (path / BDPT SPIR-V unchanged). A full
  final-gather variant for mid-roughness glossy is deferred. See
  [docs/PhotonMapping.md](docs/PhotonMapping.md).
- **GPU SPPM integrator** (change `photon-mapping-sppm`, PM-1) — a Stochastic
  Progressive Photon Mapping integrator (`INTEGRATOR_SPPM = 2`), the
  caustic-efficient third integrator after `path` and `bdpt`. **Wavefront-only**,
  **flat materials only**, on **both Vulkan and native Metal**
  (`WavefrontSppmPass` / `MetalWavefrontSppmPass`). Selectable via `--integrator sppm`
  (requires `--execution-mode wavefront`; refused under megakernel with a clear
  message), the GUI "SPPM" mode, or the pbrt importer. One SPPM pass == one
  progressive-accumulation frame, a four-stage pipeline over eight kernels in
  `integrators/wavefront_sppm.slang` — **eye** (one stochastic visible point per
  pixel + per-pass NEE direct), **grid build** (counting-sort spatial hash:
  count → exclusive prefix sum → scatter), **photon** (power-weighted emission,
  RR trace, bare-f_r deposit at non-specular vertices `depth ≥ 1` via a 3×3×3 cell
  scan with uint fixed-point atomics), and **update** (the SPPM radius/flux
  reduction `N'=N+γM`, `r'=r·√(N'/(N+M))`, `τ'=(τ+Φ)(r'/r)²`, then
  `L = ld + τ/(N_emitted·π·r²)` composited into the film). The per-pixel
  estimator (radius / count / flux) persists across frames; direct reuses NEE so
  the photon term is the disjoint indirect/caustic complement. `--sppm-radius`
  sets the initial search radius (default ≈ 0.1 % of the scene bbox diagonal),
  `--sppm-photons-per-pass` the photons/pass (default one per pixel). The pbrt v4
  importer now **maps** `Integrator "sppm"` / `"photonmap"` (previously skipped)
  to the skinny SPPM selection (`customLayerData["pbrt"]["skinny"]`, read by
  `api.sppm_selection`). Verified on both backends (Apple M5 Pro): Cornell-box
  SPPM/path energy ratio 1.008 (Vulkan); glass caustic parity vs the pbrt
  reference relMSE 0.025 (Vulkan and Metal). Skin/BSSRDF (PM-2) and volumetric (PM-3) photon transport
  are deferred. See [docs/PhotonMapping.md](docs/PhotonMapping.md).
- **pbrt importer imagemap-texture UVs** (change `pbrt-imagemap-uv-parity`) —
  finish imagemap-texture UV support in the pbrt v4 importer. Explicit mesh UVs
  (`trianglemesh` `uv`/`st`, PLY `u/v`/`s/t`/`texture_u,texture_v`, ascii+binary)
  pass through to per-vertex `primvars:st`, and pbrt-faithful default UVs are
  synthesized for UV-less *textured* shapes — `sphere` parametric UVs and
  per-triangle `faceVarying` `{(0,0),(1,0),(1,1)}` for `trianglemesh`/`plymesh` —
  so a bound texture samples like pbrt instead of at a constant point. Adds a
  `texture_quad` parity corpus scene (relMSE 0.0024 / FLIP 0.0119 vs pbrt v4) that
  validates end-to-end GPU texture sampling. See
  [docs/PbrtImport.md](docs/PbrtImport.md).
- **Mirrored (improper) pbrt camera support** (change `pbrt-mirrored-camera-flip`)
  — an orientation-reversing pbrt camera (negative-determinant camera-to-world,
  e.g. `Scale -1 1 1` before `LookAt`) now renders matching pbrt instead of
  horizontally mirrored. The importer already flagged it (`customData["pbrt"]
  ["mirrored"]`); the renderer now honors the flag end-to-end: the loader threads
  it to `CameraOverride.mirrored`, it rides `FrameConstants.cameraMirror`, and
  `zoomedNDC` negates `ndc.x` (a single chokepoint covering pinhole + thick-lens,
  megakernel + wavefront, Metal + Vulkan), with a matching `sampleWi` flip for
  BDPT/light-tracing camera connections. New `mirrored_arealight` parity corpus
  scene gated at relMSE 0.009 / FLIP 0.021 vs the pbrt v4 reference. Driven solely
  by the imported metadata (no GUI/CLI knob).

### Fixed

- **Vulkan FrameConstants UBO silent truncation** — the Vulkan uniform buffer was
  pinned to exactly 512 B (the then-current scalar-blob size) while
  `UniformBuffer.upload` memmoves `min(len, size)`, so any scalar-tail field past
  512 B was dropped on Vulkan (Metal, sized from reflection, was unaffected). The
  buffer is now sized via `_VK_UNIFORM_BUFFER_BYTES` (768 B, with an import-time
  guard tying it to the `_FC_SCALAR_FIELDS` table) so the tail can't be truncated
  unnoticed. Surfaced while wiring `cameraMirror`.

- **pbrt v4 scene import** (change `pbrt-v4-scene-import`) — read a pbrt v4 text
  scene and convert it to a skinny-loadable USD stage via the new `skinny.pbrt`
  package and `skinny-import-pbrt` CLI (`python -m skinny.pbrt` / `import_pbrt()`).
  Covers triangle/ply/sphere shapes + instancing, diffuse/conductor/dielectric/
  coated/transmissive materials (with the pbrt-v4 `sqrt` roughness remap),
  distant/point/area/infinite lights, the `perspective` camera (shorter-axis fov),
  spectrum→RGB reduction, and homogeneous media/subsurface best-effort. Each
  import emits an exact/approx/skipped report. Ships a parity corpus
  (`tests/pbrt/corpus/`) and a relMSE/FLIP gate comparing skinny's linear-HDR
  accumulation against checked-in pbrt v4 reference EXRs (no pbrt binary needed
  at test time). See [docs/PbrtImport.md](docs/PbrtImport.md).
- **Shared CPU neural handoff** (change `shared-neural-handoff`) — a third
  `--neural-handoff` value, `shared` (env `SKINNY_NEURAL_HANDOFF`, persisted on
  the front-ends). Since the online trainer runs as a same-process daemon thread,
  `shared` hands freshly-trained weights across the trainer→render boundary
  through an in-process CPU double-buffer held in RAM (`neural_handoff_shared.py`,
  `SharedWeightPublisher`): no disk write (unlike `file`) and no CUDA /
  unified-memory device (unlike `interop`), available on **any** platform with no
  added dependency. `publish()` stores a byte-faithful private copy via the new
  `serialize_neural_weights` / `deserialize_neural_weights` core (the same path
  `write_neural_weights` / `load_neural_weights` now wrap — NFW1 on-disk format
  unchanged), so the bytes the renderer consumes are identical to a `file` publish
  and a post-publish trainer mutation can never leak into the frozen render
  buffer. The renderer uploads swapped weights to the GPU through the normal
  post-swap path; `shared` never writes the GPU buffers directly (that is
  `interop`). Same frame-boundary swap + `networkVersion`-increment contract as
  the other backends. `file` stays the default; existing runs are byte-identical.
  (`tests/test_neural_handoff_shared.py`.)
- **MLX neural trainer** (change `mlx-neural-trainer`) — the reserved
  `--neural-trainer mlx` token is now a working GPU training backend: online
  neural-proposal training runs on the Metal GPU of an Apple-Silicon host via
  Apple MLX (optional `[mlx]` extra), no CUDA. `MlxTrainingBackend` mirrors the
  numpy oracle's spline-flow math op-for-op on `mlx.core` arrays — the parity
  target — but with MLX autodiff and a hand-rolled bias-corrected Adam (same
  global grad-norm clip), so a one-step update from identical weights matches
  the oracle for the bulk of the net within tolerance (the handful of outliers
  are Adam sign-flips on near-zero-gradient weights). It bakes the same fp32
  NFW1 weights and round-trips through both the `file` and `interop` handoffs
  unchanged. `--train-precision fp16` runs the flow in `float16` over fp32
  master weights with a runtime fall-back to fp32 (one-time warning) if a step's
  loss or gradients go non-finite — the fp32 masters are never corrupted, and
  the bake stays fp32. `--neural-trainer auto` precedence becomes
  `cuda > mlx > cpu`: torch CUDA when present, else MLX when the `[mlx]` extra
  is importable on a Metal host, else the numpy oracle; the supported Mac combo
  is now `--neural-trainer mlx`. The numpy and torch backends, the dataset
  contract, and the handoff format are unchanged; MLX is never imported
  unconditionally, so hosts without the extra behave exactly as before.

### Changed

- **Power-weighted emissive-mesh NEE, no triangle cap** (change
  `emissive-mesh-nee`) — next-event estimation now selects an emissive (mesh)
  area-light triangle with probability proportional to its power
  `w_i = area_i × Rec.709-luminance(emission_i)` via a cumulative-power CDF, in
  place of uniform-by-index, and **every** emissive triangle in the scene
  participates (the silent 256-triangle cap in `_upload_emissive_triangles` is
  gone — the buffer grows and rebinds to the actual count, like
  `material_capacity`, and logs it). This fixes imported interiors that rendered
  **dark** (high-poly emissive meshes lost energy past 256 triangles) and
  **noisy** (uniform selection wastes most samples on dim/oversplit triangles).
  The CDF is packed **inline** in each `EmissiveTriangle` record's spare `.w`
  lanes (`cw`, `pSel`) — the same one-slot trick the environment distribution
  uses — so no new descriptor binding is added (which would exceed native-Metal's
  31-buffer argument-table cap). The selection probability flows through
  `selectionPdf = p_i`; the area→solid-angle pdf and MIS power heuristic are
  unchanged in form, so NEE stays unbiased and the no-double-count gate is
  untouched. The shared `sampleEmissiveTriangle` helper covers the megakernel and
  wavefront path/BDPT integrators (one `nee.slang`), the skin direct-lighting
  path, and ReSTIR DI (whose emissive candidate draw now matches the power pdf it
  reports — reservoir/RIS/GRIS code unchanged). Raising the NEE baseline narrowed
  ReSTIR's variance-reduction margin on the demo scene (still a real reduction;
  `test_restir_variance` threshold relaxed 0.85→0.90 accordingly). New GPU gate
  `tests/pbrt/test_emissive_nee.py` (correctness/energy, equal-spp variance,
  unbiasedness, `diffuse_arealight` no-regression). See
  [docs/Megakernel.md](docs/Megakernel.md) § 3.1 and the
  [docs/Architecture.md](docs/Architecture.md) binding map (18).
- **Neural-flow Lambert directional chart** (change
  `directional-flow-parameterization`) — the neural directional proposal's
  square↔direction map (`neural_flow.slang` `nf_square_to_hemi` /
  `nf_hemi_to_square`) is now the **Lambert azimuthal equal-area** chart (Shirley
  concentric square→disk + the lift `cosθ = 1 − r²`) in place of the cylindrical
  equal-area map. It removes the azimuth seam and puts the pole at the disk
  centre, making the guided lobe a strictly easier target for the same net (study
  `directional-flow-param-study`: BRDF 1.23×, path 1.09–1.49× equal-time
  efficiency). The chart is equal-area, so `|J| = 2π` is unchanged — `NF_LOG2PI`,
  `sampleNeural` / `pdfNeural`, and the whole MIS pdf path are byte-identical;
  only the baked direction differs, so a chart-matched (V1) `.nrec` is required
  (the parity goldens are rebaked with `chart="V1"` and a chart-mismatch gate
  guards against silently running a V0 net).

### Backend

- **Metal record drain** (change `metal-record-drain`) — wavefront path-record
  emission and the live online-training drain now run on the native Metal
  backend, closing the loop that `metal-neural-interop` opened: online neural
  training is **fully single-device on Apple Silicon** (records drain → numpy
  trainer → UMA interop weight handoff → frame-end swap; no Vulkan device, no
  megakernel record dispatch, no NFW1 file). Arming online training rebuilds
  the Metal wavefront path pass with **`SKINNY_METAL_RECORDS=1`**, un-stubbing
  the `wf_records.slang` emitters; the build re-fits Metal's 31-buffer-slot
  argument table (the neural build sits exactly at it) by compiling out the
  two resolve globals inert on a training render (`lightSplatBuffer`,
  `gizmoSegments` — no gizmo overlay on training frames) and folding both
  counters into their data buffers: the per-lane stack count into a header
  element, and `recordCounter` into a 64-byte header of a byte-address
  `recordBuf` (capacity @0, atomic count @60, packed 64-byte records from
  byte 64 — byte-identical to the Vulkan record stream despite MSL float3
  padding). The renderer drain is backend-neutral (Metal binds by name,
  resets only the 4-byte count word per frame; `metal_compute` `upload_range`
  now does a partial encoder upload instead of a full-shadow reflush); the
  megakernel record source is refused on Metal with a clear error. Guarded
  A/B: Metal and Vulkan record streams match (equal per-depth counts,
  marginal distributions within cross-backend float tolerance), records-off
  renders are bit-identical through an arm→disarm round-trip, and records-on
  costs ≈0.35 ms (~4.6 %) per 128² Cornell frame
  (`tests/test_metal_record_drain_gpu.py`).
- **Metal neural interop** (change `metal-neural-interop`) — `--neural-handoff
  interop` now works on the native Metal backend: a new
  `MetalSharedWeightPublisher` stages freshly-trained weights host-side and the
  frame-boundary swap writes them **in place** into the binding-33/34 weight
  buffers, which are allocated as **UMA shared-storage**
  (`StorageBuffer(shared=True)`, `MemoryType.upload` → `MTLStorageModeShared`)
  — no NFW1 file round-trip, no staging upload, no semaphore (the swap runs on
  the render thread after the frame's device drain, so a frame never reads a
  half-written network). `MetalContext` gains a `supports_shared_memory`
  capability flag (probed; the external-memory / -semaphore flags stay `false`
  — Metal exports no handles), and `make_publisher("interop")` resolves the
  mechanism per backend (CUDA on Vulkan, UMA on Metal) with a clear
  `NotImplementedError` naming the `file` fallback where neither exists.
  Published bytes are byte-identical to the file path at every
  `NeuralPrecision` (fp32/fp16/fp8-e4m3); the staged in-place copy lands in
  ≤0.1 ms at the shipped fp32 size (Apple M5 Pro). The Metal
  `render`/`render_headless` paths now run the frame-end weight swap
  (previously Vulkan-only). The wavefront record drain followed in the
  `metal-record-drain` entry above, completing the fully-on-Metal online loop.
- **Metal wavefront parity** (change `metal-wavefront-parity`) — the wavefront
  execution mode now runs on the native Metal backend at parity with Vulkan:
  staged **path** and **BDPT** integrators (all three walk modes), **ReSTIR DI**
  reuse, and the **neural directional proposal**. The loop stage orders moved to
  a backend-neutral driver (`wavefront_driver.py`: `record_path_loop` /
  `record_bdpt_loop` over a `WavefrontRecorder` protocol) that the existing
  Vulkan recorder reproduces byte-for-byte; `metal_wavefront.py` adds the Metal
  side — per-entry in-process Slang→Metal pipelines, queue buffers sized from
  the **reflected MSL strides** (`float3` pads to 16 B: `WavefrontPathState`
  96 B, `BDPTVertex` 176 B), one `MetalFrameEncoder` per frame with global
  barriers, and a CPU slot-count-readback fallback for the per-material
  indirect dispatches behind a logged empirical probe (slang-rhi 0.42's Metal
  indirect dispatch is a silent no-op). Selecting the neural proposal rebuilds
  the Metal pass with `SKINNY_METAL_NEURAL=1`, un-stubbing the frozen-weight
  buffers under Metal's 31-buffer-slot argument table by compiling out the
  three wavefront-dead globals (`toolBuffer`, `recordBuf`, `recordCounter`);
  weights upload via `set_data` (fp32 — the device fp16 probe under-reports on
  current slang-rhi). Guarded A/B: path, BDPT, and ReSTIR bit-identical to the
  Vulkan wavefront render on this host; neural rel-MSE 0.00000 / corr 1.00000
  and unbiased. Equal-time (three-materials @ 256², M5 Pro): wavefront path
  ≈ 0.22× the Metal megakernel (per-bounce host syncs from the readback
  fallback; closes when slang-rhi ships Metal indirect dispatch). Mode
  selection stays user-driven — no silent fallback. Vulkan SPIR-V
  byte-identical throughout; the wavefront record drain (online training)
  remains Vulkan-only.

- **Metal megakernel render parity** (change `metal-megakernel-parity`, P2) — the
  full megakernel renderer now runs natively on Metal. `metal_compute.py` grew to
  resource-layer parity with `vk_compute` (`StorageBuffer` / `StorageImage` /
  `SampledImage` / `UniformBuffer` / `HostStorageBuffer` / `ComputePipeline`), and
  the Metal `ComputePipeline` compiles `main_pass.slang` (`mainImage`) to Metal
  **in-process** (no `slangc`, no `.metallib`) after `emit_megakernel_sources`,
  reflects the binding map, and dispatches by **binding resources by name** (no
  Vulkan descriptor sets). New `_pack_uniforms_msl` relocates the `FrameConstants`
  block to the reflected MSL layout (**592 B** vs the **512 B** Vulkan scalar —
  `float3` pads to 16 B), `set_data` only. Metal-target shader adaptations (Vulkan
  SPIR-V byte-unchanged): bindless `Texture2D[120]` + shared `commonSampler`,
  per-map discrete samplers (bindings 38–43), and an `NRI` macro for the
  compute-stage `NonUniformResourceIndex`. **`auto` now resolves to Metal on
  Apple-Silicon macOS** (else Vulkan); the foundation-phase `--backend metal`
  refusal is removed from all four front-ends. The renderer resolves its resource
  layer once via `backend_select.resource_module(ctx)` and drains teardown through
  a backend-neutral `ctx.wait_idle()` seam. Mac-verified (guarded): megakernel
  pipeline build + binding-map reflection, buffer/bindless/sampler binds, a head
  render with no hang, MSL offset pins, and shaded Metal↔Vulkan parity; byte-
  identical *structural* parity (geom/instance-ID AOV) is deferred to a follow-up
  change. New tests `tests/test_metal_megakernel_{binding_map,dispatch,head_render,
  rebuild}.py`, `tests/test_metal_vulkan_shaded_parity.py`.

- Native **Metal backend foundation** (change `add-metal-backend-foundation`,
  P1 of a phased plan) — a native Metal device built on SlangPy's
  `DeviceType.metal` (slang-rhi, no MoltenVK, no raw PyObjC) in new
  `metal_context.py` (`MetalContext`) + minimal `metal_compute.py`
  (`StorageBuffer` / `StorageImage` / `ComputePipeline`). Proves a trivial Slang
  compute dispatch on Metal that is **bit-identical** to the same kernel on
  Vulkan, plus a windowed clear+present whose GPU fence signals every frame.
  Slang compiles to Metal **in-process** (no `slangc` shell-out); the present
  path drives the slang-rhi `Surface` bridged from the GLFW Cocoa `NSWindow`
  (no manual `CAMetalLayer`). Pipeline params go via `set_data` byte blobs only,
  never per-field cursor writes (fence-hang discipline). New shared
  `--backend {auto,metal,vulkan}` flag (env `SKINNY_BACKEND`, persisted on the
  interactive front-ends) resolved once by `backend_select.select_backend` /
  `make_context` and routed through all four front-ends (`skinny`, `skinny-gui`,
  `skinny-web`, `skinny-render`). **Scope:** device + dispatch + present +
  selection only — the full renderer is not yet ported to Metal, so `auto`
  resolves to Vulkan and `--backend metal` on a real front-end reports the
  foundation phase and exits; `vulkan` is byte-identical to before on every
  platform. Mac-tested: `tests/test_backend_select.py`,
  `tests/test_metal_foundation.py`.

### Neural guiding

- Online-training scaffolding for the neural directional proposal (Stage 2, change
  `neural-online-training`) — the sampling-layer seam for continuous, recency
  -weighted training that tracks animation. New `sampling/neural_replay.py`
  (recency-weighted ring buffer over the shipped `PathRecord` layout, evicts stale
  records on motion), `sampling/neural_trainer.py` (warm-started async trainer
  skeleton on the shipped flow arch; the PyTorch optimisation step reuses
  `spline_flow` and is filled on the CUDA box), and a `NeuralWeightPublisher`
  handoff seam (`sampling/neural_handoff*.py`) with two flag-selectable backends:
  a **file double-buffer** (writes `NFW1`, hot-reloads via `neural_weights`,
  swaps + bumps `networkVersion` at frame end — works on any platform) and a
  **GPU-shared-memory interop** path (`VK_KHR_external_memory` +
  `cudaImportExternalMemory`, CUDA-guarded, raises off-CUDA). Added a canonical
  `write_neural_weights` NFW1 writer. The async swap stays unbiased by evaluating
  each sample against its stamped `networkVersion`; staleness raises variance only.
  Renderer-runtime wiring (live GPU-counter drain, frame-end swap point, the
  `--neural-handoff` flag, external-memory buffer export) and the CUDA trainer/
  interop internals are the NVIDIA-box follow-up. Mac-tested:
  `tests/test_neural_online.py` (replay recency/capacity/eviction, file-handoff
  swap + version, end-to-end drain→train→publish→swap loop, interop guard).

### Rendering

- Unified flat / `std_surface` BSDF onto one composable lobe set —
  `FlatMaterial.sample()` and `.evaluate()` now share `{coat, spec, diffuse}`
  (`materials/flat/flat_lobes.slang`), so `sample().pdf == evaluate().pdf` and
  `response / pdf` stays bounded (`F·G₁` / Lambert, no clamp). This is the
  canonical BSDF for the path tracer **and** BDPT in both megakernel and wavefront
  modes, and removes the directional-proposal-mixture bias on layered coat+metal
  materials (brass under the BSDF+Env / Env presets: +4.6% → −0.2% vs the BDPT
  reference; megakernel and wavefront converge identically). The MaterialX
  `std_surface` closure (`evalStdSurfaceBSDF`, binding 19) is retained only for the
  raster preview pass. Each lobe carries a runtime-pluggable sampler id (native
  strategies only for now) — the seam a later `per-lobe-sampler-registry` change
  populates with a host registry + alternative samplers.

- Per-lobe sampler registry for the flat / `std_surface` BSDF — populates that
  `samplerId` seam. Each lobe's draw/density strategy is runtime-selectable per
  lobe (`--lobe-samplers coat=…,spec=…,diff=…` / GUI / `SKINNY_LOBE_SAMPLERS`),
  folded into one `FrameConstants.flatLobeSamplers` uint with **no** new
  descriptor bindings. Registered alternates: the Heitz-2018 basis-form VNDF
  (coat/spec — a different warp of the *same* GGX visible-normal distribution, so
  the pdf is shared and converged radiance is identical to native, mega ≡ wave,
  PT ≡ BDPT) and uniform-hemisphere (diffuse — unbiased, bounded weight,
  demonstrably higher-variance than cosine). `sample()` and `evaluate()` read the
  same per-lobe id so unbiasedness holds for any strategy; `flatBsdfResponse`
  (= f·cos) is sampler-invariant. Registry in `sampling/lobe_samplers.py`; gate in
  `tests/test_sampling_parity.py`.

- Neural directional proposal (wavefront-only) — a learned, position-conditioned
  rational-quadratic **neural spline flow** (proposal bit2,
  `--proposals bsdf,neural` / GUI "BSDF + Neural") that proposes the BSDF bounce
  direction with an exact solid-angle pdf and MIS-mixes into the scene-sampling
  proposal seam. Frozen, offline-trained per scene (standalone `spline_flow` repo)
  in the **NFW1** weight format. Architecture is **Option A** — a compute pre-pass
  (`WavefrontNeuralProposalPass`, `wavefront/neural_proposal_pass.slang`) draws one
  forward sample per live lane between scatter and shade; the flat shade kernel
  reads it and evaluates the arbitrary-direction inverse pdf inline
  (`sampling/{neural_flow,neural_proposal,proposal}.slang` + `nee.slang`). Scoped to
  the flat wavefront shade kernel; unbiased regardless of net quality
  (`proposalWeights` renormalises, the same effective weights drive the bounce and
  NEE companion pdfs). New weight buffers at **bindings 33/34/35** (above the
  MaterialX graph range) — always bound with an all-zero dummy net so the inline
  inverse resolves everywhere; the megakernel strips the bit and falls back to its
  analytic proposal subset (mirroring ReSTIR DI → identity). FrameConstants gained
  a scalar tail (scene AABB + net version), UBO now 508 B.

- Neural proposal — offline training pipe (Stage 1b/1c). `Renderer.dump_path_records`
  emits per-vertex `(position, normal, wo, wiLocal, contribution)` training records
  to a `.nrec` file via a second megakernel entry `mainImageRecord`
  (`integrators/path_record.slang`, an RR-free path tracer that backward-attributes
  the tail radiance `contribution = (L_final−L_k)/beta_in_k = f·cos·Li`); records
  land on new **bindings 36/37** (`mainImage` never references them → byte-identical).
  The standalone `spline_flow/render_records.py` trains the flow from those records by
  contribution-weighted MLE (`q ∝ f·Li·cos`) using the exact `neuralCondition`
  encoding and bakes NFW1. Verified end-to-end on Mac MPS (4.36M Cornell records →
  trained net, pdf ∫≈1). The equal-time gate is **measured, not won on Mac**: the
  net is unbiased (mixture-MIS) but the MLP pre-pass is ~28× a bsdf bounce on
  MoltenVK/MPS (the deferred CUDA-perf goal) and the flat ceiling-lit Cornell box is
  broad-indirect (cosine already near-optimal), so the guide ≈ cosine and adds a
  firefly tail with no offsetting win. Follow-ups: GPU-optimised inference,
  guiding-iteration training, a concentrated-indirect scene, a pdf-floor / lower-α
  firefly measure.

- Neural proposal — **build-time-configurable size + precision** + a
  size×precision quality-vs-cost **study** (the MLP pre-pass cost, not quality,
  is what loses the equal-time gate on Mac — so: how small can the net get, and
  does Apple-Silicon fp16 cut the cost?). The network size (`NF_LAYERS/NF_BINS/
  NF_HIDDEN`) and inference precision become `slangc -D`-selectable via one
  `NeuralBuildConfig(layers, bins, hidden, precision)`
  (`skinny.sampling.neural_weights`) threaded into every neural compile + the
  weight upload; the default `fp32 @ 6/24/96` build is **byte-identical** to the
  shipped proposal (no `-D` flags). Mixed fp16 via two compile-time aliases —
  `NF_WT` (weight storage, the binding-33/34 element type) and `NF_CT` (MLP GEMM
  accumulate) — giving **fp32** / **fp16-storage** (`half`/`float`, ½-byte
  weights) / **fp16-compute** (`half`/`half`); the RQ-spline math + the returned
  pdf stay `float` in every mode. NFW1 stays fp32 on disk (host casts to half at
  upload — no new format); `vk_context` probes `shaderFloat16` +
  `uniformAndStorageBuffer16BitAccess` and falls back to fp32 (logged) where
  absent. Unbiasedness holds in every mode (mixture-MIS;
  `test_fp16_unbiased_gate`). The two-track study
  (`tests/study_size_precision.py` + `spline_flow/bake_grid.py`) maps fp16
  pdf-parity drift (negligible: `~4e-4`/`~1e-3` on the Cornell net) + held-out
  NLL against MoltenVK ms/frame + weight bytes → a Pareto table + a recommended
  config (a measurement; *shipping* a config is a later change). No new
  descriptor bindings (33/34 only change element type).

### Fixed

- **Path tracer specular→area-light bias** (change `path-bdpt-convergence`) — a
  BSDF-sampled ray that hit an emissive triangle (area light) after a
  delta/perfectly-specular bounce was dropped: emission was only added at the
  primary bounce, and next-event estimation cannot sample a delta lobe (a smooth
  dielectric's mirror reflect/refract), so the **reflection of an area light in
  glass** and the **specular leg of a caustic** contributed through neither path.
  The path tracer was biased dark (FLIP 0.058 → 0.025 vs the pbrt reference on
  `glass_arealight`). Fixed by carrying the spawning bounce's delta-ness
  (`spawnedBySpecular` in `integrators/path.slang` + `path_record.slang`; the
  `PATH_FLAG_SPECULAR` path-state bit + `pathSpecular()` gate in
  `wavefront/wf_shade_common.slang`) and adding the emission at full weight on a
  delta bounce — mirroring the existing sphere-light delta branch and BDPT's
  `deltaBounce`. Non-delta transport is bit-unchanged (no double-count with NEE).
  Verified equal on Vulkan and native Metal, megakernel and wavefront. The
  checked-in `main_pass.spv` is regenerated by the in-process runtime compile. A
  headless path-vs-pbrt convergence gate (`tests/pbrt/test_convergence.py`) locks
  it. (Apply note: the gate was re-anchored from BDPT to the pbrt reference —
  skinny's BDPT is ~1.7× too bright vs pbrt even on a diffuse scene, a separate
  normalization bug tracked on its own; the pbrt parity / convergence harness now
  resolves the GPU backend via `select_backend` so it exercises native Metal on
  Apple Silicon, not only MoltenVK-under-Vulkan.)
- **BDPT ~1.7× over-brightness vs pbrt** (change `bdpt-energy-convergence`) — the
  separate BDPT normalization bug noted above. BDPT's `t = 0` strategy (the
  eye/BSDF subpath landing on an emissive triangle) was accumulated at **full
  weight** while `connectT1` (the `t = 1` NEE) already counted the same area light
  power-heuristic-weighted, double-counting direct area-light transport — measured
  `mean(bdpt)/mean(ref)` ×1.76 on a *purely diffuse* scene (no delta bounces) and
  ×1.49 on `glass_arealight`. The corpus parity gate missed it because it
  exposure-aligns before comparing, dividing out a uniform scale. Fixed by gating
  the emissive eye hit exactly like the path tracer — full weight only at the
  primary/first hit (`s == 2`), a delta bounce into the light (`eye[s - 2].isDelta`),
  or a scene with no emissive-triangle NEE — in both `integrators/bdpt.slang` and
  `wavefront/wavefront_bdpt.slang`. Post-fix BDPT tracks the path tracer (energy
  ×0.88, `mean(bdpt)/mean(path)` ≈ 1.00) on Metal and Vulkan, megakernel and
  wavefront; raw relMSE vs pbrt 0.346 → 0.017. New un-aligned absolute-energy gate
  `tests/pbrt/test_bdpt_energy.py` locks it; the path tracer and corpus parity
  gates are unchanged (no regression).
- **BDPT display brighter than the path tracer** (change `bdpt-mis-unification`) —
  the `bdpt-energy-convergence` fix corrected the accumulation, but the *display*
  also composites the `s = 1` light-tracer splat (`main_pass.slang`), which was
  added at full weight and double-counted every diffuse path the eye side already
  had (displayed BDPT ~1.12× the path tracer on a pure-diffuse scene, ~1.19× on
  `bathroom.usda`). Two changes put every BDPT strategy into one MIS partition:
  `splatLightVertex` now MIS-weights each splat (`splatMisWeight`, with the camera
  as the `s = 1` eye endpoint and the pinhole camera-importance reverse pdf), and
  `connectT1` (`t = 1` NEE) now uses `misWeight` like the `t ≥ 2` connections
  instead of a standalone 2-strategy power heuristic (it previously over-weighted
  indirect transport ~2%). After both, BDPT tracks the path tracer (diffuse display
  ×1.12 → ×1.02, accum ×1.02 → ×0.99; bathroom ×1.19 → ×1.14, the residual being
  genuine caustic/glossy energy the path tracer is biased dark on). Both BDPT
  surfaces (megakernel `bdpt.slang` + wavefront via the shared `splatLightWalk` /
  `connectT1`). New **display** gate
  `test_diffuse_arealight_bdpt_display_tracks_path` (the accum gates exclude the
  splat); accum + convergence + corpus parity gates stay green. Splat camera
  importance is the analytic pinhole `We`; abstracting it over `ICamera` for
  thick-lens parity is a tracked follow-up.
- **Metal 31-buffer argument-table overflow** with neural online training on
  multi-graph scenes (change `combine-graph-param-buffers`). `--backend metal
  --execution-mode wavefront --neural-trainer mlx --neural-handoff interop
  --online-training` on a scene with ≥2 MaterialX graph materials (e.g.
  `three_materials_demo.usda`) crashed at pipeline creation
  (`wfPathShadeFlat` … `'buffer' attribute parameter is out of bounds: must be
  between 0 and 30` → `SLANG_FAIL`): each scene graph emitted its own
  `StructuredBuffer<GraphParams_X>` slot, an unbounded contributor that pushed
  the neural weight buffers past Metal's cap. Fix: collapse all scene graphs into
  one matId-major `ByteAddressBuffer graphParamsCombined` (read
  `Load<GraphParams_X>(matId * GRAPH_PARAM_STRIDE)`, scalar layout identical on
  Metal/SPIR-V — also retires the per-backend MSL graph-param repack), and fold
  the two env importance-sampling CDFs into one `envDistCdf` (binding 32 retired)
  to reclaim the last baseline slot. Vulkan and the megakernel are functionally
  unchanged (fewer descriptors).
- Transform gizmo (and the HUD) never drew in **wavefront** execution mode (so
  with ReSTIR on, which forces wavefront). The gizmo overlay was composited only
  in `main_pass.slang`, which wavefront skips — the wavefront display tail
  (`wavefront/wf_display.slang::wfWriteDisplay`) composited the HUD but
  deliberately omitted the gizmo. It now runs the same screen-space gizmo line
  composite (binding 22 / `numGizmoSegments`), so the editor gizmo is visible in
  wavefront mode (path, BDPT, and ReSTIR) just like megakernel. The segment list
  is already rebuilt + uploaded every frame regardless of mode.
- Transform gizmo never appeared when an analytic gprim (`UsdGeom.Sphere`,
  `Cube`, `Cylinder`, `Cone`, `Capsule`, `Plane`) was selected in the
  scene-graph dock. The loader tessellates those into renderable instances just
  like `UsdGeom.Mesh` prims, but the scene-graph builder and
  `populate_instance_refs` only attached the `instance` `renderer_ref` to nodes
  typed `"Mesh"`, so selecting a gprim cleared the gizmo target
  (`set_gizmo_target(-1)`). Both now match a baked instance by prim path
  regardless of prim type (`build_scene_graph` covers the reload path,
  `populate_instance_refs` the streaming path). Authored `Mesh` prims were
  unaffected.
- USD scenes used a degenerate `(0,1)` AABB for the neural-proposal condition's
  position normalisation (the per-frame `Scene` snapshot has no instances for USD —
  geometry streams straight to the GPU); `_neural_scene_bounds` now falls back to the
  streamed `_usd_scene` instances, used by both inference and the dump header.
- `Renderer.cleanup()` never freed the env importance-sampling CDF buffers
  (bindings 31/32) — fixed (surfaced by the record-dump's clean-teardown check).
- ReSTIR DI reuse mode (wavefront-only) — reservoir resampling of primary-hit
  direct lighting over the unified light set (sphere + emissive-triangle + env,
  light- and BSDF-sampled candidates) with deferred visibility. Spatial reuse
  uses the unbiased generalized balance heuristic (GRIS) and reduces variance on
  many-light scenes; the path tracer's depth-0 BSDF-hits-light terms are gated so
  ReSTIR owns primary direct (converges to stock NEE). Selectable regimes
  (default Spatial only; Spatial+Temporal / Temporal only are progressive-limited,
  reprojected temporal is a follow-on), a biased ΣM toggle, and live tuning
  (candidate counts, neighbours, radius, M-cap). Capability-gates to identity on
  megakernel/Metal. Built on the pluggable scene-sampling reuse seam.
- Environment importance sampling: equirect HDR sampled by a sin θ-weighted
  2D piecewise-constant distribution (CDF buffers at bindings 31/32) for env
  next-event estimation + MIS — both the path tracer and BDPT consume it
- GGX specular now uses visible-normal (VNDF) importance sampling
  (Heitz 2018/2023); the BRDF×cos/pdf weight reduces to F·G₁, eliminating
  grazing-angle specular fireflies
- BDPT connections evaluate the real `standard_surface` BSDF instead of a
  Lambertian approximation, and use the same env importance sampling as the
  path tracer so the two integrators converge to the same image
- Exposure (EV stops) and a selectable tonemap operator (ACES filmic /
  Reinhard / Hable / linear) as post-process knobs that do not reset
  accumulation

### Materials

- OpenPBR material support in the USD loader, including resolution of
  connected shader inputs to their authored constant
- UsdPreviewSurface texture bindings: per-input channel selection, normal-map
  `scale`/`bias` (OpenGL vs DirectX Y convention), wrap modes, and source
  colour space, carried on a new `TextureBinding` (`scene.py`)
- `FlatMaterialParams` grew 96 B → 128 B to carry `normalScale`, `normalBias`,
  and a packed `channelMask`
- Cutout vs alpha-blend opacity split in `fetchFlatHitData` to match
  UsdPreviewSurface `opacityThreshold` semantics
- Python-authored materials: SlangPile `python_materials/*.py` compile to GPU
  `IMaterial` structs dispatched as material type 3 (id in bits 24–31 of
  `materialTypes`), editable live in the Qt material editor
- Removed the dedicated `ProceduralParams` buffer (was binding 20); binding 20
  is now `DistantLight`

### Animation

- USD animation playback: a `PlaybackClock` maps wall-clock time onto the
  stage's authored time range and re-evaluates *cheap* time-sampled prims each
  frame — transform tracks, camera, and lights — by re-uploading only the TLAS
  instance records / light buffers (no mesh rebake). A load-time animated-prim
  index keeps per-frame cost proportional to the animated set. `current_time_code`
  feeds the accumulation hash, so playback renders at 1 spp and converges when
  paused (`playback.py`, `usd_loader.build_animation_index`)
- A `usd` camera mode follows an animated USD camera; the user can switch back to
  Orbit/Free at any time
- UsdSkel skeletal animation: skinned meshes deform per frame via linear blend
  skinning. CPU computes per-joint matrices (pxr, validated against
  `UsdSkelSkinningQuery.ComputeSkinnedPoints`); a GPU compute pass
  (`shaders/skin.slang`) blends rest vertices into the shared vertex buffer and a
  GPU BVH refit (`shaders/bvh_refit.slang`) keeps the path tracer correct over
  the deformed geometry — no readback. Standalone Vulkan pipelines
  (`vk_skinning.py`) with their own descriptor sets leave `main_pass` untouched;
  a CPU skinning path is the fallback on non-Vulkan backends

### Scene editing

- Runtime scene-graph editing API on `Renderer`: `add_model()` (reference a USD
  file under a parent prim), `remove_node()` (non-destructive deactivation),
  `set_transform()` (fast transform-only resync), `save_edits()`, and
  `list_nodes()`. The loaded `Usd.Stage` is the source of truth; edits are
  authored to an in-memory edit sublayer so the original file is never modified
  until `save_edits()`. `MeshInstance` now carries its `prim_path`, and
  `apply_instance_transform` / `apply_node_enabled` are keyed by prim path
- Scene-graph panels are now live editors in both front-ends (Qt dock + Panel
  card): an "Add model" picker (references a USD file under the selected group
  or `/World`), a "Delete node" action (context-menu + `Delete` key in Qt;
  button in Panel; guarded against synthesized `/Skinny/*` nodes), a "Save edits"
  button, and per-node transform edits that author to the stage (persisted by
  "Save edits"). Add/remove now rebuild the scene graph and re-read lights +
  cameras, so deleting a light or camera drops it and the panels refresh; lights
  carry a `prim_path` so a runtime enable-toggle survives an unrelated edit

### UI and interaction

- ReSTIR controls now live in a dedicated **ReSTIR** sidebar group — the `Reuse`
  selector plus the regime, biased-combine, M-light/M-bsdf, neighbours, radius,
  and M-cap tuning — split out of the general **Render** group and placed
  directly after it. Defined once in `ui/build_app_ui.build_main_ui`
  (`_classify`), so the Qt, web, and debug front-ends pick it up identically; the
  group stays visible at identity reuse so you can switch into ReSTIR from it
- Transform gizmo: the viewport manipulator now has four modes — rotate and
  translate, each in world or local space — cycled with `Space`
  (rotate-world → rotate-local → translate-world → translate-local). A `W`/`L`
  glyph above the pivot hints the coordinate space; rings vs arrows hint the
  type. The active mode persists in `~/.skinny/settings.json`. **`Space` no
  longer toggles the HUD — use `F1`.** Rotation drag is now a true axis-angle
  rotation about the (world or local) ring axis rather than a per-Euler-axis
  add, so world-axis rotation behaves slightly differently (more correct)
- Built-in animation transport (play/pause, normalized time scrubber, fps) in the
  shared control tree, shown only when the loaded stage has animation
- USD-driven Scene Controls: a stage can declare its own control panel via
  `skinny:ui:*` prims (slider/toggle/combo/color). Each control's prefix-typed
  target binds to a renderer parameter (`renderer:`/`mtlx:`), a material input
  (`material:`), or a USD attribute (`usd:`); editing a `usd:` control writes the
  stage and refreshes the live light/transform/camera state. Controls appear in a
  "Scene Controls" section across the Qt, web, and debug front-ends
  (`usd_loader.extract_ui_controls` + `resolve_control_binding`)
- Live Python material editing in the Qt material editor
- Camera debug viewport (`F2`) with frustum, lens rings, focus plane, DOF
  planes, render-area outline, ground grid, mesh wireframes, AABBs, and
  camera-body glyph
- Screen-space HUD inside the debug viewport listing its keyboard
  shortcuts; toggleable with `Space`
- Lens focus overlay (`L`), lens-vignette debug visualisation (`V`),
  zoom-rectangle drag (`Z` to arm, `X` to reset) hotkeys on the main
  window
- Updated on-screen HUD and `H` help text to list the full key set

### Tooling

- Unified render-selection flags across every front-end (`skinny`,
  `skinny-gui`, `skinny-web`, `skinny-render`), defined once in
  `skinny.cli_common`: `--integrator {path,bdpt}`, `--execution-mode
  {megakernel,wavefront}`, and `--bdpt-walk {fused,eye,eye_light}`. The
  interactive front-ends gained `--integrator` (sets the initial integrator,
  still runtime-cycleable); `skinny-render` gained `--execution-mode` /
  `--bdpt-walk`
- The wavefront-bdpt single-kernel subpath build is now `--bdpt-walk fused`
  (was `megakernel`), so `megakernel` names only the execution mode. The old
  `megakernel` walk value is still accepted as a deprecated alias for `fused`
  (CLI, `SKINNY_BDPT_WALK` env, and persisted values)
- Headless render API (`skinny.headless`) and `skinny-render` CLI for offscreen
  USD rendering — accepts a file path or a live `Usd.Stage` mutated per frame;
  saves PNG/JPEG/BMP/EXR/HDR or returns a numpy array; supports USD-time and an
  animation loop. `examples/render_image.py` and `examples/render_turntable.py`
  are thin wrappers over the new API.

### Fixes

- Windowed Vulkan swapchain is now created at the surface's `currentExtent`
  instead of the window point-size. On MoltenVK/Retina the two differ (backing
  pixels), which made `vkAcquireNextImageKHR` return `VK_SUBOPTIMAL_KHR` and the
  windowed app crash on the first frame; the offscreen→swapchain blit already
  scales, so decoupled render resolution is unaffected (`vk_context.py`)

## [0.1.0] - 2026-05-02

First release. Skinny is a physically based renderer built on a Vulkan compute
shader pipeline. It started as a human skin rendering testbed; this release
ships the full skin feature set alongside generic MaterialX material support
and OpenUSD scene loading.

### Rendering

- Three-layer biological skin model: epidermis (melanin absorption), dermis
  (hemoglobin + blood oxygenation + optional tattoo ink), subcutaneous fat
- GGX microfacet specular with Fresnel
- Point-BSSRDF subsurface scattering (quantized diffusion / normalized
  diffusion profiles)
- Delta-tracked (Woodcock) volume transport through the layered skin medium
- Scattering mode selector: BSSRDF + Volume, BSSRDF only, Volume only, Off
- Four sampling strategies: path tracing, MIS, bidirectional, stored BDPT
- Image-based lighting from Radiance HDR environments
- Analytic light (distant) and area lights (sphere, rect, emissive triangles)
- Statistical pore and vellus hair detail layer
- Furnace mode for energy conservation verification
- ACES filmic tone mapping

### MaterialX

- Custom nodedefs for skin layers: `ND_skinny_skin_epidermis`,
  `ND_skinny_skin_dermis`, `ND_skinny_skin_subcut`, plus generic
  `ND_skinny_scattering_layer` escape hatch
- `ND_skinny_layered_skin_stack` combiner producing `surfaceshader` +
  `volumeshader` outputs
- Function-form Slang implementations (`mtlx/genslang/`) referenced by
  `<implementation target="genslang">` tags
- MaterialX runtime (`materialx_runtime.py`) for document loading, Slang code
  generation, uniform block reflection, and scalar-layout buffer packing
- All skin biological parameters exposed as MaterialX inputs, driveable by
  constants or UV-sampled images

### OpenUSD

- USD scene loader (`usd_loader.py`): `UsdGeom.Mesh` triangulation, transform
  baking, `UsdShade.Material` binding resolution
- Light import: `DomeLight`, `DistantLight`, `SphereLight`, `RectLight`
- Per-prim material assignment with `materialId` dispatch
- Flat material path for `UsdPreviewSurface` and MaterialX `standard_surface`
  prims alongside skin materials
- Example scenes in `assets/`: Cornell box variants, demo head, dual-skin demo,
  sphere light demo, multi-material test scene

### Scene and geometry

- Scene graph abstraction (`scene.py`): `MeshInstance`, `Material`, `Light*`,
  `Scene` data classes
- TLAS + per-instance BLAS pool with transform and material ID
- OBJ mesh loading with optional midpoint subdivision and displacement baking
- BVH construction (median-split) for ray/triangle intersection
- Analytic SDF head fallback (Loomis-style proportions)
- Head texture maps: normal, roughness, displacement (auto-discovered by
  filename keyword)
- Tattoo support: alpha-driven ink density in dermis layer

### UI and interaction

- GLFW window with orbit and free camera modes
- Tk control panel with collapsible per-material sections
- Colour picker for light and material diffuse colour
- Light direction picker (hemisphere widget)
- Keyboard parameter navigation (Tab/Shift+Tab, arrow keys, number jump)
- Fitzpatrick I--VI presets (male/female variants, 12 total)
- User preset save/load to `~/.skinny/presets/`
- Persistent settings across sessions (`skinny.settings`)

### Infrastructure

- Vulkan 1.2 compute pipeline with scalar block layout
  (`VK_EXT_scalar_block_layout`)
- Descriptor indexing for bindless texture arrays
  (`VK_EXT_descriptor_indexing`)
- Slang shader compilation via `slangpy`
- Python 3.11+, packaged via `pyproject.toml` with `skinny` entry point
- Optional `usd-core` dependency (`pip install -e ".[usd]"`)
