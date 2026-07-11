# Design — spectral wavefront integrators

## Context

Spectral rendering is megakernel-only. `path_spectral.slang` (v1) and `bdpt_spectral.slang`
(change `spectral-bdpt-megakernel`) are **separate** integrator modules compiled under
`-DSKINNY_SPECTRAL` that carry `Spectrum` (float4) radiance/throughput, reuse the RGB flat
machinery for wavelength-independent geometry (directions, pdfs, MIS stay scalar), and recolor
per wavelength via the shared `integrators/spectral_flat_common.slang` helpers
(`flatBsdfResponseSpectral`, `spectralAllLightsNEE`, upsampling, exact conductor/blackbody/SPD
sources) + `spectrum.slang` (`SampledWavelengths`, `sampleWavelengths`,
`spectrumResolveToLinearSRGB`). Bindings 45–51 (sigmoid tables, D65, conductor η/k, emitter
records, illuminant SPDs, Planck scales) are already uploaded for every spectral session.

The wavefront runs **staged** compute dispatches, not one in-kernel loop. Per-path state lives
in GPU-resident stream buffers, sized in Python (`wavefront_layout.py`) from the Slang struct
byte layout, and threaded across kernels (generate → intersect → shade/logic → scatter →
resolve). The color carriers that must become spectral:

- **Path** — `WavefrontPathState` (`shaders/wavefront/wavefront_state.slang:15`): `float3
  throughput` (24..36), `float3 radiance` (36..48). One record per live lane.
- **BDPT** — `WfBdptAux` (`shaders/wavefront/wavefront_bdpt.slang:42`): `float3 escaped`,
  `radiance`, `ewThroughput`; per-vertex `WfBdptVertex.throughput`/`emission`; light-origin
  `beta`. Staged eye/light subpath kernels + a camera-splat kernel (`_wfbdpt_splat`).
- **SPPM** — `VisiblePoint` (`shaders/integrators/sppm_state.slang:50`): `float3
  beta`/`ld`/`tau`; and the **fixed-point** per-pass photon-deposit accumulator `SppmAccum`
  (`:79`): `uint phiR/phiG/phiB` atomically summed by the photon stage, decoded by
  `sppmDecodeFlux`. Photon flux `beta` in `sppmEmitPhoton` is `float3`.

The wavefront already compiles as **multiple variants** (neural `_L6B24H96_*`, fp16/fp32) with
per-variant `.spv` checked in — a spectral variant is one more compile define, exactly the
model spectral was built for.

Constraints: RGB `.spv` byte-identical for every wavefront kernel; Metal dispatch hygiene
(short staged kernels + kill harness — spectral kernels are longer); Metal record-drain +
indirect-dispatch CPU-readback fallback must carry the wider records; parity matrix must gate
the three new combos, never silently admit them.

## Goals / Non-Goals

**Goals:**
- `--spectral --execution-mode wavefront` renders `path`, `bdpt`, and `sppm` (flat materials,
  BSDF proposal, no reuse) on both backends with all v1 spectral features (sigmoid upsampling,
  exact conductor Fresnel, blackbody/authored SPD sources, hero-λ dispersion).
- Spectral **SPPM** works — the first spectral integrator with a photon pass (per-λ photon
  flux, per-λ gather, resolve at the visible-point measurement).
- Parity-matrix admission of the three combos under the standard dual gates; mega≡wave is now
  an assertable equivalence for spectral (not a recorded skip) once wavefront spectral lands.
- Remaining spectral rejections (neural proposal, ReSTIR reuse, skin/subsurface/volume) stay.

**Non-Goals:**
- No neural directional proposal, no ReSTIR reuse, no skin/subsurface/heterogeneous-volume
  under spectral — each still refused at startup.
- No change to RGB wavefront behavior; no change to the CIE tables or bindings 45–51. The
  spectral **film-resolve helper** (`spectrumResolveToLinearSRGB`) is reused unchanged, but
  spectral SPPM does introduce a new *fold order* — per-pass resolve into a spectral-invariant
  progressive τ (D5) — which is not the per-path resolve the path/BDPT integrators use. That
  fold order is in scope; the resolve math is not changed.
- No new spectral capability math — transport reuses the merged megakernel helpers verbatim.

## Decisions

### D1 — Retype the wavefront carriers to `Spectrum`, reuse the staged kernels

Unlike the megakernel (one kernel → a *separate* `*_spectral.slang` module preserves
byte-identity), the wavefront's color carriers are **shared stream structs** consumed by
~14 staged kernels per integrator. Duplicating every kernel spectrally is ~40 new kernels and
a second stream-layout path. Instead: retype the color fields to the existing `Spectrum`
typealias and add spectral branches inside the existing kernels.

- `Spectrum` (`common.slang:19`) is already `float4`+`SPECTRUM_N=4` under `SKINNY_SPECTRAL`
  and `float3`+`SPECTRUM_N=3` otherwise. So the color roles — `WavefrontPathState.throughput`/
  `radiance` (`wavefront_state.slang:18-19`), `WfBdptAux.escaped`/`radiance`/`ewThroughput`
  (`wavefront_bdpt.slang:49-55`), `BDPTVertex.throughput`/`emission` (`bdpt.slang:62-63`),
  `VisiblePoint.beta`/`ld`/`tau` (`sppm_state.slang:55-71`) — change `float3` → `Spectrum`.
  In the RGB build `Spectrum` **is** `float3`, so the struct layout and `.spv` are
  byte-identical (hostless guard per kernel). No `#if` on the field itself.
- The **added** per-lane state is `SampledWavelengths` (`spectrum.slang:27`, two `float4`),
  gated `#if defined(SKINNY_SPECTRAL)` on the carrier struct so RGB layout is untouched.
- The **arithmetic** sites gain a `.w` term **under `#if defined(SKINNY_SPECTRAL)`** rather
  than a rewritten `SPECTRUM_N` loop: the `wfFinishShade` Russian-roulette max-component
  (`wf_shade_common.slang:135`) keeps its exact RGB `max(max(x,y),z)` expression textually
  unchanged and adds `max(·, w)` only under the spectral define — a rewritten loop risks a
  non-bit-identical N=3 lowering (review, minor). The RR survival `p` must match
  `path_spectral`/`bdpt_spectral` (`max(…,.w)`) for the anchor to hold. The resolve
  (`wfPathResolve`) gains `spectrumResolveToLinearSRGB` at the film write.
- `wavefront_layout.py` (`FIELDS`/`VISIBLE_POINT_FIELDS`/`SPPM_ACCUM_FIELDS`), its
  **scalar + MSL** stride asserts, `tests/test_wavefront_state.py`, and both backends'
  stride checks (`metal_wavefront.py:515-526`) move in lockstep. Note the two variants diverge
  **by construction** (review, minor): scalar layout grows +4 B per retyped color field
  (`float3`→`float4`) plus the `SampledWavelengths`; MSL already pads `float3`→16 B so the
  color retype is **0-byte** on Metal and only `SampledWavelengths` (32 B) grows the MSL
  stride. The asserts must expect the divergence, not equal strides.

*Alternative — separate spectral kernels/structs (the megakernel approach)*: rejected —
40 duplicated kernels, a parallel layout path, and a second variant matrix; the
retype-plus-compile-define branch is the model the wavefront's neural/fp16 variants already use.

### D2 — Hero wavelengths drawn once at raygen, carried in the path-state record

For path/BDPT, `wfPathGenerate`/`wfBdptGenEye` draws
`SampledWavelengths sw = sampleWavelengths(rng.next())` **after** `cam.generateRay` (matching
the megakernel/`path_spectral` ordering — generateRay consumes the RNG first) and stores it in
the widened record. Every downstream stage (intersect/shade/scatter/connect/resolve) reads `sw`
from the record — the 4 wavelengths are fixed for that camera path's life; BDPT's light subpath
shares the eye path's `sw` so connections form the per-λ product. (The `#if`-guarded draw keeps
the RGB stream byte-identical; the mega≡wave anchor is tolerance-based convergence, not
stream-identity, so exact ordering only matters for correctness, not for a bit match.) SPPM's
wavelength scoping is per-*pass*, not per-path — see D5.

### D3 — Scalar pdfs and MIS, per-λ responses (same split as the megakernel)

Directions, solid-angle pdfs, and MIS weights come from the unchanged RGB `FlatMaterial`
sample/pdf machinery; only the color-carrying evaluations recolor via
`spectral_flat_common.slang`. This keeps wavefront spectral MIS-consistent with megakernel
spectral (same hero-wavelength estimator), which is exactly what the self-consistency gate
(mega≡wave) will now assert.

### D4 — BDPT camera splat resolves λ → linear sRGB before the atomic add

The `_wfbdpt_splat` kernel computes its contribution as a `Spectrum`, applies
`spectrumResolveToLinearSRGB(sw, contribution)`, and atomic-adds the resulting `float3` into
the existing splat buffer — identical to the megakernel spectral splat (change
`spectral-bdpt-megakernel` D4), inheriting the same per-splat clamp-bias note.

**Mega≡wave equivalence caveat (review, major).** `spectrumResolveToLinearSRGB` clamps to gamut
(`max(·,0)`) per value, and the splat resolves+clamps **per splat** before the atomic add. The
clamp is nonlinear, so `E[clamp(splat)]` depends on the per-splat distribution, which differs
between the fused megakernel and the staged wavefront (different splat granularity). Two
spectrum-unbiased estimators therefore converge to **different** clamped images exactly on the
out-of-gamut hero-collapsed dispersion caustics the prism scene targets. So spectral BDPT
mega≡wave is asserted tight only on **in-gamut** (non-dispersion) scenes; on the dispersion
splat scene it is a **recorded self-consistency skip** with the reason above (path has no splat
→ always tight). The full linear fix — accumulate **signed XYZ (3-wide)** and clamp once at
composite — is deferred to a splat-format change taken on gate evidence (megakernel + wavefront
together), out of scope here.

### D5 — Spectral SPPM: **one shared hero-λ set per pass**, resolve before the progressive fold

The hard part, and the review's critical correction. A photon deposit computes
`phi = photon_beta ⊗ f_r(VP)` (`wavefront_sppm.slang:525`), and **one photon deposits onto
many visible points**. If photons drew wavelengths at emission and each VP drew its own at the
eye pass, the product `photon_beta[k]·f_VP[k]` would multiply flux at λ_photon by a BSDF at a
**different** λ_VP — physically incoherent — and `VisiblePoint.tau` **persists across passes**,
so per-pass-rotating λ would sum flux measured at different wavelengths into the same channel.
Both are wrong.

**Design: hero wavelengths are drawn once per SPPM *pass*, shared by every photon and every
eye visible point of that pass.**
- A pass-global `SampledWavelengths sw` (seeded from the pass index) is used by
  `sppmEmitPhoton` (`Spectrum beta`, light SPD upsampled as illuminant / exact source) **and**
  by the eye pass's VP BSDF recolor at the **same** 4 λ — so `phi = beta ⊗ f_r` is a coherent
  per-λ product. The photon-deposit BSDF `mat.evaluate(...).response` recolors per-λ via
  `spectral_flat_common` at `sw` (task owns it explicitly).
- `SppmAccum` grows from 3 fixed-point channels (`phiR/G/B`) to **4** (`phi0..phi3`), atomically
  summed at deposit; `sppmDecodeFlux` returns a `Spectrum`. Flux is non-negative per λ, so
  4-wide unsigned fixed point is clean (no signed-gamut problem — contrast the BDPT splat).
- **Resolve λ→linear sRGB (or XYZ) at the end of each pass, *before* folding into the
  progressive estimator.** `VisiblePoint.tau` stays a spectral-**invariant** RGB/XYZ quantity;
  the per-pass per-λ `phi` is resolved at `sw` then `sppmUpdate`'s progressive reduction runs on
  the resolved 3-wide value (the `(r'/r)²` ratio is scalar geometry — unchanged). Rotating `sw`
  across passes then integrates the spectrum exactly as the megakernel integrates λ across
  frames per pixel. `VisiblePoint.beta`/`ld` are per-pass spectral (resolved into the pass's
  direct term); only `tau` and the estimator persist, and they persist as resolved RGB/XYZ.
- The fixed-point scale `SPPM_FLUX_FIXED_SCALE` is re-checked for per-λ range (hero-collapsed
  dispersion concentrates flux into one λ) against the numpy mirror — recorded risk, gated.

This is a genuine change to *how spectral radiance folds into the SPPM accumulation* (see the
reconciled Non-Goal) — the pass loop is the spectral integrator for SPPM, distinct from the
per-path resolve the path/BDPT integrators use.

### D6 — Startup gate + renderer pin widen together

`reject_spectral_unsupported` drops the `wavefront`-refused and `sppm`-refused branches; it
now accepts `path`/`bdpt`/`sppm` under `wavefront` (still refusing neural proposal, reuse,
non-BSDF proposal, and skin/volume scenes). The renderer's spectral integrator pin
(`_active_integrator_index`, execution-mode resolution, `_collect_config_rows`) widens to admit
`{path, bdpt, sppm}` under wavefront so the compiled spectral kernels are reachable — without
this the new branches are dead code. `resolve_execution_mode` already sends `sppm` → wavefront.

### D7 — Parity matrix: three combos in, a **spectral-aware anchor**, mega≡wave assertable

`parity.spectral_envelope`/`combo_is_valid` admit `(path|bdpt|sppm, wavefront, spectral)`.

**The harness anchor must become spectral-aware (review, major — missing task).** Today
`parity.ANCHOR` is the single RGB `(path, wavefront)` image and `combo_axis_class` compares
every combo against it. Comparing a **spectral** combo to the **RGB** anchor conflates the
RGB→spectral delta with the integrator/mode delta — invalid. So this change adds a per-variant
anchor: for the spectral axis the anchor is the **megakernel spectral path** image (which this
change makes renderable in wavefront too). Then:
- Spectral wavefront **path/bdpt** anchor to the spectral megakernel path image
  (self-consistency) — the existing "mega≡wave is a spectral-only skip" is **lifted** (except
  the BDPT dispersion-splat skip, D4).
- Spectral **SPPM** anchors to the **spectral path** anchor at the `sppm` tolerance class —
  **not** the RGB golden (review, major: spectral ≠ RGB by construction on spectrum-authored
  scenes, and RGB↔spectrum round-trip is not identity). If spectral-path-anchoring proves
  infeasible for SPPM, the fallback is **pbrt-truth-only** gating with the RGB-golden
  self-consistency claim dropped.

Neural/reuse/skin spectral stay recorded exclusions.

## Risks / Trade-offs

- **RGB byte-identity blast radius.** ~14 kernels × 3 integrators gain `#if SKINNY_SPECTRAL`
  branches; any stray edit outside the guard breaks a checked-in `.spv`. **No byte-identity
  guard exists today** (review, major): `test_spectrum_compile.py` only checks the variants
  *compile*, not that RGB output is unchanged. Mitigation is **net-new** infrastructure — a
  before/after **compile-and-compare** for each touched wavefront kernel (compile RGB at the
  pre-edit and post-edit source, assert byte-equal), *not* "equals the checked-in `.spv`"
  (that won't survive a slangc bump). Explicit task (8.1).
- **SPPM fixed-point flux precision (D5).** 4-wide per-λ atomics with a hero-collapsed
  dispersion spike can overflow/underflow the fixed-point scale differently than RGB.
  Mitigation: re-measure `SPPM_FLUX_FIXED_SCALE` under spectral against the numpy mirror; gate
  the dispersion suite scene.
- **Stream memory growth.** The widened record (Spectrum + `SampledWavelengths` ≈ +40 B/lane)
  raises per-lane stream memory; at a fixed stream size, spectral uses more. Mitigation: the
  tiled-streaming requirement already bounds by stream size; trim the spectral default if a
  Metal argument-table / buffer-size limit is hit, recorded.
- **Metal watchdog.** Spectral shade/connect/photon kernels are longer per lane. Mitigation:
  the wavefront's naturally-short staged kernels are the watchdog-friendly path; kill harness
  (`tests/test_metal_cleanup.py -m gpu`) must pass with the spectral variant compiled.
- **Metal indirect-dispatch CPU fallback + record-drain** must carry the wider records — a
  latent stride bug there shows only on Metal. Mitigation: Metal self-consistency A/B in the
  GPU gate, not just Vulkan.
