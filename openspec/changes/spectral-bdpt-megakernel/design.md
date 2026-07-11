# Design — spectral BDPT (megakernel)

## Context

Spectral v1 (`spectral-rendering`, merged) renders `(path, megakernel, flat)` under
`--spectral`: `integrators/path_spectral.slang` is a **separate** integrator compiled only
under `-DSKINNY_SPECTRAL`, carrying `Spectrum` (float4) radiance/throughput while reusing the
RGB flat machinery for wavelength-independent geometry (sampled `wi`, solid-angle pdfs, MIS
weights all stay scalar) and recoloring per wavelength via `flatBsdfResponseSpectral` plus the
upsample/SPD helpers (bindings 45–51: sigmoid tables, D65, conductor η/k, emitter records,
illuminant SPDs, Planck scales). The RGB build never imports the module, so its SPIR-V is
byte-identical — a hostless test guards this.

The RGB BDPT megakernel (`integrators/bdpt.slang`, 1118 lines) is already flat-only: a
non-flat first hit bails, `BDPTVertex.throughput` is `float3`, strategies are eye random walk,
light random walk, s≥2/t=0 emissive hits, t=1 NEE (`connectT1`), t≥2 generic connections
(`connectGeneric` + `misWeight`), and the s=1 light tracer (`splatLightWalk`) that atomic-adds
RGB into `lightSplatBuffer` composited by `main_pass.slang`.

The spectral branch of `main_pass.slang` currently dispatches `SpectralPathTracer`
unconditionally — there is no BDPT branch under `SKINNY_SPECTRAL`.

Constraints: RGB SPIR-V byte-identical; Metal dispatch hygiene (megakernel row-band tiling +
kill harness — BDPT-over-flat already wedged the GPU once, change
`metal-megakernel-watchdog-tiling`); parity matrix must gate the new combo, never silently
admit it.

## Goals / Non-Goals

**Goals:**
- `--spectral --integrator bdpt` renders on both backends (megakernel, flat materials, BSDF
  proposal, no reuse) with all v1 spectral features: sigmoid upsampling, exact conductor
  Fresnel, blackbody Planck emission, authored illuminant SPDs, hero-λ glass dispersion.
- All five BDPT strategy families transport spectrally: eye walk, light walk, emissive hits,
  t=1 NEE, t≥2 connections, s=1 camera splat.
- Parity-matrix admission with the standard dual gates; SPPM stays a recorded exclusion.

**Non-Goals:**
- No spectral wavefront (prerequisite for SPPM spectral — separate change).
- No SPPM spectral, no skin/subsurface/volume spectral, no neural/ReSTIR under spectral.
- No change to the spectral film resolve, accumulation, or splat compositing pipelines.
- No RGB BDPT behavior change of any kind.

## Decisions

### D1 — Separate `bdpt_spectral.slang`, not a widened shared BDPT

Mirror the `path_spectral.slang` precedent exactly: a new module compiled only under
`SKINNY_SPECTRAL` with its own `SpectralBDPTVertex` (Spectrum `throughput`/`emission`) and
spectral strategy functions.

- *Alternative — widen `BDPTVertex.throughput` to Spectrum in `bdpt.slang`*: breaks the RGB
  build's byte-identical guarantee and every float3 assignment across the connection code;
  rejected for the same reason v1 rejected widening the shared carriers.
- *Alternative — Slang generics over a color carrier*: `bdpt.slang` is deeply imperative
  (fixed vertex arrays, atomic splats); a generic carrier would still change the RGB
  compilation output. Rejected.

The MIS code cannot simply be called from the spectral module: `misWeight`,
`splatMisWeight`, and `convertSAtoArea` take full `BDPTVertex` / `BDPTVertex[7]` arrays with
the float3 colors embedded. Options: (a) duplicate the MIS chain over `SpectralBDPTVertex`,
(b) keep parallel RGB `BDPTVertex` arrays purely for pdf bookkeeping, (c) refactor
`bdpt.slang` onto a color-free pdf view (touches the RGB compile → byte-identity risk).
**Decision: (a) — duplicate, with a drift guard**: a hostless test asserts the spectral MIS
copy stays in sync with the RGB original (textual slice compare over `DOC:`-style markers, or
a shared color-free struct extracted later). (c) stays a post-change refactor opportunity
once the byte-identity guard can be re-baselined deliberately. Only color-carrying paths
(walk throughput updates, `connectT1`, `connectGeneric` contribution, splat) are rewritten
spectrally.

### D2 — One hero-wavelength draw per pixel path, shared by both subpaths

`SampledWavelengths sw = sampleWavelengths(rng.next())` is drawn once in
`estimateRadiance` (first RNG consumption, matching `path_spectral` so the RGB stream is
untouched) and threaded through the eye walk, light walk, every connection, and the splat.
Both subpaths of one sample transport the same 4 wavelengths — required for connections to
form the per-λ product `f_eye(λ)·G·f_light(λ)·L(λ)`.

### D3 — Scalar pdfs and MIS, per-λ responses (same split as v1)

Directions and pdfs come from the RGB `FlatMaterial.sample()`/pdf machinery; `misWeight` and
the pdf-rev chain are unchanged scalar code. Only responses recolor: connection endpoints
evaluate `flatBsdfResponseSpectral` (via the hoisted helpers, D5), light radiance upsamples
as illuminant (or takes the exact Planck/SPD override), surface reflectances upsample bound.
This keeps spectral BDPT MIS-consistent with spectral path (same estimator family as Wilkie
et al. hero-wavelength MC).

### D4 — Splats resolve to linear sRGB before the atomic add

`splatLightVertex` computes its contribution as a Spectrum, then applies
`spectrumResolveToLinearSRGB(sw, contribution)` and atomic-adds the resulting float3 into
`lightSplatBuffer` exactly as today, keeping the splat buffer format, compositing, and
readback untouched.

This is **not** exactly equivalent to accumulating spectra and resolving later: the resolve
is CMF dot products (linear) **followed by a gamut clamp** (`max(xyzToLinearSRGB(…), 0)`),
and the splat buffer itself is unsigned Q22.10 fixed point — clamp-per-splat ≠ clamp-of-sum.
A hero-collapsed (near-monochromatic) dispersion splat resolves outside the sRGB gamut, so
dispersed-caustic splats carry a systematic per-splat clamp bias. Accepted for v1 because it
is the same estimator the eye side already uses (per-sample resolve+clamp into the RGBA32F
accumulation) — the two sides stay consistent — and the bias is recorded at the splat site.
**Escalation path** if the dispersion-caustic splat gate shows hue/energy bias vs pbrt:
accumulate signed **XYZ** splats (the large negatives only appear after the XYZ→sRGB matrix;
still 3-wide) and convert at composite — a splat-format change, taken only on gate evidence.

- *Alternative — a 4-wide spectral splat buffer + resolve pass*: new binding, new pass;
  rejected for v1 (the signed-XYZ option above dominates it if escalation is needed).

### D5 — Hoist shared spectral-flat helpers out of `path_spectral.slang`

`SpectralFlatColors`, `upsampleFlatColors`, `flatResponseS`, `flatResponseNEE`,
`spectralLightNEE`, `spectralAllLightsNEE`, and the blackbody/SPD override lookups move to a
shared spectral module (e.g. `integrators/spectral_flat_common.slang`) imported by both
spectral integrators. Pure code motion for the path integrator — its spectral render output
must be bit-unchanged (guarded by an A/B in the tasks).

### D6 — Light-subpath origin emission is spectral at the source

`sampleLightOrigin`'s β seed upsamples the light's RGB radiance as an illuminant, with the
exact-source overrides applied where they can actually transport: blackbody emissive
triangles and sphere lights seed their spectral radiance at the origin (`planckSpectrum(sw,
T)·scale` for blackbody; upsample otherwise) — observable through splats and t≥2
connections. Directional lights get **no** light walk (`BDPT_VK_LIGHT_DIR` origins spawn no
random walk, and a lone origin vertex feeds no strategy — connections need t≥2, the splat
needs a surface vertex), and there is no environment light-subpath origin at all — so
authored-SPD distant lights and env metamerism reach the image **only via t=1 NEE and the
eye walk's env/light hits**, which already carry the v1 overrides. This keeps light subpaths
metamerism-consistent with NEE to the same emitter without inventing transport the RGB BDPT
doesn't have.

### D7 — Dispersion: hero-λ collapse on either walk

The dispersive delta-refraction rule from `path_spectral.slang` (Cauchy `n(λ0)`, refract at
hero λ, `terminateSecondary(sw)`, achromatic TIR fallback) applies wherever a walk crosses a
dispersive interface — eye or light subpath. `sw` is shared (D2), and the **evaluation order
is pinned**: eye walk → light walk → all s=1 splats → all connections. A collapse during
either walk therefore hero-collapses every splat and connection (they all evaluate after
both walks complete), including strategies whose own path never crossed the dispersive
interface — unbiased under the uniform hero-rotation argument, but ~4× spectral variance on
dispersion-dominated scenes. Contributions deposited *during* a walk before the collapse
(env-miss, sphere-light hits, escaped radiance) keep their 4-λ weights; v1 validated these
semantics for the unidirectional walk only (pbrt-v4 has no BDPT — there is no direct pbrt
anchor for the bidirectional collapse), so the prism pbrt-truth gate must expect a
noisy-but-unbiased result at matched sample counts, not misread variance as bias.

### D8 — Gates widen by wording, not by new mechanism

`reject_spectral_unsupported` changes its integrator check from `== "path"` to
`in ("path", "bdpt")` (SPPM refusal wording names the wavefront follow-up);
`parity.spectral_envelope` admits BDPT into the envelope; combo enumeration then emits
`(bdpt, megakernel, spectral)` automatically. `SPECTRAL_IMPLEMENTED` stays the single
capability gate (already True). No new flags, no new capability constants.

**Critically, the renderer-side integrator pin must widen too**: `_active_integrator_index`
returns `0 if self._spectral` today — what `_pack_uniforms` writes into `fc.integratorType` —
so without widening it to `{path, bdpt}` the new shader branch is dead code and every
spectral session still dispatches `SpectralPathTracer` (the BDPT-vs-path anchor A/B would
then trivially pass — a silent green). SPPM (and any future integrator index) stays pinned →
path, mirroring the RGB megakernel's silent path fallback for non-BDPT types. The inlined
second copy of the pin in `_collect_config_rows` (kept field-only for the observability
tests) widens identically and keeps reporting the pin row when a pin actually applies.
Runtime path↔bdpt switching in a spectral session works via the existing accumulation reset
(`_current_state_hash` includes `integrator_index`; the reset clears `light_splat_buffer`,
so bdpt→path leaves no stale splats).

## Risks / Trade-offs

- [BDPT MIS bug hidden by spectral recoloring] → self-consistency gate: spectral BDPT vs
  spectral path anchor on suite scenes reachable by both (relMSE), plus RGB-BDPT vs
  spectral-BDPT on a metamer-free scene (spectral of an sRGB-lit sRGB scene ≈ RGB). The
  anchor A/B is only meaningful once the renderer pin is widened (D8) — the pin task is
  ordered before the gate tasks.
- [Duplicated MIS chain (D1a) drifting from the RGB original] → hostless drift-guard test
  (textual slice compare); genericizing `bdpt.slang` recorded as post-change refactor.
- [Megakernel register/length growth — Spectrum vertex arrays (2×7 float4 throughputs) on top
  of BDPT's already-heavy kernel wedging Metal] → existing row-band watchdog tiling covers
  dispatch length, **but the band budget (`_METAL_MEGAKERNEL_BAND_PIXELS`) was sized for RGB
  BDPT and is keyed by integrator only — re-measure it under `SKINNY_SPECTRAL`**; kill
  harness (`tests/test_metal_cleanup.py -m gpu`) is a required task; if occupancy regresses
  badly, reduce spectral BDPT depth only (recorded, gated — and re-record the anchor A/B's
  sampling effort, which the shared depth constants otherwise match: `BDPT_MAX_DEPTH` =
  `SPECTRAL_MAX_BOUNCES` = 6 today).
- [Helper hoist (D5) silently perturbing spectral path output] → primary gate = byte compare
  of the *spectral* SPIR-V before/after the code motion; fixed-seed image A/B as fallback
  (slangc may reassociate across module boundaries, so image identity alone is over-strict
  in one direction and under-diagnostic in the other).
- [Per-splat gamut clamp bias on dispersed caustics (D4)] → recorded at the splat site;
  dispersion-caustic splat scene gated vs pbrt; signed-XYZ splat accumulation is the
  evidence-triggered escalation.
- [Light-walk dispersion order effects (collapse after some walk deposits already recorded)]
  → evaluation order pinned in D7; weights before termination stay valid (pbrt
  terminateSecondary semantics, unidirectionally validated in v1); prism scene gets a
  pbrt-truth gate under BDPT with variance expectations recorded.
- [Backend divergence (Vulkan vs Metal) in the new branch] → gated Vulkan≡Metal A/B on a
  spectral BDPT suite scene, mirroring the v1 path requirement's backend-parity scenario.

## Open Questions

- Whether `connectGeneric`'s co-planar/Lambertian-eval shortcuts need per-λ opacity handling
  beyond `flatResponseNEE` (resolve during implementation against `FlatMaterial.evaluate`).
- Exact split of hoisted module vs `import path_spectral` re-export — settle at
  implementation time; the A/B gate (D5) is what matters.
