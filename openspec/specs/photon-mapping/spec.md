# photon-mapping Specification

## Purpose
TBD - created by archiving change photon-mapping-sppm. Update Purpose after archive.
## Requirements
### Requirement: SPPM integrator selectable, wavefront-only

The renderer SHALL provide a Stochastic Progressive Photon Mapping (SPPM)
integrator selectable as a third integrator (`INTEGRATOR_SPPM`) alongside the
path tracer and BDPT, wired end to end through the same selection seam: the
shared CLI `--integrator sppm`, the GUI integrator list, and the `integratorType`
GPU dispatch. SPPM SHALL run only under the wavefront execution mode; a
megakernel selection of SPPM SHALL be refused or fall back with a clear message,
the same way the neural directional proposal is wavefront-gated. Switching to or
from SPPM SHALL reset progressive accumulation (the integrator index is part of
the accumulation state hash).

#### Scenario: SPPM is selectable from every front-end

- **WHEN** any front-end is launched with `--integrator sppm --execution-mode wavefront`
- **THEN** the renderer starts on the SPPM integrator, and on the interactive
  front-ends the integrator remains runtime-cycleable among path / bdpt / sppm

#### Scenario: SPPM under the megakernel is refused

- **WHEN** a front-end is launched with `--integrator sppm --execution-mode megakernel`
- **THEN** it exits (or refuses to start SPPM) with a clear message that SPPM
  requires `--execution-mode wavefront`, naming the fix, without initializing a
  broken pipeline

#### Scenario: Switching to SPPM resets accumulation

- **WHEN** the interactive renderer is cycled from path or bdpt to sppm
- **THEN** the progressive accumulation buffer resets to frame 0 and begins a
  fresh SPPM accumulation

### Requirement: Per-pass SPPM pipeline produces a consistent caustic estimate

Each SPPM accumulation pass SHALL execute four stages over the wavefront driver:
(1) an **eye stage** that traces one stochastic camera path per pixel through
specular/perfectly-glossy bounces and stores a single visible point at the first
non-specular surface hit (position, shading normal, path throughput to that
point, **the full evaluated flat material — every `FlatHitMat` field the lobe
model reads, per the visible-point mirror requirement —** search radius, photon
count, accumulated flux);
(2) a **grid-build stage** that indexes visible points in a uniform spatial hash
by a deterministic counting sort (per-cell count → prefix sum → scatter);
(3) a **photon stage** that emits photons from the scene lights using the
existing power-weighted emissive / light importance distribution, traces them
with Russian roulette, and deposits flux into every visible point found within
its radius via the grid, **evaluating f_r from the visible point's stored
material fields only (no per-photon texture refetch or graph re-run)**; and
(4) an **update stage** that applies the SPPM radius-reduction and
flux-accumulation rules per visible point and resolves each pixel's radiance
estimate. Direct lighting SHALL continue to use the existing NEE path; SPPM
SHALL contribute the indirect / caustic term. The estimator SHALL be
**consistent**: as the per-pixel search radius shrinks across passes, the SPPM
contribution SHALL converge toward the reference solution.

#### Scenario: Radius shrinks monotonically across passes

- **WHEN** SPPM runs for multiple accumulation passes on a fixed scene and camera
- **THEN** each visible point's search radius is non-increasing across passes per
  the SPPM reduction rule, and the accumulated photon count is non-decreasing

#### Scenario: Indirect caustic estimate converges as radius shrinks

- **WHEN** a glass-over-diffuse caustic scene is rendered with increasing pass
  budgets (shrinking radius)
- **THEN** the SPPM caustic estimate's error against the reference trends
  downward (a radius-sweep convergence trend), rather than plateauing at a biased
  value

#### Scenario: Direct lighting is not double-counted

- **WHEN** a scene with both direct illumination and caustics is rendered under SPPM
- **THEN** direct lighting is supplied by NEE and the photon contribution adds the
  indirect/caustic term only, with overall energy matching the reference within
  the parity tolerance (no double-counted direct term)

### Requirement: SPPM scope is surface flat materials; skin and volumetric are deferred

PM-1 SPPM SHALL apply to flat materials only (`UsdPreviewSurface`,
`standard_surface`, `OpenPBR`, and Python flat materials), using the same
material gate as neural guiding and the BDPT flat path. The layered skin /
BSSRDF estimator chain SHALL be left untouched by this change. Photon transport
through participating media / volumes SHALL NOT be attempted in PM-1. These two
extensions — skin/BSSRDF photon deposition and volumetric/media photon transport
— are explicitly deferred to follow-up changes that extend this same capability.

#### Scenario: SPPM on a flat-material scene

- **WHEN** SPPM renders a scene whose surfaces are all flat materials
- **THEN** the integrator runs and produces a caustic estimate

#### Scenario: Skin and volume paths are unchanged

- **WHEN** the SPPM change is applied
- **THEN** the layered skin/BSSRDF estimator chain and the volume-transport path
  produce byte-identical output to before the change for path/bdpt rendering, and
  SPPM does not route photons through them

### Requirement: SPPM runs on both backends, Metal first, at parity

SPPM SHALL be implemented and verified on the native Metal backend first and
SHALL reach parity on the Vulkan backend within this change, so both backends
produce equivalent results within the parity tolerance. The new GPU buffers
(visible points, spatial-hash grid, photon records) SHALL be folded so the Metal
argument-table 31-slot limit is not exceeded even with SPPM active alongside
multiple graph materials. Per-pass queue compaction MAY use the Metal CPU
read-back fallback already used by the wavefront driver; this cost SHALL be
documented in the compatibility matrix.

#### Scenario: SPPM compiles and renders on Metal with multiple graph materials

- **WHEN** SPPM is active on the Metal backend in a scene with two or more
  MaterialX graph materials
- **THEN** the Metal pipeline compiles and dispatches without exceeding the
  argument-table slot limit

#### Scenario: Metal and Vulkan SPPM agree on the caustic scene

- **WHEN** the caustic parity scene is rendered under SPPM on Metal and on Vulkan
  at the same pass budget
- **THEN** the two outputs agree within the parity tolerance

### Requirement: pbrt importer maps the sppm integrator

The pbrt→USD importer SHALL recognize `Integrator "sppm"`: it SHALL translate
the integrator parameters (`numiterations`, `maxdepth`, `photonsperiteration`,
`radius`, `seed`) into USD metadata **and** select the skinny SPPM integrator on
the imported stage. The translation report SHALL classify `sppm` as *mapped*
(for the surface case) rather than *skipped*, and the prior "sppm / photon
integrators out of scope" limitation SHALL be lifted for surfaces. When the pbrt
`radius` parameter is present it SHALL seed the initial SPPM search radius;
otherwise a scene-bounds-derived default SHALL be used.

#### Scenario: Importing an sppm scene selects the SPPM integrator

- **WHEN** a pbrt scene declaring `Integrator "sppm"` is imported
- **THEN** the resulting USD stage carries the sppm parameters in metadata and
  records the skinny SPPM integrator as the selected integrator, and the report
  lists the integrator as mapped

#### Scenario: sppm radius parameter seeds the initial search radius

- **WHEN** the imported sppm integrator specifies a `radius` parameter
- **THEN** the SPPM initial search radius is seeded from that value rather than
  the scene-bounds default

### Requirement: Caustic parity gate against pbrt v4 sppm

The change SHALL include a caustic parity scene (a glass object over a diffuse
plane) with a pbrt v4 `sppm` reference image, compared to skinny's SPPM output
through the existing parity harness using relMSE and FLIP thresholds at a fixed
pass budget, on both Metal and Vulkan. The comparison SHALL also report an
energy ratio to guard against gross bias, and SHALL surface a labelled
side-by-side image (reference · skinny) artifact.

#### Scenario: SPPM matches the pbrt sppm reference within tolerance

- **WHEN** the caustic parity scene is rendered under skinny SPPM and compared to
  the pbrt v4 sppm reference at the gate's pass budget
- **THEN** relMSE and FLIP are within the configured thresholds and the energy
  ratio is within its tolerance band, on both Metal and Vulkan

### Requirement: SPPM reconstructs glossy / near-specular reflectors

SPPM SHALL reconstruct sharp inter-object reflections on glossy, near-specular
flat reflectors (e.g. polished metals) rather than losing them to the photon
gather. The eye-walk continue-vs-store decision SHALL treat a sampled lobe whose
roughness is below a configurable threshold (`sppmGlossyContinueRoughness`) as a
caustic carrier: the walk SHALL follow the BSDF-sampled direction one bounce and
store the visible point at the next surface that is not itself glossy-continued,
so the reflection is reconstructed at the reflected surface and accumulated
across progressive passes. A glossy-continued vertex SHALL be treated like a
specular vertex by the photon stage (no photon deposit at it), preserving the
disjoint direct (NEE) / indirect (photon) split. A threshold of `0` SHALL
reproduce the prior delta-only behavior.

The default threshold SHALL be expressed in perceptual (USD) roughness and SHALL
reach pbrt-imported polished metals: pbrt perceptual roughness `r` imports as
`usd = r**0.25`, so the default SHALL be high enough that a polished
pbrt-roughness-`0.1` conductor (usd `≈ 0.562`) is glossy-continued while a
pbrt-roughness-`0.3` metal (usd `≈ 0.740`) remains on the photon-gather side.

A glossy-continued (non-delta) carrier vertex runs both env and emissive-triangle
NEE, so any light its carrier ray subsequently reaches SHALL be MIS-weighted
against that NEE (never taken at full weight, which would double-count):

- On an **environment escape**, the escaped-env radiance SHALL be weighted by
  `powerHeuristic(bsdfPdf, envPdf(dir))`.
- On an **emissive-triangle hit**, the emission SHALL be weighted by
  `powerHeuristic(bsdfPdf, pdfLightSA)`, where `pdfLightSA` is the NEE solid-angle
  pdf reconstructed without the triangle index (`lum·d² / (emissiveTotalPower·cosLight)`),
  matching the path tracer's emissive-hit MIS.

Only a **perfectly-specular (delta)** carrier (`bs.pdf <= 0`) has no NEE partner
and SHALL take the reached light at full weight; transmitted lobes and furnace
mode SHALL suppress the env companion. (Prior to this change the eye walk treated
every continued vertex — glossy included — as delta for the emissive-hit gate,
which double-counted an emitter seen in a glossy metal; the higher default
threshold makes that path reachable, so it is corrected here.) This makes a glossy
metal reflecting an environment or an emitter converge to the path-traced reference.

The roughness gating and the reached-light MIS SHALL be the only behavioral
changes — flat-only scope, wavefront-only execution, both-backend parity, and the
caustic-parity gate from PM-1 remain in force.

#### Scenario: Glossy metal reflects neighbouring objects under SPPM

- **WHEN** the three-materials demo (a polished-metal sphere beside diffuse/wood
  spheres) is rendered under SPPM with `sppmGlossyContinueRoughness` at its
  default
- **THEN** the metal sphere SHALL show the neighbouring spheres reflected in it
- **AND** the reflected content SHALL trend toward the path-traced reference as
  passes accumulate (not remain absent as with delta-only continuation)

#### Scenario: Polished pbrt metal under an environment converges to path

- **WHEN** a polished pbrt-roughness-`0.1` conductor sphere lit only by a constant
  infinite light (`conductor_infinite`) is rendered under SPPM
- **THEN** the sphere SHALL be glossy-continued (its usd roughness `≈ 0.562` is
  below the default threshold), so its env reflection is reconstructed via the
  carrier ray rather than a degenerate photon gather (no photon deposits on the
  sole metal surface)
- **AND** the escaped-env radiance SHALL be MIS-weighted against the vertex's env
  NEE, so the `sppm|wavefront` render matches the `(path, wavefront)`
  self-consistency anchor within tolerance

#### Scenario: Threshold of zero preserves PM-1 behavior

- **WHEN** `sppmGlossyContinueRoughness` is `0`
- **THEN** the eye walk SHALL continue only through perfectly-specular (delta)
  lobes, reproducing the PM-1 visible-point placement and the existing caustic
  parity result

#### Scenario: Direct lighting is still not double-counted

- **WHEN** a scene is rendered under SPPM with glossy continuation enabled
- **THEN** photons SHALL still deposit only at non-specular, non-glossy-continued
  vertices after at least one bounce
- **AND** the SPPM-vs-path energy ratio SHALL remain within the PM-1 tolerance
  band (no double-counted direct term)

### Requirement: The visible point mirrors every deposit-relevant FlatHitMat field

The SPPM `VisiblePoint` SHALL store every `FlatHitMat` field the flat lobe
model reads at evaluation time (the evaluated values, including texture- and
graph-driven ones), and `sppmLoadMaterial` SHALL rebuild the deposit-time
`FlatMaterial` exclusively from those stored fields, so the photon deposit
evaluates the *exact* BSDF the eye pass stored. `emission` is the sole
documented exemption (direct, not BRDF; rebuilt as zero). A structural
(hostless) lock SHALL fail the test suite when `FlatHitMat` grows a field that
lacks a `VisiblePoint` slot, is not written by `sppmStoreVisiblePoint`, or is
not rebuilt by `sppmLoadMaterial` — an un-slotted field feeds undefined values
into the deposit-time `evaluate()` and can silently zero the photon term (the
Stage-2 rich inputs `transmissionColor`/`specularColor`/`diffuseRoughness` did
exactly that: `τ == 0` at 100% of visible points scene-wide, SPPM degraded to
its eye-pass direct term, and indirect-lit surfaces such as the bathroom walls
rendered black).

#### Scenario: FlatHitMat grows a field without a VisiblePoint slot

- **WHEN** a field is added to `FlatHitMat` in `flat_shading.slang` without a
  matching `VisiblePoint` slot, `sppmStoreVisiblePoint` write, and
  `sppmLoadMaterial` rebuild
- **THEN** the hostless parse-lock tests in `tests/test_sppm_state.py` fail,
  naming the missing field and the three places it must be wired

#### Scenario: Photon deposits carry non-zero flux on a lit diffuse scene

- **WHEN** SPPM renders a scene with lights and diffuse receivers (e.g.
  `cornell_box_sphere.usda`) for several passes
- **THEN** the accumulated visible-point flux `τ` is non-zero over a
  substantial fraction of active visible points, and the SPPM/path mean-energy
  ratio stays inside the energy gate's band (`test_sppm_energy_matches_path_tracer`,
  no xfail)

#### Scenario: Rich-input materials deposit with the authored values

- **WHEN** the eye pass stores a visible point on a material that authors
  Stage-2 rich inputs (e.g. bathroom's `standard_surface` with
  `specular_color` / `transmission_color`)
- **THEN** the visible point carries the evaluated authored values and the
  photon deposit's f_r uses them (not defaults), keeping SPPM consistent with
  the path tracer on those surfaces

### Requirement: SPPM terminal visible points capture complete env direct lighting

The wavefront SPPM eye stage SHALL add the BSDF-sampled env-miss MIS companion at
each terminal diffuse visible point, so that env direct there equals NEE +
BSDF-miss, matching the path tracer. Because the eye terminates at the first
diffuse visible point (photons carry the indirect), the companion of that
vertex's MIS-weighted environment NEE is otherwise never traced by a subsequent
bounce, leaving the env NEE down-weighted with no counterpart and env direct
systematically under-counted under a broad environment. The companion SHALL be
one BSDF sample whose escaped environment radiance is MIS-weighted against the env
NEE. Small analytic lights SHALL be unaffected (their BSDF-sampling pdf is ~0, so
the NEE is already effectively full weight).

#### Scenario: env-lit diffuse surface matches the path anchor

- **WHEN** an env-only scene (a diffuse plane under the dome) renders under
  `--integrator sppm` at a converged, firefly-robust sample count
- **THEN** the flat-plane radiance matches the `(path, wavefront)` reference
  within tolerance (not ~0.75× of it)

#### Scenario: small analytic lights unchanged

- **WHEN** a scene lit only by a sphere/emissive/distant light (no environment)
  renders under `--integrator sppm`
- **THEN** the result is unchanged — the added env-miss contributes nothing
  (the escaped ray carries zero env radiance), so sphere-lit parity is preserved

### Requirement: Update stage applies eye throughput at resolve

The update stage SHALL apply the visible point's stored eye throughput
(`VisiblePoint.beta` — camera-to-visible-point path throughput excluding the
visible point's BSDF) to the per-pass photon flux when resolving the pixel's
indirect radiance estimate, i.e. `L_indirect = (beta ⊗ Φ) / (N_emitted · π · r²)`,
matching pbrt-v4's fold of `vp.beta` at pass end. In spectral mode the product
SHALL be per-λ at the shared pass wavelengths used by both the eye and photon
stages, applied before the λ→sRGB resolve. The radius/N advance
(`sppmUpdate`) is flux-independent and unchanged.

#### Scenario: Directly viewed visible point is unchanged

- **WHEN** the visible point is reached directly from the camera (throughput ≈ 1)
- **THEN** the resolved SPPM estimate is unchanged within accumulation noise
  relative to the pre-change renderer

#### Scenario: Photon term through a tinted/lossy eye chain is not over-counted

- **WHEN** the photon-map indirect term is observed through a tinted or lossy
  specular/glossy eye chain (throughput ≠ 1, e.g. tinted glass)
- **THEN** the photon-indirect term is scaled by that throughput exactly as the
  NEE direct term already is (no over-bright, un-tinted photon contribution),
  and the SPPM render agrees with the BDPT reference on that region within the
  parity tolerance

### Requirement: Environment light emits photons
The SPPM photon pass SHALL include the environment light as a photon-emission
group whenever the environment is active (`furnaceMode == 0` and
`envIntensity > 0`). Env photons SHALL be emitted from the scene-bounding disk
along an importance-sampled environment direction (`sampleEnvDir`), with flux
`beta = L_env(ω) · πR² / (p_sel · p_dir(ω))` where `R` is the bounding-sphere
radius, `p_sel` the group-selection pmf, and `p_dir` the solid-angle pdf
reported by `sampleEnvDir` — the pbrt `ImageInfiniteLight::SampleLe`
normalization. Emission SHALL reject samples with a non-positive direction
pdf (`p_dir <= 0`, e.g. at the equirect poles) before dividing, mirroring the
validity guards of the other emission groups. Under `SKINNY_SPECTRAL` the env
flux SHALL use `upsampleIlluminantBound` with the shared per-pass wavelengths.

#### Scenario: Env-lit scene closes the indirect gap against the path anchor
- **WHEN** the fair null-sun glass-caustic scene (env light active, zero-power
  authored DistantLight) is rendered by SPPM and by the path anchor at matched
  spp
- **THEN** the shadow-region median-of-ratio SPPM/path SHALL be within a
  measured tolerance of 1.0, improving on the recorded 0.78, and the
  caustic-mask ratio SHALL improve on the recorded 0.936

#### Scenario: Flux normalization validated by forced-env probe
- **WHEN** the photon pass is probed with all photons forced to the env group
  (selection pmf set to 1 for the probe) on a flat-ground env-lit scene
- **THEN** the median deposited indirect flux SHALL match the path integrator's
  indirect-only term on the same region within [0.9, 1.1]

#### Scenario: Furnace mode emits no env photons
- **WHEN** furnace mode is active
- **THEN** the environment group SHALL be absent from photon emission and the
  furnace-closure suite results SHALL be unchanged

#### Scenario: Disabled environment does not dilute the photon budget
- **WHEN** `envIntensity == 0`
- **THEN** the environment group SHALL be absent and group selection SHALL be
  identical to the pre-change behavior

### Requirement: Env direct/indirect partition has no double count
Environment DIRECT lighting SHALL remain owned exclusively by the eye stage
(env NEE at the visible point plus the terminal env-miss companion); env
photons SHALL contribute only deposits with at least one prior scatter
(`depth >= 1`), so the union of the two terms neither omits nor double-counts
any env transport path.

#### Scenario: Env-lit indirect scene total matches the path anchor
- **WHEN** an env-only scene with genuine indirect transport (two diffuse
  surfaces, e.g. open box or plane + occluder, so photons deposit at
  depth ≥ 1) is rendered by SPPM and by the path anchor at matched spp
- **THEN** the SPPM total (eye direct + photon indirect) median-of-ratio
  against path SHALL be within the recorded per-scene tolerance of 1.0, with
  no systematic excess attributable to first-hit env deposits

#### Scenario: Single-plane scene gains nothing from env photons
- **WHEN** an env-only single diffuse-plane scene (no second surface, so no
  depth ≥ 1 deposit is possible) is rendered with env photons enabled
- **THEN** the photon-indirect contribution SHALL be zero and the SPPM total
  SHALL equal the eye-stage-only render

### Requirement: Photon emission group selection is power-proportional

The SPPM photon pass SHALL select the photon-emission group (emissive triangles,
sphere lights, distant lights, environment) with probability proportional to each
group's total emitted power, computed host-side and uploaded per frame:
emissive `π·Σ(area·luminance)`, sphere `4π²·Σ(luminance·r²)`, distant
`πR²·Σ luminance`, environment `πR²·envIntensity·∫L dω` (sin θ-weighted luminance
integral of the equirect map), with `R` the scene bounding-sphere radius used by the
emission geometry. Each emission branch SHALL divide its flux by the **actual**
selection probability (replacing the uniform `1/G`), so the estimator stays unbiased
while per-photon flux equalises across groups (`Φ_g / p_g ≈ Φ_total`). A group SHALL
receive zero probability when absent under the same presence predicates the shader
uses (`count > 0`; env: `furnaceMode == 0 && envIntensity > 0`). When the total
power is zero or non-finite the pmf SHALL fall back to uniform over the present
groups. Selection SHALL be wavelength-independent (the spectral build shares the
identical scalar pmf).

#### Scenario: Mixed weak-local-light + environment scene loses the env fireflies

- **WHEN** `glass_caustics_test.usda` (sphere light r = 0.2 + environment) is
  rendered by SPPM at matched spp against the path anchor, before and after the
  change
- **THEN** whole-image `noise_sigma` (from `metrics.compute_metrics`) SHALL drop
  versus the uniform-selection baseline (measured −11%: 4.66e-4 → 4.14e-4 at
  384²/48 spp; the group power ratio on this scene is ~5.4×, so the equalisation
  win is bounded), **AND** the caustic-region `noise_sigma` (masked to the
  caustic footprint — the whole-image number is background-dominated and blind
  to a caustic regression) SHALL NOT regress versus the uniform-selection
  baseline (measured −11%: 1.06e-3 → 9.4e-4), and the median-of-ratio SPPM/path
  SHALL remain within the recorded unbiasedness tolerance of 1.0 (measured:
  median 1.0000, mean 1.0120 vs the uniform baseline's 1.0118)

#### Scenario: Single-group scene is unchanged

- **WHEN** a scene with exactly one present emission group is rendered
- **THEN** that group's selection probability is 1 and the per-photon flux matches
  the pre-change value (selection pmf cancels identically)

#### Scenario: Probe override forces a group

- **WHEN** the host probe override (`_sppm_group_pmf_override`) is set to
  `[0, 0, 0, 1]`
- **THEN** the packed pmf is exactly that value, preserving the existing
  forced-env flux-normalization probe under the new selection path

#### Scenario: Zero total power falls back to uniform

- **WHEN** the host pmf helper receives all-zero (or non-finite) group powers with
  at least one present group
- **THEN** it returns the uniform distribution over the present groups and zeros
  for absent groups

### Requirement: Per-pass photon budget is environment-aware

The SPPM per-pass photon count SHALL be
`round(pixels / max(1 − pmfEnv, 1/CAP))` with `CAP = 8`, where `pixels` is
`width·height` and `pmfEnv` is the environment entry of the power-proportional
photon-group pmf, so the expected non-environment photon count stays exactly
`pixels` (uncapped regime) and the environment group's photons ride on top of —
never dilute — the local-light budget. When `pmfEnv` is zero (no live
environment) the count SHALL equal `pixels`, keeping env-free SPPM renders
bit-identical to the flat budget. An explicit photons-per-pass override SHALL
take absolute precedence over the formula. The update-stage flux normalisation
SHALL use the actually-emitted per-pass count so the estimator remains unbiased
for every budget value.

#### Scenario: Env-lit weak-local-light scene loses the env speckle component

The guarantee is a variance reduction of the ENV noise component by ≈ √(budget
multiplier) — not "reaches the pre-env floor" in general: a scene whose pmfEnv
saturates the cap keeps a residual (unbiased) env noise term.

- **WHEN** `assets/glass_caustics_test.usda` (sphere light r = 0.2 + live
  environment, pmfEnv ≈ 0.84 → ×6.25 budget) is rendered by SPPM at 384²/48 spp
  before and after the change
- **THEN** the env noise component
  `sqrt(noise_sigma² − noise_sigma_floor²)` (from `metrics.compute_metrics` vs
  the path anchor; floor = the pmf-forced-sphere-only render, measured 0.0154)
  SHALL shrink by ≈ √6.25 (flat-budget component measured 0.0224; whole-image
  `noise_sigma` 0.0272 → expected ≈ 0.0165 on this scene), **AND** the image
  mean SHALL stay within the recorded unbiasedness tolerance of the path anchor
  (flat-budget measured mean ratio ≈ 1.003)

#### Scenario: Env-free scene is bit-identical

- **WHEN** a scene with no live environment (`pmfEnv = 0`) is rendered by SPPM
- **THEN** the per-pass photon count equals `width·height` and the render is
  bit-identical to the pre-change renderer

#### Scenario: Override wins

- **WHEN** a photons-per-pass override is set
- **THEN** the emitted count equals the override regardless of `pmfEnv`

### Requirement: Photon dispatches never exceed the portable workgroup-count limit

The shared SPPM photon-dispatch loop SHALL bound every single dispatch to at
most `65535 × 64` photons (Vulkan's minimum-guaranteed
`maxComputeWorkGroupCount[0]` times the threadgroup width), tiling larger
budgets into additional 64-aligned sub-dispatches on every backend, so a driver
can never silently clamp `groupCountX` and drop photons while the update stage
divides by the full emitted count (dim bias).

#### Scenario: Oversized budget is tiled, not clamped

- **WHEN** the per-pass photon budget exceeds 4,194,240 on a backend that does
  not request watchdog batching (`photon_batch <= 0`)
- **THEN** the photon stage records multiple sub-dispatches whose 64-aligned
  batch sizes sum to the full budget, bit-identically to a single dispatch

