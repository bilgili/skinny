# Delta — photon-mapping (fix-sppm-bathroom-black-walls)

## ADDED Requirements

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

## MODIFIED Requirements

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
