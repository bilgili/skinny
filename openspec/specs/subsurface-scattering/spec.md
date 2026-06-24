# subsurface-scattering Specification

## Purpose
TBD - created by archiving change pbrt-subsurface-volumetric. Update Purpose after archive.
## Requirements
### Requirement: pbrt subsurface imports as a volumetric interior medium

A pbrt `Material "subsurface"` SHALL import as a dedicated subsurface material — a
smooth dielectric boundary (`eta`) plus a homogeneous interior medium
(`σ_a`, `σ_s`, Henyey-Greenstein `g`) — and SHALL NOT be lowered to the flat
material with `opacity = 0` (which renders as clear glass with no interior
transport). The renderer SHALL carry the medium coefficients per material and
shade the object via a new subsurface material type.

The importer SHALL derive `(σ_a, σ_s, g)` using pbrt's input precedence: explicit
`sigma_a`/`sigma_s` (× `scale`); else a named preset (`Skin1`, …) from pbrt's
measured scattering table; else `reflectance` + `mfp` via the diffuse-albedo
inversion. The `-mtlx` / `standard_surface` inputs SHALL map to the same
coefficients — `subsurface_color` → single-scatter/diffuse albedo,
`subsurface_radius` → per-channel mean free path, `subsurface_scale` → `1/mfp`
scale, `subsurface_anisotropy` → `g` — so the native-USD and `-mtlx` import paths
produce identical `(σ_a, σ_s, g)`.

#### Scenario: subsurface dragon renders milky, matching pbrt

- **WHEN** the `sssdragon` scene (`Material "subsurface" "string name" "Skin1"`) is
  imported and rendered to convergence
- **THEN** the dragon is a soft, light-diffusing translucent solid (not clear
  glass), and its exposure-aligned relMSE / FLIP versus a pbrt v4 reference are
  within the scene's parity tolerance

#### Scenario: native-USD and -mtlx imports agree

- **WHEN** the same pbrt subsurface material is imported via the UsdPreviewSurface
  path and via the `-mtlx` standard_surface sidecar
- **THEN** both yield the same `(σ_a, σ_s, g)` interior medium and converge to the
  same image within noise

### Requirement: Volumetric subsurface transport is unbiased and bounded

The estimator SHALL transport light through the interior as a random walk: refract
at the dielectric boundary, delta-track (Woodcock) the medium with per-channel
`σ_t` and HG scattering, and Fresnel-split internal-reflection vs refraction on
boundary hits — **at both the entry boundary and at every interior face the walk
reaches**: on reaching a face the walk SHALL transmit with probability `Ft` (carry
the environment out) and otherwise internally reflect back into the medium and
continue, rather than discarding the reflected fraction. It SHALL be a single-pdf,
bounded-throughput (no clamp), Russian-roulette-terminated estimator, so the path
/ BDPT / ReSTIR invariants hold. It SHALL run in both megakernel and wavefront
execution modes and on both the Vulkan and native Metal backends.

The interior bounce cap SHALL be large enough to conserve energy across the
optical-depth range typical subsurface scenes hit (a high-albedo walk needs on the
order of `τ²` scatter events to escape a medium of optical depth `τ`), subject to
the platform GPU-watchdog limit on the Metal single-dispatch path.

#### Scenario: energy conservation (furnace) across optical depth

- **WHEN** a homogeneous subsurface sphere with `σ_a → 0` is rendered in a constant
  environment, at thin **and** thick optical depth (e.g. `τ ≈ 1`, `20`, `200`)
- **THEN** the converged result is ~unity at every depth (energy is neither created
  nor lost) — it does NOT collapse at high `τ` — and the albedo inversion
  round-trips within tolerance

#### Scenario: boundary energy is conserved, not discarded

- **WHEN** a subsurface object with a dielectric boundary (`η > 1`) is rendered
- **THEN** the internally-reflected fraction at each boundary escape is reflected
  back into the medium (not dropped), so the converged radiance is independent of
  whether the path happens to reach a face at a high-Fresnel / total-internal-
  reflection angle

#### Scenario: path tracer and BDPT agree

- **WHEN** a subsurface sphere is rendered under the path tracer and BDPT, in both
  execution modes
- **THEN** both converge to the same radiance per pixel within noise, on both
  backends

#### Scenario: non-subsurface scenes are unchanged

- **WHEN** a scene with no subsurface material (the pbrt parity corpus, or a true
  `dielectric` glass) is rendered
- **THEN** the result is byte/behaviour-identical to the pre-change renderer — the
  flat opacity/refraction path is untouched and the corpus parity is unchanged

### Requirement: Volume transport is forward-compatible with heterogeneous free-standing media

The volume transport and medium representation SHALL be chosen so that
heterogeneous, free-standing participating media (e.g. the pbrt `disney-cloud`
NanoVDB model) can be added later as a separate change **without reworking the
transport loop**. Specifically: the random walk SHALL read the medium ONLY through
a density seam — `densityAt(medium, p)` (local density multiplier) and
`mediumMajorant(medium, segment)` — dispatched by a medium `kind` tag, so adding a
volume source (NanoVDB, etc.) is a new `kind` plus two `case` bodies with no change
to the transport equations, the walk, NEE, RR, or integrator wiring. The medium
SHALL be a handle-referenced registry entry (not hardwired to a surface's
interior), the transport SHALL be majorant /
null-collision (Woodcock) tracking (so a constant `σ_t` is the degenerate case of a
spatially-varying density field), the boundary crossing SHALL be parameterized by
mode (dielectric refract vs index-matched pass-through), the per-collision
throughput SHALL be per-channel (`float3`), and the phase function SHALL be the
general Henyey-Greenstein already used. This change implements only the
homogeneous, dielectric-bounded interior; NanoVDB grid ingestion, `MediumInterface`
free-standing attachment, and spectral σ are out of scope.

#### Scenario: homogeneous interior is the degenerate null-collision case

- **WHEN** a homogeneous subsurface interior is transported
- **THEN** it is handled by the same majorant/null-collision walk a heterogeneous
  grid would use (with `σ_max = σ_t` and a constant density), so no closed-form
  homogeneous-only transmittance path is introduced that a grid could not reuse

#### Scenario: medium is reachable independently of the bounding surface material

- **WHEN** the renderer resolves the interior medium at a subsurface hit
- **THEN** it does so through a medium handle into a medium registry (not by reading
  the bounding material's BRDF params), so a future free-standing
  `MediumInterface` can register and attach a named medium to geometry or the
  camera through the same registry without changing the walk

#### Scenario: distinct media of different kinds coexist without conflict

- **WHEN** a scene contains two disjoint media of different kinds (e.g. a
  homogeneous subsurface object and — once added — a heterogeneous free-standing
  volume)
- **THEN** each is a separate registry entry resolved by its own handle/kind and
  transported by the same segment walk with its own `(kind, boundaryMode)`, with no
  shared mutable state — so they render correctly together; and the per-segment
  traversal is factored as a standalone function so a future current-medium /
  `MediumInterface` path-loop reuses it unchanged (overlapping / nested media,
  needing a medium priority stack, are explicitly out of scope)

### Requirement: Imported subsurface optical density is unit-correct

The pbrt importer SHALL emit the subsurface interior medium coefficients
(`subsurface_sigma_a`, `subsurface_sigma_s`) such that the renderer's volumetric
walk reproduces pbrt's per-scene-unit optical depth `τ = σ · L`, where `σ` are
the pbrt medium coefficients (mm⁻¹, × `scale`) and `L` is the path length in
world units. The packed coefficients SHALL therefore be the pbrt coefficients
divided by the stage's `mm_per_unit` (so that the walk's
`σ_packed · L_world · mm_per_unit` equals `σ_pbrt · L_world`), cancelling the
generic camera/SDF unit factor that pbrt does not apply to media.

The divisor SHALL be derived from the importer's declared stage unit
(`SetStageMetersPerUnit`) so the medium scale cannot drift from it.

This requirement governs only the optical-density **units**. Full pixel-mean
parity with a pbrt reference image additionally depends on the env-light
application and high-optical-depth walk fidelity, which are out of scope here.

#### Scenario: packed coefficients cancel the stage unit factor

- **WHEN** a pbrt `Material "subsurface" "string name" "Skin1" "float scale" 10`
  is imported and the stage declares `metersPerUnit = 1.0` (so the loader derives
  `mm_per_unit = 1000`)
- **THEN** the emitted `subsurface_sigma_a` / `subsurface_sigma_s` equal the pbrt
  Skin1 × scale coefficients divided by 1000, so that the walk's
  `σ_packed · mm_per_unit` recovers the original mm⁻¹ coefficients

#### Scenario: optical depth matches pbrt geometry

- **WHEN** the `sssdragon` scene is imported and the medium optical depth across
  the dragon is computed analytically as `σ_t · L` for the baked world-space
  extent
- **THEN** it matches pbrt's optical depth for the same geometry within a small
  tolerance, rather than being inflated by the ~1000× `mm_per_unit` factor

#### Scenario: dragon no longer renders at catastrophic density

- **WHEN** the `sssdragon` scene is imported through the production path
  (`mm_per_unit = 1000`) and rendered to convergence in wavefront mode
- **THEN** the dragon is a light-diffusing translucent solid at its true
  geometric optical depth, not the opaque gold/brown solid produced by the
  ×1000-inflated density

#### Scenario: energy conservation is unchanged

- **WHEN** the furnace energy-conservation gate for the subsurface walk is run
  after the coefficient rescale
- **THEN** the white-environment furnace albedo remains within tolerance of unity
  (the density rescale does not introduce or remove energy)

### Requirement: Subsurface opacity refraction gate requires an interior medium

The loader bridge that lowers a material's `opacity` to `0` on account of a `subsurface` input SHALL fire ONLY for a material that carries a non-zero interior medium (`subsurface_sigma_a` or `subsurface_sigma_s`).

This opacity bridge exists so the flat path's delta-dielectric refraction branch
(`flat_material.slang: if (m.opacity < 1.0)`) fires for a pbrt subsurface
boundary; it MUST gate on exactly the materials that
`renderer._material_is_subsurface` classifies as `MATERIAL_TYPE_SUBSURFACE` and
routes through the volumetric interior walk.

A plain Autodesk `standard_surface` (or OpenPBR) `subsurface` *weight* with no
interior medium (e.g. the `three_materials_demo` marble: `subsurface = 0.4`,
`subsurface_color`, no σ_a/σ_s) is a diffuse subsurface-scattering shading term,
not a transmissive boundary. The bridge SHALL leave its `opacity` untouched so it
remains an opaque flat material; forcing `opacity = 0` there would turn it into a
clear dielectric that refracts the environment (rendering as a dark, near-black,
speckled ball that ignores the lights). The gate SHALL remain a no-op when
`subsurface` is absent/zero, when σ_a and σ_s are both zero, or when an explicit
`opacity` was already authored. This applies identically on both intake paths (the
native-USD parse and the `.mtlx` API fallback), every backend, execution mode and
integrator, because it is a host-side material-loading invariant.

#### Scenario: a standard_surface subsurface weight stays opaque

- **WHEN** a `standard_surface` material with `subsurface = 0.4` and no
  `subsurface_sigma_a`/`subsurface_sigma_s` is loaded
- **THEN** no `opacity` override is derived (the material stays opaque, routes to
  `MATERIAL_TYPE_FLAT`, and renders as a lit diffuse surface — the marble sphere
  shows its grey base with blue veining and responds to the environment)

#### Scenario: a genuine subsurface medium still opens the gate

- **WHEN** a material carrying a non-zero interior medium
  (`subsurface_sigma_a`/`subsurface_sigma_s`, e.g. an imported pbrt
  `Material "subsurface"`) with a `subsurface` weight is loaded and no explicit
  `opacity` was authored
- **THEN** `opacity` is set to `0` so the refraction boundary of the volumetric
  subsurface walk fires, unchanged from before this gate

