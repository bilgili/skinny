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
boundary hits. It SHALL be a single-pdf, bounded-throughput (no clamp), Russian-
roulette-terminated estimator, so the path / BDPT / ReSTIR invariants hold. It
SHALL run in both megakernel and wavefront execution modes and on both the Vulkan
and native Metal backends.

#### Scenario: energy conservation (furnace)

- **WHEN** a homogeneous subsurface sphere with `σ_a → 0` is rendered in a constant
  environment
- **THEN** the converged result is ~unity (energy is neither created nor lost) and
  the albedo inversion round-trips within tolerance

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

