## ADDED Requirements

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
