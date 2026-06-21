## ADDED Requirements

### Requirement: Equal-area octahedral env maps are reprojected to equirectangular

The pbrt importer SHALL reproject an `infinite` light's image map from pbrt v4's
equal-area octahedral square parameterization to the equirectangular layout that
skinny's dome-light path and `environment.slang` consume, so that the radiance
the shader samples for a direction matches the radiance pbrt's
`ImageInfiniteLight` returns for that same direction. The reprojection SHALL port
pbrt's `EqualAreaSquareToSphere` / `EqualAreaSphereToSquare` and invert skinny's
`directionToEquirectUV` (`u = atan2(dx,dz)/2π + 0.5`, `v = acos(dy)/π`, `+y` up).

#### Scenario: square image map is reprojected

- **WHEN** a `LightSource "infinite"` references a **square** `.exr`/`.pfm` image
  map (equal-area octahedral, as pbrt v4 always authors)
- **THEN** the emitted `.hdr` is the equirectangular reprojection of that map (not
  a verbatim pixel copy), at height = source edge and width = 2·height, and the
  conversion is reported as an equal-area→equirect resample

#### Scenario: a source texel lands at the matching equirect direction

- **WHEN** a single bright texel at square-uv `s` is reprojected
- **THEN** its energy appears at the equirect pixel whose shader direction
  `directionToEquirectUV⁻¹` maps to a direction that `EqualAreaSphereToSquare`
  returns to `s` (within one texel), confirming directional consistency

#### Scenario: non-square / constant maps pass through unchanged

- **WHEN** the infinite light has a **non-square** image map (already lat-long) or
  no map (constant `rgb L`)
- **THEN** the importer does not reproject — non-square maps are written as
  equirectangular as before (with a note that equirect is assumed), constant
  lights are unchanged, and the existing parity corpus output is byte-identical

### Requirement: Equal-area chart math is exact and isolated

The square↔sphere chart functions SHALL live in a dependency-free module
(`src/skinny/pbrt/equiarea.py`, numpy only, no USD/torch/GPU) so they are
unit-testable under `.venv`, and `EqualAreaSphereToSquare` SHALL be the exact
inverse of `EqualAreaSquareToSphere`.

#### Scenario: round-trip is identity

- **WHEN** a square-uv `p ∈ [0,1]²` is mapped sphere→square→sphere (and
  square→sphere→square)
- **THEN** the result equals `p` (resp. the unit direction) to floating-point
  tolerance, and the six axis directions map to the expected square corners and
  edge midpoints

### Requirement: Imported equal-area env renders at pbrt orientation

A pbrt scene whose `infinite` light uses a non-uniform equal-area map SHALL,
after import, render with environment lighting at the pbrt v4 orientation and
radiance, not a scrambled lat-long misread.

#### Scenario: sss_dragon environment matches pbrt reference

- **WHEN** `sss_dragon_small.pbrt` (infinite light
  `small_rural_road_equiarea.exr`) is imported and rendered headless
- **THEN** the env-lit result aligns with the pbrt-v4 reference in orientation
  (feature/silhouette alignment under shared exposure) and falls within the parity
  corpus relMSE/FLIP tolerance, improving markedly over the pre-change verbatim
  copy
