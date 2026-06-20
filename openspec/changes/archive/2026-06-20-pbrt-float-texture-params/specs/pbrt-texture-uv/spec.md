## ADDED Requirements

### Requirement: Texture-valued FloatTexture material parameters never crash the importer

The importer SHALL accept a pbrt material whose `FloatTexture`-typed scalar
parameter (`roughness`, `uroughness`, `vroughness`, the dielectric `eta`, or
`coatedconductor` `interface.roughness`) is bound to a **named texture** rather
than a constant. A texture-typed parameter SHALL NOT be coerced with `float()`;
the importer SHALL instead resolve it to a texture connection (when the texture
is a supported `imagemap`, including a `scale`-wrapped imagemap) or fall back to
the parameter's defined scalar default, and SHALL never raise on such input.

#### Scenario: roughness bound to a named texture imports without error

- **WHEN** a `Material "conductor"` (or `dielectric`/`coateddiffuse`/
  `coatedconductor`) declares `"texture roughness" ["<name>"]`
- **THEN** import completes without raising, and the emitted USD material's
  `roughness` is texture-connected when `<name>` resolves to a supported image,
  or set to the scalar default otherwise

#### Scenario: nested `scale`→`imagemap` roughness texture resolves to its image

- **WHEN** the bound roughness texture is a `Texture "<n>" "float" "scale"` whose
  `tex` references a `Texture "<m>" "float" "imagemap"`
- **THEN** the importer connects the USD `roughness` input to the inner
  `imagemap`'s resolved image path (the `scale` factor unwrap is recorded as an
  APPROX note)

#### Scenario: unsupported FloatTexture class falls back to scalar default

- **WHEN** a scalar parameter is bound to a texture class the importer does not
  support (e.g. `checkerboard`, `mix`, `constant`) or an unresolvable name
- **THEN** the importer uses the parameter's scalar default, records an APPROX
  note naming the parameter and texture, and does not fail the import

#### Scenario: a textured parameter maps to its own USD input, not diffuse

- **WHEN** a material binds a texture to `roughness` (and, separately, a material
  binds a texture to `reflectance`)
- **THEN** the `roughness`-bound texture connects to the USD `roughness` input via
  the `UsdUVTexture` scalar (`.r`) output, and the `reflectance`-bound texture
  connects to `diffuseColor` via the color (`.rgb`) output — each driven by the
  single texture→parameter map, with no parameter assumed to be `diffuseColor`

#### Scenario: constant scalar parameters are unaffected

- **WHEN** a scalar parameter is bound to a constant (e.g. `"float roughness" 0.1`)
- **THEN** the resolved value matches the prior behavior exactly (no texture
  connection authored, no regression)

#### Scenario: crown.pbrt imports

- **WHEN** `crown.pbrt` (texture-valued roughness across multiple materials,
  including nested `scale`/`imagemap`) is imported
- **THEN** import completes without raising and produces a USD stage with the
  expected number of shapes
