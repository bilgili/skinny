# Delta — spectral-rendering: dispersion showcase asset

## ADDED Requirements

### Requirement: High-dispersion named flint glass

The named-glass Cauchy table SHALL include at least one physically real
high-dispersion flint glass (`sf11`, Schott SF11 family) whose Cauchy `B`
coefficient is strictly greater than BK7's, fitted from the glass's published
Sellmeier curve over the visible band (400–700 nm). The entry SHALL resolve
through the same `named_glass_cauchy` lookup and `glassCauchyB` GPU binding path
as BK7, and unknown names SHALL continue to fall back to `"default"`.

#### Scenario: sf11 resolves with stronger dispersion than bk7

- **WHEN** `named_glass_cauchy("sf11")` is queried
- **THEN** it returns a `(A, B)` fit with `B > named_glass_cauchy("bk7")[1]`,
  and `cauchy_ior` for that fit decreases monotonically from 400 nm to 700 nm

#### Scenario: unknown glass name still falls back

- **WHEN** a material carries an unrecognized `glass_dispersion` name
- **THEN** the lookup resolves to the `"default"` (BK7-family) fit, unchanged
  from pre-existing behavior

### Requirement: Dispersion showcase asset

The repository SHALL ship a self-contained demo asset
`assets/dispersion_prism.usda` — a dark-room scene with a triangular
high-dispersion (`sf11`) delta-glass prism, a narrow white slit light aimed
through it, and a matte screen catching the refracted fan — framed so both the
through-prism dispersed slit image and the projected fan are visible from the
authored camera. The asset SHALL have no external file dependencies and SHALL
author dispersion via the existing `skinnyOverrides["glass_dispersion"]` seam.

#### Scenario: showcase asset demonstrates the dispersion requirement

- **WHEN** the asset is rendered with `--spectral` (path or bdpt, either
  execution mode)
- **THEN** both authored cues — the through-prism slit image and the projected
  screen fan — exhibit the spatial wavelength separation already required by
  "Dispersion for wavelength-dependent dielectrics", visibly absent from the
  RGB render of the identical scene

#### Scenario: asset integrity is CI-checked hostless

- **WHEN** the hostless test suite runs
- **THEN** the asset loads through the USD loader, the prism material's
  `glass_dispersion` override is `"sf11"`, and the scene references no external
  texture/HDR/volume files
