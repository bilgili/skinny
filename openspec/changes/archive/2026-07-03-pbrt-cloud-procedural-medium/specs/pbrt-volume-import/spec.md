# pbrt-volume-import (delta)

## ADDED Requirements

### Requirement: procedural cloud media import as volumes

The pbrt importer SHALL convert `MakeNamedMedium` of type `cloud` into a supported procedural
volume: the medium's `sigma_a`/`sigma_s` (RGB via the existing spectrum resolution), `g`,
`density`, `wispiness` (default 1), and `frequency` (default 5) SHALL be carried as `skinnyOverrides`
keys (`volume_cloud`=True, `cloud_density`, `cloud_wispiness`, `cloud_frequency`, `volume_sigma_a`,
`volume_sigma_s`, `volume_g`) on the bound interface material, and the medium's composed CTM SHALL
be preserved (the renderer maps world→medium-local so the altitude falloff reads correctly). The
current "heterogeneous media unsupported" skip SHALL be removed for `cloud`; other unsupported
heterogeneous types (`uniformgrid`, `rgbgrid`, …) remain reported skips.

#### Scenario: clouds.pbrt medium imports as a cloud volume

- **WHEN** `clouds.pbrt` is imported
- **THEN** the import report contains no "heterogeneous media unsupported" entry for the `cloud`
  medium, and the bound sphere's material carries `volume_cloud`=True with `cloud_density`=2 and
  the σ_a/σ_s/g overrides

#### Scenario: non-cloud unsupported media still report as skipped

- **WHEN** a scene declares `MakeNamedMedium` of type `rgbgrid`
- **THEN** the importer skips it with the existing report entry, unchanged

## MODIFIED Requirements

### Requirement: interface material imports as a null boundary

The importer SHALL treat both pbrt `Material "interface"` and an empty-string pbrt material on a
shape carrying a `MediumInterface` as a null boundary material: the bound shape is emitted as
geometry whose material carries no BSDF lobes and whose `skinnyOverrides` reference the shape's
`MediumInterface` interior medium (with the `volume_interface` marker), so the renderer routes it
to the free-standing volume path. Such a material MUST NOT fall back to the default diffuse or
dielectric material. An empty-string material with no `MediumInterface` SHALL keep the
default-material behavior.

#### Scenario: interface-bounded sphere binds the named medium

- **WHEN** `bunny-cloud.pbrt` is imported (sphere r=45, `MediumInterface "foo" ""`,
  `Material "interface"`)
- **THEN** the emitted sphere's material resolves with the `foo` medium's `volume_*` overrides and
  no diffuse/dielectric lobes, and `_resolve_medium` returns heterogeneous overrides instead of
  skipping

#### Scenario: empty-string material with a medium interface is a null boundary

- **WHEN** `clouds.pbrt` is imported (sphere, `MediumInterface "c" ""`, `Material ""`)
- **THEN** the emitted sphere's material carries the `c` medium overrides + `volume_interface`
  marker with no BSDF lobes (not the grey `UsdPreviewSurface` fallback)

#### Scenario: empty-string material without a medium keeps the default

- **WHEN** a shape has `Material ""` and no `MediumInterface`
- **THEN** it imports with the default material, unchanged
