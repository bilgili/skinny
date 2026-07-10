# pbrt-spectrum-conversion Specification (delta)

## ADDED Requirements

### Requirement: Spectral payloads are preserved alongside the RGB reduction

For every spectrum-valued parameter it reduces to RGB, the importer SHALL additionally
preserve the raw spectral payload for renderer consumption in spectral mode, riding the
established `skinnyOverrides` side-channel: the authored temperature for `blackbody`
parameters, the sample pairs (densely resampled onto the 360–830 nm / 5 nm grid) for
illuminant `spectrum` parameters, the named-spectrum identity for named spectra (including
conductor eta/k and glass IOR names), and wavelength-dependent IOR fits for named glasses.
The existing RGB reduction SHALL remain unchanged and remain the value consumed by the RGB
pipeline. Scenes that author no spectra SHALL carry no new payload.

#### Scenario: blackbody temperature preserved

- **WHEN** a light authors `"blackbody L" [3000]`
- **THEN** the imported USD carries both the existing RGB reduction and the temperature
  `3000` in the light's spectral payload

#### Scenario: authored illuminant SPD preserved

- **WHEN** a light authors a non-constant `spectrum L` inline or from a file
- **THEN** the imported USD carries the SPD resampled onto the 5 nm grid alongside the RGB
  reduction

#### Scenario: named conductor identity preserved

- **WHEN** a material references a named metal spectrum (e.g. `metal-Au-eta`)
- **THEN** the imported USD records the spectrum name so the renderer can bind the vendored
  spectral eta/k curve

#### Scenario: RGB-authored scenes unchanged

- **WHEN** a scene authors only `rgb`/`float` values
- **THEN** the imported USD is byte-identical to the pre-change import (no empty payloads
  authored)

### Requirement: Vendored spectral data tables

The importer package SHALL vendor the spectral data the renderer's spectral mode requires,
under `src/skinny/pbrt/data/`: full spectral eta/k curves for the pbrt named metals, the CIE
D65 illuminant SPD, wavelength-dependent IOR fits for pbrt named glasses, and the
RGB→spectrum sigmoid-coefficient table with its checked-in generation script. A hostless test
SHALL validate the coefficient table against pbrt-published values.

#### Scenario: table validation is hostless

- **WHEN** the data-validation test runs on a host with no GPU
- **THEN** upsampled round-trip error and spot-checked coefficients match pbrt-published
  values within recorded tolerances
