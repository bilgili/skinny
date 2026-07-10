# pbrt-spectrum-conversion Specification

## Purpose
TBD - created by syncing change constant-spectrum-achromatic-rgb. Update Purpose after archive.
## Requirements
### Requirement: Constant sampled spectra reduce to achromatic RGB

The importer SHALL detect a constant pbrt sampled spectrum (`spectrum [l v l v
…]` where all authored sample values are exactly equal) and reduce it to the
achromatic linear RGB `[v, v, v]`, bypassing the CIE XYZ → sRGB projection.
This SHALL apply in both reflectance and illuminant modes and to every
spectrum-valued parameter routed through the shared conversion (media
`sigma_a`/`sigma_s`, material reflectance, light `L`/`I`, area-light `L`).

#### Scenario: Constant medium scattering spectrum

- **WHEN** a pbrt scene authors `"spectrum sigma_s" [200 10 900 10]`
- **THEN** the imported RGB coefficient is exactly `[10, 10, 10]` (before any
  unit/scale factors), not a tinted projection

#### Scenario: Constant reflectance spectrum

- **WHEN** a material authors `"spectrum reflectance" [200 0.2 900 0.2]`
- **THEN** the imported reflectance is exactly `[0.2, 0.2, 0.2]`

#### Scenario: Constant illuminant spectrum

- **WHEN** a light authors a constant `L` spectrum with value `v`
- **THEN** the imported color is `[v, v, v]` and the light's separate
  scale/intensity handling is unchanged

### Requirement: Colored sampled spectra keep the CIE projection

A sampled spectrum whose authored values are not all exactly equal SHALL be
reduced through the existing pipeline unchanged: linear interpolation onto the
360–830 nm grid, integration against the CIE colour-matching functions,
equal-energy normalisation for reflectance, XYZ → linear sRGB, clamp to ≥ 0.

#### Scenario: Genuinely colored spectrum unchanged

- **WHEN** a spectrum with unequal sample values (e.g. a measured conductor or
  a ramp `[400 0.1 700 0.9]`) is imported
- **THEN** the resulting RGB is bit-identical to the pre-change projection

#### Scenario: Near-constant spectrum is colored

- **WHEN** a spectrum's values differ by any nonzero amount (e.g.
  `[200 10 900 10.000001]`)
- **THEN** it takes the projection path, not the achromatic shortcut

### Requirement: Corpus baselines track the achromatic fix

The recorded pbrt-truth `measured` metrics MUST be re-measured for corpus
scenes whose imported values change under the achromatic shortcut
(`disney_cloud`, `bunny_cloud`), with the standard parity harness against the unchanged
pbrt reference EXRs, and the re-measured values MUST NOT be worse than the
previously recorded ones. Megakernel ≡ wavefront self-consistency MUST be
re-verified on those scenes.

#### Scenario: Baselines improve, never loosen

- **WHEN** the cloud corpus scenes are re-rendered after the fix and asset
  regeneration
- **THEN** the manifest `measured` relMSE/FLIP values are updated to the new
  (lower or equal) measurements, and no tolerance or baseline is raised

#### Scenario: Self-consistency preserved

- **WHEN** `disney_cloud` and `bunny_cloud` render in megakernel and wavefront
  modes after asset regeneration
- **THEN** the two modes match within the recorded self-consistency gate
  (EXACT parity expected, as before)

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

