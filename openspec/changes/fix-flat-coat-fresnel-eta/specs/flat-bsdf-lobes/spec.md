## ADDED Requirements

### Requirement: Coat lobe Fresnel uses the entering dielectric eta

The flat / `std_surface` coat lobe SHALL compute its dielectric Fresnel
selection term for a view ray **entering** the coat from air, i.e. with relative
index `1 / coatIOR` (the same convention as the flat glass-refraction branch and
the subsurface boundary, which pass `1/ior` when entering a denser medium). It
SHALL NOT pass `coatIOR` raw to a Fresnel routine whose convention is
`eta = η_incident / η_transmitted`, because that evaluates the coat→air
(exiting) direction and produces spurious total internal reflection at moderate
view angles. The coat selection probability `pCoat` SHALL therefore equal the
coat's true reflectance (≈ `F0 = ((coatIOR−1)/(coatIOR+1))²` near normal
incidence, rising to 1 only at true grazing), consistent with the Schlick `F0`
already used by the coat reflection weight.

#### Scenario: coated diffuse conserves energy under a uniform environment

- **WHEN** a fully-weighted coat (`coat = 1`, white `coat_color`, default
  `coatIOR = 1.5`) over a mid-grey diffuse base is rendered under a uniform
  (furnace-like) environment
- **THEN** the result is within a few percent of the same material with
  `coat = 0` (the thin dielectric coat barely darkens a diffuse base), and SHALL
  NOT lose a large fraction of the base energy (no dark region from a saturated
  `pCoat`)

#### Scenario: non-coated flat materials are unaffected

- **WHEN** a flat material with `coat = 0` is rendered
- **THEN** its output is byte-identical to before the fix (the coat Fresnel term
  is gated on `coat > 0`)
