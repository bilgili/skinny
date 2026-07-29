# flat-bsdf-lobes Specification

## ADDED Requirements

### Requirement: The coat attenuates the layer below it

The coat lobe SHALL remove energy from the layers beneath it in proportion to
what those layers reflect, not only through the probability that the coat lobe
was selected.

`pCoat` accounts for one Fresnel reflection at the outer interface. A real coat
also loses energy at every interface crossing and returns light to the base by
internal reflection, where it is multiplied by the base's reflectance again. A
model that stops at `pCoat` under-attenuates a base in proportion to how much
that base reflects, so the error is small on a Lambert base and large on a
metal.

The attenuation SHALL be computed by ONE function that every estimator consumes:
`sample()`'s lobe weight (BSDF-only transport), `evaluate()`'s response (NEE,
BDPT, ReSTIR, and the env / neural proposals) and the spectral response
(`--spectral` transport). A copy that reaches some consumers and not others makes
those estimators disagree on one material — an RGB and a `--spectral` render
would differ, and a neural-proposal path would differ from a BSDF one.

Its parameters SHALL admit the directions and the base lobe's reflectance. An
internal-reflection term depends on what the base sends back up through the
interface, so it is not expressible as a per-material constant.

This is a **response-only** change: no pdf changes, so lobe selection and MIS
weights are untouched, and `weight = response / pdf` keeps the pdf a valid
sampling density.

A material with `coat = 0` SHALL be byte-identical to before this requirement:
every coat term stays gated on `coat > 0`.

#### Scenario: A coated metal is darkened like the reference layers it

- **WHEN** a gold `coatedconductor` is rendered and its sphere-region mean
  luminance is divided by that of the same gold uncoated, under identical
  lighting
- **THEN** the coat's effect on the metal approaches pbrt's, rather than
  retaining ~1.64x more of the base's energy than pbrt does

#### Scenario: A coated diffuse stays close to its uncoated form

- **WHEN** a fully-weighted coat (`coat = 1`, white `coat_color`, default
  `coatIOR = 1.5`) sits over a mid-grey diffuse base
- **THEN** it is still within a few percent of the same material with
  `coat = 0` — the attenuation scales with what the base reflects, and a Lambert
  base reflects little, so the same rule that darkens a metal substantially
  leaves a diffuse base nearly unchanged

#### Scenario: Every estimator sees the same coated response

- **WHEN** a coated material is rendered through BSDF-only transport, through
  NEE / BDPT / ReSTIR / an env or neural proposal, and under `--spectral`
- **THEN** all of them agree on its energy to the existing self-consistency
  tolerances, because all of them read one coat-transfer function rather than
  their own copy of it

#### Scenario: Uncoated materials do not move

- **WHEN** any flat material with `coat = 0` is rendered
- **THEN** its output is byte-identical, and the uncoated parity controls
  (`mat_conductor`, `furnace_conductor`, `furnace_lambert`,
  `furnace_rough_conductor`) do not move — a change reaching them means the coat
  gate leaked

## MODIFIED Requirements

### Requirement: Bounded per-lobe weight without clamping

`evaluate().response / evaluate().pdf` SHALL reduce to the lobe's bounded native
importance weight (`F·G₁` for the GGX coat/spec lobes, the diffuse albedo term
for the Lambert lobe) **times the coat transfer for that direction pair**, which
is itself bounded by 1 — so the product stays bounded and the guarantee this
requirement exists for is unchanged. The unified BSDF SHALL stay firefly-free
**by construction** and SHALL NOT rely on a weight clamp, firefly cap, or other
biasing safeguard to bound throughput.

For an uncoated material (`coat = 0`) the transfer is exactly 1 and the weight
reduces to the native term unchanged.

#### Scenario: no spec-lobe fireflies under the proposal mixture

- **WHEN** the `{bsdf, env}` or `{env}` proposal renders a glossy or coated
  surface
- **THEN** no firefly appears, because the weight is bounded by construction —
  the coat transfer multiplies a bounded native weight by a factor ≤ 1 and
  cannot introduce an unbounded term

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

`pCoat` is the coat's **selection** term and SHALL NOT be treated as the coat's
whole effect on the layers below it. Attenuating the base by nothing more than
"the coat lobe was not chosen" is what leaves a coated metal ~1.64x too bright.

#### Scenario: coated diffuse conserves energy under a uniform environment

- **WHEN** a fully-weighted coat (`coat = 1`, white `coat_color`, default
  `coatIOR = 1.5`) over a mid-grey diffuse base is rendered under a uniform
  (furnace-like) environment
- **THEN** the result is within a few percent of the same material with
  `coat = 0` (the thin dielectric coat barely darkens a diffuse base), and SHALL
  NOT lose a large fraction of the base energy (no dark region from a saturated
  `pCoat`)
- **AND** this closeness is a property of a Lambert base's low reflectance, NOT
  a licence for the coat to leave a high-reflectance base equally unchanged

#### Scenario: non-coated flat materials are unaffected

- **WHEN** a flat material with `coat = 0` is rendered
- **THEN** its output is byte-identical to before the fix (the coat Fresnel term
  is gated on `coat > 0`)
