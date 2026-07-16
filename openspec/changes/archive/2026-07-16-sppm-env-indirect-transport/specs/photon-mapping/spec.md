# photon-mapping Delta: sppm-env-indirect-transport

## ADDED Requirements

### Requirement: Environment light emits photons
The SPPM photon pass SHALL include the environment light as a photon-emission
group whenever the environment is active (`furnaceMode == 0` and
`envIntensity > 0`). Env photons SHALL be emitted from the scene-bounding disk
along an importance-sampled environment direction (`sampleEnvDir`), with flux
`beta = L_env(ω) · πR² / (p_sel · p_dir(ω))` where `R` is the bounding-sphere
radius, `p_sel` the group-selection pmf, and `p_dir` the solid-angle pdf
reported by `sampleEnvDir` — the pbrt `ImageInfiniteLight::SampleLe`
normalization. Emission SHALL reject samples with a non-positive direction
pdf (`p_dir <= 0`, e.g. at the equirect poles) before dividing, mirroring the
validity guards of the other emission groups. Under `SKINNY_SPECTRAL` the env
flux SHALL use `upsampleIlluminantBound` with the shared per-pass wavelengths.

#### Scenario: Env-lit scene closes the indirect gap against the path anchor
- **WHEN** the fair null-sun glass-caustic scene (env light active, zero-power
  authored DistantLight) is rendered by SPPM and by the path anchor at matched
  spp
- **THEN** the shadow-region median-of-ratio SPPM/path SHALL be within a
  measured tolerance of 1.0, improving on the recorded 0.78, and the
  caustic-mask ratio SHALL improve on the recorded 0.936

#### Scenario: Flux normalization validated by forced-env probe
- **WHEN** the photon pass is probed with all photons forced to the env group
  (selection pmf set to 1 for the probe) on a flat-ground env-lit scene
- **THEN** the median deposited indirect flux SHALL match the path integrator's
  indirect-only term on the same region within [0.9, 1.1]

#### Scenario: Furnace mode emits no env photons
- **WHEN** furnace mode is active
- **THEN** the environment group SHALL be absent from photon emission and the
  furnace-closure suite results SHALL be unchanged

#### Scenario: Disabled environment does not dilute the photon budget
- **WHEN** `envIntensity == 0`
- **THEN** the environment group SHALL be absent and group selection SHALL be
  identical to the pre-change behavior

### Requirement: Env direct/indirect partition has no double count
Environment DIRECT lighting SHALL remain owned exclusively by the eye stage
(env NEE at the visible point plus the terminal env-miss companion); env
photons SHALL contribute only deposits with at least one prior scatter
(`depth >= 1`), so the union of the two terms neither omits nor double-counts
any env transport path.

#### Scenario: Env-lit indirect scene total matches the path anchor
- **WHEN** an env-only scene with genuine indirect transport (two diffuse
  surfaces, e.g. open box or plane + occluder, so photons deposit at
  depth ≥ 1) is rendered by SPPM and by the path anchor at matched spp
- **THEN** the SPPM total (eye direct + photon indirect) median-of-ratio
  against path SHALL be within the recorded per-scene tolerance of 1.0, with
  no systematic excess attributable to first-hit env deposits

#### Scenario: Single-plane scene gains nothing from env photons
- **WHEN** an env-only single diffuse-plane scene (no second surface, so no
  depth ≥ 1 deposit is possible) is rendered with env photons enabled
- **THEN** the photon-indirect contribution SHALL be zero and the SPPM total
  SHALL equal the eye-stage-only render

