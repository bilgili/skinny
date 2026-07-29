# furnace-closure Specification

## ADDED Requirements

### Requirement: A coated material is furnace-probed

The suite SHALL carry a lossless **coated** material in the per-material furnace
probe, so the coat's energy is gated by an absolute invariant and not only
against a pbrt reference.

A coat over a lossless base loses no energy, so the coated probe SHALL close the
furnace to the same degree as its uncoated control. The difference between the
two is the coat's energy error, measured with no reference renderer in the loop
— which is what distinguishes an energy fix from a coincidence at one coat IOR,
one base and one roughness.

The probe SHALL use the **per-material** furnace path. Plain furnace mode
replaces every material in the scene, so a coated scene and an uncoated one
render identically under it and the probe passes without testing anything.

#### Scenario: The coated probe is measured against its uncoated control

- **WHEN** a lossless coated material and its uncoated control render under the
  per-material furnace
- **THEN** the coated one closes the furnace to the same degree, and any
  shortfall is recorded as the coat's energy error rather than absorbed into the
  scene's non-uniformity baseline

#### Scenario: Plain furnace mode cannot stand in for the per-material path

- **WHEN** a coated scene, an uncoated scene and a coated-metal scene are
  rendered under plain (whole-scene) furnace mode
- **THEN** all three return the same value, because the mode overrides every
  material — so a probe built on it SHALL NOT be accepted as covering the coat
