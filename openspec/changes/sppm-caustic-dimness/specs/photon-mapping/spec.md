# photon-mapping — SPPM sample-count invariance delta

## ADDED Requirements

### Requirement: SPPM radiance is sample-count-invariant and matches the path anchor

The wavefront SPPM integrator SHALL produce radiance that converges to the
`(path, wavefront)` anchor and is invariant (within a noise tolerance) to the
number of progressive passes. Doubling the sample count SHALL reduce noise, not
scale the radiance. The per-pass direct term and the photon-map indirect term
SHALL each composite so that neither dominates at low pass counts nor is divided
away at high pass counts.

#### Scenario: caustic-scene energy does not depend on sample count

- **WHEN** `assets/glass_caustics_test.usda` renders under `--integrator sppm`
  at {64, 256, 1024} spp on Metal, linear film
- **THEN** the mean linear radiance of a flat-lit control region agrees across
  the three sample counts within tolerance and matches the sample-invariant
  `(path, wavefront)` reference for the same region
