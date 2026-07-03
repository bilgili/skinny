# heterogeneous-media (delta)

## ADDED Requirements

### Requirement: procedural cloud density goes through the same medium seam

The `MEDIUM_CLOUD` medium kind SHALL be implemented exclusively as a new `densityAt(medium, p)`
`case` (pbrt `CloudMedium::Density`: a 5-octave classic-Perlin fBm with a 2-iteration wispiness
domain warp and the altitude falloff), with `mediumMajorant` returning the packed σ_t (the density
clamps to [0,1], so σ_t is the exact global majorant — identical to the grid case). The transport
walk, phase function, NEE, RR, per-channel throughput, boundary handling, and integrator wiring
SHALL be unchanged, and it SHALL require no new GPU binding (the density is analytic). The ported
Perlin `Noise`/`DNoise` (the 256→512 `NoisePerm` table, `Grad`, quintic `NoiseWeight`) SHALL match
pbrt's implementation so the density is pbrt's, not a look-alike.

#### Scenario: ported noise matches the pbrt algorithm

- **WHEN** the Slang-ported `cloudDensity` constants are evaluated at a grid of medium-local points
  and compared to a CPU reference of the identical pbrt algorithm
- **THEN** the two agree to float tolerance at every sampled point

#### Scenario: cloud renders structured, not homogeneous

- **WHEN** `clouds.pbrt` renders inside its interface-bounded sphere under the sky environment
- **THEN** the medium shows fBm cloud structure with the top-lit altitude falloff (denser low,
  wispy top), not a uniform blob, and a zero-**σ** cloud renders as an invisible boundary
  (matches the same scene without the sphere within Monte Carlo noise). Note: pbrt's `density 0`
  is NOT empty — `CloudMedium::Density` keeps the altitude floor term `2·max(0, 0.5−p.y)`
  regardless of `density`, and the port matches pbrt exactly, so the empty-boundary check uses
  σ=0 instead

#### Scenario: other medium kinds unchanged

- **WHEN** the homogeneous and `MEDIUM_NANOVDB` parity scenes render after `MEDIUM_CLOUD` lands
- **THEN** results are unchanged (same seeds → same image) and their gates stay green

## MODIFIED Requirements

### Requirement: heterogeneous media render at parity across modes and backends

Heterogeneous-volume scenes SHALL pass the standing dual parity gates for the Path integrator:
megakernel ≡ wavefront self-consistency on Metal, and pbrt-truth against checked-in pbrt v4
references (with any known divergence recorded as a per-combo `baseline`, never a loosened
self-consistency tolerance). `disney_cloud`, `bunny_cloud` (grid) **and** `clouds` (procedural
`MEDIUM_CLOUD`) SHALL be corpus scenes; integrator combos that do not support media (BDPT, SPPM)
SHALL be recorded exclusions in `combo_is_valid`, not silent skips.

#### Scenario: dual gate on the cloud corpus scenes

- **WHEN** the parity matrix runs `disney_cloud`, `bunny_cloud`, and `clouds` with the Path
  integrator
- **THEN** megakernel and wavefront agree within the self-consistency threshold and the pbrt-truth
  metric passes at its recorded baseline

#### Scenario: unsupported combos are recorded

- **WHEN** the matrix enumerates BDPT or SPPM against a volume scene
- **THEN** the combo is excluded by a `combo_is_valid` rule (visible in the coverage meta-test),
  not attempted and not silently dropped
