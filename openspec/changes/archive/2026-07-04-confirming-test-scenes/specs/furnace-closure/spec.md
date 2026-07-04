# furnace-closure

## ADDED Requirements

### Requirement: Global furnace uniformity gate
The test suite SHALL verify white-furnace energy closure via **spatial uniformity**: with
`furnace_index` enabled (constant-white environment, direct lights disabled) a lossless
material lit by a constant environment is indistinguishable from that environment, so the
object vanishes and the linear accumulation image is spatially uniform. The gate SHALL
measure the relative non-uniformity (coefficient of variation) of the linear accumulation
image over lit pixels and pass when it is within a pinned tolerance (or a recorded baseline).
The gate SHALL read the linear accumulation image, not the tonemapped display output.

Uniformity is used rather than an absolute "equals 1.0" because skinny's constant furnace
environment carries its own radiance constant (integrator-dependent) that is an
environment-normalization detail, whereas uniformity is normalization- and integrator-
independent and is the true closure statistic — a lossless object either disappears or it
does not.

#### Scenario: lossless material vanishes into the furnace
- **WHEN** a lossless-material scene renders with furnace mode on
- **THEN** the linear accumulation image's relative non-uniformity is within tolerance (the
  object is indistinguishable from the constant environment)

#### Scenario: energy regression is caught
- **WHEN** a shader change loses or gains energy in a swept lobe so the furnace object
  appears (a darker or brighter region)
- **THEN** the furnace gate for that material variant fails with the measured non-uniformity

### Requirement: Furnace matrix over materials, integrators, and execution modes
The furnace gates SHALL sweep material variants × integrators × execution modes through the
parity-matrix validity table, so every valid combo is asserted and every invalid combo is a
recorded exclusion with a reason, never a silent skip.

#### Scenario: valid combos are all asserted
- **WHEN** the furnace matrix runs
- **THEN** every (material variant, integrator, execution mode) combo marked valid is
  rendered and gated

#### Scenario: exclusions are recorded
- **WHEN** a combo is invalid (e.g. an integrator lacking transport for the variant)
- **THEN** the matrix records the exclusion reason and the coverage meta-test accepts it

### Requirement: Per-material furnace gate
The suite SHALL verify the per-material furnace flag (material flag bit 10): in a scene with
two materials where only one has the flag enabled, arming the flag on one material SHALL
change that material's rendered response distinguishably from the unflagged neighbour (the
per-material path has a measurable, material-local effect).

#### Scenario: flag has a material-local effect
- **WHEN** the two-material scene renders with the furnace bit set on one material only
- **THEN** the flagged material's rendered response diverges from the unflagged material's by
  at least the recorded minimum

### Requirement: Recorded furnace baselines
The measured non-uniformity SHALL be recorded as a numeric baseline in the manifest whenever
a material variant does not reach ideal (zero-non-uniformity) closure for a known reason
(e.g. edge darkening, an environment-normalization residual, or a rough conductor lacking
multiple-scattering compensation), and the gate SHALL assert against that baseline. Baselines
SHALL only ever be tightened toward ideal closure by later changes, never loosened.

#### Scenario: known residual does not mask new regressions
- **WHEN** a material closes at its recorded baseline non-uniformity (above zero)
- **THEN** the gate passes at the baseline but fails if non-uniformity drifts further from
  zero

#### Scenario: a fix tightens the baseline
- **WHEN** a follow-up change reduces the furnace residual
- **THEN** the recorded baseline is tightened toward zero and never relaxed afterwards
