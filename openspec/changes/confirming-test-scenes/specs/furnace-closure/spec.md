# furnace-closure

## ADDED Requirements

### Requirement: Global furnace closure gate
The test suite SHALL verify white-furnace closure: with `furnace_index` enabled (constant-
white environment, direct lights disabled) and a lossless material (Lambert albedo 1.0,
smooth conductor reflectance 1.0, or clear dielectric), the linear accumulation image SHALL
equal 1.0 per pixel within a pinned relative tolerance. The gate SHALL read the linear
accumulation image, not the tonemapped display output.

#### Scenario: lossless diffuse closes the furnace
- **WHEN** an albedo-1.0 diffuse scene renders with furnace mode on
- **THEN** the accumulation image mean and per-pixel values are 1.0 within tolerance

#### Scenario: energy regression is caught
- **WHEN** a shader change loses or gains energy in a swept lobe
- **THEN** the furnace gate for that material variant fails with the measured deviation

### Requirement: Furnace matrix over materials, integrators, and execution modes
The furnace gates SHALL sweep material variants × integrators × execution modes through the
parity-matrix validity table, so every valid combo is asserted and every invalid combo is a
recorded exclusion with a reason, never a silent skip.

#### Scenario: valid combos are all asserted
- **WHEN** the furnace matrix runs
- **THEN** every (material variant, integrator, execution mode) combo marked valid is
  rendered and gated, and megakernel/wavefront results agree per the self-consistency policy

#### Scenario: exclusions are recorded
- **WHEN** a combo is invalid (e.g. an integrator lacking transport for the variant)
- **THEN** the matrix records the exclusion reason and the coverage meta-test accepts it

### Requirement: Per-material furnace gate
The suite SHALL verify the per-material furnace flag (material flag bit 10): in a scene with
two materials where only one has the flag enabled, the flagged object SHALL close to 1.0
within tolerance while the unflagged object renders its normal appearance.

#### Scenario: flag isolates one material
- **WHEN** the two-material scene renders with the furnace bit set on material A only
- **THEN** material A's pixels close to 1.0 and material B's pixels match its non-furnace
  reference within tolerance

### Requirement: Legitimate energy loss is a recorded baseline
The measured closure value SHALL be recorded as a numeric baseline in the manifest whenever
a material variant fails closure for a known physical-model reason (e.g. rough conductor
without multiple-scattering energy compensation), and the gate SHALL assert against that
baseline. Baselines SHALL only ever be tightened toward 1.0 by later changes, never loosened.

#### Scenario: known energy loss does not mask new regressions
- **WHEN** rough conductor closes at its recorded baseline (below 1.0)
- **THEN** the gate passes at the baseline but fails if closure drifts further from 1.0

#### Scenario: a fix tightens the baseline
- **WHEN** a follow-up change adds energy compensation
- **THEN** the recorded baseline is updated toward 1.0 and never relaxed afterwards
