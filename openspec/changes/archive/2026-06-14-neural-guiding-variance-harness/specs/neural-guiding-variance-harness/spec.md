## ADDED Requirements

### Requirement: Headless configuration sweep over the parameterization axes
The harness SHALL render a fixed set of known scenes headlessly across a declarative matrix of
configurations — guiding proposal set, reuse mode, chart, conditioner encoding, temporal mode,
inference precision, and budget — building the renderer with the matching defines and loading a
parametrization-tag-matched network for each cell. A cell whose build or network is unavailable
SHALL be skipped and logged, never silently omitted.

#### Scenario: A configuration cell is rendered
- **WHEN** the sweep reaches a cell with an available matching build and network
- **THEN** the renderer is built/selected for that cell's defines, the tag-matched network is
  loaded, and the scene is rendered headlessly at the cell's budget

#### Scenario: An unavailable cell is logged, not hidden
- **WHEN** a cell's required build or network is missing
- **THEN** the cell is skipped and recorded in the run log as skipped, so coverage gaps are
  visible

### Requirement: Variance is measured against an asserted-converged reference over seeds
For each scene and integrator the harness SHALL render a converged reference and assert its
convergence before use, and SHALL compute each cell's error/variance against that reference
from the linear-HDR accumulation image over multiple independent seeds, reporting a spread
rather than a single-run value.

#### Scenario: Reference convergence is gated
- **WHEN** a reference render is produced for a scene/integrator
- **THEN** the harness asserts the reference has converged before any cell is compared against
  it, and refuses to report numbers otherwise

#### Scenario: Per-cell variance carries a spread
- **WHEN** a cell's variance/error is computed
- **THEN** it is aggregated over multiple independent seeds and reported with a spread, not from
  a single run, and is computed from the linear-HDR accumulation image

### Requirement: Equal-time, equal-variance, and efficiency outputs
The harness SHALL emit equal-time (variance at a fixed budget), equal-variance (time to reach a
target variance), and `1/(var·t)` efficiency results — the efficiency metric identical to the
`spline_flow` parameterization study — as checked-in tables and SVG plots.

#### Scenario: The three views are produced
- **WHEN** a sweep slice completes
- **THEN** the harness produces an equal-time comparison, an equal-variance comparison, and a
  `1/(var·t)` efficiency table for that slice

#### Scenario: Plots are SVG and tables are transcribable
- **WHEN** the harness writes its outputs
- **THEN** plots are emitted as SVG per the repository diagram convention and tables carry a
  provenance note suitable for transcription into the paper

### Requirement: Reproducible runs
The sweep SHALL be reproducible: deterministic seeds, the configuration and scene set and
reference identities checked in, and a documented headless invocation environment.

#### Scenario: Re-running reproduces the results
- **WHEN** the harness is re-run with the same checked-in configuration and seeds
- **THEN** it reproduces the same scene set, references, and reported numbers within tolerance
