## ADDED Requirements

### Requirement: BDPT integrator refused for online training

The online-training prerequisite check SHALL refuse when the active integrator
is BDPT, because BDPT does not consume the neural directional proposal (its
shaders do not import the proposal seam) and has no wavefront record source — so
training under it would guide nothing and the record drain would fall back to a
record source absent on the Metal backend. The refusal SHALL name the fix
(`--integrator path`). This makes the interactive path — where the integrator
can be selected after startup, past the CLI validation — report the
incompatibility cleanly (the configuration matrix's online-training row shows
`REFUSED`) instead of crashing in the record drain.

#### Scenario: BDPT refuses online training at the gate

- **WHEN** online training is requested, the execution mode is wavefront, and
  the active integrator is BDPT
- **THEN** the prerequisite check returns not-ok with a reason naming
  `--integrator path`, training is not enabled, and the record drain does not run

#### Scenario: Matrix shows the BDPT refusal

- **WHEN** the configuration matrix is emitted with online training requested
  under the BDPT integrator
- **THEN** the online-training row's status is `REFUSED` naming `--integrator
  path`, rather than `APPROVED`
