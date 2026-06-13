## ADDED Requirements

### Requirement: Reject impossible render-flag combinations at startup

The CLI front-ends SHALL validate the parsed render flags at startup and SHALL
exit with a clear error when a combination cannot work, rather than crashing
later or silently doing nothing. Specifically, when the integrator is explicitly
`bdpt` AND either `--online-training` is set or the directional-proposal mixture
includes the neural proposal, the front-ends SHALL raise a usage error naming
the incompatible option and the fix (`--integrator path`), because BDPT does not
consume the neural directional proposal on any backend or execution mode. The
validation SHALL be shared across all front-ends (`skinny`, `skinny-gui`,
`skinny-web`, `skinny-render`) and SHALL NOT trip when the integrator is the
default/persisted path (not explicitly `bdpt`).

#### Scenario: bdpt + online-training exits with an error

- **WHEN** `skinny` (or any front-end) is launched with `--integrator bdpt
  --online-training`
- **THEN** it prints a clear error stating the combination is incompatible and
  to use `--integrator path`, and exits without initializing the GPU

#### Scenario: bdpt + neural proposal exits with an error

- **WHEN** a front-end that exposes `--proposals` is launched with `--integrator
  bdpt --proposals bsdf,neural`
- **THEN** it exits with the same clear error

#### Scenario: compatible combinations are unaffected

- **WHEN** a front-end is launched with `--integrator path` (with or without a
  neural proposal / `--online-training`), or with `--integrator bdpt` and no
  neural proposal and no `--online-training`, or with no explicit `--integrator`
- **THEN** startup proceeds normally with no error
