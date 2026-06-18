## ADDED Requirements

### Requirement: In-renderer NIS proposal
The renderer SHALL provide a Neural Importance Sampling (Müller et al. 2019)
proposal — a piecewise-quadratic coupling flow with one-blob conditioning — as a
controlled in-renderer re-implementation of NIS's sampler, runnable on the same
scenes and under the same wavefront mixture-MIS path as the rational-quadratic
neural proposal. It SHALL hold the directional chart fixed at `V1` so the only
difference from the `neural` proposal is the coupling and the encoding.

#### Scenario: NIS renders as a proposal
- **WHEN** the renderer is built with `NF_COUPLING=nis-pq` and run with the `nis`
  proposal enabled
- **THEN** the wavefront path uses the NIS coupling under the V1 chart and
  produces an unbiased render through the same mixture-MIS pdf division as the
  `neural` proposal

#### Scenario: Scope deltas stated
- **WHEN** the in-renderer NIS result is reported
- **THEN** it is labelled a controlled re-implementation of NIS's coupling and
  encoding (offline-trained, V1 chart), not full NIS (no online-during-render
  training, no primary-sample-space full-path mode)

### Requirement: NIS render passes the unbiasedness gate
A converged NIS render SHALL match the reference under the same renderer-path
unbiasedness gate as the `neural` proposal before any scene number is reported.
Because the coupling is exactly invertible with an analytic log-determinant under
the equal-area chart, a poorly trained NIS net SHALL cost variance, not bias.

#### Scenario: Gate precedes any scene number
- **WHEN** a NIS scene variance/efficiency number is recorded
- **THEN** the converged NIS render for that scene has already matched the
  reference within the gate tolerance
