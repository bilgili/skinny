## ADDED Requirements

### Requirement: Live-updated weights with per-sample version
The neural directional proposal SHALL accept a weight swap between frames (rather
than only frozen offline weights), SHALL stamp each emitted sample with the network
version that produced it, and SHALL evaluate the density of a sample against its
stamped version so that an asynchronous swap leaves the estimator unbiased.

#### Scenario: Weights swap between frames
- **WHEN** new weights are published while the proposal is active
- **THEN** the proposal uses them from the next frame boundary without re-allocating its
  buffers or interrupting rendering

#### Scenario: Sample carries and uses its version
- **WHEN** a sample is drawn and later has its density evaluated in the mixture after a swap
- **THEN** the density is computed from the network version stamped on the sample, not the
  current render-side version

### Requirement: Frozen-weight behavior preserved when training is off
With online training disabled the neural proposal SHALL behave exactly as the frozen
Stage 1 proposal, loading baked weights and never swapping.

#### Scenario: Online disabled is frozen
- **WHEN** the neural proposal runs without online training enabled
- **THEN** weights are loaded once and never swapped, and behavior matches the frozen
  proposal
