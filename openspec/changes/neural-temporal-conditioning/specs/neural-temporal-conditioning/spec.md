## ADDED Requirements

### Requirement: Optional temporal conditioning dimension
The neural directional proposal SHALL support an optional temporal conditioning dimension,
selected by a `--temporal {off,on}` flag, that appends a normalized scene-time scalar to
the network's condition vector (condition dimension 9 → 10). When temporal conditioning is
off, the proposal SHALL be byte-identical to the non-temporal behavior.

#### Scenario: Temporal off reproduces non-temporal behavior
- **WHEN** the renderer runs with `--temporal off` (the default)
- **THEN** the neural proposal builds with condition dimension 9 and produces results
  identical to the behavior before this change

#### Scenario: Temporal on adds the time scalar to the condition
- **WHEN** the renderer runs with `--temporal on` and a neural proposal is active
- **THEN** the neural shaders build with condition dimension 10 and the condition vector's
  tenth component is the normalized scene time

### Requirement: Canonical normalized time encoding
The renderer SHALL encode scene time as the stage's current time code normalized to
`[startTimeCode, endTimeCode]` mapped to `[-1, 1]`, exposed to the shader as a frame-
constant scalar with no new descriptor binding, and the network's condition encoding and
the trainer's dataset encoding SHALL use this identical definition. When the stage has no
animation, the normalized time SHALL be a fixed constant.

#### Scenario: Time normalization matches between shader and trainer
- **WHEN** a path record generated at a given time code is used to train the network and
  the same time code drives the shader's condition at render time
- **THEN** both use the same `[-1,1]` normalized value, so the trained and applied
  conditions agree

#### Scenario: Static stage yields a constant time condition
- **WHEN** the loaded stage reports no animation
- **THEN** the normalized time condition is a fixed constant and temporal conditioning does
  not perturb the static result

### Requirement: Time-stamped records and time-aware replay
Each drained batch of path records SHALL be associated with the time code of the frame that
produced it, without widening the GPU record structure, and the replay buffer SHALL provide
a time-stratified sampling mode so the network retains guiding knowledge across the visited
time range rather than only the most recent frames.

#### Scenario: A drained batch carries its frame time
- **WHEN** the renderer drains the GPU path records for a frame
- **THEN** the batch is stamped with that frame's normalized time code and the stamp is
  available to the trainer's dataset builder

#### Scenario: Time-stratified sampling retains multiple times
- **WHEN** temporal conditioning is on and the trainer samples a batch from the replay
  buffer
- **THEN** the batch is drawn across the visited time range (not purely the most recent
  frames), so training does not overfit the current time slice

### Requirement: Temporal conditioning leaves the sampler unbiased
Conditioning on time SHALL NOT change the flow's solid-angle Jacobian or pdf normalization,
and querying the network at a time it was not trained on SHALL raise variance without
biasing the estimate, because the sampling density and the multiple-importance-sampling pdf
are evaluated from the same network at the same condition.

#### Scenario: Jacobian and normalization are unchanged by time
- **WHEN** the solid-angle pdf is evaluated with temporal conditioning on
- **THEN** the Jacobian term and the pdf normalization are identical to the non-temporal
  path (time enters only the conditioner, never the measure transform)

#### Scenario: Untrained time raises variance, not bias
- **WHEN** the network is queried at a scene time for which it has no training data
- **THEN** the rendered estimate remains unbiased (matching the no-guiding reference in
  expectation) and only its variance increases, as the proposal's importance-sampling
  weights fall back toward the analytic samplers

### Requirement: Architecture tag rejects a condition-dimension mismatch
The network record's architecture descriptor SHALL include the condition dimension, and the
loader SHALL reject a network whose condition dimension does not match the build (a static
network into a temporal build, or vice-versa) rather than rendering a mismatched result.

#### Scenario: Mismatched condition dimension is refused
- **WHEN** a network exported with condition dimension 9 is loaded into a temporal build
  expecting condition dimension 10 (or vice-versa)
- **THEN** the loader refuses the network with a clear architecture-mismatch error instead
  of silently producing a biased or malformed sampler
