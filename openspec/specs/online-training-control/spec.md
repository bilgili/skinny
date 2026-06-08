# online-training-control Specification

## Purpose
TBD - created by archiving change online-training-trigger. Update Purpose after archive.
## Requirements
### Requirement: User-facing trigger to start online training
The system SHALL expose a `--online-training` flag (env `SKINNY_ONLINE_TRAINING`)
that, when set on an interactive front-end, enables the online neural-proposal
training loop after the scene is ready. When the flag is absent the renderer
SHALL behave byte-identically to today (online training off). The flag SHALL be
persisted in the interactive settings with the same CLI/env-wins-then-restore
precedence as `--neural-handoff`.

#### Scenario: Flag enables the loop
- **WHEN** an interactive front-end starts with `--online-training` and the
  prerequisites are met
- **THEN** the renderer's online training is enabled (the `NeuralTrainer` is
  constructed and its configuration is logged) and training runs

#### Scenario: Flag absent is a no-op
- **WHEN** a front-end starts without `--online-training`
- **THEN** online training stays off and rendering is unchanged from before this
  change

### Requirement: Per-frame training driver
The system SHALL provide a single renderer entry point the render loop calls once
per frame to advance online training. The entry point SHALL drain GPU path
records into the replay buffer on the render thread, SHALL be a no-op when online
training is off, and SHALL rely on the existing frame-end double-buffer swap to
promote newly trained weights. The actual per-cycle training SHALL NOT run on the
render thread, so a slow cycle (e.g. the numpy reference oracle) does not stall
rendering.

#### Scenario: Render loop drives the loop without stalling
- **WHEN** online training is on and the render loop calls the per-frame driver
- **THEN** records are drained into the replay buffer each frame and rendering
  continues at frame rate while training proceeds off the render thread

#### Scenario: Driver is a no-op when off
- **WHEN** the per-frame driver is called while online training is off
- **THEN** it does nothing and rendering is unaffected

### Requirement: Prerequisite gating with clear refusal
The system SHALL require `--execution-mode wavefront` and an active neural
proposal in the directional-proposal mixture before enabling online training.
When `--online-training` is requested but a prerequisite is missing, the system
SHALL refuse with a clear message naming the missing prerequisite rather than
silently doing nothing. An unsupported trainer backend or weight-handoff combo
(e.g. `mlx`, or `interop` without CUDA) SHALL surface its existing clear error
rather than being swallowed.

#### Scenario: Missing prerequisite refuses clearly
- **WHEN** `--online-training` is set but the execution mode is not wavefront or
  no neural proposal is in the mixture
- **THEN** the front-end logs a clear message naming the missing prerequisite and
  does not enable training (no silent no-op)

#### Scenario: Unsupported combo surfaces its error
- **WHEN** `--online-training` is set with `--neural-trainer mlx` (reserved) or
  `--neural-handoff interop` on a host without CUDA
- **THEN** enabling raises the existing clear error rather than starting a broken
  loop

