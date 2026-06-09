## MODIFIED Requirements

### Requirement: Prerequisite gating with clear refusal
The system SHALL require `--execution-mode wavefront` and an active neural
proposal in the directional-proposal mixture before enabling online training.

The execution-mode prerequisite is fixed for the session: when it is not met and
`--online-training` was requested, the system SHALL refuse with a clear message
naming the missing prerequisite rather than silently doing nothing, and SHALL
stop trying for the session.

The active-neural-proposal prerequisite is runtime-selectable on interactive
front-ends that expose a proposal control (e.g. the `skinny-gui` Proposals
combobox). When `--online-training` is requested, the execution mode is
wavefront, but no neural proposal is active yet, the system SHALL keep polling
and SHALL enable training the moment a neural proposal becomes active, rather
than refusing permanently. The system SHALL emit a one-time message making clear
training is armed and waiting for a neural proposal.

An unsupported trainer backend or weight-handoff combo (e.g. `mlx`, or `interop`
without CUDA) SHALL surface its existing clear error rather than being swallowed.

#### Scenario: Wrong execution mode refuses permanently
- **WHEN** `--online-training` is set but the execution mode is not wavefront
- **THEN** the front-end logs a clear message naming the missing prerequisite,
  does not enable training (no silent no-op), and does not retry for the session

#### Scenario: Neural proposal selected at runtime starts training
- **WHEN** `--online-training` is set with `--execution-mode wavefront` on a
  front-end with a proposal combobox, no neural proposal is active at startup,
  and the user later selects a neural proposal in the combobox
- **THEN** the front-end enables online training at that point (it did not give
  up at startup), having emitted a one-time armed-and-waiting message beforehand

#### Scenario: Unsupported combo surfaces its error
- **WHEN** `--online-training` is set with `--neural-trainer mlx` (reserved) or
  `--neural-handoff interop` on a host without CUDA
- **THEN** enabling raises the existing clear error rather than starting a broken
  loop

### Requirement: User-facing trigger to start online training
The system SHALL expose a `--online-training` flag (env `SKINNY_ONLINE_TRAINING`)
that, when set on an interactive front-end, enables the online neural-proposal
training loop after the scene is ready. When the flag is absent the renderer
SHALL behave byte-identically to today (online training off). The flag SHALL be
persisted in the interactive settings with the same CLI/env-wins-then-restore
precedence as `--neural-handoff`.

On front-ends that expose a runtime proposal control (the `skinny-gui` Proposals
combobox), proposal selection SHALL be owned by that control and NOT by a
`--proposals` CLI flag; `skinny-gui` SHALL NOT define `--proposals`. Front-ends
without such a control (`skinny` GLFW, `skinny-web`, `skinny-render`) SHALL keep
the `--proposals` flag.

#### Scenario: Flag enables the loop
- **WHEN** an interactive front-end starts with `--online-training` and the
  prerequisites are met
- **THEN** the renderer's online training is enabled (the `NeuralTrainer` is
  constructed and its configuration is logged) and training runs

#### Scenario: Flag absent is a no-op
- **WHEN** a front-end starts without `--online-training`
- **THEN** online training stays off and rendering is unchanged from before this
  change

#### Scenario: skinny-gui has no proposals flag
- **WHEN** `skinny-gui` is launched with `--proposals …`
- **THEN** it errors with an unrecognized-argument message, because proposal
  selection is owned by the Proposals combobox (and the persisted setting)
