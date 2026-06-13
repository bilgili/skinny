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

### Requirement: Startup configuration matrix with resolved selections and approval status

The system SHALL emit, on every front-end, a human-readable configuration
matrix when the renderer is ready. The matrix SHALL contain one row per
render-selection axis — backend, execution mode, integrator, directional
proposals, neural trainer, neural handoff, train precision, and online training
— and for each row SHALL show the **requested** value (CLI/env/persisted), the
**resolved** value the renderer actually uses (e.g. `auto`→`metal`,
`auto`→`mlx`, the handoff mechanism, the train precision after any
support-fallback), and a **status**. The online-training row's status SHALL be
one of `APPROVED`, `REFUSED (<reason>)`, or `WAITING (<reason>)`, naming the
missing prerequisite when not approved; other rows use `ON`/`OFF`/`n/a`.

The matrix SHALL be reprinted when, and only when, the status signature changes
— for example when a neural proposal is selected at runtime and online training
flips from `WAITING` to `APPROVED`. The matrix subsumes the previous one-shot
refused/armed console lines.

#### Scenario: Matrix printed at startup with resolved values

- **WHEN** any front-end starts and the renderer becomes ready
- **THEN** a configuration matrix is printed showing each axis's requested and
  resolved value and its status, including the online-training row's
  `APPROVED`/`REFUSED (reason)`/`WAITING (reason)`

#### Scenario: Matrix reprints when approval flips

- **WHEN** online training is `WAITING (select a neural proposal)` and the user
  selects a neural proposal at runtime so the prerequisites are now met
- **THEN** the matrix is reprinted with the online-training status now
  `APPROVED`, and it is not reprinted on frames where the status signature is
  unchanged

#### Scenario: Off is informational only

- **WHEN** a front-end starts without `--online-training`
- **THEN** the matrix is still printed (online-training row `OFF`) and no other
  behavior changes

### Requirement: Online-training lifecycle logging with run summary

The system SHALL emit a one-time message when online training first begins
actually training (the first warm-started cycle), naming the resolved trainer
backend, handoff mechanism, and train precision. When online training stops —
either via an explicit disable or at process exit while still active — the
system SHALL emit a single summary reporting the wall-clock duration it ran, the
total number of training cycles, total optimizer steps, total samples trained
on, the final loss, and the trainer backend. The stop summary SHALL be emitted
at most once per training session (the explicit-disable and exit paths SHALL not
double-print).

#### Scenario: Active marker on first real cycle

- **WHEN** online training is enabled and the first warm-started training cycle
  runs
- **THEN** a one-time `online training ACTIVE` message is logged naming the
  trainer backend, handoff, and precision

#### Scenario: Stop summary reports run statistics

- **WHEN** online training is disabled, or the process exits while training is
  active
- **THEN** a single summary is logged with the run duration, total
  cycles/steps/samples, final loss, and backend, and it is not printed twice if
  both the disable and exit paths run

### Requirement: Thread-safe online-training status snapshot for the GUI

The system SHALL provide a renderer method returning a cheap, lock-free snapshot
of the current online-training state — at minimum whether training is armed and
whether it is actively training, the last loss, and the cycle count — suitable
for polling from a UI thread. A front-end with a window (`skinny-gui`) SHALL
surface this snapshot in the UI (e.g. a status label or window-title suffix) so
the training state is visible without a terminal. The snapshot read SHALL NOT
block the render thread.

#### Scenario: GUI surfaces training state

- **WHEN** `skinny-gui` is running with online training active
- **THEN** the window surfaces the training status (active state, last loss,
  cycle count) updated as training proceeds, without requiring console access

#### Scenario: Snapshot is a non-blocking read

- **WHEN** the UI thread polls the status snapshot each frame
- **THEN** the call returns immediately without acquiring a lock that the render
  or trainer thread could hold, and rendering is unaffected

