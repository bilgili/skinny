## ADDED Requirements

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
