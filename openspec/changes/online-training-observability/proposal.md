## Why

Online neural-proposal training is wired correctly but **near-silent**. Today
the only console feedback is `[skinny] --online-training refused: …` /
`… armed; waiting …` one-shot lines, a `[neural] trainer ready: …` line, and a
loss line throttled to one every 2 s. There is:

1. **No single view of what was selected vs what the renderer actually
   resolved** — backend `auto`→`metal`, trainer `auto`→`mlx`, handoff
   mechanism, train precision after a support fallback, whether a neural
   proposal is active — and **no statement of whether online training was
   approved, refused, or is waiting**, with the reason. A user running the
   documented Mac combo cannot tell at a glance why training is or isn't
   happening.
2. **No lifecycle signal that training actually ran.** Nothing says "training
   is now ACTIVE", and when the session ends nothing reports **how long it ran,
   how many cycles/steps/samples, or the final loss**. The user asked for
   exactly this ("if training is done, how long did it work").
3. **Nothing is visible without a terminal.** `skinny-gui` launched from the
   app icon surfaces none of the above.

## What Changes

- **Config matrix at startup + on change.** A new pure builder renders a
  console table, one row per axis (backend, execution-mode, integrator,
  proposals/neural-active, neural-trainer, neural-handoff, train-precision,
  online-training) with `requested`, `resolved`, and a `status` column
  (`ON`/`OFF`/`APPROVED`/`REFUSED (reason)`/`WAITING (reason)`/`n/a`). The
  renderer emits it at startup and **reprints only when the status signature
  changes** — i.e. when a runtime selection flips approval (selecting a neural
  proposal in the `skinny-gui` Proposals combobox arms training; switching
  execution mode). The existing one-shot refused/armed lines are subsumed by
  the matrix.
- **Training lifecycle logging.** A one-time `[neural] online training ACTIVE
  (backend=…, handoff=…, precision=…)` at the first real training cycle (the
  warm-start), and a `[neural] online training STOPPED: ran <s>, <cycles>
  cycles, <steps> steps, <samples> samples, final loss=…, backend=…` summary
  on `disable_online_training()` **and** at process exit while training is
  still active.
- **GUI status line.** A thread-safe `online_training_status()` snapshot
  (armed/active, last loss, cycles) the `skinny-gui` worker polls per frame to
  update a status label / window-title suffix (`— neural: ACTIVE 410cyc
  loss=0.012`), so state is visible without a terminal.
- **Front-end parity.** The console matrix is emitted on every front-end
  (`skinny` GLFW, `skinny-gui`, `skinny-web`, `skinny-render`); the GUI status
  line is `skinny-gui`-only (the others have no window).

No change to training math, the handoff format, or any default — when
`--online-training` is off the only new output is the startup config matrix
(informational), and the renderer behavior is otherwise unchanged.

## Capabilities

### Modified Capabilities
- `online-training-control`: adds the startup/reprint **config matrix**, the
  **training lifecycle** ACTIVE/STOPPED logging with a run summary, and the
  GUI **status snapshot**. The existing trigger/gating/refusal requirements are
  unchanged in behavior; their user-facing surface is folded into the matrix.

## Impact

- **Code:** new `src/skinny/config_report.py` (pure matrix builder +
  status-signature helper, no renderer imports). `src/skinny/renderer.py`
  (`_emit_config_matrix(reason)` with dedup signature; emit at scene-ready and
  on the existing arm/refuse/wait transitions; `online_training_status()`
  snapshot; STOPPED summary in `disable_online_training` + an `atexit` guard;
  pass trainer totals through). `src/skinny/sampling/neural_trainer.py`
  (accumulate total samples + monotonic start at first warm-start; one-time
  ACTIVE marker; `summary()`). Front-ends: `src/skinny/app.py`,
  `src/skinny/ui/qt/{app.py,viewport.py}`, and the `skinny-web`/`skinny-render`
  entry points call the startup matrix; `ui/qt` adds the status label.
- **Tests:** `tests/` unit-cover `config_report` (row set, resolved vs
  requested, status vocab incl. REFUSED/WAITING reasons, signature dedup) and
  `NeuralTrainer.summary()` (cycles/steps/samples/duration/final-loss). No GPU
  needed for either.
- **Docs:** `docs/NeuralGuiding.md` gains an Observability subsection (the
  matrix + lifecycle lines + GUI status); `README.md` notes the matrix and the
  STOPPED summary near the online-training flag description.
- **Behaviour:** additive logging only. Off-by-default training path is
  unchanged except for the informational startup matrix.
- **Risk:** low. The matrix builder is pure and unit-tested; the `atexit`
  summary must be idempotent with `disable_online_training` (guard so it prints
  once); the status snapshot must be a cheap lock-free read so it cannot stall
  the render thread.
