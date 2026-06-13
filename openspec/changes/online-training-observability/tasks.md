## 1. Config matrix builder (pure)

- [x] 1.1 Add `src/skinny/config_report.py`: a pure `build_config_matrix(rows)` that formats an aligned table (axis / requested / resolved / status) and a `matrix_signature(rows)` helper returning a stable string for dedup. No renderer/Qt imports.
- [x] 1.2 Define the row model (a small dataclass or tuple: `axis`, `requested`, `resolved`, `status`) and the status vocab constants (`ON`/`OFF`/`APPROVED`/`REFUSED`/`WAITING`/`n/a`), with a `refused(reason)` / `waiting(reason)` formatter.

## 2. Renderer wiring

- [x] 2.1 `renderer.py`: `_collect_config_rows()` assembling the rows from the resolved state — backend (`select_backend` requested vs `is_metal`), execution mode (requested vs `effective_execution_mode_index`), integrator, proposals (requested vs active + `neural ACTIVE/—`), neural-trainer (requested vs resolved backend name once known), neural-handoff, train-precision (requested vs effective after support fallback), online-training (requested vs `can_online_train()` → APPROVED/REFUSED/WAITING).
- [x] 2.2 `renderer.py`: `_emit_config_matrix(reason="")` builds rows, computes the signature, prints via `config_report.build_config_matrix` only when the signature differs from the last printed one (store `_last_config_sig`). Emit once when the scene/backend is ready.
- [x] 2.3 Call `_emit_config_matrix` at the existing arm/refuse/wait transitions so the online-training row reprints on flip; remove or route the old one-shot `--online-training refused`/`armed; waiting` prints through it (keep wording in the matrix reason).

## 3. Training lifecycle

- [x] 3.1 `sampling/neural_trainer.py`: accumulate `_total_samples` and record a monotonic `_started_t` at the first warm-start; print the one-time `[neural] online training ACTIVE (backend=…, handoff=…, precision=…)` there (handoff passed in or set by the renderer).
- [x] 3.2 `sampling/neural_trainer.py`: `summary()` returning `{duration_s, cycles, steps, samples, final_loss, backend}` from the accumulators.
- [x] 3.3 `renderer.py`: in `disable_online_training()` print the `[neural] online training STOPPED: …` summary from `trainer.summary()`, guarded by a `_train_summary_printed` flag so it fires once. Register an `atexit` (or front-end shutdown) hook that prints the same summary if training is still active, sharing the guard.

## 4. GUI status snapshot + label

- [x] 4.1 `renderer.py`: `online_training_status()` returning a cheap dict snapshot (`armed`, `active`, `last_loss`, `cycles`, `backend`) read without locking (plain attribute reads of trainer counters).
- [x] 4.2 `ui/qt/viewport.py` + `ui/qt/app.py`: poll `online_training_status()` from the render worker each frame and surface it in a status label / window-title suffix (`— neural: ACTIVE 410cyc loss=0.012`); show `armed/waiting/off` states too.

## 5. Front-end parity

- [x] 5.1 Ensure the startup matrix is emitted on `skinny` (GLFW `app.py`), `skinny-gui`, `skinny-web`, and `skinny-render` (console only off-GUI). Drive it from the renderer's scene-ready path so each front-end gets it without duplicating logic.

## 6. Tests

- [x] 6.1 `tests/` cover `config_report`: full row set, requested≠resolved rendering, every status token incl. `REFUSED (reason)` / `WAITING (reason)`, and `matrix_signature` dedup (same state → same signature, flipped status → different).
- [x] 6.2 `tests/` cover `NeuralTrainer.summary()`: after N fake cycles the duration is positive, cycles/steps/samples/final-loss match, and the ACTIVE marker prints once (capture stdout).

## 7. Docs + verification

- [x] 7.1 `docs/NeuralGuiding.md`: add an Observability subsection (matrix columns + status vocab, ACTIVE/STOPPED lines, GUI status). `README.md`: note the matrix + STOPPED summary by the `--online-training` description.
- [x] 7.2 `.venv/bin/ruff check src/` clean; `.venv/bin/pytest` for the new tests green; `openspec validate online-training-observability --strict` passes.
