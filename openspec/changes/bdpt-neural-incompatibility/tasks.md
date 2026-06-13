## 1. Shared CLI validator

- [x] 1.1 `cli_common.validate_render_flags(args)`: raise `SystemExit` with a clear message when explicit `--integrator bdpt` is combined with `--online-training` or a neural proposal (`neural` in `--proposals`). No-op for default/path integrator; tolerant of a Namespace without a `proposals` attribute.
- [x] 1.2 Call it from every front-end `main()` right after `parse_args`: `app.py` (GLFW), `ui/qt/app.py` (skinny-gui), `web_app.py` (skinny-web), `headless.py` (skinny-render).

## 2. Renderer runtime refusal (defensive)

- [x] 2.1 `renderer.can_online_train()`: refuse when `integrator_index == 1` (bdpt), naming `--integrator path`.
- [x] 2.2 `renderer._collect_config_rows()`: the online-training matrix row shows `REFUSED (requires --integrator path …)` under bdpt instead of `APPROVED`.

## 3. Tests

- [x] 3.1 `tests/test_cli_common.py`: validator exits on bdpt+online and bdpt+neural; is OK for bdpt-alone, bdpt+non-neural, path+neural, default integrator, and a Namespace lacking `proposals`.
- [x] 3.2 `tests/test_online_training_observability.py`: `can_online_train` refuses bdpt; the matrix online-training row is REFUSED under bdpt.

## 4. Docs + verification

- [x] 4.1 `README.md` + `docs/NeuralGuiding.md`: note bdpt is incompatible with the neural proposal / online training (path-integrator only).
- [x] 4.2 `ruff` clean on touched files; new tests green; the previously-crashing command now errors-and-exits cleanly; `openspec validate bdpt-neural-incompatibility --strict` passes.
