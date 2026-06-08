## 1. Renderer driver + background trainer thread

- [x] 1.1 Add `Renderer.online_training_tick()` — drains GPU records into the replay buffer on the render thread (`online_drain()`); no-op when `_online_training` is off. The existing `_online_frame_end_swap()` in `render()`/`render_headless()` keeps promoting weights.
- [x] 1.2 In `enable_online_training`, start a daemon trainer thread that loops `online_train_and_publish()` + a short sleep; in `disable_online_training`, signal stop and join it. Guard start/stop so re-enabling is safe.
- [x] 1.3 Guard `ReplayBuffer` `add` (render-thread drain) vs `sample` (trainer-thread) with a lock so the cross-thread access can't race; keep all `vkQueue*`/GPU work on the render thread (KnownBugs #1).
- [x] 1.4 Add a renderer prerequisite check `can_online_train() -> (bool, reason)` — true only when execution mode is wavefront and a neural proposal is active (`_neural_active()`).

## 2. CLI flag

- [x] 2.1 Add `--online-training` (store_true, env `SKINNY_ONLINE_TRAINING`) to `cli_common.add_render_flags`, next to `--neural-trainer`/`--train-precision`, with a help string noting the wavefront + neural-proposal prerequisites.

## 3. GLFW front-end (app.py)

- [x] 3.1 After the USD scene is ready, if `args.online_training` is set: check `renderer.can_online_train()`, and on failure print a clear one-line refusal naming the missing prerequisite (do not enable); on success call `renderer.enable_online_training(...)` (it surfaces the existing `mlx`/`interop` errors).
- [x] 3.2 Call `renderer.online_training_tick()` once per frame in the main loop (next to `renderer.update`/`render`).
- [x] 3.3 Persist `online_training` in `~/.skinny/settings.json` (mirror `neural_handoff` save + CLI/env-wins-then-restore).

## 4. Qt front-end (skinny-gui)

- [x] 4.1 Forward `args.online_training` through `main()` → `MainWindow` (alongside the neural flags already wired).
- [x] 4.2 In the `_RenderWorker` loop, once the scene is ready, enable online training (with the same `can_online_train()` gate + clear refusal) and call `renderer.online_training_tick()` each iteration under the render lock.

## 5. Tests

- [x] 5.1 `online_training_tick` is a no-op when off; when on (mocked/headless) it drains into the replay buffer and does not block (returns promptly).
- [x] 5.2 `can_online_train()` returns false + reason when execution mode is not wavefront or no neural proposal is active; true when both hold.
- [x] 5.3 The background trainer thread starts on enable and stops/joins on disable; weights advance via the publisher without the render thread calling `train_cycle`.

## 6. Docs

- [x] 6.1 `README.md`: document `--online-training` + its prerequisites (wavefront + `--proposals …,neural`) and the supported Mac combo (`--neural-trainer cpu`/`auto`, `--neural-handoff file`).
- [x] 6.2 `docs/NeuralGuiding.md`: an end-to-end "run online training" recipe (flags, what the `[neural]` logs mean, the async-trainer-thread model).
- [x] 6.3 `docs/PythonAPI.md`: `Renderer.online_training_tick` / `can_online_train` and the enable/disable lifecycle.

## 7. Validation

- [x] 7.1 `ruff check src/` clean on touched files; `pytest` green (incl. new tests).
- [x] 7.2 `openspec validate online-training-trigger --strict` passes.
- [x] 7.3 Manual smoke: `skinny-gui --execution-mode wavefront --proposals bsdf,neural --online-training --neural-trainer cpu` shows the `[neural] trainer ready …` + per-cycle logs; the bad combo `--neural-trainer mlx`/`--neural-handoff interop` errors clearly.
