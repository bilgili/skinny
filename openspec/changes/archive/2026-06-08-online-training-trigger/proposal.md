## Why

The online neural-proposal training machinery (change `neural-online-training`)
is fully built — `Renderer.enable_online_training()`, `online_drain()`,
`online_train_and_publish()`, the double-buffered publisher, and the frame-end
swap — but **nothing calls it**. No front-end (`app.py` GLFW, `skinny-gui` Qt,
headless, web) ever enables training or drives the per-cycle loop, and there is
no CLI flag or GUI control to start it. As a result `--neural-trainer` /
`--train-precision` / `--neural-handoff` set renderer attributes that are never
consumed, the `NeuralTrainer` is never constructed, and its config + per-cycle
logging never appears. The feature is unreachable by users.

## What Changes

- Add a `--online-training` CLI flag (env `SKINNY_ONLINE_TRAINING`, persisted
  like `--neural-handoff`) that turns the loop on.
- Add a renderer-side per-frame driver `Renderer.online_training_tick()` that
  the render loop calls once per frame: it drains GPU records into the replay
  buffer every frame and lets the background trainer publish; the existing
  frame-end swap promotes new weights. Training runs on a **dedicated trainer
  thread** (not the render thread) so a slow cycle — e.g. the ~seconds-per-cycle
  numpy reference oracle on CPU — never stalls the render; the double-buffered
  publisher already provides the safe cross-thread handoff.
- Enable from the front-ends after the scene is ready: `app.py` (GLFW) and
  `skinny-gui` (Qt `_RenderWorker` loop). Headless/web call the same renderer
  API (documented; not auto-enabled).
- **Refuse with a clear message** when prerequisites are unmet: online training
  requires `--execution-mode wavefront` **and** a neural proposal in the mixture
  (`--proposals …,neural`). Surface the existing clear errors for unsupported
  backend/handoff combos (`mlx` reserved; `interop` CUDA-only) rather than
  silently doing nothing.
- Persist `--online-training` in `app.py` settings (mirror `neural_handoff`).

## Capabilities

### New Capabilities
- `online-training-control`: the user-facing trigger + per-frame driver that
  starts/stops online training, its prerequisite gating (wavefront + neural
  proposal), its background-trainer cadence, and the front-end wiring + CLI flag
  + persistence.

### Modified Capabilities
- `neural-online-training`: the async trainer is now **reachable from the
  front-ends** — a CLI flag enables it and a per-frame driver runs the
  drain→train→publish→swap loop, where before the loop existed but had no caller.

## Impact

- **Code**: `src/skinny/renderer.py` (`online_training_tick`, background trainer
  thread lifecycle tied to `enable/disable_online_training`, prerequisite gate);
  `src/skinny/cli_common.py` (`--online-training`); `src/skinny/app.py` (enable
  after scene-ready + per-frame tick + settings persistence); `src/skinny/ui/qt/`
  (`app.py` pass-through + `viewport.py` `_RenderWorker` tick + enable).
- **Threading**: a new trainer thread; must coordinate with the render thread
  via the existing publisher double-buffer (render thread drains + swaps; trainer
  thread trains + publishes). Audit for queue/replay races (see KnownBugs #1 for
  an existing queue-threading caution).
- **Behavior**: with the flag off (default) everything is byte-identical to
  today. Mac (no CUDA) uses the numpy backend with `--neural-handoff file`.
- **Docs**: `README.md` (the flag + prerequisites), `docs/NeuralGuiding.md`
  (how to run online training end-to-end), `docs/PythonAPI.md`
  (`online_training_tick`).
