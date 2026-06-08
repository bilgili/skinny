## Context

Online neural-proposal training (`neural-online-training`) ships a complete
machinery on `Renderer`:

- `enable_online_training(*, handoff, trainer_backend, train_precision, …)` —
  builds the `ReplayBuffer`, the `NeuralTrainer` (selected `TrainingBackend`),
  and the `NeuralWeightPublisher`; sets `_online_training = True`; resolves the
  record source (wavefront-native drain).
- `online_drain()` — drains one frame of GPU path records into the replay buffer.
- `online_train_and_publish(rng)` — one `train_cycle` + `publisher.publish`.
- `_online_frame_end_swap()` — already called inside `render()` /
  `render_headless()` when `_online_training` is set; promotes pending weights at
  the frame boundary and stamps the network version.

**The gap:** no front-end calls `enable_online_training`, and nothing calls
`online_drain` / `online_train_and_publish` per frame. The frame loops are:
`app.py` (GLFW, main thread: `update` → `render`) and `skinny-gui`
(`ui/qt/viewport.py` `_RenderWorker.run`, a background `QThread`:
`update` → `render_headless` under `_render_lock`). The trainer runs
**synchronously** inside `train_cycle`; the numpy reference oracle is
~seconds-per-cycle on CPU, so calling it on the render thread would freeze the
viewport.

## Goals / Non-Goals

**Goals:**
- A user-facing trigger (`--online-training`) that actually starts the loop.
- A single renderer-side per-frame entry point the front-ends call, so cadence
  and threading live in one place, not duplicated per front-end.
- Training that does not stall rendering (Mac numpy backend especially).
- Clear refusal when prerequisites are unmet (wavefront + neural proposal), and
  the existing clear errors for `mlx` / `interop` surfaced, not swallowed.
- Off-by-default: byte-identical behavior when the flag is absent.

**Non-Goals:**
- Changing the trainer, backends, precision, or publisher (owned by
  `neural-trainer-backends` / `neural-online-training`).
- A GUI on/off toggle widget (CLI flag only this change; a control can follow).
- Auto-enabling in headless/web (they call the renderer API directly; only the
  interactive front-ends auto-enable from the flag).

## Decisions

### One renderer entry point: `online_training_tick()`
The render loop calls `renderer.online_training_tick()` once per frame. It
`online_drain()`s GPU records into the replay buffer (cheap; must run on the
render thread because it touches the GPU/queue) and is a no-op when training is
off. Front-ends stay thin (one call); cadence + threading are centralized.

*Alternative — each front-end calls `online_drain` + `online_train_and_publish`
directly:* duplicates the cadence/threading policy in every loop and invites
drift. Rejected.

### Training on a dedicated background thread
`enable_online_training` starts a daemon trainer thread that loops:
`online_train_and_publish()` (which `train_cycle`s on the host replay buffer and
`publish`es), then sleeps a short interval. The render thread keeps draining and
performs the frame-end swap. The double-buffered publisher already provides the
safe trainer→render handoff (publish stages; swap promotes at the frame
boundary), so no new locking around weights is needed. `disable_online_training`
signals the thread to stop and joins it.

This keeps the render loop smooth regardless of cycle cost (numpy ~seconds, CUDA
fast). The replay buffer is host numpy written by the render thread (drain) and
read by the trainer thread (sample); its access is guarded so a drain append and
a trainer sample don't race.

*Alternative — train inline every N frames/ms on the render thread:* trivially
simple, but a multi-second numpy cycle freezes the viewport for seconds. Rejected
for the default (Mac) path; the background thread is the "async trainer" the spec
already names.

### Prerequisite gate, fail loud
Online training requires `--execution-mode wavefront` (the record drain +
neural pre-pass are wavefront-only) and a neural proposal in the mixture
(`_neural_active()`). If the flag is set but a prerequisite is missing, the
front-end logs a clear one-line refusal and does not enable — never a silent
no-op. Unsupported backend/handoff combos already raise clearly inside
`enable_online_training` (`mlx` → `NotImplementedError`, `interop` off CUDA →
`NotImplementedError`); those propagate as a clear startup error.

### CLI + persistence mirror `--neural-handoff`
`--online-training` (store_true) in `cli_common.add_render_flags`, env
`SKINNY_ONLINE_TRAINING`. `app.py` reads it, enables after the USD scene is
ready, ticks each frame, and persists it in `~/.skinny/settings.json` with the
same CLI/env-wins-then-restore precedence as `neural_handoff`. `skinny-gui`
forwards the flag and enables in the `_RenderWorker` once the scene is ready.

## Risks / Trade-offs

- **Trainer thread vs. render thread race on the replay buffer / GPU queue** →
  the replay buffer guards `add` (drain, render thread) vs `sample` (trainer
  thread) with a lock; all GPU work (drain, swap, upload) stays on the render
  thread. Weight handoff goes only through the publisher double-buffer. (See
  KnownBugs #1: an existing queue-multithread caution — keep every `vkQueue*`
  call on the render thread.)
- **Scene-ready timing** → USD loads async; enabling before the wavefront record
  drain exists would no-op the drain. Enable only after the scene is built (the
  front-ends already pump `update()` until instances exist) or make
  `online_training_tick` enable lazily on the first frame the prerequisites hold.
- **Weight upload cost on swap** → `_apply_render_weights` uploads on the render
  thread at frame end; for the default net this is small, but a very large net
  could hitch. Swap already only fires when a new version is pending.
- **numpy cycle is slow** → acceptable: it runs off the render thread and only
  affects how often new weights appear, not frame rate. Logged (ms/cycle) so the
  cost is visible.

## Migration Plan

- Flag defaults off → no behavior change for existing invocations.
- No spec/format changes to weights, handoff, or precision.
- Rollback: drop the flag; the renderer methods are untouched callers-of-existing
  API plus one new tick method.

## Open Questions

- Trainer-thread cadence: a fixed small sleep vs. train-as-fast-as-data-allows.
  Resolve during implementation; default to a short sleep so a fast CUDA backend
  doesn't spin on an empty/duplicate replay buffer.
- Whether to also add a runtime GUI toggle (deferred to a follow-up).
