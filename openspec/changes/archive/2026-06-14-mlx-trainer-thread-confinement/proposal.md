## Why

`tests/test_neural_headless.py::test_online_file_handoff_swap_unbiased` fails
deterministically on an Apple-Silicon Metal host with:

    [neural] trainer cycle failed: There is no Stream(gpu, 0) in current thread.

and the controlled swap assertion `r._neural_network_version == 1` instead sees
a much larger version (e.g. 7).

Investigation (reproduced on this host) shows **one** root cause behind *both*
symptoms — and it is **not** the MLX stream configuration the error text
suggests:

- `enable_online_training` auto-starts the background daemon trainer thread
  (added in `online-training-trigger`, one day *after* this test was written).
  The test then drives `online_train_and_publish` **manually on the main
  thread** while that daemon runs concurrently. Two threads now drive one MLX
  backend.
- MLX arrays and streams are **thread-affine**: whichever thread warm-starts the
  flow binds the leaf arrays to its own thread-local GPU stream; the other
  thread's ops then raise `There is no Stream(gpu, N) in current thread.`
- The daemon's extra `publish()` calls coalesce into the single frame-end swap,
  so the promoted `networkVersion` runs far past the one manual publish the test
  expects.

Evidence that the stream config is a red herring:

- The **numpy/cpu** backend (no MLX, no stream affinity) fails the *same*
  `version == 1` assertion (observed version 2) — so no MLX-side stream fix can
  make the test pass.
- The **daemon-only** path (exactly what production does — only the trainer
  thread ever touches MLX) runs **clean, zero stream errors**. Production is
  single-threaded over the backend, so there is no production stream bug.
- A stream created on one thread is **not** usable from another even under
  `with mx.stream(...)`, so "pin the worker thread's default stream" cannot fix
  cross-thread sharing.

The defect is therefore a **test-harness concurrency race** (manual stepping vs.
the auto-started daemon), plus a latent footgun in `MlxTrainingBackend` that
crashes — rather than serializes — when driven from more than one thread.

## What Changes

- **Confine all MLX work in `MlxTrainingBackend` to a single owned worker
  thread.** `warm_start`, `update`, and `export` marshal their MLX ops onto one
  dedicated single-worker executor the backend owns, so the leaf arrays are
  always created and used on the same thread regardless of which thread calls
  the backend. Multi-thread access can no longer raise
  `There is no Stream(gpu, N)`; it is serialized instead. Production behavior
  (single daemon thread) is unchanged; the executor is created lazily and only
  for the MLX backend.
- **Stop the auto-started daemon in the manual-control tests.** The tests that
  drive `online_train_and_publish` directly for deterministic single-step
  assertions call `_stop_trainer_thread()` immediately after
  `enable_online_training`, so the manual publishes are the sole driver and the
  frame-end swap promotes exactly the staged version. Applies to
  `test_neural_headless.py::test_online_file_handoff_swap_unbiased`,
  `test_neural_interop.py`, and `test_metal_record_drain_gpu.py`.

Production code paths (`app.py`, `viewport.py`) that rely on the daemon
auto-start are untouched.

## Impact

- Affected specs: `neural-training-backends` (MLX backend gains a
  thread-confinement guarantee).
- Affected code: `src/skinny/sampling/training_backends.py` (`MlxTrainingBackend`
  thread confinement); the three manual-control tests above.
- No change to the weight-handoff format, the numpy/torch backends, or any
  production renderer path.
