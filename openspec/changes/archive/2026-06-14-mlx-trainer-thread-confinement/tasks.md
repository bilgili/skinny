## 1. Confine MLX work to one owned thread

- [x] 1.1 Give `MlxTrainingBackend` a lazily-created single-worker executor (one
  dedicated thread it owns). Route the MLX-touching bodies of `warm_start`,
  `update`, and `export` through it (submit + block on the result), so the leaf
  arrays are always created and evaluated on that one thread.
- [x] 1.2 Verify the confined backend no longer raises
  `There is no Stream(gpu, N) in current thread.` when called from two threads
  (a direct main-thread call concurrent with the daemon), and that single-thread
  (production) use is byte-for-byte unchanged.

## 2. De-race the manual-control tests

- [x] 2.1 In `test_neural_headless.py::test_online_file_handoff_swap_unbiased`,
  call `r._stop_trainer_thread()` immediately after `enable_online_training` so
  the manual `online_train_and_publish` is the sole publisher and the swap
  promotes exactly version 1.
- [x] 2.2 Apply the same daemon-stop to `test_neural_interop.py` (expects manual
  publishes 1 then 2) and `test_metal_record_drain_gpu.py`.

## 3. Verification

- [x] 3.1 `test_online_file_handoff_swap_unbiased` passes deterministically on
  the Metal host (version → 1, no stream errors).
- [x] 3.2 The neural online-training test set passes (`test_neural_headless.py`,
  `test_neural_interop.py`), and the MLX numpy-parity test still converges.
- [x] 3.3 `ruff check src/` is clean; `openspec validate
  mlx-trainer-thread-confinement --strict` passes.
