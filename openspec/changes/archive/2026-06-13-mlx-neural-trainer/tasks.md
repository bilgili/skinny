# Tasks — mlx-neural-trainer

## 1. Packaging and availability gate

- [x] 1.1 Add `mlx` optional-dependency extra to `pyproject.toml` (floor-pinned, e.g. `mlx>=0.30`); install into the dev venv with `pip install -e ".[mlx,dev]"`
- [x] 1.2 Add import/platform availability helper in `training_backends.py` (lazy `import mlx.core` + Apple-Silicon Metal check, no top-level import), mirroring `_torch_cuda()`

## 2. MlxTrainingBackend

- [x] 2.1 Implement the `ConditionalSplineFlow2D` model as an `mlx.nn.Module` mirroring the torch layout (6 couplings × 3 Linears, `_LINEAR_IDX` mapping) with `warm_start` loading NFW1 weights into MLX arrays
- [x] 2.2 Implement the contribution-weighted MLE loss + `mlx.nn.value_and_grad` + `mlx.optimizers.Adam` step in `update`, with `mx.eval` barrier per step (design D2) and optimizer state retained across cycles
- [x] 2.3 Implement `export` → in-memory fp32 NFW1 `NeuralWeights` (per-tensor `np.array` readback), byte-format-identical to numpy/torch exports
- [x] 2.4 Implement `is_available` / `supports_precision` (fp32 always; fp16 per design D5)

## 3. Selection and CLI

- [x] 3.1 Replace the `mlx` raise in `make_training_backend` with construction + clear unavailability error naming the missing piece (package vs Metal device)
- [x] 3.2 Extend `auto` precedence to cuda > mlx > numpy (design D4) and log the selected backend at startup
- [x] 3.3 Update `--neural-trainer` help text in `cli_common.py` (drop "reserved"), and the stale comments at `renderer.py:1227` / `neural_trainer.py` docstring

## 4. fp16 on MLX

- [x] 4.1 Implement fp16 compute path: fp32 master weights, fp16 cast for forward/backward, fp32 loss accumulation and gradient apply (design D5)
- [x] 4.2 Implement non-finite-loss fallback to fp32 with one-time warning; verify exported weights stay fp32 NFW1

## 5. Tests (all MLX tests `pytest.importorskip("mlx")`)

- [x] 5.1 Extend `tests/test_training_backends.py`: explicit `mlx` selection succeeds on this host, clear error path when forced unavailable (monkeypatched import), `auto` → mlx precedence, capability gating
- [x] 5.2 Add MLX↔numpy parity test (fixed batch + seed, one cycle, weights within the documented tolerance used by `test_neural_parity.py`); pin/match Adam hyperparameters, fall back to hand-rolled Adam in MLX if the library variant cannot match
- [x] 5.3 Add multi-cycle warm-state test: second cycle continues from prior model/optimizer state; per-cycle time/memory stable (no lazy-graph growth)
- [x] 5.4 Add fp16 tests: fp16 cycle exports fp32 NFW1; injected non-finite loss triggers fp32 fallback + single warning
- [x] 5.5 Run full suite `pytest -m 'not gpu'` plus the new MLX tests; ruff clean

## 6. Integration check

- [x] 6.1 End-to-end Mac combo smoke run: `--backend metal --execution-mode wavefront --proposals bsdf,neural --online-training --neural-trainer mlx --neural-handoff interop` — training cycles complete, weights publish, no regression vs `--neural-trainer cpu`

## 7. Docs

- [x] 7.1 Update compatibility matrices: `README.md` and `CLAUDE.md` `--neural-trainer` rows (`mlx` → ✅ Apple-Silicon Metal; `auto` on Mac → mlx if importable else numpy; supported Mac combo example)
- [x] 7.2 Update `docs/NeuralGuiding.md` trainer-backend section and `CHANGELOG.md`

## 8. Finalize

- [x] 8.1 `openspec validate mlx-neural-trainer` passes; review artifacts vs implementation for drift
