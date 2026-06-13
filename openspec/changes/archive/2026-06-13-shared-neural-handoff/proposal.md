## Why

Online neural-proposal training has two weight-handoff transports today: `file`
(an NFW1 disk round-trip the renderer hot-reloads — portable but pays disk I/O
on every publish) and `interop` (zero-copy GPU memory, but only on CUDA-with-
Vulkan-external-memory or a Metal unified-memory device). The trainer already
runs as a same-process daemon thread, so on every host the `file` backend writes
weights to disk only to read them straight back into the same address space.
There is no transport that skips the disk yet stays portable — a host without a
CUDA/UMA `interop` path is forced onto the file round-trip even though both ends
share memory.

## What Changes

- Add a third `--neural-handoff` value, **`shared`**: an in-process CPU
  double-buffer that holds `NeuralWeights` in RAM and hands them across the
  trainer→render boundary with no disk write and no GPU-interop requirement.
- Publish deep-copies the staged `NeuralWeights` (its three numpy arrays) so the
  trainer may keep mutating its working set while the render buffer stays frozen
  for the duration of a frame — same frozen-buffer / frame-boundary-swap /
  version-increment contract as `file` and `interop`.
- The renderer uploads swapped weights to its GPU buffers through the existing
  swap path (identical to what `file` does after a load); `shared` does not write
  GPU buffers directly (that is `interop`'s job).
- Wire `shared` through `make_publisher`, the `--neural-handoff` argparse choices
  + `SKINNY_NEURAL_HANDOFF` env, and the front-end persistence of
  `_neural_handoff_kind`.
- Update the compatibility matrix and handoff docs (README, NeuralGuiding,
  Architecture, Wavefront, PythonAPI, CHANGELOG).
- No change to `file`, `interop`, the trainer, the replay buffer, the record
  drain, or the on-disk/handoff weight format. Default stays `file`; behavior is
  byte-identical unless `shared` is selected.

## Capabilities

### New Capabilities
<!-- none — this extends an existing capability -->

### Modified Capabilities
- `neural-online-training`: the "Two selectable weight-handoff backends"
  requirement becomes three — add the in-process `shared` CPU double-buffer
  backend (no disk, no CUDA/UMA), available on any platform, byte-faithful to a
  `file` publish of the same weights, behind the same publisher interface and
  frame-boundary swap contract.

## Impact

- **Code**: new `src/skinny/sampling/neural_handoff_shared.py`
  (`SharedWeightPublisher`); `make_publisher` in
  `src/skinny/sampling/neural_handoff.py` (+ module docstring);
  `--neural-handoff` choices/help in `src/skinny/cli_common.py`;
  `renderer.enable_online_training` only needs the new value to flow through
  (no `interop`-style buffer kwargs for `shared`). Persistence in `app.py` /
  `ui/qt/app.py` already keys off `_neural_handoff_kind` — only the choices list
  widens.
- **APIs / CLI**: `--neural-handoff {file,interop,shared}` (+ env
  `SKINNY_NEURAL_HANDOFF`). `make_publisher("shared", …)` public factory value.
- **Dependencies**: none added — numpy only, already a hard dependency.
- **Docs**: README compatibility matrix + `--neural-handoff` row,
  `docs/NeuralGuiding.md`, `docs/Architecture.md`, `docs/Wavefront.md`,
  `docs/PythonAPI.md`, `CHANGELOG.md`.
- **Tests**: a publisher unit test (publish→swap→acquire, version increment,
  frozen-buffer isolation under trainer mutation, byte-parity vs the `file`
  backend) plus a `make_publisher("shared")` factory test.
