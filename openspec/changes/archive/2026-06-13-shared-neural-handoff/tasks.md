## 1. SharedWeightPublisher

- [x] 1.1 Add `src/skinny/sampling/neural_handoff_shared.py` with
  `SharedWeightPublisher(NeuralWeightPublisher)` — `__init__(self, initial=None,
  expect_arch=None)`, double-buffer fields (`_render`/`_render_version`,
  `_pending`/`_pending_version`, `_staged_version`) mirroring
  `FileWeightPublisher`.
- [x] 1.2 Implement `publish(weights)`: bump `_staged_version`, store a
  byte-faithful private copy as pending (per design D3 — prefer routing the
  staged weights through `write_neural_weights`→`load_neural_weights` over an
  in-memory `io.BytesIO`; fall back to copying the three `NeuralWeights` numpy
  arrays). Return the staged version. No filesystem writes.
- [x] 1.3 Implement `swap()` (promote pending→render, clear pending, return
  whether a swap occurred), `acquire_for_render()` (return `(_render,
  _render_version)`), and `current_version()` — identical contract to
  `FileWeightPublisher`.
- [x] 1.4 If 1.2 needs a bytes/`BytesIO` overload, add a minimal file-like entry
  point to `write_neural_weights`/`load_neural_weights` in
  `src/skinny/sampling/neural_weights.py` without changing the on-disk format or
  existing call sites.

## 2. Factory and CLI wiring

- [x] 2.1 In `src/skinny/sampling/neural_handoff.py` `make_publisher`, add a
  `kind == "shared"` branch importing and returning `SharedWeightPublisher`
  (passing `initial`/`expect_arch`, no interop kwargs). Update the unknown-kind
  error to list `'file', 'interop', 'shared'`, and update the module + factory
  docstrings to describe the third backend.
- [x] 2.2 In `src/skinny/cli_common.py`, widen the `--neural-handoff` `choices`
  to `("file", "interop", "shared")` and extend the help text to describe
  `shared` (in-process CPU double-buffer, no disk, no GPU interop, any platform;
  contrast with `interop`'s direct GPU-buffer write).
- [x] 2.3 Confirm `renderer.enable_online_training` flows `kind == "shared"`
  through unchanged (interop-only kwargs stay gated on `kind == "interop"`); add
  no `shared`-specific kwargs.
- [x] 2.4 Confirm front-end persistence round-trips `shared`: `app.py` and
  `ui/qt/app.py` already save/restore `_neural_handoff_kind` as a string —
  verify the restore path accepts `shared` (no allow-list narrower than the CLI
  choices).

## 3. Tests

- [x] 3.1 Add `tests/` unit test for `SharedWeightPublisher`: publish→swap→
  `acquire_for_render` returns the published weights and `current_version`
  increments; `swap()` with nothing pending returns False.
- [x] 3.2 Test frozen-buffer isolation: publish, then mutate the source
  `NeuralWeights` arrays in place, assert the acquired render weights are
  unchanged until a later publish+swap.
- [x] 3.3 Test byte-parity with the `file` backend: publish identical weights
  through `FileWeightPublisher` and `SharedWeightPublisher`, assert
  `weights`/`biases`/`headers` arrays are byte-equal after swap.
- [x] 3.4 Test `make_publisher("shared", …)` returns a `SharedWeightPublisher`
  and that an unknown kind error message names all three backends.

## 4. Docs

- [x] 4.1 `README.md` — add the `shared` row/entry to the Compatibility matrix
  `--neural-handoff` table and any `--neural-handoff` usage line; keep the
  CLAUDE.md compatibility matrix wording in sync.
- [x] 4.2 `CLAUDE.md` — update the `--neural-handoff` row in the compatibility
  matrix to list `file` | `interop` | `shared`.
- [x] 4.3 `docs/NeuralGuiding.md` — document the `shared` handoff backend
  alongside `file` and `interop` (transport, when to use, no-disk/no-interop
  trade-off).
- [x] 4.4 `docs/Architecture.md` and `docs/Wavefront.md` — note the third
  publisher backend where the handoff seam is described.
- [x] 4.5 `docs/PythonAPI.md` — document `make_publisher("shared", …)` /
  `SharedWeightPublisher` if the handoff factory is part of the public Python
  surface there.
- [x] 4.6 `CHANGELOG.md` — add an entry for the `shared` neural-handoff backend.

## 5. Verification

- [x] 5.1 `.venv/bin/ruff check src/` clean on the new/modified files.
- [x] 5.2 `.venv/bin/pytest` for the new handoff tests passes (CPU-only, no GPU).
- [x] 5.3 `openspec validate shared-neural-handoff --strict` passes.
