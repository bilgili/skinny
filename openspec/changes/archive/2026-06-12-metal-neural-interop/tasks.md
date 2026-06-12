# Tasks — metal-neural-interop

## 1. Metal buffer plumbing (D3, D5)

- [x] 1.1 Probe the pinned slangpy build: can an upload-heap (`MemoryType.upload`)
      buffer carry storage usage and bind to a compute kernel, and is a mapped
      pointer / persistent map exposed? Record findings in design.md Open
      Questions (resolves the zero-copy vs one-memcpy question).
- [x] 1.2 `metal_compute.StorageBuffer`: add `shared: bool = False` ctor kwarg —
      shared mode allocates host-visible storage; add `write_in_place(data,
      offset=0)` (mapped-pointer write when available, `copy_from_numpy`
      fallback; device-local fallback path logs once and degenerates to
      `upload_sync`). Keep `external` no-op and `export_handle() -> None`.
- [x] 1.3 `MetalContext`: add `supports_shared_memory` capability flag (probe
      unified memory; conservative `False` when unknown); leave
      `supports_external_memory`/`supports_external_semaphore` `False`.
- [x] 1.4 GPU-marked test: shared-storage buffer in-place host write observed by
      a compute dispatch; non-shared buffer behavior unchanged; flag scenario
      (`supports_shared_memory` true on Apple Silicon, external flags false).

## 2. Metal publisher (D1, D2, D6)

- [x] 2.1 New `sampling/neural_handoff_interop_metal.py`:
      `MetalSharedWeightPublisher(NeuralWeightPublisher)` — `publish()` stages
      precision-cast bytes (`weight_bytes_for`/`bias_bytes_for`) under a lock and
      bumps the staged version; `swap()` (render thread, frame boundary) writes
      staged bytes via `write_in_place` into the bindings-33/34 buffers and
      promotes the version; `acquire_for_render()` returns
      `(None, render_version)`; `close()` idempotent.
- [x] 2.2 Availability guard `metal_interop_available()` — requires a Metal
      shared-capable context; failure message names `--neural-handoff file`.
- [x] 2.3 `make_publisher("interop", ...)` dispatches by `weights_buffer` backend:
      `metal_compute.StorageBuffer` → Metal publisher, else CUDA
      `InteropWeightPublisher`; no-backend error names both requirements
      (CUDA+external-memory or Metal UMA).
- [x] 2.4 Unit tests (no GPU): staging/swap version protocol, precision
      byte-faithfulness vs the file path (fp32/fp16/fp8 e4m3), torn-write guard
      (publish during swap never yields mixed-version bytes), arch mismatch
      raises via `assert_matches_shader`.

## 3. Renderer wiring (D4)

- [x] 3.1 Allocate bindings 33/34 with `shared=True` on Metal when
      `_neural_handoff_kind == "interop"` and the context reports
      `supports_shared_memory` (mirror of the Vulkan `external=_ext_neural`
      pattern); binding 35 stays device-local; initial upload unchanged.
- [x] 3.2 `enable_online_training`: extend the `kind == "interop"` block — on
      Metal pass the Metal weight/bias buffers (no timeline-semaphore kwarg);
      verify the existing `nw is None` frame-end re-stamp branch covers the
      Metal publisher.
- [x] 3.3 Confirm the frame-end swap call-site ordering on the Metal wavefront
      driver (previous command buffer complete before `swap()`); add a
      `wait_for_idle()` guard in `write_in_place` only if the contract is weaker
      than Vulkan's (design Open Question 2).
- [x] 3.4 `cli_common.py`: `--neural-handoff` help text mentions Metal UMA
      interop; validation unchanged (same two kinds).

## 4. Parity verification

- [x] 4.1 GPU-marked end-to-end test (`tests/test_neural_interop_metal.py`):
      Metal backend, online training a few cycles with `--neural-handoff
      interop` vs `--neural-handoff file` from the same seed/records — network
      versions advance, published buffer bytes match, renders match. Respect the
      thermal rule: single guarded Metal-compile process, mark `gpu`.
- [x] 4.2 Measure the swap-frame cost of the staged copy (publish-frame vs
      steady-state frame time); note the number in the change; split
      weights/biases across two boundaries only if visible.
- [x] 4.3 Guard test: `--neural-handoff interop` on a host with neither CUDA nor
      Metal UMA raises `NotImplementedError` naming the file fallback (mock the
      availability probes).

## 5. Docs + housekeeping

- [x] 5.1 `docs/NeuralGuiding.md`: handoff section gains the Metal UMA interop
      path (publish→stage, swap→in-place write, sync rationale).
- [x] 5.2 `docs/Architecture.md`: capability-flag table adds
      `supports_shared_memory`; note bindings 33/34 shared-storage mode on
      Metal.
- [x] 5.3 `README.md` flag docs + `CHANGELOG.md` entry.
- [x] 5.4 `ruff check src/`, full `pytest -m 'not gpu'` sweep, then the guarded
      GPU tests; `openspec validate metal-neural-interop`.
