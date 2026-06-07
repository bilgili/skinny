# Neural interop CUDA weight handoff (task 5.2) — design

## Context

Change `neural-online-training` ships two weight-handoff backends behind
`NeuralWeightPublisher` (`--neural-handoff`):

- `file` — trainer writes an NFW1 file; renderer hot-reloads it (CPU round-trip).
  Implemented, cross-platform.
- `interop` — CUDA writes new weights straight into the Vulkan-exported weight
  buffer; no CPU round-trip. The Vulkan **export** half (task 5.1) is done
  (`StorageBuffer(external=True)` + `export_handle()`); the CUDA **import** half
  (task 5.2) is a documented stub raising `NotImplementedError`.

This design fills the CUDA half and the small Vulkan-side timeline-semaphore
plumbing it needs, and verifies it on the target box (RTX 4090, driver 596.21,
CUDA-13-class).

### Environment facts that shape the design

- The runnable env (Python 3.13 venv with `vulkan` 1.3.275.1) is the **repo
  root itself** (`Scripts/`, `Lib/`, `pyvenv.cfg`). Work happens on a feature
  branch in the main checkout, not a separate worktree, because a worktree would
  not carry this env and "run & verify here" needs it.
- `vulkan` 1.3.275.1 exposes every required struct (timeline semaphore,
  external-semaphore win32, `VkMemoryDedicatedAllocateInfo`,
  `VkPhysicalDeviceIDProperties`). The two extension entry points
  `vkGetMemoryWin32HandleKHR` / `vkGetSemaphoreWin32HandleKHR` are loaded via
  `vkGetDeviceProcAddr` (the pattern the existing `export_handle()` already uses).
- `cuda-python` 13.3.1 is installable into the venv; torch does **not** expose the
  external-memory API, so cuda-python is the conduit for the handoff. torch stays
  the trainer's dependency only.

## Goal & data flow

```
trainer bakes host NeuralWeights
  └─ publish(nw):
        cudaMemcpyAsync(host weight/bias bytes → mapped binding 33/34)   ┐ GPU-ordered
        cudaSignalExternalSemaphoresAsync(stream, ++staged_version)       │ on one stream
renderer frame-end _online_frame_end_swap():
        publisher.swap(): timeline.wait(staged_version) → render_version=v ┘
        acquire_for_render() → (None, v): exported buffer already bound;
                                          bump FrameConstants + pass version stamp
```

Unbiasedness is unchanged from the file backend: weights are frozen per frame,
the swap happens only at the frame boundary, and the per-sample version stamp
keys each sample's density to the version that drew it. The online loop is
host-serialized (`drain → train → publish → render → swap`), so the host-side
timeline wait at the frame boundary is sufficient to guarantee CUDA writes have
landed before the next frame binds the buffers — no concurrent-thread tear.

## Components

### Vulkan side (additive, guarded — `vk_context.py`, `vk_compute.py`)

1. **Timeline + external-semaphore extensions** (`vk_context.py`): enable
   `VK_KHR_timeline_semaphore` (feature `timelineSemaphore`) and
   `VK_KHR_external_semaphore_win32` (`_fd` on Linux) with the same probe →
   try-enable → fall-back-without pattern the external-memory enable uses. Expose
   `supports_external_semaphore` and `_external_semaphore_handle_type`. Never
   destabilise the default render device.

2. **Dedicated allocation for external buffers** (`vk_compute.py`,
   `StorageBuffer`): when `external=True`, chain `VkMemoryDedicatedAllocateInfo`
   (and the dedicated buffer-create requirement) onto the device-local
   allocation. NVIDIA's CUDA import of an OPAQUE_WIN32 buffer requires a dedicated
   allocation. Expose `self.alloc_size` (the padded `buf_reqs.size`), which is the
   size CUDA imports (not the logical `self.size`).

3. **`ExternalTimelineSemaphore`** (`vk_compute.py`): a timeline `VkSemaphore`
   (`VkSemaphoreTypeCreateInfo(TIMELINE)`) created with
   `VkExportSemaphoreCreateInfo`. Methods:
   - `export_handle()` — `vkGetSemaphoreWin32HandleKHR` via `vkGetDeviceProcAddr`;
     None where unsupported.
   - `wait(value, timeout_ns)` — `vkWaitSemaphores`.
   - `value()` — `vkGetSemaphoreCounterValue`.
   - `destroy()`.

### CUDA half (`neural_handoff_interop.py`)

- `interop_available()` → `(ok, reason)`: require importable `cuda.bindings.runtime`
  (cuda-python) **and** a present CUDA device. Drop the torch dependency.
- `InteropWeightPublisher(weights_buffer, biases_buffer, timeline_semaphore,
  expect_arch, precision)`:
  - `_import()` (lazy, once, cached): select the CUDA device whose UUID matches
    `VkPhysicalDeviceIDProperties.deviceUUID`; for **both** weights and biases,
    `cudaImportExternalMemory` (win32, dedicated flag) +
    `cudaExternalMemoryGetMappedBuffer(0, alloc_size)` → device pointers;
    `cudaImportExternalSemaphore` (win32, timeline) for the semaphore. Coerce the
    Vulkan Win32 HANDLE to the integer the cuda-python handle field expects.
  - `publish(nw)`: cast to the storage dtype via `weight_bytes_for(precision)` /
    `bias_bytes_for(precision)` (fp16 in the half modes, fp32 otherwise);
    `cudaMemcpyAsync` host→mapped for weights and biases on the publisher stream;
    `cudaSignalExternalSemaphoresAsync(stream, ++staged_version)`. Return
    staged_version. Non-blocking — ordering rides the semaphore.
  - `swap()`: `timeline.wait(staged_version)` (host wait at the frame boundary →
    writes provably landed), promote `render_version = staged_version`, return
    True iff it advanced.
  - `acquire_for_render()` → `(None, render_version)`: weights live in the bound
    exported buffers; the renderer re-stamps the version without re-uploading.
  - `current_version()` → `render_version`.
- Off the implemented path (no cuda-python / no CUDA / no external-memory),
  construction raises a clear `NotImplementedError` pointing at
  `--neural-handoff file`.

### Renderer wiring (`renderer.py`)

- Under `--neural-handoff interop`: allocate the exportable
  `ExternalTimelineSemaphore`; bindings 33/34/35 are already allocated external.
- `enable_online_training`: pass `weights_buffer=neural_weights_buffer`,
  `biases_buffer=neural_biases_buffer`, `timeline_semaphore=…`, and `precision`
  to the publisher (replacing the single `exported_buffer` kwarg).
- `_online_frame_end_swap` is unchanged: its existing `nw is None` branch already
  handles the interop case (version stamp bump, no re-upload).

## Verification (on this 4090)

`pip install cuda-python` into the venv, then `tests/test_neural_interop.py`,
hardware-guarded (skip if no cuda-python / no CUDA / no external-memory Vulkan):

- **memory roundtrip**: headless `VulkanContext` → external `StorageBuffer` →
  CUDA import → `cudaMemcpy` known bytes in → copy back → assert equal.
- **semaphore**: export VK timeline → CUDA import →
  `cudaSignalExternalSemaphoresAsync(v)` → `timeline.wait(v)` returns within
  timeout.
- **publish→swap**: `InteropWeightPublisher.publish(nw)` then `swap()` advances
  the version; the mapped buffer holds `nw.weight_bytes_for(precision)`.
- **guard**: forced-unavailable `interop_available()` still raises the clear
  `NotImplementedError`.

`file` handoff stays fully intact. `ruff check src/` and `py_compile` stay green.

## Docs to update

- `docs/Architecture.md` — binding-map note on 33/34 (dedicated external alloc) +
  timeline-semaphore; online-training section marks the interop CUDA half live.
- `docs/Wavefront.md` — frame-end swap section (interop path).
- `README.md` — `--neural-handoff {file,interop}` note (interop now functional on
  CUDA + external-memory Vulkan).
- `openspec/changes/neural-online-training/tasks.md` — close 5.2; update 7.3 with
  the interop measurement once run.

## Risks tracked during build

- Dedicated-allocation requirement for CUDA import (mitigated above).
- Win32 HANDLE → cuda-python handle-field coercion (int vs void*).
- CUDA/Vulkan device-UUID match (single GPU here; still matched defensively).
- cuda-python 13 API shape (`cudaExternalMemoryHandleDesc` nested `.handle.win32`,
  `cudaExternalSemaphoreHandleDesc`, async signal/wait param structs).

## Out of scope

- Concurrent (separate-thread) trainer overlapping a frame submit — the shipped
  loop is host-serialized; double-region ping-pong is a later optimisation.
- torch in the renderer venv / the full file-vs-interop online benchmark (task
  7.3) beyond what the interop seam itself needs to be exercised.
