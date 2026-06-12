# Design — Metal Neural Interop

## Context

Online neural-proposal training hands trained weights to the renderer through the
`NeuralWeightPublisher` seam (`sampling/neural_handoff.py`): `publish()` on the
trainer thread, `swap()` on the render thread at the frame boundary,
`acquire_for_render()` for the frozen render-side weights. Two backends exist:

- `file` (`neural_handoff_file.py`) — NFW1 serialize → disk → reload → staged
  re-upload. Works everywhere; the only option on Metal today.
- `interop` (`neural_handoff_interop.py`) — CUDA imports the Vulkan-exported
  binding-33/34 memory (`VK_KHR_external_memory`) and an exported timeline
  semaphore; `publish()` is `cudaMemcpyAsync` + signal, `swap()` host-waits the
  timeline and bumps the version. `acquire_for_render()` returns `(None, version)`
  — the renderer re-stamps the version without re-uploading (`nw is None` branch).

On Metal, `metal_compute.StorageBuffer` allocates `MemoryType.device_local`, sets
`external=False`, and `export_handle()` returns `None` (`metal_compute.py:160,203`);
`metal_context.py` keeps `supports_external_memory`/`supports_external_semaphore`
`False` per the metal-backend spec. The weight/bias/layer buffers (bindings 33–35,
`renderer.py:3350–3367`) are plain device-local uploads. So `--neural-handoff
interop` on Metal raises (no CUDA), and online training pays the file round-trip
per publish.

Apple-Silicon Metal is a unified-memory architecture: a shared-storage
(`MTLStorageModeShared`) buffer is the same physical memory for CPU and GPU. The
interop opportunity is not an exported-handle import like CUDA — it is making the
weight buffers host-writable and copying staged bytes in at the frame boundary.

## Goals / Non-Goals

**Goals:**

- `--neural-handoff interop` works on the Metal backend: weight publish with no
  disk round-trip and no NFW1 serialize/parse, same unbiasedness contract
  (a sample drawn under version *N* is evaluated against version *N*'s density;
  staleness raises variance only).
- Same bytes in the buffer as the file path for every `NeuralPrecision`
  (fp32 / fp16-storage / fp8-e4m3): reuse `weight_bytes_for` / `bias_bytes_for`.
- Clear capability surface: new `supports_shared_memory` flag; the existing
  unavailability guard stays for hosts with neither CUDA nor Metal UMA.
- No shader or descriptor-binding changes.

**Non-Goals:**

- True trainer-GPU→render-GPU sharing (importing torch-MPS's MTLBuffer into the
  slang-rhi device). The trainer still produces host bytes (`.cpu()` on an MPS
  tensor); on UMA that is one memcpy, not a PCIe transfer.
- Zero-copy path-record drain (trainer reading bindings 36/37 via shared
  storage). Follow-up.
- `supports_external_memory=true` on Metal — no exported handles are involved.
- Vulkan/CUDA interop path changes — `InteropWeightPublisher` is untouched.

## Decisions

**D1 — Shared-storage staging at swap, not a timeline semaphore.**
`MetalSharedWeightPublisher.publish()` only stages the precision-cast bytes in
host memory (double-buffered, lock-protected) and bumps the staged version;
`swap()` — already called on the render thread at the frame boundary — copies the
staged bytes into the shared weight/bias buffers and promotes the version. The
write happens on the only thread that encodes GPU work, between frames, so the
GPU never reads a half-written network; no MTLSharedEvent needed.
*Alternative considered:* trainer-thread writes + MTLSharedEvent fencing (CUDA
mirror). Rejected: cross-thread writes into memory a in-flight command buffer may
read require event plumbing through slang-rhi that it does not expose; the staged
copy is a few hundred KB at most and runs in microseconds on UMA.

**D2 — One user-facing handoff kind; factory dispatches by buffer backend.**
The flag stays `--neural-handoff file|interop`. `make_publisher("interop", ...)`
inspects the passed `weights_buffer`: a `metal_compute.StorageBuffer` selects
`MetalSharedWeightPublisher`; otherwise the CUDA `InteropWeightPublisher` (which
keeps its own availability guard). Construction with no usable backend raises
`NotImplementedError` naming both requirements ("needs CUDA+external-memory or a
Metal UMA device — use --neural-handoff file").
*Alternative:* a third kind `interop-metal`. Rejected: the user intent is "GPU
handoff, no file"; which mechanism applies is a backend property, like
`--backend auto`.

**D3 — `StorageBuffer(shared=True)` mode + `write_in_place()`.**
`metal_compute.StorageBuffer` gains `shared: bool = False`. When set, the buffer
allocates from the host-visible heap (slang-rhi `MemoryType.upload`, which maps to
`MTLStorageModeShared` on Apple Silicon) and exposes `write_in_place(data,
offset=0)`. Implementation prefers a mapped-pointer write if the installed
slangpy/slang-rhi build exposes one (`map`/`cursor`); otherwise falls back to
`copy_from_numpy` — still no disk, no NFW1 parse, one host copy on UMA. The
host-shadow bookkeeping stays so `upload_range` composes. `external` remains a
no-op and `export_handle()` still returns `None`.
*Alternative:* ctypes into the raw MTLBuffer `contents()` pointer via the native
handle. Rejected as primary path: fragile across slangpy versions; acceptable
later as an optimization behind the same `write_in_place` signature.

**D4 — Renderer allocates bindings 33/34 shared when Metal interop is active.**
Mirroring the Vulkan `external=_ext_neural` pattern (`renderer.py:3364`): on the
Metal backend, when `_neural_handoff_kind == "interop"` and the context reports
`supports_shared_memory`, the weight/bias buffers are constructed with
`shared=True`. Binding 35 (layer headers) is immutable after build and stays
device-local. Initial weight upload goes through the existing `upload_sync`
(works on either heap).

**D5 — Capability flag `supports_shared_memory`.**
`MetalContext` probes the device (slangpy device-info unified-memory query when
available; Apple-Silicon Metal default `True`, conservative `False` otherwise).
`VulkanContext` does not grow the flag; consumers read it with
`getattr(ctx, "supports_shared_memory", False)`. `supports_external_memory`
stays `False` on Metal.

**D6 — `acquire_for_render()` returns `(None, version)`** exactly like the CUDA
publisher, so the renderer's existing `nw is None` re-stamp branch
(`renderer.py:7702` area) works unchanged. `enable_online_training` extends its
`kind == "interop"` block to pass the Metal buffers (no timeline semaphore
kwarg on Metal).

## Risks / Trade-offs

- [swap() copies on the render thread → frame-time blip on publish frames] →
  Bytes are small (typ. ≤ a few hundred KB for the MLP at fp32; less at
  fp16/fp8); copy is host-memcpy on UMA. Measure in the parity test; if visible,
  split weights/biases across two frame boundaries.
  **Measured (task 4.2, Apple M5 Pro):** shipped-size fp32 weights+biases
  (418 776 B) land in 0.04–0.09 ms per swap — invisible against a frame; no
  split needed.
- [A command buffer in flight could still read the shared buffer while swap()
  writes] → swap() runs at the renderer's frame-end point, after the frame's
  submit-and-wait; the Metal driver (`metal_wavefront.py`) synchronizes the frame
  before the swap hook runs. Verify ordering in implementation and add a
  `wait_for_idle()` guard in `write_in_place` if the frame-end contract is
  weaker on Metal than on Vulkan.
- [slangpy `MemoryType.upload` buffer might reject storage-usage binding on some
  builds] → probe at allocation; on failure fall back to device-local +
  `upload_sync` (publisher still works — degenerates to the staged-upload path,
  logged once).
- [Torch-MPS `.cpu()` sync stalls the trainer thread] → trainer is async by
  design; the stall bounds publish latency, not render frame time. fp8/fp16
  casts already happen host-side in the file path (`weight_bytes_for`).
- [Thermal/compile discipline] — Metal GPU tests follow the repo rule: one
  guarded Metal-compile process at a time; sweeps run `-m 'not gpu'`.

## Open Questions

- Does the pinned slangpy build expose a mapped pointer / persistent map for
  upload-heap buffers? (Determines whether `write_in_place` is zero-copy or one
  memcpy. Resolve in task 1; both satisfy the spec.)
  **Resolved (task 1.1, slangpy 0.42.0, Apple M5 Pro):** no `map()`/`unmap()` on
  the Python `Buffer` surface; `write_in_place` is one host memcpy via
  `copy_from_numpy`. A `MemoryType.upload` buffer accepts full storage usage
  (`shader_resource | unordered_access | copy_*`), binds to a compute kernel, and
  dispatches observe host `copy_from_numpy` rewrites between submissions with no
  staging call. `Buffer.native_handle` returns the raw `MTLBuffer`
  (`NativeHandleType.MTLBuffer`), so a ctypes `contents()` zero-copy write stays
  available later behind the same `write_in_place` signature.
- Exact frame-end swap call-site ordering on the Metal wavefront driver — confirm
  the previous command buffer is complete before `swap()` (drives whether the
  `wait_for_idle()` guard is needed).
