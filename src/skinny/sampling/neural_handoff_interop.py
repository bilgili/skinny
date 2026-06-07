"""GPU-shared-memory weight handoff (``--neural-handoff interop``).

Change ``neural-online-training`` (task 5.2). The real-time "best path": Vulkan
exports the neural weight (33) + bias (34) buffers with ``VK_KHR_external_memory``
(dedicated allocation); CUDA imports them with ``cudaImportExternalMemory`` →
``cudaExternalMemoryGetMappedBuffer`` and ``cudaMemcpyAsync``'s fresh weights
straight in with **no CPU round-trip**, then signals an exported timeline
semaphore (``cudaImportExternalSemaphore`` + ``cudaSignalExternalSemaphoresAsync``)
at the staged network version. The renderer's frame-end swap host-waits that value
(``vkWaitSemaphores``), so the CUDA write is provably resident before the buffers
are re-bound — tear-free.

The conduit is ``cuda-python`` (the ``[interop]`` extra); torch does not expose
the external-memory/-semaphore API. The module imports cleanly everywhere; every
operation is guarded by a capability check and raises a clear
``NotImplementedError`` off CUDA / off an external-memory Vulkan device, where
``--neural-handoff file`` is the fallback. Implemented and verified on an RTX 4090
(``tests/test_neural_interop.py``).
"""

from __future__ import annotations

import sys

import numpy as np

from .neural_handoff import NeuralWeightPublisher
from .neural_weights import NeuralWeights

__all__ = [
    "InteropWeightPublisher",
    "interop_available",
    "CudaExternalBuffer",
    "CudaExternalTimeline",
]


# ── cuda-python helpers ─────────────────────────────────────────────────────

def _rt():
    """The cuda-python runtime module (``cuda.bindings.runtime``)."""
    from cuda.bindings import runtime as rt
    return rt


def _check(ret):
    """cuda-python calls return ``(err, *out)``; raise on a non-success err and
    return the unwrapped output (scalar, tuple, or None)."""
    rt = _rt()
    if isinstance(ret, tuple):
        err, *out = ret
    else:
        err, out = ret, []
    if err != rt.cudaError_t.cudaSuccess:
        name = getattr(err, "name", err)
        raise RuntimeError(f"CUDA error {name}")
    if not out:
        return None
    return out[0] if len(out) == 1 else tuple(out)


def _handle_to_int(handle) -> int:
    """Coerce a Vulkan Win32 HANDLE / POSIX fd (as the python ``vulkan`` binding
    returns it) to the integer cuda-python's handle field expects. The ``vulkan``
    binding is cffi-based, so the handle comes back as a ``cdata 'void *'``."""
    if isinstance(handle, int):
        return handle
    try:
        import vulkan as vk
        return int(vk.ffi.cast("uintptr_t", handle))
    except Exception:  # noqa: BLE001 — fall through to ctypes / direct int
        pass
    import ctypes
    if isinstance(handle, ctypes.c_void_p):
        return int(handle.value or 0)
    return int(ctypes.cast(handle, ctypes.c_void_p).value or 0)


def _vk_device_uuid(ctx) -> bytes | None:
    """16-byte ``VkPhysicalDeviceIDProperties.deviceUUID`` for ``ctx``'s GPU, used
    to import on the matching CUDA device. None if it can't be queried."""
    try:
        import vulkan as vk

        id_props = vk.VkPhysicalDeviceIDProperties()
        props2 = vk.VkPhysicalDeviceProperties2(pNext=id_props)
        vk.vkGetPhysicalDeviceProperties2(ctx.physical_device, props2)
        return bytes(bytearray(id_props.deviceUUID)[:16])
    except Exception:  # noqa: BLE001 — UUID match is best-effort
        return None


def _select_cuda_device(ctx) -> int:
    """Set (and return) the CUDA device whose UUID matches the Vulkan device, so
    the imported memory lives on the same physical GPU. Falls back to device 0."""
    rt = _rt()
    want = _vk_device_uuid(ctx) if ctx is not None else None
    chosen = 0
    if want is not None:
        count = _check(rt.cudaGetDeviceCount())
        for dev in range(int(count)):
            props = _check(rt.cudaGetDeviceProperties(dev))
            uuid = bytes(bytearray(props.uuid.bytes)[:16])
            if uuid == want:
                chosen = dev
                break
    _check(rt.cudaSetDevice(chosen))
    return chosen


def _win32_handle_type() -> bool:
    """True on a platform whose external handles are Win32 (else POSIX fd)."""
    return sys.platform == "win32"


class CudaExternalBuffer:
    """A Vulkan-exported ``StorageBuffer`` imported into CUDA as a device pointer.

    Wraps ``cudaImportExternalMemory`` + ``cudaExternalMemoryGetMappedBuffer`` so
    the trainer can ``cudaMemcpy`` weights straight into the GPU memory the Vulkan
    buffer backs (no CPU round-trip). ``read`` exists for verification."""

    def __init__(self, ext_mem, device_ptr: int, size: int, alloc_size: int):
        self._ext_mem = ext_mem
        self.device_ptr = int(device_ptr)
        self.size = int(size)
        self.alloc_size = int(alloc_size)

    @classmethod
    def from_storage_buffer(cls, sb) -> CudaExternalBuffer:
        rt = _rt()
        handle = sb.export_handle()
        if handle is None:
            raise RuntimeError("StorageBuffer has no export handle (not external?)")
        _select_cuda_device(getattr(sb, "ctx", None))

        desc = rt.cudaExternalMemoryHandleDesc()
        if _win32_handle_type():
            desc.type = (rt.cudaExternalMemoryHandleType
                         .cudaExternalMemoryHandleTypeOpaqueWin32)
            desc.handle.win32.handle = _handle_to_int(handle)
        else:
            desc.type = (rt.cudaExternalMemoryHandleType
                         .cudaExternalMemoryHandleTypeOpaqueFd)
            desc.handle.fd = _handle_to_int(handle)
        desc.size = int(sb.alloc_size)
        desc.flags = int(rt.cudaExternalMemoryDedicated)
        ext_mem = _check(rt.cudaImportExternalMemory(desc))

        bd = rt.cudaExternalMemoryBufferDesc()
        bd.offset = 0
        bd.size = int(sb.alloc_size)
        bd.flags = 0
        device_ptr = _check(rt.cudaExternalMemoryGetMappedBuffer(ext_mem, bd))
        return cls(ext_mem, device_ptr, sb.size, sb.alloc_size)

    def write(self, data, stream=None) -> None:
        """Copy host ``data`` (bytes / np array) into the device buffer."""
        rt = _rt()
        buf = np.frombuffer(memoryview(data).cast("B"), dtype=np.uint8)
        n = int(buf.nbytes)
        if n > self.alloc_size:
            raise ValueError(f"write {n}B > buffer {self.alloc_size}B")
        kind = rt.cudaMemcpyKind.cudaMemcpyHostToDevice
        if stream is None:
            _check(rt.cudaMemcpy(self.device_ptr, buf.ctypes.data, n, kind))
        else:
            _check(rt.cudaMemcpyAsync(self.device_ptr, buf.ctypes.data, n, kind, stream))

    def read(self, nbytes: int) -> bytes:
        """Copy ``nbytes`` back from the device buffer (verification path)."""
        rt = _rt()
        out = np.empty(int(nbytes), dtype=np.uint8)
        _check(rt.cudaMemcpy(out.ctypes.data, self.device_ptr,
                             int(nbytes), rt.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return out.tobytes()

    def close(self) -> None:
        if self._ext_mem is not None:
            try:
                _check(_rt().cudaDestroyExternalMemory(self._ext_mem))
            finally:
                self._ext_mem = None


class CudaExternalTimeline:
    """A Vulkan-exported timeline semaphore imported into CUDA.

    The trainer's CUDA stream ``signal``s this at the staged network version after
    the weight ``cudaMemcpy``; the renderer host-waits the same value at the frame
    boundary (``ExternalTimelineSemaphore.wait``) so the write is provably resident
    before the buffers are re-bound — tear-free, no CPU round-trip."""

    def __init__(self, ext_sem):
        self._ext_sem = ext_sem

    @classmethod
    def from_semaphore(cls, sema) -> CudaExternalTimeline:
        rt = _rt()
        handle = sema.export_handle()
        if handle is None:
            raise RuntimeError("semaphore has no export handle (not external?)")
        _select_cuda_device(getattr(sema, "ctx", None))

        sd = rt.cudaExternalSemaphoreHandleDesc()
        if _win32_handle_type():
            sd.type = (rt.cudaExternalSemaphoreHandleType
                       .cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32)
            sd.handle.win32.handle = _handle_to_int(handle)
        else:
            sd.type = (rt.cudaExternalSemaphoreHandleType
                       .cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd)
            sd.handle.fd = _handle_to_int(handle)
        sd.flags = 0
        ext_sem = _check(rt.cudaImportExternalSemaphore(sd))
        return cls(ext_sem)

    def signal(self, value: int, stream=None) -> None:
        """Signal the timeline to ``value`` on ``stream`` (after the weight copies
        on the same stream)."""
        rt = _rt()
        sp = rt.cudaExternalSemaphoreSignalParams()
        sp.params.fence.value = int(value)
        sp.flags = 0
        _check(rt.cudaSignalExternalSemaphoresAsync(
            [self._ext_sem], [sp], 1, stream if stream is not None else 0))

    def wait(self, value: int, stream=None) -> None:
        """CUDA-side wait until the timeline reaches ``value`` (host-side wait lives
        on the Vulkan semaphore; this is for symmetry / a CUDA-ordered consumer)."""
        rt = _rt()
        wp = rt.cudaExternalSemaphoreWaitParams()
        wp.params.fence.value = int(value)
        wp.flags = 0
        _check(rt.cudaWaitExternalSemaphoresAsync(
            [self._ext_sem], [wp], 1, stream if stream is not None else 0))

    def close(self) -> None:
        if self._ext_sem is not None:
            try:
                _check(_rt().cudaDestroyExternalSemaphore(self._ext_sem))
            finally:
                self._ext_sem = None


def interop_available() -> tuple[bool, str]:
    """(supported, reason-if-not). True only on a box with ``cuda-python`` and a
    present CUDA device.

    The conduit is ``cuda-python`` (``cuda.bindings.runtime``), not torch: torch
    does not expose the external-memory/-semaphore API this handoff needs, and is
    only the *trainer's* dependency. The Vulkan-side capability probe
    (``VK_KHR_external_memory_{win32,fd}`` + the timeline / external-semaphore
    extensions on the render device) is wired by the renderer at buffer-export
    time; this module-level check only gates the CUDA half."""
    try:
        from cuda.bindings import runtime as rt
    except Exception:  # noqa: BLE001 — cuda-python absent → interop unavailable
        return False, "cuda-python not installed (CUDA runtime unavailable)"
    try:
        err, count = rt.cudaGetDeviceCount()
    except Exception as exc:  # noqa: BLE001 — driver/runtime load failure
        return False, f"CUDA runtime unavailable: {exc}"
    if err != rt.cudaError_t.cudaSuccess or count < 1:
        return False, "no CUDA device — interop requires CUDA"
    return True, ""


class InteropWeightPublisher(NeuralWeightPublisher):
    """GPU-shared weight handoff: CUDA writes weights straight into the
    Vulkan-exported binding-33/34 device memory and signals an exported timeline
    semaphore; the renderer's frame-end swap host-waits it. No CPU round-trip.

    The import is lazy (first ``publish``) so the publisher constructs cheaply and
    raises a clear ``NotImplementedError`` off the implemented path (no
    cuda-python / no CUDA), keeping ``--neural-handoff file`` the fallback."""

    # Generous host-wait ceiling at the frame boundary — a training cycle's
    # cudaMemcpy + signal completes in well under this; a timeout means the GPU
    # work never landed, which is a hard error, not a normal swap.
    _SWAP_WAIT_NS = 5_000_000_000

    def __init__(self, weights_buffer=None, biases_buffer=None,
                 timeline_semaphore=None,
                 expect_arch: tuple[int, int, int, int] | None = None,
                 precision=None, initial=None):
        # `initial` is accepted for a uniform `make_publisher` signature but unused:
        # the interop render buffers are seeded by the renderer's own upload, and
        # new weights flow GPU-side, never through a host-side initial snapshot.
        ok, reason = interop_available()
        if not ok:
            raise NotImplementedError(
                f"interop weight handoff unavailable: {reason}. "
                f"Use --neural-handoff file on this platform."
            )
        from .neural_weights import NeuralPrecision
        self._weights_buffer = weights_buffer
        self._biases_buffer = biases_buffer
        self._timeline = timeline_semaphore           # Vulkan side (host wait)
        self._expect = expect_arch
        self._precision = precision or NeuralPrecision.FP32
        self._render_version = 0
        self._staged_version = 0
        # Imported lazily on first publish.
        self._imported = False
        self._stream = None
        self._weights_cuda = None
        self._biases_cuda = None
        self._timeline_cuda = None

    def _ensure_imported(self) -> None:
        if self._imported:
            return
        if self._weights_buffer is None or self._biases_buffer is None:
            raise RuntimeError(
                "interop publisher needs the exported weights+biases buffers "
                "(bindings 33/34); none were provided")
        if self._timeline is None:
            raise RuntimeError(
                "interop publisher needs the exported timeline semaphore")
        rt = _rt()
        _select_cuda_device(getattr(self._weights_buffer, "ctx", None))
        self._stream = _check(rt.cudaStreamCreate())
        self._weights_cuda = CudaExternalBuffer.from_storage_buffer(self._weights_buffer)
        self._biases_cuda = CudaExternalBuffer.from_storage_buffer(self._biases_buffer)
        self._timeline_cuda = CudaExternalTimeline.from_semaphore(self._timeline)
        self._imported = True

    def publish(self, weights: NeuralWeights) -> int:
        """Trainer side: write the new weights+biases into the exported GPU buffers
        (storage dtype for the active precision) and signal the timeline at the
        staged version, all on one stream so the signal follows the copies."""
        if self._expect is not None:
            weights.assert_matches_shader(self._expect)
        self._ensure_imported()
        self._weights_cuda.write(weights.weight_bytes_for(self._precision), self._stream)
        self._biases_cuda.write(weights.bias_bytes_for(self._precision), self._stream)
        self._staged_version += 1
        self._timeline_cuda.signal(self._staged_version, self._stream)
        return self._staged_version

    def swap(self) -> bool:
        """Renderer side, at frame end: host-wait the timeline so the CUDA write is
        provably resident, then promote the render version. No buffer copy — the
        exported memory already holds the new weights."""
        if self._staged_version <= self._render_version:
            return False
        if not self._timeline.wait(self._staged_version, timeout_ns=self._SWAP_WAIT_NS):
            raise RuntimeError(
                f"interop swap timed out waiting for timeline {self._staged_version} "
                f"(CUDA weight-write did not complete)")
        self._render_version = self._staged_version
        return True

    def acquire_for_render(self) -> tuple[NeuralWeights | None, int]:
        # Weights live in the bound exported buffers; the renderer re-stamps the
        # version without re-uploading (the `nw is None` branch in the frame-end
        # swap), rather than acquiring a host-side NeuralWeights.
        return None, self._render_version

    def current_version(self) -> int:
        return self._render_version

    def close(self) -> None:
        """Release the imported CUDA resources (idempotent)."""
        for obj in (self._weights_cuda, self._biases_cuda, self._timeline_cuda):
            if obj is not None:
                try:
                    obj.close()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass
        self._weights_cuda = self._biases_cuda = self._timeline_cuda = None
        if self._stream is not None:
            try:
                _check(_rt().cudaStreamDestroy(self._stream))
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        self._imported = False
