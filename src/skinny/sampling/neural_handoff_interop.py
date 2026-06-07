"""GPU-shared-memory weight handoff (``--neural-handoff interop``).

Change ``neural-online-training``. The real-time "best path": Vulkan exports the
neural weight buffer (bindings 33/34/35) with ``VK_KHR_external_memory``; CUDA
imports it with ``cudaImportExternalMemory`` and writes updated weights into it
with **no CPU round-trip**. A timeline semaphore (also external) orders
CUDA-writes vs Vulkan-reads so a swap is tear-free.

This is hardware-bound and **untestable on Mac** (no CUDA, no Vulkan external
-memory on MoltenVK). The module imports cleanly everywhere; every operation is
guarded by a capability check and raises a clear ``NotImplementedError`` off-CUDA.
The NVIDIA box fills the import + sync internals and benchmarks file-vs-interop.
"""

from __future__ import annotations

from .neural_handoff import NeuralWeightPublisher
from .neural_weights import NeuralWeights

__all__ = ["InteropWeightPublisher", "interop_available"]


def interop_available() -> tuple[bool, str]:
    """(supported, reason-if-not). True only on a CUDA box whose Vulkan device
    advertises the external-memory + external-semaphore extensions."""
    try:
        import torch
    except ImportError:
        return False, "torch not installed (CUDA runtime unavailable)"
    if not torch.cuda.is_available():
        return False, "no CUDA device — interop requires CUDA"
    # The Vulkan-side capability probe (VK_KHR_external_memory_{fd,win32},
    # VK_KHR_timeline_semaphore on the render device) is wired by the renderer at
    # buffer-export time; this module-level check only gates the CUDA half.
    return True, ""


class InteropWeightPublisher(NeuralWeightPublisher):
    def __init__(self, exported_buffer=None, timeline_semaphore=None,
                 expect_arch: tuple[int, int, int, int] | None = None):
        ok, reason = interop_available()
        if not ok:
            raise NotImplementedError(
                f"interop weight handoff unavailable: {reason}. "
                f"Use --neural-handoff file on this platform."
            )
        # On CUDA: import the Vulkan-exported buffer once and keep the mapping.
        self._exported_buffer = exported_buffer
        self._timeline = timeline_semaphore
        self._expect = expect_arch
        self._version = 0
        self._cuda_ptr = None  # set by _import_external_memory on the NVIDIA box
        # The Vulkan side (task 5.1) allocates bindings 33/34/35 as exportable and
        # hands the OS memory handle here; the CUDA import below consumes it.
        self._export_handle = (
            exported_buffer.export_handle() if exported_buffer is not None else None)
        self._buffer_size = getattr(exported_buffer, "size", 0)

    def _import_external_memory(self):
        # NVIDIA-box implementation seam (task 5.2). The Vulkan export half is wired
        # (task 5.1): `self._export_handle` is the OS handle to the device memory
        # backing bindings 33/34/35 and `self._buffer_size` its byte size. The
        # remaining CUDA step:
        #   desc = cudaExternalMemoryHandleDesc(
        #       type=cudaExternalMemoryHandleTypeOpaqueWin32,    # *OpaqueFd on Linux
        #       handle.win32.handle=self._export_handle, size=self._buffer_size)
        #   extMem = cudaImportExternalMemory(desc)
        #   self._cuda_ptr = cudaExternalMemoryGetMappedBuffer(extMem, 0, self._buffer_size)
        # via cuda-python / a ctypes shim to cudart (torch does not expose the
        # external-memory API). publish() then cudaMemcpy's weights into _cuda_ptr.
        raise NotImplementedError(
            "interop external-memory import is implemented on the NVIDIA box; "
            f"Vulkan export handle is {'present' if self._export_handle else 'unavailable'} "
            f"(size={self._buffer_size}B) — cudaImportExternalMemory consumes it."
        )

    def publish(self, weights: NeuralWeights) -> int:
        # Copy weights into the imported device buffer (cudaMemcpyAsync) and
        # signal the timeline semaphore so the renderer reads the new version.
        raise NotImplementedError(
            "interop publish is implemented on the NVIDIA box (CUDA→Vulkan write)"
        )

    def swap(self) -> bool:
        # Renderer waits on the timeline value for the staged version at frame end;
        # no CPU buffer copy — the GPU memory already holds the new weights.
        # Timeline-semaphore sync contract (task 5.2): the trainer's CUDA write
        # signals an exported VK_KHR_external_semaphore (timeline) at the staged
        # version value; the renderer's frame-end swap waits that value before
        # binding the weights, so a CUDA write in flight can never tear a frame.
        raise NotImplementedError(
            "interop swap is implemented on the NVIDIA box (timeline-semaphore wait)"
        )

    def acquire_for_render(self) -> tuple[NeuralWeights | None, int]:
        raise NotImplementedError(
            "interop weights live in GPU memory; the renderer binds the exported "
            "buffer directly rather than acquiring a host-side NeuralWeights"
        )

    def current_version(self) -> int:
        return self._version
