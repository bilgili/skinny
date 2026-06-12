"""Metal UMA shared-storage weight handoff (``--neural-handoff interop`` on the
native Metal backend; change ``metal-neural-interop``).

The CUDA interop publisher imports Vulkan-exported memory and fences with an
exported timeline semaphore; Metal exposes no exported handles, but Apple-Silicon
GPUs are unified-memory — the renderer's binding-33/34 weight buffers can live in
shared storage (``StorageBuffer(shared=True)``) the host writes in place. The
synchronization model is therefore different (design D1): ``publish()`` only
stages the precision-cast bytes host-side under a lock; the actual buffer write
happens inside ``swap()``, which the renderer already calls on the render thread
at the frame boundary — the only point where no in-flight command buffer reads
the weights — so the GPU never samples a half-written network and no
MTLSharedEvent plumbing is needed.

Same unbiasedness contract as every publisher: a sample drawn under version *N*
is evaluated against version *N*'s density, so trainer staleness raises variance
only, never bias. ``acquire_for_render()`` returns ``(None, version)`` exactly
like the CUDA publisher — the weights already live in the bound buffers and the
renderer's frame-end swap re-stamps ``networkVersion`` without re-uploading.
"""

from __future__ import annotations

import threading

from .neural_handoff import NeuralWeightPublisher
from .neural_weights import NeuralWeights

__all__ = ["MetalSharedWeightPublisher", "metal_interop_available"]


def metal_interop_available(ctx) -> tuple[bool, str]:
    """Whether the context supports the UMA shared-storage handoff.

    Requires the native Metal backend reporting ``supports_shared_memory`` (the
    upload-heap storage-usage probe in ``MetalContext``). Returns ``(ok, reason)``
    like the CUDA-side ``interop_available()``."""
    if ctx is None:
        return False, "no GPU context"
    if not getattr(ctx, "is_metal", False):
        return False, f"not a Metal context ({type(ctx).__name__})"
    if not getattr(ctx, "supports_shared_memory", False):
        return False, "Metal device reports no shared-storage support"
    return True, ""


class MetalSharedWeightPublisher(NeuralWeightPublisher):
    """UMA shared-buffer weight handoff: stage on publish, write at swap.

    ``weights_buffer``/``biases_buffer`` are the renderer's binding-33/34
    ``metal_compute.StorageBuffer`` objects, allocated ``shared=True`` when this
    publisher is selected (renderer task 3.1). A non-shared buffer still works —
    ``write_in_place`` degrades to the staged upload path with a one-shot log —
    so a fallback allocation never breaks training."""

    def __init__(self, weights_buffer=None, biases_buffer=None,
                 expect_arch: tuple[int, int, int, int] | None = None,
                 precision=None, initial=None):
        # `initial` is accepted for a uniform `make_publisher` signature but
        # unused: the render buffers are seeded by the renderer's own upload, and
        # published weights land in the same buffers at swap.
        if weights_buffer is None or biases_buffer is None:
            raise RuntimeError(
                "Metal interop publisher needs the shared weights+biases buffers "
                "(bindings 33/34); none were provided")
        ok, reason = metal_interop_available(getattr(weights_buffer, "ctx", None))
        if not ok:
            raise NotImplementedError(
                f"Metal interop weight handoff unavailable: {reason}. "
                f"Use --neural-handoff file on this platform.")
        from .neural_weights import NeuralPrecision
        self._weights_buffer = weights_buffer
        self._biases_buffer = biases_buffer
        self._expect = expect_arch
        self._precision = precision or NeuralPrecision.FP32
        self._lock = threading.Lock()
        self._staged: tuple[bytes, bytes] | None = None  # (weights, biases)
        self._staged_version = 0
        self._render_version = 0

    def publish(self, weights: NeuralWeights) -> int:
        """Trainer side: precision-cast the weights+biases and stage the byte
        blobs (no GPU touch — the render thread owns the buffer write)."""
        if self._expect is not None:
            weights.assert_matches_shader(self._expect)
        wbytes = weights.weight_bytes_for(self._precision)
        bbytes = weights.bias_bytes_for(self._precision)
        with self._lock:
            self._staged_version += 1
            self._staged = (wbytes, bbytes)
            return self._staged_version

    def swap(self) -> bool:
        """Renderer side, at frame end: write the staged bytes into the shared
        buffers in place and promote the version. Runs on the render thread
        between frames, so no in-flight command buffer reads mid-write."""
        with self._lock:
            staged, version = self._staged, self._staged_version
            self._staged = None
        if staged is None or version <= self._render_version:
            return False
        wbytes, bbytes = staged
        self._weights_buffer.write_in_place(wbytes)
        self._biases_buffer.write_in_place(bbytes)
        self._render_version = version
        return True

    def acquire_for_render(self) -> tuple[NeuralWeights | None, int]:
        # Weights live in the bound shared buffers; the renderer re-stamps the
        # version without re-uploading (the `nw is None` frame-end branch).
        return None, self._render_version

    def current_version(self) -> int:
        return self._render_version

    def close(self) -> None:
        """Drop staged state (idempotent; buffers belong to the renderer)."""
        with self._lock:
            self._staged = None
