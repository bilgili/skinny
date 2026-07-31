"""Weight-handoff seam for online neural-proposal training (Stage 2).

Change ``neural-online-training``. The async trainer publishes updated weights;
the renderer swaps them in at a frame boundary. Three backends sit behind one
``NeuralWeightPublisher`` interface, selectable at runtime (``--neural-handoff``):

* ``file``    — trainer writes an ``NFW1`` file; renderer hot-reloads via the
                existing ``neural_weights`` loader; double-buffered, CPU round-trip
                through disk. Works on any platform (Mac-testable end to end).
* ``shared``  — in-process CPU double-buffer held in RAM; the trainer (a
                same-process daemon thread) hands a byte-faithful copy across with
                no disk write and no CUDA / unified-memory device. Any platform, no
                added dependency; renderer uploads via the normal post-swap path
                (change shared-neural-handoff).
* ``interop`` — CUDA writes weights straight into the Vulkan-exported weight
                buffer (``VK_KHR_external_memory`` + ``cudaImportExternalMemory``),
                no CPU round-trip. The real-time path; CUDA-only, guarded.

Common contract: ``publish`` is called by the trainer (any time); ``swap`` is
called by the renderer at the frame boundary; ``acquire_for_render`` returns the
frozen render-side weights + their ``networkVersion``. A sample drawn under one
version is always evaluated against that version's density — staleness raises
variance only, never bias.
"""

from __future__ import annotations

import abc

from .neural_weights import NeuralWeights

__all__ = ["NeuralWeightPublisher", "make_publisher"]


class NeuralWeightPublisher(abc.ABC):
    """Double-buffered weight handoff between the async trainer and the renderer."""

    @abc.abstractmethod
    def publish(self, weights: NeuralWeights) -> int:
        """Trainer side: stage new weights as pending. Returns the staged version."""

    @abc.abstractmethod
    def swap(self) -> bool:
        """Renderer side, at frame end: promote pending→render if any. Returns
        True if a swap occurred (then ``current_version`` has incremented)."""

    @abc.abstractmethod
    def acquire_for_render(self) -> tuple[NeuralWeights | None, int]:
        """Renderer side: the frozen render-buffer weights + their version."""

    @abc.abstractmethod
    def current_version(self) -> int:
        """The render-side ``networkVersion`` (baseline 0)."""


def make_publisher(kind: str, **kwargs) -> NeuralWeightPublisher:
    """Factory for the ``--neural-handoff`` flag value (``file`` | ``shared`` |
    ``interop``).

    ``shared`` is the in-process CPU double-buffer (change shared-neural-handoff):
    no disk, no GPU-interop requirement, any platform. ``interop`` resolves per
    GPU backend (change metal-neural-interop, design D2): a Metal ``weights_buffer``
    selects the UMA shared-storage publisher, anything else the Vulkan↔CUDA
    external-memory publisher. The user intent is "GPU handoff, no file"; which
    mechanism applies is a backend property."""
    if kind == "file":
        from .neural_handoff_file import FileWeightPublisher
        return FileWeightPublisher(**kwargs)
    if kind == "shared":
        from .neural_handoff_shared import SharedWeightPublisher
        return SharedWeightPublisher(**kwargs)
    if kind == "interop":
        from skinny.gpu_backend import capabilities
        ctx = getattr(kwargs.get("weights_buffer"), "ctx", None)
        # A three-way decision, each arm selected by the capability it needs
        # (change gpu-backend-adapter; codex pre-merge review HIGH 1):
        #   - in-place UMA write, where the device shares memory with the host;
        #   - exported-memory handoff, which needs external memory AND an
        #     external timeline semaphore — `vk_context`'s graded device build
        #     probes them independently and can enable memory without the
        #     semaphore, so neither implies the other;
        #   - otherwise refuse HERE, naming the missing capability, instead of
        #     allocating silently-non-exportable buffers and deferring the
        #     failure to the first publish.
        if ctx is not None:
            caps = capabilities(ctx)
            if caps.has_shared_in_place_write:
                from .neural_handoff_interop_metal import (
                    MetalSharedWeightPublisher,
                )
                kwargs.pop("timeline_semaphore", None)  # no exported semaphores
                return MetalSharedWeightPublisher(**kwargs)
            missing = [
                cap for cap, ok in (
                    ("external memory", caps.has_external_memory),
                    ("an external timeline semaphore",
                     caps.has_external_semaphore),
                ) if not ok
            ]
            if missing:
                raise NotImplementedError(
                    "interop weight handoff unavailable on the "
                    f"{caps.name} backend: it has no "
                    f"{' and no '.join(missing)}. Use --neural-handoff shared "
                    "(in-process RAM) or file (disk) on this device."
                )
        from .neural_handoff_interop import InteropWeightPublisher
        try:
            return InteropWeightPublisher(**kwargs)
        except NotImplementedError as exc:
            raise NotImplementedError(
                "interop weight handoff unavailable: needs CUDA with Vulkan "
                "external memory, or the native Metal backend on a unified-memory "
                f"device ({exc}). Use --neural-handoff file on this platform."
            ) from exc
    raise ValueError(
        f"unknown neural handoff backend {kind!r} (want 'file', 'shared' or 'interop')")
