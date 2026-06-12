"""Weight-handoff seam for online neural-proposal training (Stage 2).

Change ``neural-online-training``. The async trainer publishes updated weights;
the renderer swaps them in at a frame boundary. Two backends sit behind one
``NeuralWeightPublisher`` interface, selectable at runtime (``--neural-handoff``):

* ``file``    — trainer writes an ``NFW1`` file; renderer hot-reloads via the
                existing ``neural_weights`` loader; double-buffered, CPU round-trip.
                Works on any platform (Mac-testable end to end).
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
    """Factory for the ``--neural-handoff`` flag value (``file`` | ``interop``).

    ``interop`` resolves per GPU backend (change metal-neural-interop, design
    D2): a Metal ``weights_buffer`` selects the UMA shared-storage publisher,
    anything else the Vulkan↔CUDA external-memory publisher. The user intent is
    "GPU handoff, no file"; which mechanism applies is a backend property."""
    if kind == "file":
        from .neural_handoff_file import FileWeightPublisher
        return FileWeightPublisher(**kwargs)
    if kind == "interop":
        ctx = getattr(kwargs.get("weights_buffer"), "ctx", None)
        if getattr(ctx, "is_metal", False):
            from .neural_handoff_interop_metal import MetalSharedWeightPublisher
            kwargs.pop("timeline_semaphore", None)  # no exported semaphores on Metal
            return MetalSharedWeightPublisher(**kwargs)
        from .neural_handoff_interop import InteropWeightPublisher
        try:
            return InteropWeightPublisher(**kwargs)
        except NotImplementedError as exc:
            raise NotImplementedError(
                "interop weight handoff unavailable: needs CUDA with Vulkan "
                "external memory, or the native Metal backend on a unified-memory "
                f"device ({exc}). Use --neural-handoff file on this platform."
            ) from exc
    raise ValueError(f"unknown neural handoff backend {kind!r} (want 'file' or 'interop')")
