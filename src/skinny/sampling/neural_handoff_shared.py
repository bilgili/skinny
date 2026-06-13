"""In-process shared CPU weight handoff (``--neural-handoff shared``).

Change ``shared-neural-handoff``. The async trainer runs as a same-process daemon
thread, so trainer and renderer share one address space. This backend hands
weights across that boundary through a CPU double-buffer held in RAM — no disk
write (unlike ``file``) and no CUDA / unified-memory device (unlike ``interop``),
available on every platform with no added dependency.

``publish`` stores a byte-faithful private copy of the staged weights so the
trainer may keep mutating its own working ``NeuralWeights`` without touching the
frozen render buffer; ``swap`` promotes pending→render at the frame boundary,
bumping ``networkVersion``. The copy round-trips through ``serialize`` /
``deserialize`` — the same path ``file`` takes minus the filesystem — so the
bytes the renderer consumes are identical to a ``file`` publish of the same
weights (catching any serialise/parse drift in memory rather than on disk). The
renderer uploads swapped weights to the GPU through the same post-swap path it
uses for ``file``; this backend never writes the GPU buffers directly (that is
``interop``).
"""

from __future__ import annotations

from .neural_handoff import NeuralWeightPublisher
from .neural_weights import (
    NeuralWeights,
    deserialize_neural_weights,
    serialize_neural_weights,
)

__all__ = ["SharedWeightPublisher"]


class SharedWeightPublisher(NeuralWeightPublisher):
    def __init__(self, initial: NeuralWeights | None = None,
                 expect_arch: tuple[int, int, int, int] | None = None):
        self._expect = expect_arch
        self._render: NeuralWeights | None = initial   # frozen during a frame
        self._render_version = 0
        self._pending: NeuralWeights | None = None
        self._pending_version = 0
        self._staged_version = 0

    def _copy(self, weights: NeuralWeights) -> NeuralWeights:
        # Byte-faithful private copy via the file backend's serialiser, in RAM:
        # decouples the pending buffer from the trainer's working weights and
        # applies the canonical <f4>/<u4> casts so the bytes match a file publish.
        return deserialize_neural_weights(
            serialize_neural_weights(weights), self._expect)

    def publish(self, weights: NeuralWeights) -> int:
        self._staged_version += 1
        self._pending = self._copy(weights)
        self._pending_version = self._staged_version
        return self._staged_version

    def swap(self) -> bool:
        if self._pending is None:
            return False
        self._render = self._pending
        self._render_version = self._pending_version
        self._pending = None
        return True

    def acquire_for_render(self) -> tuple[NeuralWeights | None, int]:
        return self._render, self._render_version

    def current_version(self) -> int:
        return self._render_version
