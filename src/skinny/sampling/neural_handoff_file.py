"""File double-buffer weight handoff (``--neural-handoff file``).

Change ``neural-online-training``. Trainer writes a new ``NFW1`` file and stages
it; the renderer hot-reloads it through the existing ``load_neural_weights`` and
swaps at the frame boundary, bumping ``networkVersion``. A CPU round-trip — fine
for correctness and slow animation, the wrong cost for real-time (see the interop
backend). Reuses the shipped weight format end to end; runs on any platform.
"""

from __future__ import annotations

from pathlib import Path

from .neural_handoff import NeuralWeightPublisher
from .neural_weights import NeuralWeights, load_neural_weights, write_neural_weights

__all__ = ["FileWeightPublisher"]


class FileWeightPublisher(NeuralWeightPublisher):
    def __init__(self, weights_dir: str | Path = ".skinny_neural",
                 initial: NeuralWeights | None = None,
                 expect_arch: tuple[int, int, int, int] | None = None):
        self._dir = Path(weights_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._expect = expect_arch
        self._render: NeuralWeights | None = initial   # frozen during a frame
        self._render_version = 0
        self._pending: NeuralWeights | None = None
        self._pending_version = 0
        self._staged_version = 0

    def publish(self, weights: NeuralWeights) -> int:
        # round-trip through disk so the published bytes are exactly what the
        # renderer would load (catches any serialise/parse drift).
        self._staged_version += 1
        path = self._dir / f"weights_v{self._staged_version:06d}.nfw1"
        write_neural_weights(path, weights)
        self._pending = load_neural_weights(path, self._expect)
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
