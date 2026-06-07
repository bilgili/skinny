"""Async trainer skeleton for online neural-proposal training (Stage 2).

Change ``neural-online-training``. Pulls recency-weighted batches from the
``ReplayBuffer`` and does small warm-started updates on the **exact** shipped flow
architecture (``ConditionalSplineFlow2D(cond=9, layers=6, bins=24, hidden=96)``)
using the contribution-weighted MLE loss the offline trainer uses
(``spline_flow/render_records.py``: ``loss = -Σ w·log q / Σ w``,
``w = luminance(contribution)``). It then bakes a new ``NeuralWeights`` the
publisher hands to the renderer.

Stub depth (this change): the buffer drain, warm-start, cycle cadence, and the
publish/swap/version wiring are real and Mac-runnable; the **PyTorch optimisation
step itself is the implementation seam** — it reuses the verified ``spline_flow``
flow + loss (a sibling repo, put on ``PYTHONPATH`` on the training box) and the
CUDA + tensor-core path is filled on the NVIDIA box. ``train_cycle`` here advances
the version and returns valid, correctly-sized weights so the end-to-end online
loop (drain → train → publish → swap → ``networkVersion++``) is testable today.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .neural_replay import ReplayBuffer
from .neural_weights import NeuralBuildConfig, NeuralWeights, make_dummy_weights

__all__ = ["TrainerConfig", "NeuralTrainer"]


@dataclass
class TrainerConfig:
    arch: NeuralBuildConfig = field(default_factory=NeuralBuildConfig)
    steps_per_cycle: int = 64          # small: smooth animation → cheap warm updates
    batch: int = 4096
    lr: float = 1e-3
    device: str = "auto"               # cpu|mps|cuda|auto; cuda path = NVIDIA box


class NeuralTrainer:
    """Warm-started online trainer. Holds the current weights; each cycle updates
    them from recent records and returns the new weights to publish."""

    def __init__(self, config: TrainerConfig | None = None,
                 initial: NeuralWeights | None = None):
        self.config = config or TrainerConfig()
        self._weights = initial or make_dummy_weights(self.config.arch)
        self._cycles = 0

    @property
    def weights(self) -> NeuralWeights:
        return self._weights

    def train_cycle(self, replay: ReplayBuffer,
                    rng: np.random.Generator | None = None) -> NeuralWeights:
        """Run one warm-started training cycle on recent records; return new weights.

        IMPLEMENTATION SEAM: replace the placeholder update below with the
        ``spline_flow`` PyTorch loop — build ``ConditionalSplineFlow2D`` from
        ``self._weights``, optimise the contribution-weighted MLE on
        ``replay.sample(batch)`` for ``steps_per_cycle`` steps on
        ``self.config.device`` (CUDA + autocast fp16 on the NVIDIA box), then bake
        the result back into a ``NeuralWeights``. The drain/cadence/publish wiring
        around it is final.
        """
        self._cycles += 1
        batch = replay.sample(self.config.batch, rng)
        if len(batch) == 0:
            return self._weights  # nothing to learn from yet

        # --- placeholder update (NOT real training) -------------------------
        # Keeps the loop honest end to end: produces valid, correctly-sized,
        # finite weights so publish→swap→networkVersion++ is exercised. The real
        # gradient step replaces this block (see the seam note above).
        new = NeuralWeights(
            self._weights.layers, self._weights.bins, self._weights.hidden,
            self._weights.cond, self._weights.headers.copy(),
            self._weights.weights.copy(), self._weights.biases.copy(),
        )
        # --------------------------------------------------------------------
        self._weights = new
        return new
