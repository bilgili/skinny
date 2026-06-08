"""Async trainer for online neural-proposal training (Stage 2).

Change ``neural-online-training`` (+ ``neural-trainer-backends``). Pulls
recency-weighted batches from the ``ReplayBuffer`` and does small warm-started
updates on the **exact** shipped flow architecture
(``ConditionalSplineFlow2D(cond=9, layers=6, bins=24, hidden=96)``) using the
contribution-weighted MLE loss the offline trainer uses
(``loss = -Σ w·log q / Σ w``, ``w = luminance(contribution)``). It then bakes a
new ``NeuralWeights`` the publisher hands to the renderer.

The orchestrator owns replay sampling, the dataset build (``build_dataset_np``),
version/loss bookkeeping and publishing; the *per-cycle gradient step* runs
behind a selectable :class:`~skinny.sampling.training_backends.TrainingBackend`
(``cpu`` → numpy reference oracle, ``cuda`` → torch on CUDA, ``mlx`` → later).
There is no longer a torch/placeholder two-tier branch: a torch-free host trains
for real through the numpy backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .neural_replay import ReplayBuffer  # noqa: F401  (type clarity for callers)
from .neural_weights import NF_COND, NeuralBuildConfig, NeuralWeights, make_dummy_weights
from .training_backends import (
    TrainingBackend,
    build_dataset_np,
    make_training_backend,
)

__all__ = ["TrainerConfig", "NeuralTrainer"]


@dataclass
class TrainerConfig:
    arch: NeuralBuildConfig = field(default_factory=NeuralBuildConfig)
    steps_per_cycle: int = 64          # small: smooth animation → cheap warm updates
    batch: int = 4096
    lr: float = 1e-3
    backend: str = "auto"              # cpu|cuda|mlx|auto (training-compute backend)
    device: str = "auto"              # torch sub-device cpu|mps|cuda|auto (cuda backend)
    train_precision: str = "fp32"      # fp32|fp16 — optimizer compute precision
    # Scene AABB (bmin, bext) for the position condition — must match the
    # renderer's neuralCondition normalisation; None ⇒ unit cube (raw position).
    bounds: tuple | None = None
    # spline_flow repo location (has train.py / export_weights.py) for the torch backend.
    spline_flow_path: str | None = None


class NeuralTrainer:
    """Warm-started online trainer. Holds the current weights and a stateful
    :class:`TrainingBackend` (warm model + optimizer); each cycle builds a
    dataset from recent records, runs the backend's update, and returns the
    new weights to publish."""

    def __init__(self, config: TrainerConfig | None = None,
                 initial: NeuralWeights | None = None,
                 backend: TrainingBackend | None = None):
        self.config = config or TrainerConfig()
        self._weights = initial or make_dummy_weights(self.config.arch)
        self._cycles = 0
        self.last_loss: float | None = None
        self._backend = backend or make_training_backend(
            self.config.backend, device=self.config.device,
            train_precision=self.config.train_precision,
            spline_flow_path=self.config.spline_flow_path)
        self._warm = False
        # Throttled per-cycle logging state (see train_cycle).
        self._trained_cycles = 0
        self._logged_cycles = 0
        self._train_ms_acc = 0.0
        self._sample_acc = 0
        self._last_log_t = 0.0
        # Precision gating: a requested train_precision the backend/device cannot
        # provide falls back to fp32 (reduced precision is variance, never bias),
        # surfacing a clear message rather than running an unsupported path.
        if not self._backend.supports_precision(self.config.train_precision,
                                                self.config.device):
            print(f"[neural] train_precision={self.config.train_precision!r} "
                  f"unsupported on backend {self._backend.name!r} "
                  f"(device {self.config.device!r}); falling back to fp32")
            self.config.train_precision = "fp32"
        a = self.config.arch
        print(f"[neural] trainer ready: backend={self._backend.name} "
              f"arch=L{a.layers}/B{a.bins}/H{a.hidden}/cond{NF_COND} "
              f"train_precision={self.config.train_precision} "
              f"infer_precision={a.precision.value} "
              f"steps/cycle={self.config.steps_per_cycle} batch={self.config.batch} "
              f"lr={self.config.lr}")

    @property
    def weights(self) -> NeuralWeights:
        return self._weights

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def torch_active(self) -> bool:
        """True when the active backend is the real torch loop (back-compat)."""
        return self._backend.name == "torch" and self._backend.is_available()

    # ── public cycle ────────────────────────────────────────────────────

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.config.bounds is not None:
            bmin, bext = self.config.bounds
            return (np.asarray(bmin, np.float32).reshape(3),
                    np.asarray(bext, np.float32).reshape(3))
        return np.zeros(3, np.float32), np.ones(3, np.float32)

    # Coalesce the per-cycle training log to at most one line every this many
    # seconds — online training runs continuously, so an unthrottled line per
    # cycle would flood the console.
    _LOG_INTERVAL_S = 2.0

    def train_cycle(self, replay: ReplayBuffer,
                    rng: np.random.Generator | None = None) -> NeuralWeights:
        """Run one warm-started training cycle on recent records; return new
        weights. The backend (torch on CUDA, or the numpy reference) owns the
        gradient step; the contract here is unchanged."""
        self._cycles += 1
        batch = replay.sample(self.config.batch, rng)
        if len(batch) == 0:
            return self._weights  # nothing to learn from yet

        cond, z, w = build_dataset_np(batch, self._bounds())
        if cond.shape[0] == 0:
            return self._weights  # no upper-hemisphere, positive-weight samples

        if not self._warm:
            t0 = time.perf_counter()
            self._backend.warm_start(self._weights, self.config)
            self._warm = True
            print(f"[neural] warm-started {self._backend.name} flow from current "
                  f"weights in {(time.perf_counter() - t0) * 1e3:.0f} ms")

        t0 = time.perf_counter()
        loss = self._backend.update(cond, z, w)
        dt_ms = (time.perf_counter() - t0) * 1e3
        if loss is not None and np.isfinite(loss):
            self.last_loss = float(loss)
        self._weights = self._backend.export()

        # Throttled progress: how long the step took, how many records it trained
        # on, the step count, and the current loss (averaged over the cycles
        # since the last line).
        self._trained_cycles += 1
        self._train_ms_acc += dt_ms
        self._sample_acc += int(cond.shape[0])
        now = time.perf_counter()
        if now - self._last_log_t >= self._LOG_INTERVAL_S:
            n = self._trained_cycles - self._logged_cycles
            loss_str = f"{self.last_loss:.4f}" if self.last_loss is not None else "n/a"
            print(f"[neural] trained {n} cycle(s) [{self._trained_cycles} total] on "
                  f"{self._sample_acc // max(n, 1)} samples/cycle × "
                  f"{self.config.steps_per_cycle} steps: loss={loss_str}, "
                  f"{self._train_ms_acc / max(n, 1):.1f} ms/cycle")
            self._last_log_t = now
            self._logged_cycles = self._trained_cycles
            self._train_ms_acc = 0.0
            self._sample_acc = 0
        return self._weights
