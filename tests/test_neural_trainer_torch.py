"""Torch/CUDA validation of the online trainer's real loop (task 2.2).

Runs ONLY where torch + the spline_flow repo are importable (the training box —
CUDA on the NVIDIA box, or CPU/MPS). In the renderer-only skinny venv (no torch)
the whole module skips; the torch-free placeholder path is covered by
``test_neural_online.py::test_online_loop_end_to_end``.

Run on the NVIDIA box (spline_flow venv has torch + CUDA)::

  SKINNY_SPLINE_FLOW=/path/to/spline_flow PYTHONPATH=src \
    <spline_flow-venv-python> -m pytest tests/test_neural_trainer_torch.py -q
  # or, without pytest in that venv:
  SKINNY_SPLINE_FLOW=/path/to/spline_flow PYTHONPATH=src \
    <spline_flow-venv-python> tests/test_neural_trainer_torch.py
"""

from __future__ import annotations

import numpy as np

try:                                   # spline_flow venv has no pytest; __main__ runs instead
    import pytest
except ModuleNotFoundError:
    pytest = None

from skinny.sampling.neural_replay import ReplayBuffer
from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig
from skinny.sampling.path_records import RECORD_DTYPE
from skinny.sampling.training_backends import TorchTrainingBackend


def _torch_ready() -> bool:
    return TorchTrainingBackend(device="cpu").is_available()


if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not _torch_ready(),
        reason="torch + spline_flow unavailable (renderer-only venv); placeholder path "
               "covered by test_neural_online",
    )


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _concentrated_records(n: int, rng, *, lobe=(0.3, 0.9, 0.2)) -> np.ndarray:
    """Synthetic records whose sampled directions cluster around `lobe` (upper
    hemisphere) with positive contribution — a target the flow can actually fit."""
    recs = np.zeros(n, dtype=RECORD_DTYPE)
    recs["pos"] = rng.uniform(0.2, 0.8, size=(n, 3)).astype(np.float32)
    recs["normal"] = np.array([0.0, 1.0, 0.0], np.float32)
    recs["wo"] = np.array([0.0, 1.0, 0.0], np.float32)
    d = np.asarray(lobe, np.float32)
    d = d / np.linalg.norm(d)
    wi = d[None, :] + 0.08 * rng.standard_normal((n, 3)).astype(np.float32)
    wi[:, 1] = np.abs(wi[:, 1]) + 0.2                      # keep upper hemisphere
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    recs["wi_local"] = wi
    recs["contrib"] = rng.uniform(0.5, 1.5, size=(n, 3)).astype(np.float32)
    return recs


def test_train_cycle_updates_weights_and_is_finite():
    """One real cycle on the target box: warm-start → MLE steps → NFW1 round-trip.
    The weights change (training happened), stay finite, and keep the arch."""
    rng = np.random.default_rng(0)
    replay = ReplayBuffer(capacity=50_000)
    replay.add(_concentrated_records(8192, rng))
    trainer = NeuralTrainer(TrainerConfig(
        steps_per_cycle=150, batch=2048, lr=2e-3, device=_device(),
        bounds=(np.zeros(3), np.ones(3))), backend=_torch_backend())
    assert trainer.torch_active
    w0 = trainer.weights.weights.copy()
    new = trainer.train_cycle(replay, rng)
    assert (new.layers, new.bins, new.hidden, new.cond) == (6, 24, 96, 9)
    assert np.all(np.isfinite(new.weights)) and np.all(np.isfinite(new.biases))
    assert new.weights.shape == w0.shape
    assert not np.allclose(new.weights, w0), "training did not update the weights"
    assert trainer.last_loss is not None and np.isfinite(trainer.last_loss)


def test_train_cycle_warm_starts_across_cycles():
    """The flow + optimiser persist across cycles (warm) and keep producing valid,
    publishable weights — the online cadence."""
    rng = np.random.default_rng(1)
    replay = ReplayBuffer(capacity=50_000)
    trainer = NeuralTrainer(TrainerConfig(
        steps_per_cycle=60, batch=1024, device=_device(),
        bounds=(np.zeros(3), np.ones(3))), backend=_torch_backend())
    losses = []
    for _ in range(3):
        replay.add(_concentrated_records(4096, rng))
        w = trainer.train_cycle(replay, rng)
        assert np.all(np.isfinite(w.weights))
        losses.append(trainer.last_loss)
    assert trainer._backend._model is not None            # warm flow persisted
    assert all(x is not None and np.isfinite(x) for x in losses)


if __name__ == "__main__":  # pragma: no cover - manual harness (no pytest needed)
    import sys
    if not _torch_ready():
        print("SKIP: torch + spline_flow unavailable in this interpreter")
        sys.exit(0)
    import torch
    print(f"device: {_device()}  cuda: {torch.cuda.is_available()}  "
          f"name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    test_train_cycle_updates_weights_and_is_finite()
    print("OK test_train_cycle_updates_weights_and_is_finite (last_loss finite)")
    test_train_cycle_warm_starts_across_cycles()
    print("OK test_train_cycle_warm_starts_across_cycles")
    print("ALL TORCH TRAINER TESTS PASSED")
