"""Stage 2 (change ``neural-online-training``) — Mac-runnable online-loop gates.

Exercises the seam modules that do NOT need CUDA or a live renderer: the
recency-weighted replay buffer, the file double-buffer handoff (swap +
``networkVersion`` increment), the full drain→train→publish→swap loop, and the
CUDA-guard on the interop backend. The CUDA trainer step + interop internals are
validated on the NVIDIA box, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.sampling.neural_replay import ReplayBuffer
from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig
from skinny.sampling.neural_handoff import make_publisher
from skinny.sampling.neural_handoff_interop import InteropWeightPublisher, interop_available
from skinny.sampling.neural_weights import make_dummy_weights
from skinny.sampling.path_records import RECORD_DTYPE


def _records(n: int, tag: float) -> np.ndarray:
    r = np.zeros(n, dtype=RECORD_DTYPE)
    r["pos"] = tag           # tag lets us tell old from new records apart
    r["wi_local"] = [0.0, 0.0, 1.0]
    r["contrib"] = 1.0
    return r


def test_replay_recency_weighting():
    """After many generations, recent records dominate a recency-weighted draw."""
    buf = ReplayBuffer(capacity=10_000, recency_decay=1.0)
    buf.add(_records(5_000, tag=1.0))          # old (generation 1)
    for _ in range(20):                        # age the old batch by 20 generations
        buf.add(_records(50, tag=2.0))         # recent
    sample = buf.sample(5_000, rng=np.random.default_rng(0))
    frac_recent = float((sample["pos"][:, 0] == 2.0).mean())
    assert frac_recent > 0.9, frac_recent


def test_replay_ring_capacity():
    buf = ReplayBuffer(capacity=1_000)
    buf.add(_records(1_500, tag=1.0))          # overflow wraps
    assert len(buf) == 1_000


def test_replay_evict_stale():
    buf = ReplayBuffer(capacity=10_000, recency_decay=0.5)
    buf.add(_records(100, tag=1.0))
    for _ in range(5):
        buf.add(_records(100, tag=2.0))
    # 6 generations total (1 old + 5 recent); max_age=2 keeps the newest 3.
    evicted = buf.evict_stale(max_age=2)
    assert evicted == 300                      # generations 1,2,3 (age 5,4,3) go
    assert len(buf) == 300


def test_file_handoff_swap_and_version():
    """publish stages; swap promotes at frame end and bumps networkVersion."""
    init = make_dummy_weights()
    pub = make_publisher("file", weights_dir="_pytest_neural_handoff", initial=init)
    assert pub.current_version() == 0
    w0, v0 = pub.acquire_for_render()
    assert w0 is init and v0 == 0

    staged = pub.publish(make_dummy_weights())
    assert staged == 1
    # not yet swapped — render side still frozen at version 0
    assert pub.current_version() == 0
    assert pub.swap() is True
    assert pub.current_version() == 1
    assert pub.swap() is False                 # nothing pending now


def test_online_loop_end_to_end():
    """drain → train_cycle → publish → swap → networkVersion++."""
    replay = ReplayBuffer(capacity=100_000)
    trainer = NeuralTrainer(TrainerConfig(steps_per_cycle=4, batch=1_024))
    pub = make_publisher("file", weights_dir="_pytest_neural_loop",
                         initial=trainer.weights)
    rng = np.random.default_rng(0)

    for frame in range(3):
        replay.add(_records(2_000, tag=float(frame)))     # renderer drains a frame
        new_w = trainer.train_cycle(replay, rng)          # warm-started update
        pub.publish(new_w)                                # stage
        assert pub.swap() is True                         # frame-end promote
    assert pub.current_version() == 3
    w, v = pub.acquire_for_render()
    assert v == 3 and w is not None
    np.testing.assert_array_equal(w.headers, trainer.weights.headers)  # arch preserved


def test_interop_guarded_off_cuda():
    """On Mac (no CUDA) the interop backend reports unavailable, never silently
    degrades."""
    ok, reason = interop_available()
    if ok:
        pytest.skip("CUDA present — interop guard not exercised here")
    with pytest.raises(NotImplementedError):
        InteropWeightPublisher()
    with pytest.raises(NotImplementedError):
        make_publisher("interop")
