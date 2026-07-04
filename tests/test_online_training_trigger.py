"""Change ``online-training-trigger`` — the front-end trigger + per-frame driver.

Mac-runnable gates for the renderer-side seam this change adds: the per-frame
``online_training_tick`` (no-op off, drains on), the ``can_online_train``
prerequisite check, and the background trainer thread lifecycle
(start-on-enable / stop-and-join-on-disable, weights advancing off the render
thread). The methods are exercised on a lightweight stand-in ``self`` so the
suite stays GPU-free; the live GPU drain is validated on the NVIDIA box.
"""

from __future__ import annotations

import time
import types

import numpy as np

from skinny.params import EXECUTION_MEGAKERNEL, EXECUTION_WAVEFRONT
from skinny.renderer import Renderer
from skinny.sampling.neural_handoff import make_publisher
from skinny.sampling.neural_replay import ReplayBuffer
from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig
from skinny.sampling.path_records import RECORD_DTYPE


def _records(n: int, tag: float) -> np.ndarray:
    r = np.zeros(n, dtype=RECORD_DTYPE)
    r["pos"] = tag
    r["wi_local"] = [0.0, 0.0, 1.0]
    r["contrib"] = 1.0
    return r


def _bind(fake, *names):
    """Bind the named real ``Renderer`` methods onto a stand-in ``self``."""
    for name in names:
        setattr(fake, name, types.MethodType(getattr(Renderer, name), fake))


# ── 5.1 per-frame driver ─────────────────────────────────────────────

def test_tick_is_noop_when_off():
    fake = types.SimpleNamespace(_online_training=False)
    assert Renderer.online_training_tick(fake) == 0


def test_tick_skips_until_scene_built():
    # Online on but the scene isn't built yet — the drain must not run (it would
    # raise "scene not built"); the tick returns 0 and stays silent.
    fake = types.SimpleNamespace(
        _online_training=True, _scene_bindings=None, descriptor_sets=None)
    assert Renderer.online_training_tick(fake) == 0


def test_tick_drains_into_replay_and_returns_promptly():
    replay = ReplayBuffer(capacity=10_000)

    def fake_drain(buf, **kw):
        buf.add(_records(128, tag=1.0))
        return 128

    fake = types.SimpleNamespace(
        _online_training=True,
        _scene_bindings=object(),
        descriptor_sets=[object()],
        _neural_replay=replay,
        drain_path_records_to_replay=fake_drain,
    )
    _bind(fake, "online_drain", "online_training_tick")

    t0 = time.perf_counter()
    drained = fake.online_training_tick()
    elapsed = time.perf_counter() - t0

    assert drained == 128
    assert len(replay) == 128
    assert elapsed < 1.0  # the render-thread tick must not block on training


# ── 5.2 prerequisite gate ────────────────────────────────────────────

def test_can_online_train_requires_wavefront():
    fake = types.SimpleNamespace(
        effective_execution_mode_index=EXECUTION_MEGAKERNEL,
        _neural_active=lambda: True,
    )
    ok, reason = Renderer.can_online_train(fake)
    assert ok is False
    assert "wavefront" in reason


def test_can_online_train_requires_neural_proposal():
    fake = types.SimpleNamespace(
        effective_execution_mode_index=EXECUTION_WAVEFRONT,
        integrator_index=0,  # path — BDPT (1) is refused before the neural check
        _neural_active=lambda: False,
    )
    ok, reason = Renderer.can_online_train(fake)
    assert ok is False
    assert "neural" in reason


def test_can_online_train_true_when_both_hold():
    fake = types.SimpleNamespace(
        effective_execution_mode_index=EXECUTION_WAVEFRONT,
        integrator_index=0,  # path
        _neural_active=lambda: True,
    )
    ok, reason = Renderer.can_online_train(fake)
    assert ok is True
    assert reason == ""


def test_can_online_train_refuses_bdpt():
    # BDPT never consumes the neural proposal and has no wavefront record
    # source, so training under it is refused even with a neural proposal active
    # (change bdpt-neural-incompatibility).
    fake = types.SimpleNamespace(
        effective_execution_mode_index=EXECUTION_WAVEFRONT,
        integrator_index=1,  # bdpt
        _neural_active=lambda: True,
    )
    ok, reason = Renderer.can_online_train(fake)
    assert ok is False
    assert "bdpt" in reason.lower()


def test_online_train_execution_supported_tracks_wavefront_only():
    # The permanent half of the gate: True iff wavefront, independent of whether
    # a neural proposal is active (that half is runtime-selectable). The Qt
    # worker uses this to keep polling on a transient (neural) miss but give up
    # on a permanent (non-wavefront) one.
    wave = types.SimpleNamespace(effective_execution_mode_index=EXECUTION_WAVEFRONT)
    mega = types.SimpleNamespace(effective_execution_mode_index=EXECUTION_MEGAKERNEL)
    assert Renderer.online_train_execution_supported(wave) is True
    assert Renderer.online_train_execution_supported(mega) is False


# ── 5.3 background trainer thread lifecycle ──────────────────────────

def test_trainer_thread_starts_and_advances_then_joins():
    replay = ReplayBuffer(capacity=100_000)
    replay.add(_records(4_000, tag=1.0))
    trainer = NeuralTrainer(TrainerConfig(steps_per_cycle=2, batch=512))
    pub = make_publisher("file", weights_dir="_pytest_online_trigger",
                         initial=trainer.weights)

    # The file publisher stages on publish (the render thread's frame-end swap
    # promotes); there's no render thread here, so spy the publish calls to prove
    # the background trainer advances weights through the publisher on its own.
    published: list[int] = []
    real_publish = pub.publish

    def spy_publish(w):
        v = real_publish(w)
        published.append(v)
        return v

    pub.publish = spy_publish

    fake = types.SimpleNamespace(
        _online_training=True,
        _neural_trainer=trainer,
        _neural_replay=replay,
        _neural_publisher=pub,
        _trainer_thread=None,
        _trainer_stop=None,
        _trainer_cadence_s=0.001,
    )
    _bind(fake, "online_train_and_publish", "_start_trainer_thread",
          "_stop_trainer_thread")

    fake._start_trainer_thread()
    assert fake._trainer_thread is not None
    assert fake._trainer_thread.is_alive()

    # The background thread (not this one) trains + publishes new weights.
    deadline = time.perf_counter() + 5.0
    while not published and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert published  # weights advanced via the publisher off the render thread
    assert published[0] >= 1

    fake._stop_trainer_thread()
    assert fake._trainer_thread is None
    assert fake._trainer_stop is None


def test_start_trainer_thread_is_idempotent():
    fake = types.SimpleNamespace(
        _online_training=True,
        _neural_trainer=None,  # loop body is a no-op without a trainer
        _trainer_thread=None,
        _trainer_stop=None,
        _trainer_cadence_s=0.001,
    )
    _bind(fake, "online_train_and_publish", "_start_trainer_thread",
          "_stop_trainer_thread")

    fake._start_trainer_thread()
    first = fake._trainer_thread
    fake._start_trainer_thread()  # second call must not spawn a new thread
    assert fake._trainer_thread is first

    fake._stop_trainer_thread()
    assert fake._trainer_thread is None


def test_disable_when_never_enabled_is_safe():
    fake = types.SimpleNamespace(_trainer_thread=None, _trainer_stop=None)
    _bind(fake, "_stop_trainer_thread")
    fake._stop_trainer_thread()  # must not raise
    assert fake._trainer_thread is None
