"""Change ``online-training-observability`` — the startup configuration matrix,
the training lifecycle (ACTIVE marker + STOPPED run summary), and the GUI status
snapshot.

GPU-free: the pure matrix builder is tested directly, the renderer-side row
assembly + summary/status helpers are exercised on a lightweight stand-in
``self`` (the pattern from ``test_online_training_trigger``), and the trainer
summary uses the always-available numpy reference backend.
"""

from __future__ import annotations

import types

import numpy as np

from skinny import config_report as cr
from skinny.params import EXECUTION_MEGAKERNEL, EXECUTION_WAVEFRONT
from skinny.renderer import Renderer
from skinny.sampling.neural_replay import ReplayBuffer
from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig
from skinny.sampling.path_records import RECORD_DTYPE

_PRESETS = [("BSDF", "bsdf"), ("BSDF + Neural", "bsdf,neural")]


def _records(n: int, tag: float) -> np.ndarray:
    r = np.zeros(n, dtype=RECORD_DTYPE)
    r["pos"] = tag
    # build_dataset_np keeps upper-hemisphere samples by wi_local[:, 1] (y-up);
    # a y-up direction with positive contribution survives the filter so a real
    # training cycle runs (and the ACTIVE marker fires).
    r["wi_local"] = [0.0, 1.0, 0.0]
    r["contrib"] = 1.0
    return r


def _matrix_fake(**overrides):
    """A stand-in ``self`` carrying every attribute ``_collect_config_rows``
    reads. Defaults describe the documented Mac combo with online training
    approved; overrides flip individual axes."""
    fake = types.SimpleNamespace(
        _requested_backend="auto",
        is_metal=True,
        _requested_execution_mode="wavefront",
        effective_execution_mode_index=EXECUTION_WAVEFRONT,
        execution_mode_fallback_active=False,
        integrator_modes=["Path", "BDPT"],
        integrator_index=0,
        proposal_preset_index=1,           # bsdf,neural
        _PROPOSAL_PRESETS=_PRESETS,
        _online_training_requested=True,
        _online_training=False,
        _neural_trainer=None,
        _neural_trainer_kind="auto",
        _neural_handoff_kind="interop",
        _train_precision="fp16",
        _spectral=False,
    )
    fake._neural_active = lambda: True
    for k, v in overrides.items():
        setattr(fake, k, v)
    return fake


def _rows(fake) -> dict:
    rows = Renderer._collect_config_rows(fake)
    return {r.axis: r for r in rows}


# ── config_report (pure) ─────────────────────────────────────────────

def test_status_helpers_format_reason():
    assert cr.refused("requires wavefront") == "REFUSED (requires wavefront)"
    assert cr.waiting("select a neural proposal") == "WAITING (select a neural proposal)"


def test_build_matrix_has_header_and_every_row():
    rows = [
        cr.ConfigRow("backend", "auto", "metal", cr.ON),
        cr.ConfigRow("online-training", "on", "—", cr.APPROVED),
    ]
    out = cr.build_config_matrix(rows)
    assert "axis" in out and "requested" in out and "resolved" in out and "status" in out
    assert "backend" in out and "metal" in out
    assert "online-training" in out and "APPROVED" in out


def test_signature_dedups_and_flips():
    base = [cr.ConfigRow("online-training", "on", "—", cr.waiting("pick neural"))]
    same = [cr.ConfigRow("online-training", "on", "—", cr.waiting("pick neural"))]
    flipped = [cr.ConfigRow("online-training", "on", "—", cr.APPROVED)]
    assert cr.matrix_signature(base) == cr.matrix_signature(same)
    assert cr.matrix_signature(base) != cr.matrix_signature(flipped)


# ── renderer-side row assembly: the online-training status payoff ────

def test_rows_approved_when_prereqs_met():
    rows = _rows(_matrix_fake())
    assert rows["online-training"].status == cr.APPROVED
    assert rows["backend"].resolved == "metal"
    assert rows["proposals"].status == "neural ACTIVE"
    # interop resolves to the UMA mechanism on a Metal host.
    assert rows["neural-handoff"].resolved == "interop(UMA)"
    # train-stack rows are live (ON) when training is requested.
    assert rows["neural-trainer"].status == cr.ON
    assert rows["train-precision"].requested == "fp16"


def test_rows_waiting_without_neural_proposal():
    fake = _matrix_fake()
    fake._neural_active = lambda: False
    fake.proposal_preset_index = 0  # bsdf only
    rows = _rows(fake)
    assert rows["online-training"].status == cr.waiting("select a neural proposal")
    assert rows["proposals"].status == cr.ON


def test_rows_refused_off_wavefront():
    fake = _matrix_fake(effective_execution_mode_index=EXECUTION_MEGAKERNEL)
    fake._neural_active = lambda: False  # megakernel ⇒ neural never active
    rows = _rows(fake)
    assert rows["online-training"].status == cr.refused(
        "requires --execution-mode wavefront")


def test_rows_off_and_train_rows_na_when_not_requested():
    fake = _matrix_fake(_online_training_requested=False)
    rows = _rows(fake)
    assert rows["online-training"].status == cr.OFF
    assert rows["neural-trainer"].status == cr.NA
    assert rows["neural-handoff"].status == cr.NA
    assert rows["train-precision"].status == cr.NA


def test_execution_mode_pin_shows_in_status():
    fake = _matrix_fake(
        _requested_execution_mode="wavefront",
        effective_execution_mode_index=EXECUTION_MEGAKERNEL,
        execution_mode_fallback_active=True,
    )
    fake._neural_active = lambda: False
    rows = _rows(fake)
    assert "pinned from wavefront" in rows["execution-mode"].status
    assert rows["execution-mode"].resolved == "megakernel"


def test_spectral_pins_proposals_to_bsdf():
    # Under --spectral the megakernel samples BSDF-only (path_spectral reuses the
    # native BSDF sampler; a non-BSDF proposal would desync the NEE MIS companion).
    # The config matrix must REPORT that pin, not echo a non-BSDF selection.
    fake = _matrix_fake(_spectral=True, proposal_preset_index=1)  # bsdf,neural
    fake._neural_active = lambda: False
    rows = _rows(fake)
    assert rows["proposals"].requested == "bsdf,neural"
    assert rows["proposals"].resolved == "bsdf"
    assert "spectral pin" in rows["proposals"].status
    # The requested token is folded into the STATUS so matrix_signature (which
    # dedups reprints on resolved+status) flips when the preset changes live —
    # otherwise the pinned resolved="bsdf" would leave the matrix stale.
    assert "bsdf,neural" in rows["proposals"].status

    # A BSDF-only preset under spectral needs no pin annotation.
    plain = _matrix_fake(_spectral=True, proposal_preset_index=0)  # bsdf
    plain._neural_active = lambda: False
    prows = _rows(plain)
    assert prows["proposals"].resolved == "bsdf"
    assert "spectral pin" not in prows["proposals"].status


# ── lifecycle: ACTIVE marker + summary ───────────────────────────────

def test_trainer_active_marker_and_summary(capsys):
    replay = ReplayBuffer(capacity=100_000)
    trainer = NeuralTrainer(TrainerConfig(
        steps_per_cycle=4, batch=512, backend="cpu", handoff="interop"))
    rng = np.random.default_rng(0)
    for frame in range(3):
        replay.add(_records(2_000, tag=float(frame)))
        trainer.train_cycle(replay, rng)

    out = capsys.readouterr().out
    # ACTIVE marker prints exactly once, naming backend/handoff/precision.
    assert out.count("online training ACTIVE") == 1
    assert "handoff=interop" in out

    s = trainer.summary()
    assert s["cycles"] == 3
    assert s["steps"] == 12               # 3 cycles × 4 steps/cycle
    assert s["samples"] > 0
    assert s["duration_s"] >= 0.0
    assert s["backend"] == trainer.backend_name


def test_summary_zero_before_any_cycle():
    trainer = NeuralTrainer(TrainerConfig(backend="cpu"))
    s = trainer.summary()
    assert s["cycles"] == 0 and s["steps"] == 0 and s["duration_s"] == 0.0


# ── renderer STOPPED summary guard + status snapshot ─────────────────

def test_print_train_summary_prints_once(capsys):
    trainer = types.SimpleNamespace(summary=lambda: {
        "duration_s": 12.5, "cycles": 5, "steps": 20, "samples": 9000,
        "final_loss": 0.0123, "backend": "mlx"})
    fake = types.SimpleNamespace(_train_summary_printed=False, _neural_trainer=trainer)
    Renderer._print_train_summary(fake)
    Renderer._print_train_summary(fake)  # idempotent — must not double-print
    out = capsys.readouterr().out
    assert out.count("online training STOPPED") == 1
    assert "5 cycles" in out and "20 steps" in out and "backend=mlx" in out


def test_print_train_summary_skips_when_never_trained(capsys):
    trainer = types.SimpleNamespace(summary=lambda: {
        "duration_s": 0.0, "cycles": 0, "steps": 0, "samples": 0,
        "final_loss": None, "backend": "cpu"})
    fake = types.SimpleNamespace(_train_summary_printed=False, _neural_trainer=trainer)
    Renderer._print_train_summary(fake)
    assert "STOPPED" not in capsys.readouterr().out


def test_online_training_status_snapshot():
    trainer = types.SimpleNamespace(
        last_loss=0.05, _trained_cycles=7, _started_t=1.0, backend_name="mlx")
    fake = types.SimpleNamespace(_online_training=True, _neural_trainer=trainer)
    st = Renderer.online_training_status(fake)
    assert st == {"armed": True, "active": True, "last_loss": 0.05,
                  "cycles": 7, "backend": "mlx"}

    off = types.SimpleNamespace(_online_training=False, _neural_trainer=None)
    st2 = Renderer.online_training_status(off)
    assert st2["armed"] is False and st2["active"] is False and st2["cycles"] == 0


# ── enable-gate readiness (change online-training-metal-enable-gate) ──

def _ready(fake) -> bool:
    return Renderer._backend_render_ready.fget(fake)


def test_metal_wavefront_ready_without_descriptor_sets():
    # The native Metal backend never allocates Vulkan descriptor_sets; readiness
    # must come from the scene bindings, or online training never enables (the
    # bug: the old gate tested descriptor_sets, permanently None on Metal).
    fake = types.SimpleNamespace(
        is_metal=True, effective_execution_mode_index=EXECUTION_WAVEFRONT,
        descriptor_sets=None, _scene_bindings=object(), pipeline=None)
    assert _ready(fake) is True
    fake._scene_bindings = None
    assert _ready(fake) is False


def test_vulkan_wavefront_still_requires_descriptor_sets():
    fake = types.SimpleNamespace(
        is_metal=False, effective_execution_mode_index=EXECUTION_WAVEFRONT,
        descriptor_sets=None, _scene_bindings=object(), pipeline=None)
    assert _ready(fake) is False
    fake.descriptor_sets = object()
    assert _ready(fake) is True


# ── bdpt × neural refusal (change bdpt-neural-incompatibility) ────────

def test_can_online_train_refuses_bdpt():
    fake = types.SimpleNamespace(
        effective_execution_mode_index=EXECUTION_WAVEFRONT, integrator_index=1)
    fake._neural_active = lambda: True
    ok, reason = Renderer.can_online_train(fake)
    assert ok is False and "--integrator path" in reason


def test_matrix_online_training_refused_under_bdpt():
    fake = _matrix_fake(integrator_index=1)
    rows = _rows(fake)
    assert rows["online-training"].status == cr.refused(
        "requires --integrator path (bdpt ignores neural)")
