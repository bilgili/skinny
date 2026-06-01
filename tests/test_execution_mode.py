"""Phase-0 tests for the execution-mode axis (megakernel | wavefront).

GPU-free. Exercises the pure capability-gate helper, the mode clamp/cycle
helpers, the shared ParamSpec + persistence wiring. The data-driven UI binding
is covered separately by test_ui_spec once the param is added.
"""

from __future__ import annotations

from skinny.params import (
    EXECUTION_MEGAKERNEL,
    EXECUTION_WAVEFRONT,
    STATIC_PARAMS,
    _apply_saved_params,
    _snapshot_params,
    clamp_mode_index,
    effective_execution_mode,
    next_mode_index,
)


def _exec_spec():
    return next(p for p in STATIC_PARAMS if p.path == "execution_mode_index")


# ── Capability gate ────────────────────────────────────────────────


def test_megakernel_never_gated():
    for integrator in (0, 1):
        assert (
            effective_execution_mode(EXECUTION_MEGAKERNEL, integrator, False)
            == EXECUTION_MEGAKERNEL
        )


def test_wavefront_path_not_gated():
    assert (
        effective_execution_mode(EXECUTION_WAVEFRONT, 0, False) == EXECUTION_WAVEFRONT
    )


def test_wavefront_bdpt_falls_back_until_supported():
    assert (
        effective_execution_mode(EXECUTION_WAVEFRONT, 1, False) == EXECUTION_MEGAKERNEL
    )


def test_wavefront_bdpt_passes_through_once_supported():
    assert (
        effective_execution_mode(EXECUTION_WAVEFRONT, 1, True) == EXECUTION_WAVEFRONT
    )


# ── Mode index clamp / cycle (Metal pin = single mode) ──────────────


def test_clamp_keeps_in_range():
    assert clamp_mode_index(0, 2) == 0
    assert clamp_mode_index(1, 2) == 1
    assert clamp_mode_index(5, 2) == 1  # over-range clamps to last
    assert clamp_mode_index(-3, 2) == 0  # under-range clamps to first


def test_clamp_pins_single_mode():
    # Metal: only "Megakernel" available — any request stays at 0.
    assert clamp_mode_index(1, 1) == 0
    assert clamp_mode_index(0, 1) == 0


def test_cycle_wraps():
    assert next_mode_index(0, 2) == 1
    assert next_mode_index(1, 2) == 0


def test_cycle_pins_single_mode():
    assert next_mode_index(0, 1) == 0


# ── ParamSpec presence ─────────────────────────────────────────────


def test_execution_param_present_and_discrete():
    spec = _exec_spec()
    assert spec.kind == "discrete"
    assert spec.choice_source == "execution_modes"


# ── Persistence ────────────────────────────────────────────────────


class _Stub:
    def __init__(self, modes, index):
        self.execution_modes = modes
        self.execution_mode_index = index


def test_execution_mode_roundtrips_through_settings():
    r = _Stub(["Megakernel", "Wavefront"], EXECUTION_WAVEFRONT)
    snap = _snapshot_params(r, [_exec_spec()])
    assert snap == {"execution_mode_index": EXECUTION_WAVEFRONT}

    restored = _Stub(["Megakernel", "Wavefront"], EXECUTION_MEGAKERNEL)
    _apply_saved_params(restored, snap, [_exec_spec()])
    assert restored.execution_mode_index == EXECUTION_WAVEFRONT


def test_restore_rejects_unavailable_mode_on_metal():
    # Metal exposes a single mode; a saved Wavefront index is out of range
    # and must be rejected, leaving the pin at megakernel.
    metal = _Stub(["Megakernel"], EXECUTION_MEGAKERNEL)
    _apply_saved_params(metal, {"execution_mode_index": EXECUTION_WAVEFRONT}, [_exec_spec()])
    assert metal.execution_mode_index == EXECUTION_MEGAKERNEL
