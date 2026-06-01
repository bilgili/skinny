"""Tests for the execution-mode axis (megakernel | wavefront).

GPU-free. Exercises the pure capability-gate helper and the mode clamp the
renderer applies to the CLI-selected (`--execution-mode`) value at
construction. The mode is fixed for the session — it is no longer a runtime
GUI toggle or a persisted setting — so the snapshot/restore + Combo coverage
that lived here is gone; the constructor selection + mutual-exclusion are
covered by the GPU tests in test_wavefront_*.py.
"""

from __future__ import annotations

from skinny.params import (
    EXECUTION_MEGAKERNEL,
    EXECUTION_WAVEFRONT,
    clamp_mode_index,
    effective_execution_mode,
)


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


# ── Mode index clamp (the constructor applies this to the CLI value) ─


def test_clamp_keeps_in_range():
    assert clamp_mode_index(0, 2) == 0
    assert clamp_mode_index(1, 2) == 1
    assert clamp_mode_index(5, 2) == 1  # over-range clamps to last
    assert clamp_mode_index(-3, 2) == 0  # under-range clamps to first


def test_clamp_pins_single_mode():
    # Metal / non-Vulkan: only "Megakernel" available — a wavefront request
    # (index 1) collapses to 0, which is how the constructor enforces the pin.
    assert clamp_mode_index(EXECUTION_WAVEFRONT, 1) == EXECUTION_MEGAKERNEL
    assert clamp_mode_index(EXECUTION_MEGAKERNEL, 1) == EXECUTION_MEGAKERNEL
