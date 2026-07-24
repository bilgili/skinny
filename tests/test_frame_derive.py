"""Hostless tests for `skinny.frame_derive` (change renderer-module-carveout,
Stage B) — the pure frame-constant derivation carved out of `_pack_uniforms`.
No GPU, no `Renderer`.
"""

from __future__ import annotations

import math

import pytest

from skinny import frame_derive as fd

_MEGA, _WAVE = 0, fd.EXECUTION_WAVEFRONT


# ── detail_flags ─────────────────────────────────────────────────────────────

def test_detail_flags_all_on():
    assert fd.detail_flags(True, True, True, True, True) == 0b11111


def test_detail_flags_master_only_when_maps_missing():
    # Master toggled on but no maps present → only bit 0 set; the shader AND-s
    # master with each per-map bit, so a missing map stays off.
    assert fd.detail_flags(True, False, False, False, False) == 0b00001


def test_detail_flags_individual_bits():
    assert fd.detail_flags(False, True, False, False, False) == 0b00010
    assert fd.detail_flags(False, False, True, False, False) == 0b00100
    assert fd.detail_flags(False, False, False, True, False) == 0b01000
    assert fd.detail_flags(False, False, False, False, True) == 0b10000


def test_detail_flags_all_off():
    assert fd.detail_flags(False, False, False, False, False) == 0


# ── film_half_height_world ───────────────────────────────────────────────────

def test_film_half_height_pinhole_is_base():
    # No lens active → 0.5 * va / mm_per_unit, focal irrelevant.
    assert fd.film_half_height_world(24.0, 50.0, 1.0, 0, 999.0) == pytest.approx(12.0)


def test_film_half_height_scales_by_lens_framing_ratio():
    # ratio = filmDistance / (focal/mm_per_unit); half *= ratio.
    got = fd.film_half_height_world(24.0, 50.0, 1.0, 2, 0.05)
    assert got == pytest.approx(12.0 * (0.05 / 50.0))


def test_film_half_height_mm_per_unit_scales_base_and_focal_together():
    base = fd.film_half_height_world(24.0, 50.0, 1000.0, 0, 0.0)
    assert base == pytest.approx(0.5 * 24.0 / 1000.0)
    lens = fd.film_half_height_world(24.0, 50.0, 1000.0, 1, 0.05)
    assert lens == pytest.approx(base * (0.05 / (50.0 / 1000.0)))


def test_film_half_height_ignores_lens_with_degenerate_focal():
    # focal ≤ 1e-3 → no ratio applied even with a lens active (guards /0).
    assert fd.film_half_height_world(24.0, 0.0, 1.0, 3, 0.05) == pytest.approx(12.0)


def test_film_half_height_guards_zero_mm_per_unit():
    # mm_per_unit clamps to 1e-6 rather than dividing by zero.
    assert math.isfinite(fd.film_half_height_world(24.0, 50.0, 0.0, 0, 0.0))


# ── exposure_stops ───────────────────────────────────────────────────────────

def test_exposure_stops_ratio_one_is_identity():
    assert fd.exposure_stops(1.5, 1.0) == pytest.approx(1.5)


def test_exposure_stops_folds_ratio_as_log2():
    assert fd.exposure_stops(0.0, 4.0) == pytest.approx(2.0)
    assert fd.exposure_stops(1.0, 0.5) == pytest.approx(0.0)


def test_exposure_stops_nonpositive_ratio_adds_zero():
    assert fd.exposure_stops(2.3, 0.0) == pytest.approx(2.3)
    assert fd.exposure_stops(2.3, -1.0) == pytest.approx(2.3)


# ── fold_sampling_capabilities ───────────────────────────────────────────────

def test_fold_bsdf_default_is_identity_on_both_modes():
    for mode in (_MEGA, _WAVE):
        m, a, r, stripped = fd.fold_sampling_capabilities(
            0x1, (1.0, 0.0, 0.0, 0.0), 0, mode)
        assert (m, a, r, stripped) == (0x1, (1.0, 0.0, 0.0, 0.0), 0, False)


def test_fold_reuse_zeroed_off_wavefront():
    _, _, r, _ = fd.fold_sampling_capabilities(0x1, (1, 0, 0, 0), 2, _MEGA)
    assert r == 0
    _, _, r, _ = fd.fold_sampling_capabilities(0x1, (1, 0, 0, 0), 2, _WAVE)
    assert r == 2


def test_fold_neural_survives_on_wavefront():
    m, a, r, stripped = fd.fold_sampling_capabilities(
        0x5, (0.5, 0.0, 0.5, 0.0), 0, _WAVE)
    assert m == 0x5 and stripped is False
    assert a == (0.5, 0.0, 0.5, 0.0)


def test_fold_neural_stripped_and_renormalised_on_megakernel():
    # bsdf+neural mixture, neural weight 0.5 → strip bit2, renormalise the
    # remaining {bsdf} to 1.0.
    m, a, r, stripped = fd.fold_sampling_capabilities(
        0x5, (0.5, 0.0, 0.5, 0.0), 0, _MEGA)
    assert m == 0x1 and stripped is True
    assert a == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_fold_neural_strip_renormalises_multilobe_mixture():
    # bsdf+env+neural = bits 0,1,2, weights (0.25, 0.25, 0.5). Strip neural →
    # mask 0x3, alpha renormalised over the surviving 0.5 mass.
    m, a, r, stripped = fd.fold_sampling_capabilities(
        0x7, (0.25, 0.25, 0.5, 0.0), 0, _MEGA)
    assert m == 0x3 and stripped is True
    assert a == pytest.approx((0.5, 0.5, 0.0, 0.0))
    assert sum(a) == pytest.approx(1.0)


def test_fold_neural_only_on_megakernel_falls_back_to_bsdf():
    # Neural-only (mask 0x4): stripping leaves an empty mixture → fall back to
    # the {bsdf} fast path, not a zero mask.
    m, a, r, stripped = fd.fold_sampling_capabilities(
        0x4, (0.0, 0.0, 1.0, 0.0), 0, _MEGA)
    assert m == 0x1 and stripped is True
    assert a == (1.0, 0.0, 0.0, 0.0)


def test_fold_does_not_mutate_caller_alpha():
    alpha = [0.5, 0.0, 0.5, 0.0]
    fd.fold_sampling_capabilities(0x5, alpha, 0, _MEGA)
    assert alpha == [0.5, 0.0, 0.5, 0.0]
