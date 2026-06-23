"""Tests for parity metrics (task 11.1)."""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import metrics


def _img(seed, shape=(8, 8, 3)):
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(np.float64)


def test_relmse_identical_is_zero():
    a = _img(1)
    assert metrics.relmse(a, a) == pytest.approx(0.0, abs=1e-12)


def test_relmse_hand_value():
    a = np.full((1, 1, 3), 2.0)
    b = np.full((1, 1, 3), 1.0)
    # (2-1)^2 / (1^2 + eps) with eps=1e-2
    assert metrics.relmse(a, b) == pytest.approx(1.0 / 1.01, abs=1e-6)


def test_flip_identical_is_zero():
    a = _img(2)
    assert metrics.flip(a, a) == pytest.approx(0.0, abs=1e-9)


def test_flip_positive_for_different():
    a = np.zeros((8, 8, 3))
    b = np.ones((8, 8, 3))
    assert metrics.flip(a, b) > 0.1


def test_align_exposure_recovers_scale():
    b = _img(3)
    a = b / 3.0  # a is darker by 3x
    aligned = metrics.align_exposure(a, b)
    assert metrics.relmse(aligned, b) < metrics.relmse(a, b)


# ─── standardized battery (compute_metrics / ImageMetrics) ────────────────

import math  # noqa: E402


def test_battery_identical_images_zero_error_inf_psnr():
    a = _img(10, (32, 32, 3))
    m = metrics.compute_metrics(a, a)
    assert m.mse == pytest.approx(0.0, abs=1e-12)
    assert m.rmse == pytest.approx(0.0, abs=1e-12)
    assert m.mae == pytest.approx(0.0, abs=1e-12)
    assert m.relmse == pytest.approx(0.0, abs=1e-12)
    assert m.flip == pytest.approx(0.0, abs=1e-9)
    assert math.isinf(m.psnr) and m.psnr > 0


def test_battery_no_reference_only_single_image_stats():
    a = _img(11, (32, 32, 3))
    m = metrics.compute_metrics(a, None)
    assert m.mse is None and m.psnr is None and m.flip is None
    assert m.variance > 0
    assert m.noise_sigma >= 0
    assert 0.0 <= m.firefly_fraction <= 1.0


def test_battery_exposure_scaled_copy_aligns_to_near_zero():
    a = _img(12, (16, 16, 3))
    m = metrics.compute_metrics(a * 3.7, a)  # pure global exposure change
    assert m.relmse < 1e-6
    assert m.mse < 1e-6


def test_firefly_fraction_flags_planted_spike():
    a = np.full((32, 32, 3), 0.2, dtype=np.float64)
    assert metrics.firefly_fraction(a) == pytest.approx(0.0)
    a[16, 16] = 50.0
    flagged = metrics.firefly_fraction(a)
    assert flagged > 0.0
    assert flagged == pytest.approx(1.0 / (32 * 32), rel=0.5)


def test_noise_sigma_rises_with_added_noise():
    rng = np.random.default_rng(13)
    smooth = np.broadcast_to(
        np.linspace(0.1, 0.9, 32)[None, :, None], (32, 32, 3)
    ).astype(np.float64)
    noisy = smooth + rng.normal(0, 0.05, size=smooth.shape)
    assert metrics.noise_sigma(noisy) > metrics.noise_sigma(smooth)


def test_psnr_decreases_with_error():
    a = _img(14, (16, 16, 3))
    assert metrics.psnr(a + 0.01, a) > metrics.psnr(a + 0.2, a)


def test_battery_as_dict_has_all_fields():
    m = metrics.compute_metrics(_img(15, (16, 16, 3)), _img(16, (16, 16, 3)))
    assert set(m.as_dict()) == {
        "mse", "rmse", "mae", "relmse", "psnr", "flip",
        "variance", "noise_sigma", "firefly_fraction",
    }
    assert "relMSE" in m.summary()
