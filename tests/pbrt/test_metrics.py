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
