"""Unit tests for volume rendering: Henyey-Greenstein phase function."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near, assert_unit_vector, vec3_dot

PI = math.pi
pytestmark = pytest.mark.gpu


def _quasi_samples(n, seed_a=0.1, seed_b=0.3):
    return [((i * 0.618034 + seed_a) % 1.0, (i * 0.414214 + seed_b) % 1.0) for i in range(n)]


class TestHenyeyGreensteinPhaseFunction:
    def test_isotropic_at_g_zero(self, volume_harness):
        expected = 1.0 / (4.0 * PI)
        for cos_t in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = float(volume_harness.test_henyeyGreenstein(cos_t, 0.0))
            assert_near(result, expected, rel=1e-3)

    def test_forward_peak(self, volume_harness):
        fwd = float(volume_harness.test_henyeyGreenstein(1.0, 0.9))
        back = float(volume_harness.test_henyeyGreenstein(-1.0, 0.9))
        assert fwd > 10.0 * back

    def test_backward_peak(self, volume_harness):
        fwd = float(volume_harness.test_henyeyGreenstein(1.0, -0.9))
        back = float(volume_harness.test_henyeyGreenstein(-1.0, -0.9))
        assert back > 10.0 * fwd

    def test_nonnegative(self, volume_harness):
        for g in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            for cos_t in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                val = float(volume_harness.test_henyeyGreenstein(cos_t, g))
                assert val >= 0.0

    def test_known_value(self, volume_harness):
        g = 0.5
        cos_t = 0.3
        g2 = g * g
        expected = (1.0 - g2) / (4.0 * PI * (1.0 + g2 - 2.0 * g * cos_t) ** 1.5)
        result = float(volume_harness.test_henyeyGreenstein(cos_t, g))
        assert_near(result, expected, rel=1e-4)


class TestSampleHenyeyGreenstein:
    fwd = [0.0, 0.0, 1.0]

    def test_returns_unit_vectors(self, volume_harness):
        for g in [0.0, 0.5, 0.9, -0.5]:
            for u in _quasi_samples(10):
                result = volume_harness.test_sampleHenyeyGreenstein(self.fwd, g, list(u))
                v = [float(result[j]) for j in range(3)]
                assert_unit_vector(v, tol=1e-3)

    def test_isotropic_distribution(self, volume_harness):
        dots = []
        for u in _quasi_samples(30):
            result = volume_harness.test_sampleHenyeyGreenstein(self.fwd, 0.0, list(u))
            v = [float(result[j]) for j in range(3)]
            dots.append(vec3_dot(v, self.fwd))
        avg = sum(dots) / len(dots)
        assert abs(avg) < 0.15, f"g=0 should be ~isotropic, avg dot={avg}"

    def test_forward_bias(self, volume_harness):
        dots = []
        for u in _quasi_samples(30):
            result = volume_harness.test_sampleHenyeyGreenstein(self.fwd, 0.9, list(u))
            v = [float(result[j]) for j in range(3)]
            dots.append(vec3_dot(v, self.fwd))
        avg = sum(dots) / len(dots)
        assert avg > 0.5, f"g=0.9 forward bias, avg dot={avg}"

    def test_pdf_consistency(self, volume_harness):
        g = 0.7
        for u in _quasi_samples(10):
            result = volume_harness.test_sampleHenyeyGreenstein(self.fwd, g, list(u))
            v = [float(result[j]) for j in range(3)]
            cos_t = vec3_dot(v, self.fwd)
            pdf_expected = float(volume_harness.test_henyeyGreenstein(cos_t, g))
            g2 = g * g
            pdf_analytical = (1.0 - g2) / (4.0 * PI * (1.0 + g2 - 2.0 * g * cos_t) ** 1.5)
            assert_near(pdf_expected, pdf_analytical, rel=1e-3)
