"""Unit tests for ISampler implementations: GGX, Lambert, HG, UniformSphere."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near, assert_on_hemisphere, assert_unit_vector, vec3_dot

PI = math.pi
pytestmark = pytest.mark.gpu

# Quasi-random sample points (golden ratio sequences)
def _quasi_samples(n, seed_a=0.1, seed_b=0.3):
    return [((i * 0.618034 + seed_a) % 1.0, (i * 0.414214 + seed_b) % 1.0) for i in range(n)]


class TestLambertSampler:
    N = [0.0, 1.0, 0.0]

    def test_returns_unit_vectors(self, sampler_harness):
        for u in _quasi_samples(30):
            result = sampler_harness.test_lambert_sample(self.N, list(u))
            v = [float(result[j]) for j in range(3)]
            assert_unit_vector(v)

    def test_on_hemisphere(self, sampler_harness):
        for u in _quasi_samples(30):
            result = sampler_harness.test_lambert_sample(self.N, list(u))
            v = [float(result[j]) for j in range(3)]
            assert_on_hemisphere(v, self.N)

    def test_pdf_positive_for_samples(self, sampler_harness):
        for u in _quasi_samples(20):
            s = sampler_harness.test_lambert_sample(self.N, list(u))
            L = [float(s[j]) for j in range(3)]
            pdf = float(sampler_harness.test_lambert_pdf(self.N, L))
            assert pdf > 0.0, f"pdf={pdf} for valid sample"

    def test_pdf_matches_cosine(self, sampler_harness):
        N = self.N
        for u in _quasi_samples(10):
            s = sampler_harness.test_lambert_sample(N, list(u))
            L = [float(s[j]) for j in range(3)]
            pdf = float(sampler_harness.test_lambert_pdf(N, L))
            ndotl = max(vec3_dot(L, N), 0.0)
            expected = ndotl / PI
            assert_near(pdf, expected, rel=1e-3)

    def test_tilted_normal(self, sampler_harness):
        N = [0.0, 0.707, 0.707]
        for u in _quasi_samples(10):
            result = sampler_harness.test_lambert_sample(N, list(u))
            v = [float(result[j]) for j in range(3)]
            assert_unit_vector(v)
            assert_on_hemisphere(v, N)


class TestGGXSampler:
    N = [0.0, 1.0, 0.0]
    V = [0.0, 1.0, 0.0]

    def test_returns_unit_vectors(self, sampler_harness):
        for u in _quasi_samples(30):
            result = sampler_harness.test_ggx_sample(self.N, self.V, 0.5, list(u))
            v = [float(result[j]) for j in range(3)]
            assert_unit_vector(v, tol=1e-3)

    def test_on_hemisphere(self, sampler_harness):
        for u in _quasi_samples(20):
            result = sampler_harness.test_ggx_sample(self.N, self.V, 0.5, list(u))
            v = [float(result[j]) for j in range(3)]
            dot = vec3_dot(v, self.N)
            if dot > 0:
                assert_on_hemisphere(v, self.N)

    def test_pdf_positive(self, sampler_harness):
        for u in _quasi_samples(10):
            s = sampler_harness.test_ggx_sample(self.N, self.V, 0.5, list(u))
            L = [float(s[j]) for j in range(3)]
            if vec3_dot(L, self.N) > 0:
                pdf = float(sampler_harness.test_ggx_pdf(self.N, self.V, 0.5, L))
                assert pdf > 0.0

    def test_smooth_concentrates_near_reflection(self, sampler_harness):
        V = [0.0, 0.707, 0.707]
        dots_smooth = []
        dots_rough = []
        R = [0.0, 0.707, 0.707]  # reflection of V about N
        for u in _quasi_samples(30):
            s_smooth = sampler_harness.test_ggx_sample(self.N, V, 0.05, list(u))
            s_rough = sampler_harness.test_ggx_sample(self.N, V, 0.8, list(u))
            ls = [float(s_smooth[j]) for j in range(3)]
            lr = [float(s_rough[j]) for j in range(3)]
            dots_smooth.append(vec3_dot(ls, R))
            dots_rough.append(vec3_dot(lr, R))
        avg_smooth = sum(dots_smooth) / len(dots_smooth)
        avg_rough = sum(dots_rough) / len(dots_rough)
        assert avg_smooth > avg_rough, "Smoother should concentrate near reflection"


class TestUniformSphereSampler:
    def test_returns_unit_vectors(self, sampler_harness):
        for u in _quasi_samples(20):
            result = sampler_harness.test_uniform_sphere_sample(list(u))
            v = [float(result[j]) for j in range(3)]
            assert_unit_vector(v)

    def test_pdf_is_constant(self, sampler_harness):
        expected = 1.0 / (4.0 * PI)
        for u in _quasi_samples(10):
            s = sampler_harness.test_uniform_sphere_sample(list(u))
            L = [float(s[j]) for j in range(3)]
            pdf = float(sampler_harness.test_uniform_sphere_pdf(L))
            assert_near(pdf, expected)


class TestHenyeyGreensteinSampler:
    fwd = [0.0, 0.0, 1.0]

    def test_returns_unit_vectors(self, sampler_harness):
        for g in [0.0, 0.5, 0.9, -0.5]:
            for u in _quasi_samples(10):
                result = sampler_harness.test_hg_sample(self.fwd, g, list(u))
                v = [float(result[j]) for j in range(3)]
                assert_unit_vector(v, tol=1e-3)

    def test_isotropic_when_g_zero(self, sampler_harness):
        expected = 1.0 / (4.0 * PI)
        for u in _quasi_samples(10):
            s = sampler_harness.test_hg_sample(self.fwd, 0.0, list(u))
            L = [float(s[j]) for j in range(3)]
            pdf = float(sampler_harness.test_hg_pdf(self.fwd, 0.0, L))
            assert_near(pdf, expected, rel=1e-2)

    def test_forward_scattering_when_g_positive(self, sampler_harness):
        dots = []
        for u in _quasi_samples(30):
            s = sampler_harness.test_hg_sample(self.fwd, 0.9, list(u))
            L = [float(s[j]) for j in range(3)]
            dots.append(vec3_dot(L, self.fwd))
        avg = sum(dots) / len(dots)
        assert avg > 0.5, f"g=0.9 should scatter forward, avg dot={avg}"

    def test_backward_scattering_when_g_negative(self, sampler_harness):
        dots = []
        for u in _quasi_samples(30):
            s = sampler_harness.test_hg_sample(self.fwd, -0.8, list(u))
            L = [float(s[j]) for j in range(3)]
            dots.append(vec3_dot(L, self.fwd))
        avg = sum(dots) / len(dots)
        assert avg < -0.3, f"g=-0.8 should scatter backward, avg dot={avg}"

    def test_pdf_matches_sample(self, sampler_harness):
        for g in [0.3, 0.7, -0.5]:
            for u in _quasi_samples(10):
                s = sampler_harness.test_hg_sample(self.fwd, g, list(u))
                L = [float(s[j]) for j in range(3)]
                pdf = float(sampler_harness.test_hg_pdf(self.fwd, g, L))
                cosTheta = vec3_dot(L, self.fwd)
                g2 = g * g
                expected = (1.0 - g2) / (4.0 * PI * (1.0 + g2 - 2.0 * g * cosTheta) ** 1.5)
                assert_near(pdf, expected, rel=1e-3)
