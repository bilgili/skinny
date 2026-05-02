"""Unit tests for skin optics: melanin, hemoglobin, Fresnel, BSSRDF, specular."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near

PI = math.pi
pytestmark = pytest.mark.gpu


class TestMelaninAbsorption:
    def test_zero_melanin(self, skin_harness):
        result = skin_harness.test_melaninAbsorption(0.0)
        for i in range(3):
            assert abs(float(result[i])) < 1e-6

    def test_unit_melanin(self, skin_harness):
        result = skin_harness.test_melaninAbsorption(1.0)
        expected = [6.6, 11.0, 24.0]
        for i in range(3):
            assert_near(float(result[i]), expected[i], rel=1e-4)

    def test_half_melanin(self, skin_harness):
        result = skin_harness.test_melaninAbsorption(0.5)
        expected = [3.3, 5.5, 12.0]
        for i in range(3):
            assert_near(float(result[i]), expected[i], rel=1e-4)

    def test_monotonic(self, skin_harness):
        prev = [0.0, 0.0, 0.0]
        for f in [0.1, 0.3, 0.5, 0.7, 1.0]:
            result = skin_harness.test_melaninAbsorption(f)
            for i in range(3):
                val = float(result[i])
                assert val >= prev[i] - 1e-6
                prev[i] = val


class TestHemoglobinAbsorption:
    def test_zero_hemoglobin(self, skin_harness):
        result = skin_harness.test_hemoglobinAbsorption(0.0, 0.5)
        for i in range(3):
            assert abs(float(result[i])) < 1e-6

    def test_full_oxy(self, skin_harness):
        result = skin_harness.test_hemoglobinAbsorption(1.0, 1.0)
        expected = [0.3, 7.0, 1.0]
        for i in range(3):
            assert_near(float(result[i]), expected[i], rel=1e-4)

    def test_full_deoxy(self, skin_harness):
        result = skin_harness.test_hemoglobinAbsorption(1.0, 0.0)
        expected = [1.6, 8.5, 3.2]
        for i in range(3):
            assert_near(float(result[i]), expected[i], rel=1e-4)

    def test_linear_interpolation(self, skin_harness):
        f = 0.6
        result = skin_harness.test_hemoglobinAbsorption(f, 0.5)
        hbO2 = [0.3, 7.0, 1.0]
        hb = [1.6, 8.5, 3.2]
        for i in range(3):
            expected = f * 0.5 * (hbO2[i] + hb[i])
            assert_near(float(result[i]), expected, rel=1e-4)


class TestFresnelSchlick:
    def test_normal_incidence(self, skin_harness):
        ior = 1.4
        result = float(skin_harness.test_fresnelSchlick(1.0, ior))
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        assert_near(result, r0, rel=1e-3)

    def test_grazing_incidence(self, skin_harness):
        result = float(skin_harness.test_fresnelSchlick(0.0, 1.4))
        assert_near(result, 1.0, rel=1e-3)

    def test_ior_one_is_zero(self, skin_harness):
        result = float(skin_harness.test_fresnelSchlick(1.0, 1.0))
        assert abs(result) < 1e-5

    def test_monotonic_in_cos_theta(self, skin_harness):
        prev = 2.0
        for cos_t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            val = float(skin_harness.test_fresnelSchlick(cos_t, 1.4))
            assert val <= prev + 1e-5
            prev = val


class TestGGXDistribution:
    def test_at_normal(self, skin_harness):
        result = float(skin_harness.test_ggxDistribution(1.0, 0.5))
        a = 0.5 * 0.5
        a2 = a * a
        expected = 1.0 / (PI * a2)
        assert_near(result, expected, rel=1e-3)

    def test_peak_decreases_with_roughness(self, skin_harness):
        smooth = float(skin_harness.test_ggxDistribution(1.0, 0.1))
        rough = float(skin_harness.test_ggxDistribution(1.0, 0.9))
        assert smooth > rough

    def test_nonnegative(self, skin_harness):
        for ndoth in [0.1, 0.5, 0.9, 1.0]:
            for r in [0.1, 0.5, 0.9]:
                val = float(skin_harness.test_ggxDistribution(ndoth, r))
                assert val >= 0.0


class TestEvaluateSkinSpecular:
    def test_nonnegative(self, skin_harness):
        N = [0.0, 1.0, 0.0]
        V = [0.0, 1.0, 0.0]
        L = [0.0, 0.707, 0.707]
        result = skin_harness.test_evaluateSkinSpecular(N, V, L, 0.5, 1.4)
        for i in range(3):
            assert float(result[i]) >= -1e-6

    def test_below_hemisphere_finite(self, skin_harness):
        N = [0.0, 1.0, 0.0]
        V = [0.0, 1.0, 0.0]
        L = [0.0, -1.0, 0.0]
        result = skin_harness.test_evaluateSkinSpecular(N, V, L, 0.5, 1.4)
        for i in range(3):
            val = float(result[i])
            assert math.isfinite(val)


class TestBuildEpidermis:
    def test_absorption_includes_melanin(self, skin_harness):
        result = skin_harness.test_buildEpidermis(0.5, 0.1, [3.7, 4.4, 5.05], 0.8, 1.4)
        sigmaA = [float(result['sigmaA'][i]) for i in range(3)]
        melanin = [3.3, 5.5, 12.0]
        baseline = 0.015
        for i in range(3):
            expected = melanin[i] + baseline
            assert_near(sigmaA[i], expected, rel=1e-3)

    def test_scattering_matches_input(self, skin_harness):
        scat = [3.7, 4.4, 5.05]
        result = skin_harness.test_buildEpidermis(0.15, 0.1, scat, 0.8, 1.4)
        for i in range(3):
            assert_near(float(result['sigmaS'][i]), scat[i], rel=1e-4)

    def test_thickness_matches(self, skin_harness):
        result = skin_harness.test_buildEpidermis(0.15, 0.1, [3.7, 4.4, 5.05], 0.8, 1.4)
        assert_near(float(result['thickness']), 0.1)

    def test_nonnegative_absorption(self, skin_harness):
        result = skin_harness.test_buildEpidermis(0.0, 0.1, [3.7, 4.4, 5.05], 0.8, 1.4)
        for i in range(3):
            assert float(result['sigmaA'][i]) >= 0.0


class TestBuildDermis:
    def test_absorption_includes_hemoglobin(self, skin_harness):
        result = skin_harness.test_buildDermis(0.5, 0.75, 1.0, [3.7, 4.4, 5.05], 0.8, 1.4)
        sigmaA = [float(result['sigmaA'][i]) for i in range(3)]
        for i in range(3):
            assert sigmaA[i] > 0.015

    def test_nonnegative_scattering(self, skin_harness):
        result = skin_harness.test_buildDermis(0.05, 0.75, 1.0, [3.7, 4.4, 5.05], 0.8, 1.4)
        for i in range(3):
            assert float(result['sigmaS'][i]) >= 0.0


class TestInkOpticalContribution:
    def test_zero_density_no_absorption(self, skin_harness):
        result = skin_harness.test_inkSigmaAbsorption([1.0, 0.0, 0.0], 0.0)
        for i in range(3):
            assert abs(float(result[i])) < 1e-6

    def test_ink_absorption_formula(self, skin_harness):
        rgb = [1.0, 0.0, 0.0]
        density = 1.0
        result = skin_harness.test_inkSigmaAbsorption(rgb, density)
        r_abs = 22.0 * (1.0 - 1.0) + 1.2 * 1.0  # red channel: ink matches, low abs
        g_abs = 22.0 * (1.0 - 0.0) + 1.2 * 0.0  # green: complement, high abs
        b_abs = 22.0 * (1.0 - 0.0) + 1.2 * 0.0
        assert_near(float(result[0]), r_abs, rel=1e-3)
        assert_near(float(result[1]), g_abs, rel=1e-3)
        assert_near(float(result[2]), b_abs, rel=1e-3)

    def test_ink_scattering_nonnegative(self, skin_harness):
        result = skin_harness.test_inkSigmaScattering([0.5, 0.5, 0.5], 1.0)
        for i in range(3):
            assert float(result[i]) > 0.0


class TestBurleyDiffusionProfile:
    def test_decays_with_distance(self, skin_harness):
        d = [1.0, 1.0, 1.0]
        near = skin_harness.test_burleyDiffusionProfile(0.1, d)
        far = skin_harness.test_burleyDiffusionProfile(5.0, d)
        for i in range(3):
            assert float(near[i]) > float(far[i])

    def test_nonnegative(self, skin_harness):
        d = [0.5, 0.7, 1.0]
        for r in [0.01, 0.1, 0.5, 1.0, 5.0]:
            result = skin_harness.test_burleyDiffusionProfile(r, d)
            for i in range(3):
                assert float(result[i]) >= 0.0

    def test_larger_d_broader(self, skin_harness):
        r = 10.0
        narrow = skin_harness.test_burleyDiffusionProfile(r, [0.5, 0.5, 0.5])
        broad = skin_harness.test_burleyDiffusionProfile(r, [2.0, 2.0, 2.0])
        assert float(broad[0]) > float(narrow[0])
