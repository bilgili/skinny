"""Unit tests for pure math functions from common.slang."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near, assert_on_hemisphere, assert_unit_vector

PI = math.pi
pytestmark = pytest.mark.gpu


class TestPowerHeuristic:
    def test_equal_pdfs(self, common_harness):
        result = common_harness.test_powerHeuristic(1.0, 1.0)
        assert_near(float(result), 0.5)

    def test_dominant_primary(self, common_harness):
        result = common_harness.test_powerHeuristic(1000.0, 1.0)
        assert float(result) > 0.999

    def test_dominant_secondary(self, common_harness):
        result = common_harness.test_powerHeuristic(1.0, 1000.0)
        assert float(result) < 0.001

    def test_known_value(self, common_harness):
        result = common_harness.test_powerHeuristic(2.0, 1.0)
        assert_near(float(result), 4.0 / 5.0)

    def test_symmetric_swap(self, common_harness):
        a = float(common_harness.test_powerHeuristic(3.0, 7.0))
        b = float(common_harness.test_powerHeuristic(7.0, 3.0))
        assert_near(a + b, 1.0, rel=1e-5)


class TestUniformSpherePdf:
    def test_value(self, common_harness):
        result = common_harness.test_uniformSpherePdf()
        assert_near(float(result), 1.0 / (4.0 * PI))


class TestCosineHemispherePdf:
    def test_at_normal(self, common_harness):
        N = [0.0, 1.0, 0.0]
        result = common_harness.test_cosineHemispherePdf(N, N)
        assert_near(float(result), 1.0 / PI)

    def test_perpendicular(self, common_harness):
        N = [0.0, 1.0, 0.0]
        tangent = [1.0, 0.0, 0.0]
        result = common_harness.test_cosineHemispherePdf(tangent, N)
        assert float(result) < 1e-5

    def test_at_45_degrees(self, common_harness):
        N = [0.0, 1.0, 0.0]
        d = [0.0, math.sqrt(0.5), math.sqrt(0.5)]
        result = common_harness.test_cosineHemispherePdf(d, N)
        expected = math.sqrt(0.5) / PI
        assert_near(float(result), expected, rel=1e-3)


class TestGGXSmithG1:
    def test_at_normal(self, common_harness):
        result = common_harness.test_ggxSmithG1(1.0, 0.5)
        assert_near(float(result), 1.0, rel=1e-3)

    def test_lower_for_grazing(self, common_harness):
        at_normal = float(common_harness.test_ggxSmithG1(0.9, 0.5))
        at_grazing = float(common_harness.test_ggxSmithG1(0.1, 0.5))
        assert at_normal > at_grazing

    def test_rougher_means_less_masking(self, common_harness):
        smooth = float(common_harness.test_ggxSmithG1(0.5, 0.1))
        rough = float(common_harness.test_ggxSmithG1(0.5, 0.9))
        assert smooth > rough

    def test_range_zero_to_one(self, common_harness):
        for ndotv in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for r in [0.1, 0.5, 0.9]:
                val = float(common_harness.test_ggxSmithG1(ndotv, r))
                assert 0.0 <= val <= 1.0 + 1e-5, f"G1({ndotv}, {r}) = {val}"


class TestBuildBasis:
    @pytest.mark.parametrize("normal", [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.577, 0.577, 0.577],
    ])
    def test_orthonormal(self, common_harness, normal):
        result = common_harness.test_buildBasis(normal)
        t = [float(result['t'][i]) for i in range(3)]
        b = [float(result['b'][i]) for i in range(3)]
        n = [float(x) for x in normal]
        n_len = math.sqrt(sum(x * x for x in n))
        n = [x / n_len for x in n]

        assert_unit_vector(t, tol=1e-3)
        assert_unit_vector(b, tol=1e-3)

        tn = abs(sum(ti * ni for ti, ni in zip(t, n)))
        bn = abs(sum(bi * ni for bi, ni in zip(b, n)))
        tb = abs(sum(ti * bi for ti, bi in zip(t, b)))
        assert tn < 1e-3, f"t·n = {tn}"
        assert bn < 1e-3, f"b·n = {bn}"
        assert tb < 1e-3, f"t·b = {tb}"


class TestSampleCosineHemisphere:
    def test_unit_vector(self, common_harness):
        N = [0.0, 1.0, 0.0]
        for i in range(20):
            u = [(i * 0.618 + 0.1) % 1.0, (i * 0.414 + 0.3) % 1.0]
            result = common_harness.test_sampleCosineHemisphere(u, N)
            v = [float(result[j]) for j in range(3)]
            assert_unit_vector(v, tol=1e-3)

    def test_on_hemisphere(self, common_harness):
        N = [0.0, 1.0, 0.0]
        for i in range(20):
            u = [(i * 0.618 + 0.1) % 1.0, (i * 0.414 + 0.3) % 1.0]
            result = common_harness.test_sampleCosineHemisphere(u, N)
            v = [float(result[j]) for j in range(3)]
            assert_on_hemisphere(v, N)

    def test_tilted_normal(self, common_harness):
        N = [0.0, 0.707, 0.707]
        for i in range(10):
            u = [(i * 0.618 + 0.1) % 1.0, (i * 0.414 + 0.3) % 1.0]
            result = common_harness.test_sampleCosineHemisphere(u, N)
            v = [float(result[j]) for j in range(3)]
            assert_unit_vector(v, tol=1e-3)
            assert_on_hemisphere(v, N)


class TestSampleUniformSphere:
    def test_unit_vector(self, common_harness):
        for i in range(20):
            u = [(i * 0.618 + 0.1) % 1.0, (i * 0.414 + 0.3) % 1.0]
            result = common_harness.test_sampleUniformSphere(u)
            v = [float(result[j]) for j in range(3)]
            assert_unit_vector(v, tol=1e-3)

    def test_covers_both_hemispheres(self, common_harness):
        ys = []
        for i in range(20):
            u = [i / 20.0, 0.5]
            result = common_harness.test_sampleUniformSphere(u)
            ys.append(float(result[1]))
        assert min(ys) < -0.5
        assert max(ys) > 0.5


class TestSampleUniformDisk:
    def test_inside_unit_disk(self, common_harness):
        for i in range(20):
            u = [(i * 0.618 + 0.1) % 1.0, (i * 0.414 + 0.3) % 1.0]
            result = common_harness.test_sampleUniformDisk(u)
            x, y = float(result[0]), float(result[1])
            assert x * x + y * y <= 1.0 + 1e-5


class TestPCGHash:
    def test_deterministic(self, common_harness):
        a = int(common_harness.test_pcgHash(42))
        b = int(common_harness.test_pcgHash(42))
        assert a == b

    def test_different_inputs_different_outputs(self, common_harness):
        vals = set()
        for i in range(10):
            vals.add(int(common_harness.test_pcgHash(i)))
        assert len(vals) == 10
