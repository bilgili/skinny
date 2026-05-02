"""Unit tests for MIS (multiple importance sampling) weight functions."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near

PI = math.pi
pytestmark = pytest.mark.gpu


class TestMISWeightsLambertUniform:
    """Test misPrimaryWeight + misCompanionWeight using Lambert + UniformSphere pair."""

    N = [0.0, 1.0, 0.0]

    def test_weights_sum_to_one(self, sampler_harness):
        directions = [
            [0.0, 1.0, 0.0],
            [0.0, 0.707, 0.707],
            [0.577, 0.577, 0.577],
        ]
        for L in directions:
            result = sampler_harness.test_mis_lambert_uniform(self.N, L)
            p = float(result['primary'])
            c = float(result['companion'])
            assert_near(p + c, 1.0, rel=1e-4)

    def test_lambert_dominant_near_normal(self, sampler_harness):
        L = [0.0, 1.0, 0.0]
        result = sampler_harness.test_mis_lambert_uniform(self.N, L)
        assert float(result['primary']) > 0.5


class TestMISWeightsGGXUniform:
    """Test MIS weights using GGX + UniformSphere pair."""

    N = [0.0, 1.0, 0.0]
    V = [0.0, 1.0, 0.0]

    def test_weights_sum_to_one(self, sampler_harness):
        L = [0.0, 0.707, 0.707]
        result = sampler_harness.test_mis_ggx_uniform(self.N, self.V, 0.5, L)
        p = float(result['primary'])
        c = float(result['companion'])
        assert_near(p + c, 1.0, rel=1e-4)

    def test_smooth_ggx_dominates_near_reflection(self, sampler_harness):
        L = [0.0, 1.0, 0.0]
        result = sampler_harness.test_mis_ggx_uniform(self.N, self.V, 0.1, L)
        assert float(result['primary']) > 0.9


class TestPowerHeuristicKnownValues:
    """Direct power heuristic known-value tests via common harness."""

    def test_two_vs_one(self, common_harness):
        result = float(common_harness.test_powerHeuristic(2.0, 1.0))
        assert_near(result, 4.0 / 5.0)

    def test_one_vs_two(self, common_harness):
        result = float(common_harness.test_powerHeuristic(1.0, 2.0))
        assert_near(result, 1.0 / 5.0)

    def test_sum_symmetry(self, common_harness):
        for a, b in [(1.0, 3.0), (5.0, 2.0), (0.1, 10.0)]:
            w1 = float(common_harness.test_powerHeuristic(a, b))
            w2 = float(common_harness.test_powerHeuristic(b, a))
            assert_near(w1 + w2, 1.0, rel=1e-5)
