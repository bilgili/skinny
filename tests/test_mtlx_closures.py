"""Unit tests for MaterialX closure helpers: Fresnel, GGX NDF."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near

PI = math.pi
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="session")
def mtlx_harness(load_source):
    source = """
import common;

static const float PI_VAL = 3.14159265358979;

// Schlick Fresnel (matches mx_fresnel_schlick pattern)
float test_schlickFresnel(float cosTheta, float F0)
{
    float c = 1.0 - cosTheta;
    return F0 + (1.0 - F0) * c * c * c * c * c;
}

// F0 from IOR
float test_iorToF0(float ior)
{
    float r = (1.0 - ior) / (1.0 + ior);
    return r * r;
}

// GGX NDF (Trowbridge-Reitz)
float test_ggxNDF(float NdotH, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI_VAL * d * d);
}

// Smith G1 for GGX
float test_smithG1(float NdotV, float roughness)
{
    float a = roughness * roughness;
    float k = a / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
"""
    return load_source("test_mtlx", source)


class TestSchlickFresnel:
    def test_normal_incidence(self, mtlx_harness):
        F0 = 0.04
        result = float(mtlx_harness.test_schlickFresnel(1.0, F0))
        assert_near(result, F0, rel=1e-3)

    def test_grazing_incidence(self, mtlx_harness):
        result = float(mtlx_harness.test_schlickFresnel(0.0, 0.04))
        assert_near(result, 1.0, rel=1e-3)

    def test_half_angle(self, mtlx_harness):
        F0 = 0.04
        cos_t = 0.5
        c = 0.5
        expected = F0 + (1.0 - F0) * c**5
        result = float(mtlx_harness.test_schlickFresnel(cos_t, F0))
        assert_near(result, expected, rel=1e-3)


class TestIORToF0:
    def test_glass(self, mtlx_harness):
        result = float(mtlx_harness.test_iorToF0(1.5))
        expected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        assert_near(result, expected, rel=1e-4)

    def test_vacuum(self, mtlx_harness):
        result = float(mtlx_harness.test_iorToF0(1.0))
        assert abs(result) < 1e-6

    def test_skin(self, mtlx_harness):
        result = float(mtlx_harness.test_iorToF0(1.4))
        expected = ((1.0 - 1.4) / (1.0 + 1.4)) ** 2
        assert_near(result, expected, rel=1e-4)


class TestGGXNDF:
    def test_peak_at_normal(self, mtlx_harness):
        result = float(mtlx_harness.test_ggxNDF(1.0, 0.5))
        assert result > 0.0

    def test_decreases_away_from_normal(self, mtlx_harness):
        peak = float(mtlx_harness.test_ggxNDF(1.0, 0.5))
        off = float(mtlx_harness.test_ggxNDF(0.5, 0.5))
        assert peak > off

    def test_nonnegative(self, mtlx_harness):
        for ndoth in [0.1, 0.3, 0.5, 0.7, 1.0]:
            for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
                val = float(mtlx_harness.test_ggxNDF(ndoth, r))
                assert val >= 0.0


class TestSmithG1:
    def test_at_normal(self, mtlx_harness):
        result = float(mtlx_harness.test_smithG1(1.0, 0.5))
        assert_near(result, 1.0, rel=1e-3)

    def test_range(self, mtlx_harness):
        for ndotv in [0.1, 0.3, 0.5, 0.7, 0.9]:
            val = float(mtlx_harness.test_smithG1(ndotv, 0.5))
            assert 0.0 <= val <= 1.0 + 1e-5
