"""Unit tests for environment mapping: equirectangular UV conversion."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near

PI = math.pi
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="session")
def env_harness(load_shader):
    return load_shader("test_environment_harness.slang")


class TestDirectionToEquirectUV:
    def test_positive_y_zenith(self, env_harness):
        result = env_harness.test_directionToEquirectUV([0.0, 1.0, 0.0])
        v = float(result[1])
        assert_near(v, 0.0, abs_tol=1e-4)

    def test_negative_y_nadir(self, env_harness):
        result = env_harness.test_directionToEquirectUV([0.0, -1.0, 0.0])
        v = float(result[1])
        assert_near(v, 1.0, abs_tol=1e-4)

    def test_positive_z_center(self, env_harness):
        result = env_harness.test_directionToEquirectUV([0.0, 0.0, 1.0])
        u = float(result[0])
        assert_near(u, 0.5, abs_tol=1e-4)

    def test_horizon_at_half_v(self, env_harness):
        result = env_harness.test_directionToEquirectUV([1.0, 0.0, 0.0])
        v = float(result[1])
        assert_near(v, 0.5, abs_tol=1e-4)

    def test_negative_z(self, env_harness):
        result = env_harness.test_directionToEquirectUV([0.0, 0.0, -1.0])
        u = float(result[0])
        assert u < 0.01 or u > 0.99

    def test_uv_in_range(self, env_harness):
        directions = [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [0.577, 0.577, 0.577],
        ]
        for d in directions:
            result = env_harness.test_directionToEquirectUV(d)
            u, v = float(result[0]), float(result[1])
            assert -0.01 <= u <= 1.01, f"u={u} out of range for dir={d}"
            assert -0.01 <= v <= 1.01, f"v={v} out of range for dir={d}"
