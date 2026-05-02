"""Unit tests for ILight implementations: sphere, directional, emissive triangle."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near, assert_unit_vector, vec3_dot, vec3_length

PI = math.pi
pytestmark = pytest.mark.gpu


def _quasi_samples(n, seed_a=0.1, seed_b=0.3):
    return [((i * 0.618034 + seed_a) % 1.0, (i * 0.414214 + seed_b) % 1.0) for i in range(n)]


class TestSphereLightImpl:
    center = [0.0, 2.0, 0.0]
    radius = 0.5
    rad = [10.0, 10.0, 10.0]
    shading_pos = [0.0, 0.0, 0.0]

    def test_sample_on_sphere(self, light_harness):
        for u in _quasi_samples(20):
            s = light_harness.test_sphere_light_sample(
                self.center, self.radius, self.rad, self.shading_pos, list(u)
            )
            p = [float(s['point'][i]) for i in range(3)]
            dist = vec3_length([p[i] - self.center[i] for i in range(3)])
            assert_near(dist, self.radius, rel=1e-3)

    def test_pdf_area_correct(self, light_harness):
        s = light_harness.test_sphere_light_sample(
            self.center, self.radius, self.rad, self.shading_pos, [0.5, 0.5]
        )
        expected = 1.0 / (4.0 * PI * self.radius * self.radius)
        assert_near(float(s['pdfArea']), expected, rel=1e-4)

    def test_always_valid(self, light_harness):
        for u in _quasi_samples(10):
            s = light_harness.test_sphere_light_sample(
                self.center, self.radius, self.rad, self.shading_pos, list(u)
            )
            assert bool(s['valid'])

    def test_radiance_passthrough(self, light_harness):
        s = light_harness.test_sphere_light_sample(
            self.center, self.radius, self.rad, self.shading_pos, [0.5, 0.5]
        )
        for i in range(3):
            assert_near(float(s['radiance'][i]), self.rad[i])

    def test_normal_is_unit(self, light_harness):
        for u in _quasi_samples(10):
            s = light_harness.test_sphere_light_sample(
                self.center, self.radius, self.rad, self.shading_pos, list(u)
            )
            n = [float(s['normal'][i]) for i in range(3)]
            assert_unit_vector(n)

    def test_pdf_solid_angle_miss_is_zero(self, light_harness):
        direction = [1.0, 0.0, 0.0]
        pdf = float(light_harness.test_sphere_light_pdf(
            self.center, self.radius, self.rad, self.shading_pos, direction
        ))
        assert pdf == 0.0 or pdf < 1e-6


class TestDirectionalLightImpl:
    direction = [0.0, 1.0, 0.0]
    rad = [5.0, 5.0, 5.0]
    shading_pos = [0.0, 0.0, 0.0]

    def test_pdf_area_is_zero(self, light_harness):
        s = light_harness.test_directional_light_sample(
            self.direction, self.rad, self.shading_pos, [0.5, 0.5]
        )
        assert float(s['pdfArea']) == 0.0

    def test_pdf_solid_angle_is_zero(self, light_harness):
        pdf = float(light_harness.test_directional_light_pdf(
            self.direction, self.rad, self.shading_pos, [0.0, 1.0, 0.0]
        ))
        assert pdf == 0.0

    def test_point_far_away(self, light_harness):
        s = light_harness.test_directional_light_sample(
            self.direction, self.rad, self.shading_pos, [0.5, 0.5]
        )
        p = [float(s['point'][i]) for i in range(3)]
        dist = vec3_length(p)
        assert dist > 1e5

    def test_always_valid(self, light_harness):
        s = light_harness.test_directional_light_sample(
            self.direction, self.rad, self.shading_pos, [0.5, 0.5]
        )
        assert bool(s['valid'])


class TestEmissiveTriangleLightImpl:
    v0 = [0.0, 0.0, 0.0]
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    emission = [10.0, 8.0, 6.0]
    area = 0.5
    selection_pdf = 1.0
    shading_pos = [0.0, 0.0, -2.0]

    def test_sample_on_triangle(self, light_harness):
        for u_val in _quasi_samples(20):
            s = light_harness.test_emissive_tri_sample(
                self.v0, self.v1, self.v2, self.emission, self.area,
                self.selection_pdf, self.shading_pos, list(u_val)
            )
            p = [float(s['point'][i]) for i in range(3)]
            assert p[2] < 1e-5 and p[2] > -1e-5, "Point should be on z=0 plane"
            assert p[0] >= -1e-5 and p[1] >= -1e-5, "Barycentric coords non-negative"
            assert p[0] + p[1] <= 1.0 + 1e-4, "Inside triangle"

    def test_pdf_includes_selection(self, light_harness):
        s = light_harness.test_emissive_tri_sample(
            self.v0, self.v1, self.v2, self.emission, self.area,
            0.25, self.shading_pos, [0.5, 0.5]
        )
        expected = 0.25 / max(self.area, 1e-12)
        assert_near(float(s['pdfArea']), expected, rel=1e-4)

    def test_normal_perpendicular_to_triangle(self, light_harness):
        s = light_harness.test_emissive_tri_sample(
            self.v0, self.v1, self.v2, self.emission, self.area,
            self.selection_pdf, self.shading_pos, [0.5, 0.5]
        )
        n = [float(s['normal'][i]) for i in range(3)]
        assert_unit_vector(n)
        assert abs(n[2]) > 0.99, "Normal should be along z for triangle in xy-plane"

    def test_always_valid(self, light_harness):
        for u_val in _quasi_samples(5):
            s = light_harness.test_emissive_tri_sample(
                self.v0, self.v1, self.v2, self.emission, self.area,
                self.selection_pdf, self.shading_pos, list(u_val)
            )
            assert bool(s['valid'])
