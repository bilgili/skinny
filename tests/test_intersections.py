"""Unit tests for ray-geometry intersection routines."""

from __future__ import annotations

import math

import pytest

from tests.helpers import assert_near

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="session")
def intersection_source(load_source):
    source = """
import common;

struct SphereHitResult {
    bool hit;
    float t;
};

SphereHitResult test_raySphereIntersect(
    float3 rayOrigin, float3 rayDir, float3 center, float radius)
{
    SphereHitResult r;
    r.hit = false;
    r.t   = -1.0;
    float3 oc = rayOrigin - center;
    float b   = dot(oc, rayDir);
    float c   = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return r;
    float sd = sqrt(disc);
    float t = -b - sd;
    if (t < 0.0) t = -b + sd;
    if (t < 0.0) return r;
    r.hit = true;
    r.t   = t;
    return r;
}

// Möller-Trumbore ray-triangle
struct TriHitResult {
    bool hit;
    float t;
    float u;
    float v;
};

TriHitResult test_rayTriangleIntersect(
    float3 rayOrigin, float3 rayDir,
    float3 v0, float3 v1, float3 v2)
{
    TriHitResult r;
    r.hit = false;
    r.t = -1.0;
    r.u = 0.0;
    r.v = 0.0;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(rayDir, e2);
    float det = dot(e1, pvec);
    if (abs(det) < 1e-8) return r;
    float invDet = 1.0 / det;

    float3 tvec = rayOrigin - v0;
    float u_ = dot(tvec, pvec) * invDet;
    if (u_ < 0.0 || u_ > 1.0) return r;

    float3 qvec = cross(tvec, e1);
    float v_ = dot(rayDir, qvec) * invDet;
    if (v_ < 0.0 || u_ + v_ > 1.0) return r;

    float t = dot(e2, qvec) * invDet;
    if (t < 0.0) return r;

    r.hit = true;
    r.t   = t;
    r.u   = u_;
    r.v   = v_;
    return r;
}
"""
    return load_source("test_intersections", source)


class TestRaySphereIntersection:
    def test_hit_centered_sphere(self, intersection_source):
        result = intersection_source.test_raySphereIntersect(
            [0.0, 0.0, -5.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 1.0
        )
        assert bool(result['hit'])
        assert_near(float(result['t']), 4.0, rel=1e-4)

    def test_miss(self, intersection_source):
        result = intersection_source.test_raySphereIntersect(
            [0.0, 5.0, -5.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 1.0
        )
        assert not bool(result['hit'])

    def test_inside_sphere(self, intersection_source):
        result = intersection_source.test_raySphereIntersect(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 2.0
        )
        assert bool(result['hit'])
        assert_near(float(result['t']), 2.0, rel=1e-4)

    def test_tangent_ray(self, intersection_source):
        result = intersection_source.test_raySphereIntersect(
            [1.0, 0.0, -5.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 1.0
        )
        assert bool(result['hit'])
        assert_near(float(result['t']), 5.0, rel=1e-2)


class TestRayTriangleIntersection:
    v0 = [0.0, 0.0, 0.0]
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]

    def test_hit_center(self, intersection_source):
        result = intersection_source.test_rayTriangleIntersect(
            [0.25, 0.25, -1.0], [0.0, 0.0, 1.0], self.v0, self.v1, self.v2
        )
        assert bool(result['hit'])
        assert_near(float(result['t']), 1.0, rel=1e-4)

    def test_miss_outside(self, intersection_source):
        result = intersection_source.test_rayTriangleIntersect(
            [2.0, 2.0, -1.0], [0.0, 0.0, 1.0], self.v0, self.v1, self.v2
        )
        assert not bool(result['hit'])

    def test_hit_vertex(self, intersection_source):
        result = intersection_source.test_rayTriangleIntersect(
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], self.v0, self.v1, self.v2
        )
        assert bool(result['hit'])
        assert_near(float(result['t']), 1.0, rel=1e-4)

    def test_barycentrics_valid(self, intersection_source):
        result = intersection_source.test_rayTriangleIntersect(
            [0.25, 0.25, -1.0], [0.0, 0.0, 1.0], self.v0, self.v1, self.v2
        )
        u, v = float(result['u']), float(result['v'])
        assert u >= 0.0 and v >= 0.0 and u + v <= 1.0 + 1e-5

    def test_behind_ray_misses(self, intersection_source):
        result = intersection_source.test_rayTriangleIntersect(
            [0.25, 0.25, 1.0], [0.0, 0.0, 1.0], self.v0, self.v1, self.v2
        )
        assert not bool(result['hit'])
