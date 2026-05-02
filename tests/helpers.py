"""Shared assertion helpers for GPU shader tests."""

from __future__ import annotations

import math


def assert_near(actual, expected, *, rel=1e-4, abs_tol=1e-6):
    if isinstance(expected, (list, tuple)):
        for a, e in zip(actual, expected):
            assert math.isclose(a, e, rel_tol=rel, abs_tol=abs_tol), (
                f"expected ~{e}, got {a}"
            )
    else:
        assert math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol), (
            f"expected ~{expected}, got {actual}"
        )


def assert_unit_vector(v, tol=1e-4):
    length = math.sqrt(sum(c * c for c in v))
    assert math.isclose(length, 1.0, abs_tol=tol), f"|v| = {length}, expected 1.0"


def assert_on_hemisphere(direction, normal, tol=-1e-5):
    dot = sum(d * n for d, n in zip(direction, normal))
    assert dot >= tol, f"dot(dir, N) = {dot}, expected >= 0"


def vec3_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec3_length(v):
    return math.sqrt(vec3_dot(v, v))
