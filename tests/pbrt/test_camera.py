"""Tests for pbrt perspective camera mapping (task 5.1)."""

from __future__ import annotations

import pytest

from skinny.pbrt import camera as cam
from skinny.pbrt.parser import parse_directives
from skinny.pbrt.tokenizer import tokenize


def _cam_params(text):
    (d,) = parse_directives(tokenize(text))
    return d.params


def test_square_film_fov_roundtrips():
    params = _cam_params('Camera "perspective" "float fov" 40')
    intr = cam.perspective_to_camera(params, aspect=1.0, notes=[])
    vfov = cam.vertical_fov_from_intrinsics(intr["focal_length_mm"], intr["vertical_aperture_mm"])
    assert vfov == pytest.approx(40.0, abs=1e-6)


def test_landscape_fov_is_vertical():
    # 16:9, fov addresses the shorter (vertical) axis -> vfov == fov
    params = _cam_params('Camera "perspective" "float fov" 30')
    intr = cam.perspective_to_camera(params, aspect=16.0 / 9.0, notes=[])
    vfov = cam.vertical_fov_from_intrinsics(intr["focal_length_mm"], intr["vertical_aperture_mm"])
    assert vfov == pytest.approx(30.0, abs=1e-6)
    assert intr["horizontal_aperture_mm"] == pytest.approx(24.0 * 16.0 / 9.0)


def test_portrait_fov_is_horizontal():
    # portrait: fov addresses the horizontal axis; vertical fov must be larger
    params = _cam_params('Camera "perspective" "float fov" 30')
    intr = cam.perspective_to_camera(params, aspect=9.0 / 16.0, notes=[])
    vfov = cam.vertical_fov_from_intrinsics(intr["focal_length_mm"], intr["vertical_aperture_mm"])
    assert vfov > 30.0


def test_depth_of_field_sets_fstop():
    params = _cam_params('Camera "perspective" "float fov" 40 "float lensradius" 0.01 "float focaldistance" 3')
    notes = []
    intr = cam.perspective_to_camera(params, aspect=1.0, notes=notes)
    assert "fstop" in intr and intr["fstop"] > 0
    assert intr["focus_distance"] == 3
