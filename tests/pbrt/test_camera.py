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


def test_realistic_lens_parses_elements(tmp_path):
    lens = tmp_path / "lens.dat"
    lens.write_text(
        "# radius thickness ior aperture\n"
        "35.0  2.0  1.5  24.0\n"
        "0.0   1.0  0.0  10.0   # aperture stop\n"
        "-35.0 2.0  1.5  24.0\n"
    )
    params = _cam_params(f'Camera "realistic" "string lensfile" "{lens.name}"')
    notes = []
    elements = cam.realistic_lens(params, str(tmp_path), notes)
    assert len(elements) == 3
    assert elements[1]["role"] == "aperture" and elements[1]["radius"] == 0.0
    assert elements[0]["ior"] == 1.5
    assert elements[1]["ior"] == 1.0  # 0 in the n column -> air


def test_realistic_lens_loads_as_lenssystem(tmp_path):
    usd_loader = pytest.importorskip("skinny.usd_loader")
    from skinny.pbrt.api import import_pbrt

    (tmp_path / "lens.dat").write_text("35.0 2.0 1.5 24.0\n0.0 1.0 0.0 10.0\n-35.0 2.0 1.5 24.0\n")
    scene = tmp_path / "s.pbrt"
    scene.write_text(
        'Camera "realistic" "string lensfile" "lens.dat"\n'
        "WorldBegin\n"
        'Shape "sphere" "float radius" 1\n'
    )
    stage, report = import_pbrt(str(scene), out=str(tmp_path / "s.usda"))
    sc = usd_loader.load_scene_from_stage(stage)
    assert sc.camera_override is not None
    assert sc.camera_override.lens is not None
    assert len(sc.camera_override.lens.elements) == 3


def test_improper_camera_is_flagged_mirrored(tmp_path):
    usd_loader = pytest.importorskip("skinny.usd_loader")  # noqa: F841
    from pxr import UsdGeom

    from skinny.pbrt.api import import_pbrt

    scene = tmp_path / "s.pbrt"
    scene.write_text(
        "Scale -1 1 1\n"
        "LookAt 0 0 5  0 0 0  0 1 0\n"
        'Camera "perspective" "float fov" 40\n'
        "WorldBegin\n"
        'Shape "sphere" "float radius" 1\n'
    )
    stage, report = import_pbrt(str(scene), out=str(tmp_path / "s.usda"))
    cams = [p.GetCustomDataByKey("pbrt") for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    assert cams and cams[0].get("mirrored") is True
    assert any("mirrored" in (e.detail or "") for e in report.entries)


def test_proper_camera_not_flagged(tmp_path):
    from pxr import UsdGeom

    from skinny.pbrt.api import import_pbrt

    scene = tmp_path / "s.pbrt"
    scene.write_text(
        "LookAt 0 0 5  0 0 0  0 1 0\n"
        'Camera "perspective" "float fov" 40\n'
        "WorldBegin\n"
        'Shape "sphere" "float radius" 1\n'
    )
    stage, _ = import_pbrt(str(scene), out=str(tmp_path / "s.usda"))
    cams = [p.GetCustomDataByKey("pbrt") for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    assert cams and "mirrored" not in cams[0]


def test_depth_of_field_sets_fstop():
    params = _cam_params('Camera "perspective" "float fov" 40 "float lensradius" 0.01 "float focaldistance" 3')
    notes = []
    intr = cam.perspective_to_camera(params, aspect=1.0, notes=notes)
    assert "fstop" in intr and intr["fstop"] > 0
    assert intr["focus_distance"] == 3
