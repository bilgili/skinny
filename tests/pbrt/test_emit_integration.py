"""End-to-end: pbrt scene -> USD -> skinny usd_loader (tasks 4.1, 5, 6)."""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt.api import import_pbrt

usd_loader = pytest.importorskip("skinny.usd_loader")


SCENE = """
LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" 40
Film "rgb" "integer xresolution" 64 "integer yresolution" 64
Integrator "path" "integer maxdepth" 5
WorldBegin
LightSource "distant" "blackbody L" [6500] "point3 from" [0 0 0] "point3 to" [0 0 -1]
AttributeBegin
  Material "diffuse" "rgb reflectance" [0.6 0.6 0.6]
  Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0  -1 1 0 ] "integer indices" [0 1 2 0 2 3]
AttributeEnd
AttributeBegin
  Material "conductor" "float roughness" 0.1
  Translate 0 0 -1
  Shape "sphere" "float radius" 0.5
AttributeEnd
"""


def _import(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    stage, report = import_pbrt(str(src))
    return stage, report


def test_imports_and_loads_via_usd_loader(tmp_path):
    stage, report = _import(tmp_path)
    scene = usd_loader.load_scene_from_stage(stage)
    # two shapes -> two instances
    assert len(scene.instances) == 2
    # one distant light
    assert len(scene.lights_dir) == 1
    # camera authored
    assert scene.camera_override is not None
    assert report.count("skipped") == 0


def test_camera_position_and_forward(tmp_path):
    stage, _ = _import(tmp_path)
    scene = usd_loader.load_scene_from_stage(stage)
    cam = scene.camera_override
    # eye (0,0,5) mirrored by the z-flip -> (0,0,-5); looking back toward +z
    assert np.allclose(cam.position, [0, 0, -5], atol=1e-4)
    assert np.allclose(cam.forward / np.linalg.norm(cam.forward), [0, 0, 1], atol=1e-4)


def test_exported_usda_file_loads(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    out = tmp_path / "scene.usda"
    import_pbrt(str(src), out=str(out))
    assert out.exists()
    from pxr import Usd

    scene = usd_loader.load_scene_from_stage(Usd.Stage.Open(str(out)))
    assert len(scene.instances) == 2
