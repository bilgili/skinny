"""Tests for the pbrt-metadata carry (customData / customLayerData round-trip)."""

from __future__ import annotations

import pytest

from skinny.pbrt.api import import_pbrt

pytest.importorskip("pxr")

SCENE = """
Integrator "sppm" "integer maxdepth" 5 "float radius" 0.02
Sampler "halton" "integer pixelsamples" 64
Film "rgb" "float iso" 600 "integer xresolution" 100 "integer yresolution" 100
Camera "perspective" "float fov" 65 "float shutterclose" 1
WorldBegin
Material "conductor" "spectrum eta" "metal-Au-eta" "float roughness" 0.25 "bool remaproughness" "false"
Shape "sphere" "float radius" 1
AttributeBegin
  AreaLightSource "diffuse" "blackbody L" 4000 "float scale" 10
  Shape "trianglemesh" "point3 P" [0 0 0  1 0 0  0 1 0] "integer indices" [0 1 2]
AttributeEnd
LightSource "distant" "rgb L" [1 1 1] "point3 from" [0 0 0] "point3 to" [0 0 -1]
"""


def _stage(tmp_path):
    from pxr import Usd

    src = tmp_path / "s.pbrt"
    src.write_text(SCENE)
    out = tmp_path / "s.usda"
    import_pbrt(str(src), out=str(out))
    return Usd.Stage.Open(str(out))


def test_stage_metadata_roundtrips(tmp_path):
    stage = _stage(tmp_path)
    pbrt = stage.GetRootLayer().customLayerData["pbrt"]
    assert pbrt["integrator"]["type"] == "sppm"
    assert pbrt["integrator"]["params"]["maxdepth"] == 5
    assert pbrt["integrator"]["params"]["radius"] == pytest.approx(0.02)
    assert pbrt["sampler"]["type"] == "halton"
    assert pbrt["film"]["params"]["iso"] == pytest.approx(600)
    assert pbrt["colorSpace"] == "srgb"


def test_material_metadata_preserves_spectrum_and_type(tmp_path):
    stage = _stage(tmp_path)
    mats = [
        p.GetCustomDataByKey("pbrt")
        for p in stage.Traverse()
        if p.GetTypeName() == "Material"
    ]
    conductor = [m for m in mats if m and m.get("type") == "conductor"]
    assert conductor, "conductor material metadata missing"
    md = conductor[0]
    assert md["params"]["eta"] == "metal-Au-eta"
    assert md["paramTypes"]["eta"] == "spectrum"
    assert md["params"]["roughness"] == pytest.approx(0.25)
    assert md["params"]["remaproughness"] is False


def test_light_metadata_preserves_blackbody_and_distant(tmp_path):
    stage = _stage(tmp_path)
    from pxr import UsdLux

    lights = [
        p.GetCustomDataByKey("pbrt")
        for p in stage.Traverse()
        if p.IsA(UsdLux.DistantLight)
    ]
    distant = [m for m in lights if m and m.get("type") == "distant"]
    assert distant
    assert distant[0]["paramTypes"]["L"] == "rgb"

    # area-light emission spec is carried on the emissive mesh prim
    area = [
        p.GetCustomDataByKey("pbrtAreaLight")
        for p in stage.Traverse()
        if p.GetCustomDataByKey("pbrtAreaLight")
    ]
    assert area and area[0]["paramTypes"]["L"] == "blackbody"
    assert area[0]["params"]["L"] == pytest.approx(4000)


def test_camera_metadata(tmp_path):
    stage = _stage(tmp_path)
    from pxr import UsdGeom

    cams = [
        p.GetCustomDataByKey("pbrt")
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Camera)
    ]
    assert cams and cams[0]["type"] == "perspective"
    assert cams[0]["params"]["fov"] == pytest.approx(65)
