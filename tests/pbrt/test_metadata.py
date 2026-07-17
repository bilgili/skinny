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


# ── SPPM integrator mapping (change photon-mapping-sppm) ────────────

def test_sppm_records_skinny_selection(tmp_path):
    # The SCENE above declares Integrator "sppm" ... "float radius" 0.02.
    stage = _stage(tmp_path)
    pbrt = stage.GetRootLayer().customLayerData["pbrt"]
    assert pbrt["skinny"]["integrator"] == "sppm"
    assert pbrt["skinny"]["radius"] == pytest.approx(0.02)
    # No photonsperiteration in the scene → key absent (renderer defaults to W*H).
    assert "photons" not in pbrt["skinny"]


def test_sppm_selection_helper(tmp_path):
    from skinny.pbrt.api import sppm_selection
    stage = _stage(tmp_path)
    sel = sppm_selection(stage)
    assert sel is not None
    assert sel["integrator"] == "sppm"
    assert sel["radius"] == pytest.approx(0.02)


def test_sppm_reported_mapped(tmp_path):
    from skinny.pbrt.api import import_pbrt
    src = tmp_path / "s.pbrt"
    src.write_text(SCENE)
    _stage_obj, report = import_pbrt(str(src), out=str(tmp_path / "s.usda"))
    # The integrator is reported as mapped (exact), not skipped.
    text = str(report)
    constructs = [e.construct for e in report.entries]
    assert any("integrator:sppm" in c for c in constructs), constructs
    assert "skipped" not in [e.status for e in report.entries
                             if "integrator" in e.construct]
    assert text  # smoke


def test_photonsperiteration_maps_to_photons(tmp_path):
    from skinny.pbrt.api import import_pbrt, sppm_selection
    from pxr import Usd
    src = tmp_path / "p.pbrt"
    src.write_text(
        'Integrator "sppm" "integer photonsperiteration" 4096 "float radius" 0.05\n'
        'Camera "perspective" "float fov" 50\nWorldBegin\n'
        'Shape "sphere" "float radius" 1\n'
    )
    out = tmp_path / "p.usda"
    import_pbrt(str(src), out=str(out))
    sel = sppm_selection(Usd.Stage.Open(str(out)))
    assert sel["integrator"] == "sppm"
    assert sel["radius"] == pytest.approx(0.05)
    assert sel["photons"] == 4096


def test_non_sppm_integrator_has_no_skinny_selection(tmp_path):
    from skinny.pbrt.api import import_pbrt, sppm_selection
    from pxr import Usd
    src = tmp_path / "path.pbrt"
    src.write_text(
        'Integrator "path" "integer maxdepth" 8\n'
        'Camera "perspective" "float fov" 50\nWorldBegin\n'
        'Shape "sphere" "float radius" 1\n'
    )
    out = tmp_path / "path.usda"
    import_pbrt(str(src), out=str(out))
    stage = Usd.Stage.Open(str(out))
    assert sppm_selection(stage) is None
    assert "skinny" not in stage.GetRootLayer().customLayerData["pbrt"]


# ── MLT integrator mapping (change mlt-integrator) ──────────────────

def _mlt_stage(tmp_path, params=""):
    from skinny.pbrt.api import import_pbrt
    from pxr import Usd
    src = tmp_path / "m.pbrt"
    src.write_text(
        f'Integrator "mlt"{params}\n'
        'Camera "perspective" "float fov" 50\nWorldBegin\n'
        'Shape "sphere" "float radius" 1\n'
    )
    out = tmp_path / "m.usda"
    _stage, report = import_pbrt(str(src), out=str(out))
    return Usd.Stage.Open(str(out)), report


def test_mlt_records_selection_with_pbrt_defaults(tmp_path):
    from skinny.pbrt.api import integrator_selection, sppm_selection
    stage, _report = _mlt_stage(tmp_path)
    sel = integrator_selection(stage)
    assert sel is not None and sel["integrator"] == "mlt"
    # Self-contained selection: pbrt defaults filled in.
    assert sel["maxdepth"] == 5
    assert sel["mutationsperpixel"] == 100
    assert sel["largestepprobability"] == pytest.approx(0.3)
    assert sel["sigma"] == pytest.approx(0.01)
    assert sel["chains"] == 1000
    assert sel["bootstrapsamples"] == 100000
    # The SPPM-specific reader keeps its documented contract.
    assert sppm_selection(stage) is None


def test_mlt_records_authored_params(tmp_path):
    from skinny.pbrt.api import integrator_selection
    stage, _report = _mlt_stage(
        tmp_path,
        ' "integer mutationsperpixel" 200 "float largestepprobability" 0.5'
        ' "integer maxdepth" 7 "float sigma" 0.02'
        ' "integer chains" 4096 "integer bootstrapsamples" 50000')
    sel = integrator_selection(stage)
    assert sel["mutationsperpixel"] == 200
    assert sel["largestepprobability"] == pytest.approx(0.5)
    assert sel["maxdepth"] == 7
    assert sel["sigma"] == pytest.approx(0.02)
    assert sel["chains"] == 4096
    assert sel["bootstrapsamples"] == 50000


def test_mlt_reported_mapped(tmp_path):
    _stage, report = _mlt_stage(tmp_path)
    constructs = [e.construct for e in report.entries]
    assert any("integrator:mlt" in c for c in constructs), constructs
    assert "skipped" not in [e.status for e in report.entries
                             if "integrator" in e.construct]


def test_sppm_selection_still_returns_sppm(tmp_path):
    # Regression: the generalized reader must not change the SPPM contract.
    from skinny.pbrt.api import integrator_selection, sppm_selection
    stage = _stage(tmp_path)
    assert sppm_selection(stage)["integrator"] == "sppm"
    assert integrator_selection(stage)["integrator"] == "sppm"
