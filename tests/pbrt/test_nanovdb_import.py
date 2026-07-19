"""Tests for nanovdb heterogeneous-medium import (tasks 3.1-3.4, change
nanovdb-volume-rendering): `MakeNamedMedium "nanovdb"` -> UsdVol.Volume,
`Material "interface"` -> null boundary, disney-cloud/bunny-cloud end-to-end.
"""

from __future__ import annotations

import os

import pytest

from skinny.pbrt.api import import_pbrt
from skinny.pbrt.media import heterogeneous_overrides, is_supported_heterogeneous

pxr = pytest.importorskip("pxr")
from pxr import UsdGeom, UsdVol  # noqa: E402

DISNEY_CLOUD = os.path.expanduser(
    "~/projects/pbrt-v4-scenes/disney-cloud/disney-cloud.pbrt"
)
BUNNY_CLOUD = os.path.expanduser(
    "~/projects/pbrt-v4-scenes/bunny-cloud/bunny-cloud.pbrt"
)

INTERFACE_MEDIUM_SCENE = """
WorldBegin
MakeNamedMedium "cloud" "string type" "nanovdb"
  "string filename" "wdas_cloud_quarter.nvdb"
  "spectrum sigma_a" [200 0 900 0]
  "spectrum sigma_s" [200 1 900 1]
  "float g" [0.877]
  "float scale" [4]
AttributeBegin
  Material "interface"
  MediumInterface "cloud" ""
  Shape "sphere" "float radius" [1.44224957031]
AttributeEnd
"""

RGBGRID_MEDIUM_SCENE = """
WorldBegin
MakeNamedMedium "cloud" "string type" "rgbgrid" "rgb sigma_s" [1 1 1]
AttributeBegin
  MediumInterface "cloud" ""
  Material "interface"
  Shape "sphere" "float radius" 1
AttributeEnd
"""


def _find_volume_prim(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdVol.Volume):
            return prim
    return None


# --------------------------------------------------------------------------- #
# 3.1 media.py: heterogeneous_overrides / is_supported_heterogeneous
# --------------------------------------------------------------------------- #
def test_is_supported_heterogeneous_true_for_nanovdb_with_filename():
    from skinny.pbrt.state import PbrtMedium
    from skinny.pbrt.parser import ParamSet, Param

    params = ParamSet({"filename": Param("string", "filename", ("cloud.nvdb",))})
    medium = PbrtMedium("cloud", "nanovdb", params)
    assert is_supported_heterogeneous(medium)


def test_is_supported_heterogeneous_false_without_filename():
    from skinny.pbrt.state import PbrtMedium
    from skinny.pbrt.parser import ParamSet

    medium = PbrtMedium("cloud", "nanovdb", ParamSet({}))
    assert not is_supported_heterogeneous(medium)


def test_heterogeneous_overrides_resolves_absolute_grid_path(tmp_path):
    from skinny.pbrt.state import PbrtMedium
    from skinny.pbrt.parser import ParamSet, Param

    params = ParamSet({
        "filename": Param("string", "filename", ("cloud.nvdb",)),
        "sigma_a": Param("rgb", "sigma_a", (0.0, 0.0, 0.0)),
        "sigma_s": Param("rgb", "sigma_s", (1.0, 1.0, 1.0)),
        "g": Param("float", "g", (0.877,)),
        "scale": Param("float", "scale", (4.0,)),
    })
    medium = PbrtMedium("cloud", "nanovdb", params)
    overrides = heterogeneous_overrides(medium, str(tmp_path))
    assert overrides["pbrt_medium"] == "cloud"
    assert overrides["volume_sigma_a"] == pytest.approx([0.0, 0.0, 0.0])
    assert overrides["volume_sigma_s"] == pytest.approx([4.0, 4.0, 4.0])
    assert overrides["volume_g"] == pytest.approx(0.877)
    assert overrides["volume_grid_field"] == "density"
    assert os.path.isabs(overrides["volume_grid_asset"])
    assert overrides["volume_grid_asset"] == os.path.join(str(tmp_path), "cloud.nvdb").replace(
        os.sep, "/"
    )


# --------------------------------------------------------------------------- #
# 3.2 Material "interface" + medium routing
# --------------------------------------------------------------------------- #
def test_interface_material_has_no_diffuse_dielectric_fallback(tmp_path):
    src = tmp_path / "iface.pbrt"
    src.write_text(INTERFACE_MEDIUM_SCENE)
    stage, report = import_pbrt(str(src))

    mat_entries = [
        e for e in report.entries if e.construct.startswith("material:interface")
    ]
    assert mat_entries, "expected a material:interface report entry"
    # Never falls back to the generic "unknown material ... diffuse grey" path.
    assert not any("unknown material" in (e.detail or "") for e in mat_entries)

    mat_prim = stage.GetPrimAtPath("/World/shape_0_mat")
    assert mat_prim.IsValid()
    cd = dict(mat_prim.GetCustomDataByKey("skinnyOverrides") or {})
    assert cd.get("volume_interface") is True
    # Constant pbrt SPD -> achromatic RGB (change constant-spectrum-achromatic-rgb,
    # c1df5f0): the scene authors `"spectrum sigma_s" [200 1 900 1]`, a CONSTANT
    # spectrum, which short-circuits to [v, v, v]. Projecting it through XYZ would
    # tint it (equal-energy whitepoint != sRGB D65) — that tint is what the old
    # expectation below encoded, and a 1e-6 perturbation still reproduces it.
    assert list(cd.get("volume_sigma_s")) == pytest.approx(
        [4.0, 4.0, 4.0]
    )

    surface = stage.GetPrimAtPath("/World/shape_0_mat/Surface")
    assert surface.IsValid()
    diffuse = surface.GetAttribute("inputs:diffuseColor").Get()
    assert tuple(diffuse) == pytest.approx((0.0, 0.0, 0.0))
    # dielectric fallback would author ior + opacity==0; the interface material
    # must not carry the refraction-gate opacity==0.
    opacity = surface.GetAttribute("inputs:opacity").Get()
    assert opacity == pytest.approx(1.0)
    assert surface.GetAttribute("inputs:ior").Get() is None


def test_resolve_medium_returns_heterogeneous_overrides_not_skip(tmp_path):
    src = tmp_path / "iface.pbrt"
    src.write_text(INTERFACE_MEDIUM_SCENE)
    _stage, report = import_pbrt(str(src))
    assert not any(
        "heterogeneous media unsupported" in (e.detail or "") for e in report.entries
    )
    assert any(e.construct.startswith("medium:nanovdb") for e in report.entries)


def test_rgbgrid_medium_still_skips(tmp_path):
    src = tmp_path / "rgbgrid.pbrt"
    src.write_text(RGBGRID_MEDIUM_SCENE)
    _stage, report = import_pbrt(str(src))
    assert report.count("skipped") >= 1
    assert any(
        "heterogeneous" in (e.detail or "") and "rgbgrid" in e.construct
        for e in report.entries
    )


# --------------------------------------------------------------------------- #
# 3.3 emit.py UsdVol emission
# --------------------------------------------------------------------------- #
def test_interface_scene_emits_volume_prim(tmp_path):
    src = tmp_path / "iface.pbrt"
    src.write_text(INTERFACE_MEDIUM_SCENE)
    stage, _report = import_pbrt(str(src))

    vol_prim = _find_volume_prim(stage)
    assert vol_prim is not None
    assert vol_prim.GetPath() == "/World/volume_cloud"

    field_prim = stage.GetPrimAtPath("/World/volume_cloud/density")
    assert field_prim.IsA(UsdVol.OpenVDBAsset)
    field = UsdVol.OpenVDBAsset(field_prim)
    assert field.GetFilePathAttr().Get().path.endswith("wdas_cloud_quarter.nvdb")
    assert field.GetFieldNameAttr().Get() == "density"

    volume = UsdVol.Volume(vol_prim)
    rel = volume.GetPrim().GetRelationship("field:density")
    assert rel.IsValid()
    targets = rel.GetTargets()
    assert targets and targets[0] == field_prim.GetPath()

    # Sphere declared at identity CTM after the medium: the medium itself was
    # declared before any Translate/Scale, so its own xform stays identity.
    xf = UsdGeom.Xformable(vol_prim).GetLocalTransformation()
    for i in range(4):
        for j in range(4):
            assert xf[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)

    cd = dict(vol_prim.GetCustomDataByKey("skinnyOverrides") or {})
    assert cd.get("volume_g") == pytest.approx(0.877)
    assert cd.get("volume_grid_field") == "density"


# --------------------------------------------------------------------------- #
# 3.4 target scene end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(DISNEY_CLOUD), reason="pbrt-v4-scenes corpus not available")
def test_disney_cloud_imports_volume_and_medium():
    stage, report = import_pbrt(DISNEY_CLOUD)

    vol_prim = stage.GetPrimAtPath("/World/volume_cloud")
    assert vol_prim.IsValid() and vol_prim.IsA(UsdVol.Volume)
    field_prim = stage.GetPrimAtPath("/World/volume_cloud/density")
    field = UsdVol.OpenVDBAsset(field_prim)
    assert field.GetFilePathAttr().Get().path.endswith("wdas_cloud_quarter.nvdb")
    assert field.GetFieldNameAttr().Get() == "density"

    cd = dict(vol_prim.GetCustomDataByKey("skinnyOverrides") or {})
    assert cd.get("volume_sigma_a") is not None
    assert list(cd["volume_sigma_a"]) == pytest.approx([0.0, 0.0, 0.0])
    # sigma_s = 1 (spectrally reduced) x scale=4
    assert list(cd["volume_sigma_s"]) == pytest.approx(
        [4.0, 4.0, 4.0]
    )
    assert cd.get("volume_g") == pytest.approx(0.877)

    assert not any(
        "heterogeneous media unsupported" in (e.detail or "") for e in report.entries
    )

    # camera, distant light, and the (approximated) infinite light still present.
    assert stage.GetPrimAtPath("/World/Camera").IsValid()
    assert any(e.construct.startswith("light:distant") for e in report.entries)
    assert any(e.construct.startswith("light:infinite") for e in report.entries)


@pytest.mark.skipif(not os.path.exists(BUNNY_CLOUD), reason="pbrt-v4-scenes corpus not available")
def test_bunny_cloud_interface_sphere_and_volume_xform():
    stage, report = import_pbrt(BUNNY_CLOUD)

    mat_prim = stage.GetPrimAtPath("/World/shape_0_mat")
    assert mat_prim.IsValid()
    cd = dict(mat_prim.GetCustomDataByKey("skinnyOverrides") or {})
    assert cd.get("volume_interface") is True
    assert list(cd["volume_sigma_s"]) == pytest.approx(
        [10.0, 10.0, 10.0]
    )
    assert list(cd["volume_sigma_a"]) == pytest.approx(
        [0.5, 0.5, 0.5]
    )

    surface = stage.GetPrimAtPath("/World/shape_0_mat/Surface")
    diffuse = surface.GetAttribute("inputs:diffuseColor").Get()
    assert tuple(diffuse) == pytest.approx((0.0, 0.0, 0.0))

    # Medium CTM = Rotate 180 z . Rotate 90 x (declared inside that block, the
    # sphere's own AttributeBegin has no transform) -> B @ M @ B in USD space.
    from skinny.pbrt import transform as T
    import numpy as np

    expected = T.to_skinny(T.identity() @ T.rotate(180, 0, 0, 1) @ T.rotate(90, 1, 0, 0))
    vol_prim = stage.GetPrimAtPath("/World/volume_foo")
    assert vol_prim.IsValid()
    xf = UsdGeom.Xformable(vol_prim).GetLocalTransformation()
    got = np.array([[xf[i][j] for j in range(4)] for i in range(4)])
    assert got == pytest.approx(expected.T, abs=1e-6)

    assert not any(
        "heterogeneous media unsupported" in (e.detail or "") for e in report.entries
    )
    # bunny film sensor (nikon_d850) is a recorded approx, not a hard failure.
    # Task 3.5 made `Shape "disk"` import exact, so the scene is now skip-free.
    assert report.count("skipped") == 0
