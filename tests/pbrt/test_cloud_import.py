"""Tests for procedural `cloud` medium import + `Material ""` null boundary
(pbrt-cloud-procedural-medium, tasks 2.1-2.3): `MakeNamedMedium "cloud"` ->
skinnyOverrides with the analytic density params (no grid, no Volume prim),
empty-string material + MediumInterface -> interface null boundary, and the
recorded skips (rgbgrid, non-default cloud bounds) unchanged.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from skinny.pbrt.api import import_pbrt
from skinny.pbrt.media import cloud_overrides, is_supported_cloud
from skinny.pbrt.parser import parse_file
from skinny.pbrt.state import build_scene

pxr = pytest.importorskip("pxr")
from pxr import UsdVol  # noqa: E402

CLOUDS_PBRT = os.path.expanduser("~/projects/pbrt-v4-scenes/clouds/clouds.pbrt")

# clouds.pbrt distilled: identity medium CTM, empty-string material, the
# sphere translated to enclose the [0,1]^3 medium cube. Constant spectra
# sigma_s=10 / sigma_a=0.01 like the real scene.
CLOUD_SCENE = """
WorldBegin
MakeNamedMedium "c" "string type" "cloud"
   "spectrum sigma_s" [200 10 900 10] "spectrum sigma_a" [200 .01 900 .01]
   "float density" 2
MediumInterface "c" ""
Material ""
Translate .5 .5 .5
Shape "sphere" "float radius" 1
"""

CLOUD_CUSTOM_BOUNDS = """
WorldBegin
MakeNamedMedium "c" "string type" "cloud" "point3 p0" [-1 -1 -1] "point3 p1" [2 2 2]
MediumInterface "c" ""
Material ""
Shape "sphere" "float radius" 1
"""

RGBGRID_SCENE = """
WorldBegin
MakeNamedMedium "g" "string type" "rgbgrid" "rgb sigma_s" [1 1 1]
MediumInterface "g" ""
Material ""
Shape "sphere" "float radius" 1
"""

EMPTY_MATERIAL_NO_MEDIUM = """
WorldBegin
Material ""
Shape "sphere" "float radius" 1
"""


def _sphere_overrides(stage) -> dict | None:
    for prim in stage.Traverse():
        cd = prim.GetCustomDataByKey("skinnyOverrides")
        if cd and "pbrt_medium" in cd:
            return dict(cd)
    return None


def _import(tmp_path, text, name="s.pbrt"):
    src = tmp_path / name
    src.write_text(text)
    return import_pbrt(str(src))


# --------------------------------------------------------------------------- #
# 2.1 media.py: cloud_overrides / is_supported_cloud
# --------------------------------------------------------------------------- #

def _parse_medium(tmp_path, text):
    src = tmp_path / "m.pbrt"
    src.write_text(text)
    scene = build_scene(parse_file(str(src)))
    return next(iter(scene.media.values()))


def test_is_supported_cloud_default_bounds(tmp_path):
    medium = _parse_medium(tmp_path, CLOUD_SCENE)
    assert is_supported_cloud(medium)


def test_is_supported_cloud_rejects_custom_bounds(tmp_path):
    medium = _parse_medium(tmp_path, CLOUD_CUSTOM_BOUNDS)
    assert not is_supported_cloud(medium)


def test_cloud_overrides_params_and_defaults(tmp_path):
    medium = _parse_medium(tmp_path, CLOUD_SCENE)
    ov = cloud_overrides(medium)
    assert ov["volume_cloud"] is True
    assert ov["cloud_density"] == pytest.approx(2.0)
    assert ov["cloud_wispiness"] == pytest.approx(1.0)   # pbrt default
    assert ov["cloud_frequency"] == pytest.approx(5.0)   # pbrt default
    # Constant sampled spectra go through the importer's standing equal-energy
    # -> sRGB projection (sampled_spectrum_to_rgb), which tints an achromatic
    # constant (recorded RGB-vs-spectral floor, same as the nanovdb scenes):
    # magnitude ~the constant, and sigma_a is exactly sigma_s/1000 (same shape).
    sig_s = np.asarray(ov["volume_sigma_s"], np.float64)
    sig_a = np.asarray(ov["volume_sigma_a"], np.float64)
    assert sig_s.mean() == pytest.approx(10.0, rel=0.25)
    np.testing.assert_allclose(sig_a, sig_s / 1000.0, rtol=1e-9)
    assert ov["volume_g"] == pytest.approx(0.0)


def test_cloud_world_to_uvw_identity_ctm_is_axis_flip(tmp_path):
    """Identity medium CTM -> world->medium-local is just the USD->pbrt basis
    flip B (rows of ctm^-1 @ B): x, y pass through, z negates."""
    medium = _parse_medium(tmp_path, CLOUD_SCENE)
    rows = np.asarray(cloud_overrides(medium)["volume_world_to_uvw"],
                      np.float64).reshape(3, 4)
    expect = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0]], np.float64)
    np.testing.assert_allclose(rows, expect, atol=1e-12)


def test_cloud_world_to_uvw_folds_medium_ctm(tmp_path):
    """A Translate before MakeNamedMedium lands in the world->local rows."""
    medium = _parse_medium(tmp_path, """
WorldBegin
Translate 2 3 4
MakeNamedMedium "c" "string type" "cloud"
MediumInterface "c" ""
Material ""
Shape "sphere" "float radius" 1
""")
    rows = np.asarray(cloud_overrides(medium)["volume_world_to_uvw"],
                      np.float64).reshape(3, 4)
    # p_usd = B @ ctm @ p_medium; medium origin (0,0,0) sits at USD (2,3,-4).
    p_usd = np.array([2.0, 3.0, -4.0, 1.0])
    local = rows @ p_usd
    np.testing.assert_allclose(local, [0.0, 0.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------- #
# 2.2 + 2.3 importer end-to-end
# --------------------------------------------------------------------------- #

def test_cloud_medium_imports_without_skip(tmp_path):
    stage, report = _import(tmp_path, CLOUD_SCENE)
    assert not any(
        e.status == "skipped" and "heterogeneous" in (e.detail or "")
        for e in report.entries
    )
    assert any(e.construct.startswith("medium:cloud") and e.status == "exact"
               for e in report.entries)


def test_cloud_sphere_binds_medium_overrides(tmp_path):
    stage, _report = _import(tmp_path, CLOUD_SCENE)
    ov = _sphere_overrides(stage)
    assert ov is not None
    assert ov["volume_cloud"] is True
    assert ov["cloud_density"] == pytest.approx(2.0)
    assert ov["volume_interface"] is True                # Material "" null boundary
    assert len(ov["volume_world_to_uvw"]) == 12


def test_cloud_emits_no_volume_prim(tmp_path):
    """Procedural cloud is analytic: no grid asset, no UsdVol.Volume prim."""
    stage, _report = _import(tmp_path, CLOUD_SCENE)
    assert not any(p.IsA(UsdVol.Volume) for p in stage.Traverse())


def test_empty_material_with_medium_is_null_boundary(tmp_path):
    """Material "" + MediumInterface -> lobe-less interface encoding, not the
    grey UsdPreviewSurface fallback."""
    stage, report = _import(tmp_path, CLOUD_SCENE)
    from pxr import UsdShade
    shader = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader":
            shader = UsdShade.Shader(prim)
    assert shader is not None
    diffuse = shader.GetInput("diffuseColor").Get()
    assert tuple(diffuse) == pytest.approx((0.0, 0.0, 0.0))  # interface, not grey 0.5
    opacity = shader.GetInput("opacity").Get()
    assert opacity == pytest.approx(1.0)                     # no glass gate
    assert any("material:null" in e.construct for e in report.entries)


def test_empty_material_without_medium_keeps_default(tmp_path):
    stage, report = _import(tmp_path, EMPTY_MATERIAL_NO_MEDIUM)
    from pxr import UsdShade
    shader = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader":
            shader = UsdShade.Shader(prim)
    assert shader is not None
    diffuse = shader.GetInput("diffuseColor").Get()
    assert tuple(diffuse) == pytest.approx((0.5, 0.5, 0.5))  # default grey
    assert not any("material:null" in e.construct for e in report.entries)
    ov = _sphere_overrides(stage)
    assert ov is None


def test_rgbgrid_still_skips(tmp_path):
    _stage, report = _import(tmp_path, RGBGRID_SCENE)
    assert any(
        e.status == "skipped" and "heterogeneous" in (e.detail or "")
        for e in report.entries
    )


def test_cloud_custom_bounds_still_skips(tmp_path):
    _stage, report = _import(tmp_path, CLOUD_CUSTOM_BOUNDS)
    assert any(
        e.status == "skipped" and "heterogeneous" in (e.detail or "")
        for e in report.entries
    )


@pytest.mark.skipif(not os.path.exists(CLOUDS_PBRT),
                    reason="pbrt-v4-scenes clouds not present")
def test_real_clouds_pbrt_imports_clean():
    stage, report = import_pbrt(CLOUDS_PBRT)
    assert not any(
        e.status == "skipped" and "heterogeneous" in (e.detail or "")
        for e in report.entries
    )
    ov = _sphere_overrides(stage)
    assert ov is not None
    assert ov["volume_cloud"] is True
    assert ov["cloud_density"] == pytest.approx(2.0)
    assert ov["cloud_wispiness"] == pytest.approx(1.0)
    assert ov["cloud_frequency"] == pytest.approx(5.0)
    assert ov["volume_interface"] is True
    # identity medium CTM in clouds.pbrt: rows are the plain basis flip
    rows = np.asarray(ov["volume_world_to_uvw"], np.float64).reshape(3, 4)
    np.testing.assert_allclose(
        rows, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0]], atol=1e-12)
