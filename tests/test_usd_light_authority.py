"""Hostless tests for USD-authored lighting authority."""

from __future__ import annotations

import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


def _stage_with_triangle():
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([
        Gf.Vec3f(-1.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
    ])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return stage, mesh


def _load(stage):
    from skinny.usd_loader import load_scene_from_stage

    return load_scene_from_stage(stage)


@needs_usd
@pytest.mark.parametrize(
    "light_type",
    ["DistantLight", "SphereLight", "DomeLight", "RectLight", "DiskLight"],
)
def test_supported_usd_light_prim_records_authored_authority(light_type):
    from pxr import UsdLux

    stage, _mesh = _stage_with_triangle()
    light_cls = getattr(UsdLux, light_type)
    light_cls.Define(stage, f"/World/{light_type}")

    assert _load(stage).has_authored_lighting is True


@needs_usd
def test_zero_intensity_active_light_still_records_authority():
    from pxr import UsdLux

    stage, _mesh = _stage_with_triangle()
    light = UsdLux.SphereLight.Define(stage, "/World/BlackLight")
    light.CreateIntensityAttr().Set(0.0)

    assert _load(stage).has_authored_lighting is True


@needs_usd
def test_invisible_active_light_still_records_authority():
    from pxr import UsdGeom, UsdLux

    stage, _mesh = _stage_with_triangle()
    light = UsdLux.SphereLight.Define(stage, "/World/HiddenLight")
    UsdGeom.Imageable(light.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    assert _load(stage).has_authored_lighting is True


@needs_usd
def test_inactive_light_prim_does_not_record_authority():
    from pxr import UsdLux

    stage, _mesh = _stage_with_triangle()
    light = UsdLux.SphereLight.Define(stage, "/World/RemovedLight")
    light.GetPrim().SetActive(False)

    assert _load(stage).has_authored_lighting is False


@needs_usd
def test_emissive_material_instance_records_authority():
    from pxr import Gf, Sdf, UsdShade

    stage, mesh = _stage_with_triangle()
    material = UsdShade.Material.Define(stage, "/World/Emissive")
    shader = UsdShade.Shader.Define(stage, "/World/Emissive/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(3.0, 2.0, 1.0)
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    assert _load(stage).has_authored_lighting is True


@needs_usd
def test_stage_without_authored_lighting_records_no_authority():
    stage, _mesh = _stage_with_triangle()

    assert _load(stage).has_authored_lighting is False
