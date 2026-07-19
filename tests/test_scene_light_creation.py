"""Hostless tests for edit-layer-backed scene light creation."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdLux

from skinny.renderer import Renderer
from skinny.scene import Scene
from skinny.scene_graph import build_scene_graph, find_node_by_path, inject_default_lights
from skinny.usd_loader import load_scene_from_stage


SUPPORTED_LIGHTS = {
    "DistantLight": UsdLux.DistantLight,
    "SphereLight": UsdLux.SphereLight,
    "DomeLight": UsdLux.DomeLight,
    "RectLight": UsdLux.RectLight,
    "DiskLight": UsdLux.DiskLight,
}


def _default_light_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Skinny")
    UsdLux.DistantLight.Define(stage, "/Skinny/DefaultLight")
    UsdLux.DomeLight.Define(stage, "/Skinny/DefaultDome")
    return stage


def _paths(node) -> set[str]:
    return {node.path, *(
        path
        for child in node.children
        for path in _paths(child)
    )}


def _editor(*, resync_raises: bool = False) -> tuple[Renderer, Usd.Stage]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Lights")
    UsdGeom.Cube.Define(stage, "/World/Geometry")

    renderer = Renderer.__new__(Renderer)
    renderer._usd_stage = stage
    renderer._usd_edit_layer = None
    renderer._usd_scene = Scene()
    renderer._usd_model_index = 0
    renderer.model_index = 0
    renderer._material_version = 0
    renderer._scene_graph = build_scene_graph(stage, renderer._usd_scene)
    renderer._default_light_stage = _default_light_stage()
    renderer._attach_edit_layer()
    inject_default_lights(
        renderer._scene_graph,
        renderer._default_light_stage,
        enabled=renderer.uses_default_lights,
    )

    def resync(self) -> None:
        if resync_raises:
            raise RuntimeError("synthetic resync failure")
        self._usd_scene = load_scene_from_stage(self._usd_stage)
        self._scene_graph = build_scene_graph(self._usd_stage, self._usd_scene)
        inject_default_lights(
            self._scene_graph,
            self._default_light_stage,
            enabled=self.uses_default_lights,
        )

    renderer._resync_geometry_from_stage = MethodType(resync, renderer)
    return renderer, stage


@pytest.mark.parametrize("light_type, schema", SUPPORTED_LIGHTS.items())
def test_add_light_authors_every_supported_schema(light_type, schema):
    renderer, stage = _editor()

    path = renderer.add_light(light_type)

    assert path == f"/World/{light_type}"
    assert stage.GetPrimAtPath(path).IsA(schema)
    api = UsdLux.LightAPI(stage.GetPrimAtPath(path))
    assert tuple(api.GetColorAttr().Get()) == pytest.approx((1.0, 1.0, 1.0))
    assert api.GetIntensityAttr().Get() == pytest.approx(1.0)
    assert api.GetExposureAttr().Get() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "light_type, attribute, expected",
    [
        ("DistantLight", "inputs:angle", 0.53),
        ("SphereLight", "inputs:radius", 0.5),
        ("RectLight", "inputs:width", 1.0),
        ("RectLight", "inputs:height", 1.0),
        ("DiskLight", "inputs:radius", 0.5),
    ],
)
def test_add_light_authors_type_specific_defaults(light_type, attribute, expected):
    renderer, stage = _editor()

    path = renderer.add_light(light_type)

    assert stage.GetPrimAtPath(path).GetAttribute(attribute).Get() == pytest.approx(expected)


def test_add_light_uses_selected_parent_and_unique_names():
    renderer, stage = _editor()

    first = renderer.add_light("SphereLight", parent_prim_path="/World/Lights")
    second = renderer.add_light("SphereLight", parent_prim_path="/World/Lights")

    assert first == "/World/Lights/SphereLight"
    assert second == "/World/Lights/SphereLight_1"
    assert stage.GetPrimAtPath(first).IsA(UsdLux.SphereLight)
    assert stage.GetPrimAtPath(second).IsA(UsdLux.SphereLight)


def test_add_light_accepts_custom_name_and_transform():
    renderer, stage = _editor()
    matrix = np.eye(4, dtype=np.float32)
    matrix[3, :3] = [2.0, 3.0, 4.0]

    path = renderer.add_light("SphereLight", name="Key light", transform=matrix)

    assert path == "/World/Key_light"
    world = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    assert world.ExtractTranslation() == Gf.Vec3d(2.0, 3.0, 4.0)


@pytest.mark.parametrize("light_type", SUPPORTED_LIGHTS)
def test_created_light_transform_properties_are_editable(light_type):
    renderer, _stage = _editor()

    path = renderer.add_light(light_type)
    node = find_node_by_path(renderer.scene_graph, path)

    assert node is not None
    transform_props = {
        prop.name: prop.editable
        for prop in node.properties
        if prop.name in {"translate", "rotate", "scale"}
    }
    assert transform_props == {
        "translate": True,
        "rotate": True,
        "scale": True,
    }


def test_set_transform_resyncs_created_sphere_light_position():
    renderer, _stage = _editor()
    path = renderer.add_light("SphereLight")
    matrix = np.eye(4, dtype=np.float32)
    matrix[3, :3] = [5.0, 6.0, 7.0]

    renderer.set_transform(path, matrix)

    assert len(renderer._usd_scene.lights_sphere) == 1
    assert renderer._usd_scene.lights_sphere[0].position == pytest.approx(
        (5.0, 6.0, 7.0)
    )


def test_add_light_rejects_unknown_type_without_mutation():
    renderer, stage = _editor()
    before = [str(prim.GetPath()) for prim in stage.Traverse()]

    with pytest.raises(ValueError, match="unsupported light type"):
        renderer.add_light("LaserLight")

    assert [str(prim.GetPath()) for prim in stage.Traverse()] == before


def test_add_light_requires_editable_stage():
    renderer = Renderer.__new__(Renderer)
    renderer._usd_stage = None
    renderer._usd_edit_layer = None

    with pytest.raises(RuntimeError, match="loaded USD stage"):
        renderer.add_light("SphereLight")


def test_add_light_rolls_back_when_resync_fails():
    renderer, stage = _editor(resync_raises=True)

    with pytest.raises(RuntimeError, match="synthetic resync failure"):
        renderer.add_light("SphereLight")

    assert not stage.GetPrimAtPath("/World/SphereLight").IsValid()


def test_add_light_rolls_back_every_created_parent_ancestor():
    renderer, stage = _editor(resync_raises=True)

    with pytest.raises(RuntimeError, match="synthetic resync failure"):
        renderer.add_light(
            "SphereLight", parent_prim_path="/World/New/Lights",
        )

    assert not stage.GetPrimAtPath("/World/New").IsValid()
    assert not stage.GetPrimAtPath("/World/New/Lights").IsValid()
    assert not stage.GetPrimAtPath("/World/New/Lights/SphereLight").IsValid()


def test_first_authored_light_replaces_fallback_pair_and_rebuilds_graph():
    renderer, _stage = _editor()
    assert renderer.uses_default_lights is True
    assert {
        "/Skinny/DefaultLight",
        "/Skinny/DefaultDome",
    } <= _paths(renderer.scene_graph)

    path = renderer.add_light("DistantLight")

    assert renderer.uses_default_lights is False
    graph_paths = _paths(renderer.scene_graph)
    assert path in graph_paths
    assert "/Skinny/DefaultLight" not in graph_paths
    assert "/Skinny/DefaultDome" not in graph_paths
