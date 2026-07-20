"""Hostless tests for ``Renderer.add_primitive`` and the ``add_model``/
``add_light`` fixes from the mcp-scene-structure change: parent-complete
rollback, the ``add_model`` validator seam, and ``add_light`` intensity/color
at creation. Mirrors the ``_editor()`` pattern in test_scene_light_creation.py
-- a hand-built ``Renderer`` with ``_resync_geometry_from_stage`` monkeypatched
to the pure-USD read (``load_scene_from_stage``), no GPU context.
"""

from __future__ import annotations

from types import MethodType

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

from skinny.renderer import Renderer
from skinny.scene import Scene
from skinny.scene_graph import (
    build_scene_graph,
    compose_trs_matrix,
    find_node_by_path,
    inject_default_lights,
)
from skinny.usd_loader import load_scene_from_stage

SUPPORTED_PRIMITIVES = ("Sphere", "Cube", "Cylinder", "Cone", "Capsule", "Plane")


def _default_light_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Skinny")
    UsdLux.DistantLight.Define(stage, "/Skinny/DefaultLight")
    UsdLux.DomeLight.Define(stage, "/Skinny/DefaultDome")
    return stage


def _editor(*, resync_raises: bool = False) -> tuple[Renderer, Usd.Stage]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    # A permanent gprim: load_scene_from_stage errors on zero usable geometry,
    # so a test that removes/deactivates its only other prim would otherwise
    # hit that unrelated error instead of exercising what it's testing.
    UsdGeom.Cube.Define(stage, "/World/Geometry")

    renderer = Renderer.__new__(Renderer)
    renderer._usd_stage = stage
    renderer._usd_edit_layer = None
    renderer._usd_scene = Scene()
    renderer._usd_model_index = 0
    renderer.model_index = 0
    renderer._material_version = 0
    renderer._material_graph_ids = {}
    renderer._material_graph_overrides = {}
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

    def upload_flat_materials(self, mats) -> None:  # GPU op stubbed out
        pass

    renderer._resync_geometry_from_stage = MethodType(resync, renderer)
    renderer._upload_flat_materials = MethodType(upload_flat_materials, renderer)
    return renderer, stage


# ── add_primitive: happy path ───────────────────────────────────────

@pytest.mark.parametrize("prim_type", SUPPORTED_PRIMITIVES)
def test_add_primitive_authors_gprim_and_bound_material(prim_type):
    renderer, stage = _editor()

    path = renderer.add_primitive(prim_type)

    assert path == f"/World/{prim_type}"
    assert stage.GetPrimAtPath(path).IsValid()
    material_path = f"{path}_material"
    assert stage.GetPrimAtPath(material_path).IsA(UsdShade.Material)
    binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path))
    bound, _ = binding.ComputeBoundMaterial()
    assert bound.GetPath() == Sdf.Path(material_path)


def test_add_primitive_material_is_editable_not_fallback_slot():
    """The whole point of D4/M2: the primitive must NOT land on slot 0."""
    renderer, stage = _editor()

    path = renderer.add_primitive("Sphere", color=(1.0, 0.0, 0.0))
    material_path = f"{path}_material"
    node = find_node_by_path(renderer.scene_graph, material_path)
    assert node is not None and node.renderer_ref is not None
    material_id = node.renderer_ref.index
    assert material_id > 0, "primitive material must not be the protected slot 0"

    version_before = renderer._material_version
    renderer.apply_material_override(material_id, "roughness", 0.9)

    assert renderer._material_version > version_before
    assert renderer._usd_scene.materials[material_id].parameter_overrides["roughness"] == 0.9


def test_add_primitive_authors_given_color_roughness_metallic():
    renderer, stage = _editor()

    path = renderer.add_primitive(
        "Cube", color=(0.2, 0.4, 0.6), roughness=0.1, metallic=1.0,
    )
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{path}_material/PreviewSurface"))
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.2, 0.4, 0.6))
    assert shader.GetInput("roughness").Get() == pytest.approx(0.1)
    assert shader.GetInput("metallic").Get() == pytest.approx(1.0)


def test_add_primitive_defaults_when_omitted():
    renderer, stage = _editor()

    path = renderer.add_primitive("Sphere")
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{path}_material/PreviewSurface"))
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.8, 0.8, 0.8))
    assert shader.GetInput("roughness").Get() == pytest.approx(0.5)
    assert shader.GetInput("metallic").Get() == pytest.approx(0.0)


def test_add_primitive_uniquifies_names():
    renderer, stage = _editor()

    first = renderer.add_primitive("Sphere")
    second = renderer.add_primitive("Sphere")

    assert first == "/World/Sphere"
    assert second == "/World/Sphere_1"
    assert stage.GetPrimAtPath(first).IsValid()
    assert stage.GetPrimAtPath(second).IsValid()


def test_add_primitive_rejects_unsupported_type_without_mutation():
    renderer, stage = _editor()
    before = [str(prim.GetPath()) for prim in stage.Traverse()]

    with pytest.raises(ValueError, match="unsupported primitive type"):
        renderer.add_primitive("Torus")

    assert [str(prim.GetPath()) for prim in stage.Traverse()] == before


def test_add_primitive_requires_editable_stage():
    renderer = Renderer.__new__(Renderer)
    renderer._usd_stage = None
    renderer._usd_edit_layer = None

    with pytest.raises(RuntimeError, match="loaded USD stage"):
        renderer.add_primitive("Sphere")


# ── add_primitive: rollback ─────────────────────────────────────────

def test_add_primitive_rollback_removes_gprim_and_material():
    renderer, stage = _editor(resync_raises=True)

    with pytest.raises(RuntimeError, match="synthetic resync failure"):
        renderer.add_primitive("Sphere")

    assert not stage.GetPrimAtPath("/World/Sphere").IsValid()
    assert not stage.GetPrimAtPath("/World/Sphere_material").IsValid()


def test_add_primitive_rollback_removes_created_parents():
    renderer, stage = _editor(resync_raises=True)

    with pytest.raises(RuntimeError, match="synthetic resync failure"):
        renderer.add_primitive("Sphere", parent_prim_path="/World/New/Group")

    assert not stage.GetPrimAtPath("/World/New").IsValid()
    assert not stage.GetPrimAtPath("/World/New/Group").IsValid()
    assert not stage.GetPrimAtPath("/World/New/Group/Sphere").IsValid()


# ── add_primitive: TRS round-trip through the real builder (F13) ───

def test_add_primitive_trs_round_trips_through_scene_graph_builder():
    renderer, stage = _editor()
    translate = (2.0, 3.0, 4.0)
    rotate_deg = (0.0, 30.0, 0.0)  # single-axis to sidestep Euler-order ambiguity
    scale = (1.0, 1.0, 1.0)
    matrix = compose_trs_matrix(translate, rotate_deg, scale)

    path = renderer.add_primitive("Sphere", transform=matrix)
    node = find_node_by_path(renderer.scene_graph, path)
    props = {p.name: p.value for p in node.properties}

    assert props["translate"] == pytest.approx(translate, abs=1e-4)
    assert props["rotate"] == pytest.approx(rotate_deg, abs=1e-3)
    assert props["scale"] == pytest.approx(scale, abs=1e-4)


# ── add_primitive: inactive-prim non-resurrection (F11) ─────────────

def test_remove_then_readd_same_name_does_not_resurrect_old_subtree():
    renderer, stage = _editor()

    first = renderer.add_primitive("Sphere", color=(1.0, 0.0, 0.0))
    renderer.remove_node(first)
    assert not stage.GetPrimAtPath(first).IsActive()

    second = renderer.add_primitive("Sphere", color=(0.0, 1.0, 0.0))

    assert second != first, "the burned name must not be reused"
    assert stage.GetPrimAtPath(second).IsActive()
    # The old (deactivated) subtree's material must not have been touched.
    old_shader = UsdShade.Shader(stage.GetPrimAtPath(f"{first}_material/PreviewSurface"))
    assert tuple(old_shader.GetInput("diffuseColor").Get()) == pytest.approx((1.0, 0.0, 0.0))


# ── add_model: parent-complete rollback (F2) ────────────────────────

@pytest.fixture()
def referenceable_usda(tmp_path):
    path = tmp_path / "referenced.usda"
    stage = Usd.Stage.CreateNew(str(path))
    sphere = UsdGeom.Sphere.Define(stage, "/Referenced")
    sphere.CreateRadiusAttr().Set(2.5)  # authored, not just a schema fallback
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Referenced"))
    stage.Save()
    return path


def test_add_model_rollback_removes_created_parents_on_resync_failure(referenceable_usda):
    renderer, stage = _editor(resync_raises=True)

    with pytest.raises(RuntimeError, match="synthetic resync failure"):
        renderer.add_model(str(referenceable_usda), parent_prim_path="/World/New/Group")

    assert not stage.GetPrimAtPath("/World/New").IsValid()
    assert not stage.GetPrimAtPath("/World/New/Group").IsValid()
    assert not stage.GetPrimAtPath("/World/New/Group/referenced").IsValid()


def test_add_model_validator_veto_rolls_back_with_no_resync(referenceable_usda):
    renderer, stage = _editor()
    resync_calls = []
    original_resync = renderer._resync_geometry_from_stage

    def counting_resync(self):
        resync_calls.append(1)
        return original_resync()

    renderer._resync_geometry_from_stage = MethodType(counting_resync, renderer)

    def veto(stage_arg, added_prim):
        raise ValueError(f"vetoed: {added_prim.GetPath()}")

    with pytest.raises(ValueError, match="vetoed"):
        renderer.add_model(str(referenceable_usda), parent_prim_path="/World/New", validate=veto)

    assert not stage.GetPrimAtPath("/World/New").IsValid()
    assert resync_calls == [], "a vetoed add must not reach geometry re-sync"


def test_add_model_validator_receives_recomposed_prim_and_stage(referenceable_usda):
    renderer, _stage = _editor()
    seen = {}

    def capture(stage_arg, added_prim):
        seen["stage"] = stage_arg
        seen["prim"] = added_prim

    path = renderer.add_model(str(referenceable_usda), validate=capture)

    assert seen["prim"].GetPath() == Sdf.Path(path)
    # add_model always wraps the reference in a local Xform, so the local
    # typeName opinion wins over the referenced Sphere's -- but the Sphere
    # schema's own attributes still compose through the reference arc,
    # confirming recomposition already happened by the time validate runs.
    assert seen["prim"].GetAttribute("radius").IsValid()
    assert seen["stage"] is renderer._usd_stage


def test_add_model_no_validator_unchanged_behavior(referenceable_usda):
    renderer, stage = _editor()

    path = renderer.add_model(str(referenceable_usda))

    assert stage.GetPrimAtPath(path).IsValid()


# ── add_light: intensity/color at creation (F5) ─────────────────────

def test_add_light_authors_given_intensity_and_color():
    renderer, stage = _editor()

    path = renderer.add_light("SphereLight", intensity=250.0, color=(1.0, 0.5, 0.2))

    api = UsdLux.LightAPI(stage.GetPrimAtPath(path))
    assert api.GetIntensityAttr().Get() == pytest.approx(250.0)
    assert tuple(api.GetColorAttr().Get()) == pytest.approx((1.0, 0.5, 0.2))


def test_add_light_defaults_unchanged_when_intensity_and_color_omitted():
    renderer, stage = _editor()

    path = renderer.add_light("SphereLight")

    api = UsdLux.LightAPI(stage.GetPrimAtPath(path))
    assert api.GetIntensityAttr().Get() == pytest.approx(1.0)
    assert tuple(api.GetColorAttr().Get()) == pytest.approx((1.0, 1.0, 1.0))
