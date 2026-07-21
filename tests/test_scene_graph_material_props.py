"""Scene-graph editable-property injection for live MaterialX materials
(mcp-material-authoring, task 2.5).

A synthesized graph material surfaces its promoted logical inputs as editable
properties carrying the fan-out uniform list; a constant-shader `.mtlx` material
surfaces its parameter-override keys; a non-promoted constant is simply absent
(so `scene_set` gives the existing no-such-property error). A logical edit fans
out through `apply_scene_property` to every mapped uniform in one call.

Hostless — pure `pxr` + `skinny.scene_graph` / `skinny.ui.scene_edit_actions`;
never imports `skinny.renderer`."""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")

from skinny.scene_graph import build_scene_graph, find_node_by_path  # noqa: E402
from skinny.ui.scene_edit_actions import apply_scene_property  # noqa: E402


class _Mat:
    """Minimal stand-in for `skinny.scene.Material`."""

    def __init__(self, name, *, logical_inputs=None, parameter_overrides=None,
                 mtlx_document=None):
        self.name = name
        self.logical_inputs = logical_inputs or {}
        self.parameter_overrides = parameter_overrides or {}
        self.mtlx_document = mtlx_document


class _Scene:
    def __init__(self, materials):
        self.instances = []
        self.materials = materials
        self.lights_dir = []
        self.lights_sphere = []
        self.camera_override = None


def _stage_with_material(name):
    from pxr import Usd, UsdShade
    stage = Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(stage, f"/Materials/{name}")
    return stage


def _props(node):
    return {p.name: p for p in node.properties}


def test_graph_material_injects_logical_inputs():
    """A material with a logical→uniform mapping lists each promoted input as an
    editable property whose fan-out metadata carries the mapped uniforms."""
    stage = _stage_with_material("Marble")
    mat = _Mat(
        "Marble",
        logical_inputs={"colorA": ["blend_bg"], "scale": ["scaled_in2", "n2_in2"]},
        parameter_overrides={"blend_bg": (0.2, 0.5, 0.9)},
    )
    sg = build_scene_graph(stage, _Scene([mat]))
    props = _props(find_node_by_path(sg, "/Materials/Marble"))

    assert "colorA" in props and "scale" in props
    assert props["colorA"].type_name == "color3f"
    assert props["colorA"].value == (0.2, 0.5, 0.9)      # round-trips override
    assert props["colorA"].editable
    assert props["colorA"].metadata["fanout"] == ["blend_bg"]
    assert props["scale"].metadata["fanout"] == ["scaled_in2", "n2_in2"]
    assert props["scale"].type_name == "float"


def test_unedited_logical_input_shows_default():
    """A promoted input with no override yet still appears (editability contract),
    with a type-appropriate default value."""
    stage = _stage_with_material("Marble")
    mat = _Mat("Marble", logical_inputs={"colorB": ["blend_fg"]})
    sg = build_scene_graph(stage, _Scene([mat]))
    props = _props(find_node_by_path(sg, "/Materials/Marble"))
    assert props["colorB"].type_name == "color3f"
    assert props["colorB"].value == (0.0, 0.0, 0.0)


def test_descriptor_int_and_range_surface_typed_with_authored_default():
    """A persisted descriptor's explicit type/default/range win over inference
    (finding #2): octaves is an int spinning over 1..8 starting at its authored
    4, scale reaches its 64 max — not a 0..1 float defaulting to 0."""
    stage = _stage_with_material("Marble")
    mat = _Mat("Marble", logical_inputs={
        "octaves": {"uniforms": ["noise_octaves"], "type": "int",
                    "default": 4, "range": [1, 8]},
        "scale": {"uniforms": ["scaled_in2"], "type": "float",
                  "default": 6.0, "range": [0.01, 64.0]},
    })
    sg = build_scene_graph(stage, _Scene([mat]))
    props = _props(find_node_by_path(sg, "/Materials/Marble"))
    assert props["octaves"].type_name == "int"
    assert props["octaves"].value == 4
    assert props["octaves"].metadata["min"] == 1
    assert props["octaves"].metadata["max"] == 8
    assert props["octaves"].metadata["fanout"] == ["noise_octaves"]
    assert props["scale"].type_name == "float"
    assert props["scale"].value == 6.0
    assert props["scale"].metadata["max"] == 64.0


def test_vector3_descriptor_surfaces_as_sequence_property():
    """A reflected vector3 input surfaces as a vec3f (3-sequence) property, not a
    scalar float the runtime packer would zero (finding #2)."""
    stage = _stage_with_material("Marble")
    mat = _Mat("Marble", logical_inputs={
        "add_xyz_in2": {"uniforms": ["add_xyz_in2"], "type": "vector3",
                        "default": [0.1, 0.2, 0.3], "range": None},
    })
    sg = build_scene_graph(stage, _Scene([mat]))
    prop = _props(find_node_by_path(sg, "/Materials/Marble"))["add_xyz_in2"]
    assert prop.type_name == "vec3f"
    assert prop.value == (0.1, 0.2, 0.3)
    assert prop.metadata["fanout"] == ["add_xyz_in2"]


def test_vector2_descriptor_surfaces_as_sequence_property():
    """A reflected vector2 input surfaces as a vec2f (2-sequence) (finding #2)."""
    stage = _stage_with_material("Marble")
    mat = _Mat("Marble", logical_inputs={
        "uv": {"uniforms": ["uv"], "type": "vector2", "default": [0.5, 0.25],
               "range": None},
    })
    sg = build_scene_graph(stage, _Scene([mat]))
    prop = _props(find_node_by_path(sg, "/Materials/Marble"))["uv"]
    assert prop.type_name == "vec2f"
    assert prop.value == (0.5, 0.25)


def test_bool_descriptor_surfaces_as_bool_property():
    """A reflected boolean input surfaces as a bool, not a 0..1 float (#2)."""
    stage = _stage_with_material("Marble")
    mat = _Mat("Marble", logical_inputs={
        "flag": {"uniforms": ["flag"], "type": "bool", "default": True,
                 "range": None},
    })
    sg = build_scene_graph(stage, _Scene([mat]))
    prop = _props(find_node_by_path(sg, "/Materials/Marble"))["flag"]
    assert prop.type_name == "bool"
    assert prop.value is True


def test_vector_and_bool_fanout_route_to_material_not_trs_or_enable():
    """A vec3f/bool MATERIAL input carries fan-out, so it reaches the material-
    override path — the TRS (vec3f) / enable (bool) branches would otherwise
    capture and misroute it (finding #2). _MockRenderer has no TRS/enable verbs,
    so a misroute would raise."""
    stage = _stage_with_material("Marble")
    mat = _Mat("Marble", logical_inputs={
        "add_xyz_in2": {"uniforms": ["ax", "ay"], "type": "vector3",
                        "default": [0.0, 0.0, 0.0], "range": None},
        "flag": {"uniforms": ["fl"], "type": "bool", "default": False,
                 "range": None},
    })
    sg = build_scene_graph(stage, _Scene([mat]))
    node = find_node_by_path(sg, "/Materials/Marble")
    props = _props(node)
    idx = node.renderer_ref.index

    rend = _MockRenderer()
    assert apply_scene_property(rend, node, props["add_xyz_in2"], [1.0, 2.0, 3.0], graph=sg) is None
    assert apply_scene_property(rend, node, props["flag"], True, graph=sg) is None
    assert rend.single == []
    assert rend.batched == [
        (idx, {"ax": [1.0, 2.0, 3.0], "ay": [1.0, 2.0, 3.0]}),
        (idx, {"fl": True}),
    ]


def test_constant_preset_advertised_keys_exposed_raw_aliases_filtered():
    """For a constant-shader preset the advertised material_list keys are the
    FlatMaterialParams packer keys, and only those are surfaced on the scene-graph
    node. The loader dual-authors both the packer key (diffuseColor) and the raw
    std_surface alias (base_color) into parameter_overrides; the raw alias is an
    editable no-op in the path tracer (pack_flat_material reads only flat keys),
    so it is filtered from the editable surface (finding B, round 4)."""
    from skinny import mtlx_synthesis as msyn
    advertised = set(msyn.list_preset_inputs("chrome"))
    # the packer keys the constant preset now advertises
    assert {"diffuseColor", "metallic", "roughness"} <= advertised
    # a chrome-like loaded material: constant preset -> empty logical_inputs,
    # parameter_overrides carry BOTH the packer keys pack_flat_material reads AND
    # the dual-authored std_surface names (exactly what _load_mtlx_materials emits).
    stage = _stage_with_material("chrome")
    mat = _Mat("chrome", parameter_overrides={
        "diffuseColor": (0.9, 0.9, 0.9), "base_color": (0.9, 0.9, 0.9),
        "metallic": 1.0, "metalness": 1.0, "specular": 1.0,
        "specular_color": (1.0, 1.0, 1.0),
        "roughness": 0.05, "specular_roughness": 0.05,
    }, mtlx_document=object())
    sg = build_scene_graph(stage, _Scene([mat]))
    props = set(_props(find_node_by_path(sg, "/Materials/chrome")))
    assert advertised <= props        # every advertised (packer) key is editable
    # the dual-authored raw std_surface aliases are editable no-ops → filtered out
    assert not ({"base_color", "metalness", "specular_roughness"} & props)


def test_constant_mtlx_material_exposes_override_keys():
    """A constant-shader `.mtlx` material (no graph, has an mtlx_document) exposes
    its flat-packer override keys; a raw std_surface alias and a non-authored name
    are both absent (finding B, round 4)."""
    stage = _stage_with_material("Glass")
    mat = _Mat(
        "Glass",
        parameter_overrides={
            "roughness": 0.1, "diffuseColor": (0.9, 0.9, 1.0),
            "base_color": (0.9, 0.9, 1.0),  # raw alias → filtered
        },
        mtlx_document=object(),
    )
    sg = build_scene_graph(stage, _Scene([mat]))
    props = _props(find_node_by_path(sg, "/Materials/Glass"))
    assert "roughness" in props and "diffuseColor" in props
    assert props["diffuseColor"].type_name == "color3f"
    assert props["roughness"].type_name == "float"
    assert "base_color" not in props  # raw std_surface alias → editable no-op, filtered
    assert "opacity" not in props  # non-authored → absent (no-such-property)


def test_plain_material_without_mtlx_gets_no_injected_props():
    """A plain UsdPreviewSurface material (no mapping, no mtlx_document) is not
    injected on the Material node — its inputs surface on the child Shader."""
    stage = _stage_with_material("Preview")
    mat = _Mat("Preview", parameter_overrides={"roughness": 0.3})
    sg = build_scene_graph(stage, _Scene([mat]))
    props = _props(find_node_by_path(sg, "/Materials/Preview"))
    assert "roughness" not in props


class _MockRenderer:
    def __init__(self):
        self.single = []
        self.batched = []
        self.scene_graph = None

    def apply_material_override(self, idx, key, value):
        self.single.append((idx, key, value))

    def apply_material_overrides(self, idx, values):
        self.batched.append((idx, dict(values)))


def test_fanout_dispatch_writes_all_mapped_uniforms():
    """A logical `scene_set` fans out to every mapped uniform via one
    apply_material_overrides call (single version bump)."""
    stage = _stage_with_material("Marble")
    mat = _Mat("Marble", logical_inputs={"scale": ["scaled_in2", "n2_in2"]})
    sg = build_scene_graph(stage, _Scene([mat]))
    node = find_node_by_path(sg, "/Materials/Marble")
    prop = _props(node)["scale"]

    rend = _MockRenderer()
    err = apply_scene_property(rend, node, prop, 3.5, graph=sg)
    assert err is None
    assert rend.single == []                        # not the single-key path
    assert rend.batched == [(node.renderer_ref.index,
                             {"scaled_in2": 3.5, "n2_in2": 3.5})]


def test_nonfanout_material_prop_uses_single_write():
    """A constant-material property (no fanout metadata) routes to the single
    apply_material_override, unchanged behavior."""
    stage = _stage_with_material("Glass")
    mat = _Mat("Glass", parameter_overrides={"roughness": 0.1}, mtlx_document=object())
    sg = build_scene_graph(stage, _Scene([mat]))
    node = find_node_by_path(sg, "/Materials/Glass")
    prop = _props(node)["roughness"]

    rend = _MockRenderer()
    err = apply_scene_property(rend, node, prop, 0.7, graph=sg)
    assert err is None
    assert rend.batched == []
    assert rend.single == [(node.renderer_ref.index, "roughness", 0.7)]
