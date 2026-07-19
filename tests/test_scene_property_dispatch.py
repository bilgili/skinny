"""Tests for the shared scene-property dispatch.

This is the function the Qt dock and the MCP server both route writes through,
so the cases that are easy to get wrong from a (path, property) pair alone are
covered here: a material parameter living on a shader prim with no renderer
reference, and a transform component that has to recompose from its siblings.
"""

from __future__ import annotations

from skinny.scene_graph import RendererRef, SceneGraphNode, SceneGraphProperty
from skinny.ui.scene_edit_actions import apply_scene_property


class FakeRenderer:
    """Records the verb each edit routed to, without touching a GPU."""

    def __init__(self, *, usd_stage=None, scene_graph=None) -> None:
        self.calls: list[tuple] = []
        self._usd_stage = usd_stage
        self.scene_graph = scene_graph

    def apply_material_override(self, index, key, value):
        self.calls.append(("material", index, key, value))

    def apply_light_override(self, light_type, index, key, value):
        self.calls.append(("light", light_type, index, key, value))

    def apply_camera_param(self, key, value):
        self.calls.append(("camera", key, value))

    def apply_node_enabled(self, path, value):
        self.calls.append(("node_enabled", path, value))

    def apply_subtree_enabled(self, path, value):
        self.calls.append(("subtree_enabled", path, value))

    def apply_instance_transform(self, path, translate, rotate, scale):
        self.calls.append(("instance_transform", path, translate, rotate, scale))

    def set_transform(self, path, matrix):
        self.calls.append(("set_transform", path, matrix))

    def apply_dome_light_texture(self, index, path):
        self.calls.append(("dome_texture", index, path))


def _prop(name, type_name, value, **metadata):
    return SceneGraphProperty(
        name=name,
        display_name=name,
        type_name=type_name,
        value=value,
        editable=True,
        metadata=metadata,
    )


def _node(path, type_name="Mesh", *, ref=None, properties=None, children=None):
    return SceneGraphNode(
        path=path,
        name=path.rstrip("/").split("/")[-1] or "/",
        type_name=type_name,
        children=children or [],
        properties=properties or [],
        renderer_ref=ref,
    )


def test_material_param_on_shader_prim_routes_via_ancestor_walk() -> None:
    """The most common edit: shader prims carry no renderer_ref of their own."""
    shader = _node("/World/mat/surface", "Shader", properties=[])
    material = _node(
        "/World/mat", "Material",
        ref=RendererRef(kind="material", index=3),
        children=[shader],
    )
    root = _node("/", "Stage", children=[
        _node("/World", "Xform", children=[material]),
    ])
    renderer = FakeRenderer(scene_graph=root)

    reason = apply_scene_property(
        renderer, shader, _prop("roughness", "float", 0.5), 0.25, graph=root,
    )

    assert reason is None
    assert renderer.calls == [("material", 3, "roughness", 0.25)]


def test_transform_component_recomposes_from_siblings() -> None:
    """Writing translate must preserve the node's existing rotate and scale."""
    translate = _prop("translate", "vec3f", (0.0, 0.0, 0.0))
    rotate = _prop("rotate", "vec3f", (0.0, 90.0, 0.0))
    scale = _prop("scale", "vec3f", (2.0, 2.0, 2.0))
    node = _node(
        "/World/box", "Mesh",
        ref=RendererRef(kind="instance", index=1),
        properties=[translate, rotate, scale],
    )
    renderer = FakeRenderer()  # no _usd_stage -> runtime path

    reason = apply_scene_property(renderer, node, translate, (5.0, 0.0, 0.0))

    assert reason is None
    assert renderer.calls == [
        ("instance_transform", "/World/box", (5.0, 0.0, 0.0), (0.0, 90.0, 0.0), (2.0, 2.0, 2.0)),
    ]


def test_transform_authors_to_stage_when_one_is_loaded() -> None:
    translate = _prop("translate", "vec3f", (0.0, 0.0, 0.0))
    node = _node(
        "/World/box", "Mesh",
        ref=RendererRef(kind="instance", index=1),
        properties=[translate],
    )
    renderer = FakeRenderer(usd_stage=object())

    assert apply_scene_property(renderer, node, translate, (1.0, 2.0, 3.0)) is None
    assert renderer.calls[0][0] == "set_transform"


def test_light_property_routes_by_kind() -> None:
    node = _node("/World/sun", "DistantLight", ref=RendererRef(kind="light_dir", index=0))

    assert apply_scene_property(renderer := FakeRenderer(), node, _prop("intensity", "float", 1.0), 5.0) is None
    assert renderer.calls == [("light", "dir", 0, "intensity", 5.0)]


def test_subtree_toggle_routes_to_subtree_enable() -> None:
    node = _node("/World/group", "Xform")
    prop = _prop("visible", "bool", True, toggle="subtree")

    assert apply_scene_property(renderer := FakeRenderer(), node, prop, False) is None
    assert renderer.calls == [("subtree_enabled", "/World/group", False)]


def test_camera_vec3_fans_out_to_three_scalars() -> None:
    node = _node("/Camera", "Camera", ref=RendererRef(kind="renderer_camera", index=0))
    prop = _prop("target", "vec3f", (0.0, 0.0, 0.0), camera_axis="target")

    assert apply_scene_property(renderer := FakeRenderer(), node, prop, (1.0, 2.0, 3.0)) is None
    assert renderer.calls == [
        ("camera", "target_x", 1.0),
        ("camera", "target_y", 2.0),
        ("camera", "target_z", 3.0),
    ]


def test_unrouted_property_reports_a_reason_instead_of_silent_noop() -> None:
    """A node with no reference and no material ancestor must not look successful."""
    node = _node("/World/loose", "Xform")
    renderer = FakeRenderer(scene_graph=_node("/", "Stage"))

    reason = apply_scene_property(renderer, node, _prop("whatever", "float", 1.0), 2.0)

    assert reason is not None
    assert "/World/loose" in reason
    assert renderer.calls == []


def test_plain_camera_kind_reports_no_route() -> None:
    """`camera` (as opposed to `renderer_camera`) has no verb -- say so."""
    node = _node("/Camera", "Camera", ref=RendererRef(kind="camera", index=0))

    reason = apply_scene_property(
        renderer := FakeRenderer(), node, _prop("focalLength", "float", 50.0), 35.0,
    )

    assert reason is not None and "camera" in reason
    assert renderer.calls == []
