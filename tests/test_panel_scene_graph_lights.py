"""Focused wiring checks for Panel scene-graph light creation."""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest

pn = pytest.importorskip("panel")

from skinny.ui.panel import windows  # noqa: E402
from skinny.ui.scene_edit_actions import SUPPORTED_LIGHT_TYPES  # noqa: E402


class _Renderer:
    scene_graph = None
    _scene_graph_version = 0
    _usd_stage = object()
    _usd_edit_layer = object()
    # `has_editable_stage` now also gates on adopted scene metadata (finding #4).
    _usd_scene = object()

    def __init__(self, failure: Exception | None = None):
        self.failure = failure
        self.added = []

    def add_light(self, light_type, parent_prim_path):
        if self.failure is not None:
            raise self.failure
        self.added.append((light_type, parent_prim_path))
        return f"{parent_prim_path}/{light_type}"


def _pane(monkeypatch, renderer):
    monkeypatch.setattr(
        pn.state, "add_periodic_callback", lambda *args, **kwargs: None,
    )
    session = SimpleNamespace(renderer=renderer, _lock=Lock())
    return windows.build_scene_graph_pane(session, lambda: None)


def test_panel_scene_graph_menu_routes_every_supported_light(monkeypatch):
    renderer = _Renderer()
    pane = _pane(monkeypatch, renderer)
    menu = pane.select(pn.widgets.MenuButton)[0]

    assert tuple(value for _label, value in menu.items) == SUPPORTED_LIGHT_TYPES
    for light_type in SUPPORTED_LIGHT_TYPES:
        menu.clicked = light_type

    assert renderer.added == [
        (light_type, "/World") for light_type in SUPPORTED_LIGHT_TYPES
    ]


def test_panel_add_light_requires_edit_layer(monkeypatch):
    renderer = _Renderer()
    renderer._usd_edit_layer = None
    pane = _pane(monkeypatch, renderer)
    menu = pane.select(pn.widgets.MenuButton)[0]
    alert = pane.select(pn.pane.Alert)[0]

    assert menu.disabled is True
    menu.clicked = "SphereLight"
    assert renderer.added == []
    assert alert.visible is True
    assert "editable USD scene" in alert.object


def test_panel_surfaces_arbitrary_resync_failure(monkeypatch):
    renderer = _Renderer(OSError("synthetic loader failure"))
    pane = _pane(monkeypatch, renderer)
    menu = pane.select(pn.widgets.MenuButton)[0]
    alert = pane.select(pn.pane.Alert)[0]

    menu.clicked = "SphereLight"

    assert alert.visible is True
    assert alert.alert_type == "danger"
    assert "synthetic loader failure" in alert.object


# ── Material logical-input widgets route to the override path (finding A) ──


class _MatRenderer:
    scene_graph = None

    def __init__(self):
        self.overrides = []          # (index, {uniform: value})
        self.enabled = []            # (path, value)  — must stay empty for material props
        self.transforms = []         # (path, matrix) — must stay empty for material props
        self._usd_stage = None

    def apply_material_overrides(self, index, values):
        self.overrides.append((index, dict(values)))

    def apply_material_override(self, index, key, value):
        self.overrides.append((index, {key: value}))

    def apply_node_enabled(self, path, value):
        self.enabled.append((path, value))

    def apply_subtree_enabled(self, path, value):
        self.enabled.append((path, value))

    def set_transform(self, path, matrix):
        self.transforms.append((path, matrix))

    def apply_instance_transform(self, *a):
        self.transforms.append(a)


def _mat_node():
    from skinny.scene_graph import RendererRef, SceneGraphNode
    return SceneGraphNode(
        path="/Materials/M", name="M", type_name="Material",
        children=[], properties=[], renderer_ref=RendererRef("material", 3),
    )


def test_panel_material_bool_routes_to_override_not_enable():
    """A material logical-input bool carries fan-out uniforms; the panel must route
    it to apply_material_overrides, NOT the node-enable toggle (finding A(i))."""
    from skinny.scene_graph import SceneGraphProperty

    rend = _MatRenderer()
    session = SimpleNamespace(renderer=rend, _lock=Lock())
    node = _mat_node()
    prop = SceneGraphProperty(
        name="flag", display_name="flag", type_name="bool", value=False,
        editable=True, metadata={"fanout": ["u_flag"]},
    )
    w = windows._build_scene_prop_widget(session, node, prop)
    w.value = True
    assert rend.overrides == [(3, {"u_flag": True})]
    assert rend.enabled == []  # not misrouted to node-enable


def test_panel_material_vec3_routes_to_override_not_transform():
    """A material logical-input vector fans out to gen uniforms; the panel must
    route it to apply_material_overrides, NOT the TRS transform (finding A(i))."""
    from skinny.scene_graph import SceneGraphProperty

    rend = _MatRenderer()
    session = SimpleNamespace(renderer=rend, _lock=Lock())
    node = _mat_node()
    prop = SceneGraphProperty(
        name="dir", display_name="dir", type_name="vec3f", value=(0.0, 0.0, 0.0),
        editable=True, metadata={"fanout": ["u_dir"]},
    )
    node.properties = [prop]
    row = windows._build_scene_prop_widget(session, node, prop)
    row[0].value = 1.0  # nudge X → commits the whole vector
    assert rend.overrides and rend.overrides[-1][0] == 3
    assert set(rend.overrides[-1][1]) == {"u_dir"}
    assert rend.transforms == []  # not misrouted to a transform write
