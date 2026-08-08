"""Behavioural + coverage tests for the shared scene-property / graph-input
node mapper (change ui-spec-scene-properties).

Toolkit-free: the mapper emits ``ui.spec`` nodes and never touches Qt, Panel,
or the renderer, so a plain recording ``commit`` is enough to assert behaviour.
This is the safety net that Task 1.1 asks for before the docks migrate onto the
seam — it pins *what control each type needs* and *what value an edit commits*,
which the source-substring dock tests never checked.
"""

from __future__ import annotations

import inspect

import pytest

from skinny.scene_graph import SceneGraphProperty
from skinny.ui import spec
from skinny.ui.scene_property_nodes import (
    EDITABLE_GRAPH_INPUT_TYPES,
    EDITABLE_SCENE_PROP_TYPES,
    RENDERABLE_NODE_TYPES,
    graph_input_to_node,
    scene_property_to_node,
)


# ── Test doubles ────────────────────────────────────────────────────


class _Port:
    """Minimal ``PortView`` stand-in — the mapper reads only these three."""

    def __init__(self, name, type_name, value=None):
        self.name = name
        self.type_name = type_name
        self.value = value


class _Recorder:
    """Records ``commit(target, value)`` calls."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, target, value):
        self.calls.append((target, value))

    @property
    def last_value(self):
        return self.calls[-1][1]


def _prop(type_name, value, *, editable=True, metadata=None):
    return SceneGraphProperty(
        name=type_name, display_name=type_name.upper(), type_name=type_name,
        value=value, editable=editable, metadata=metadata or {},
    )


# ── 2.3 — the coverage guard ────────────────────────────────────────


def test_both_backends_dispatch_every_renderable_node_type():
    """Every node class either adapter can emit MUST be dispatched by BOTH
    backends. This is what makes "a type handled in one front-end but not the
    other fails the build" structural: add a node type here, forget one
    backend, and this fails.
    """
    from skinny.ui.panel.backend import PanelTreeBuilder
    from skinny.ui.qt.backend import QtTreeBuilder

    qt_src = inspect.getsource(QtTreeBuilder._build_node)
    panel_src = inspect.getsource(PanelTreeBuilder._build)
    for cls in RENDERABLE_NODE_TYPES:
        token = f"spec.{cls.__name__}"
        assert token in qt_src, f"Qt backend does not dispatch {token}"
        assert token in panel_src, f"Panel backend does not dispatch {token}"


def _one_of_each_renderable():
    """A minimal instance of every node class the adapters can emit."""
    return [
        spec.Checkbox("b", getter=lambda: False, setter=lambda v: None),
        spec.Slider("f", getter=lambda: 0.0, setter=lambda v: None, lo=0.0, hi=1.0),
        spec.Color("c", getter=lambda: (0.0, 0.0, 0.0), setter=lambda v: None),
        spec.Vector("v", 3, getter=lambda: (0.0, 0.0, 0.0), setter=lambda v: None),
        spec.IntSpin("i", getter=lambda: 0, setter=lambda v: None, lo=0, hi=8),
        spec.FilePicker("p", filters=[("a", "*.*")], on_pick=lambda p: None),
        spec.Label("l", text=lambda: "x"),
    ]


def test_both_backends_actually_render_every_renderable_node_type():
    """Stronger than the source-substring guard: instantiate one node of every
    type an adapter can emit and confirm each backend produces a real widget /
    viewable — a handler that returned None (or a token that only appears in a
    comment) would pass the substring check but fail here.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QWidget

    from skinny.ui.panel.backend import PanelTreeBuilder
    from skinny.ui.qt.backend import QtTreeBuilder

    nodes = _one_of_each_renderable()
    assert {type(n) for n in nodes} == set(RENDERABLE_NODE_TYPES)  # keep in sync

    app = QApplication.instance() or QApplication([])
    for node in nodes:
        host = QWidget()
        builder = QtTreeBuilder(spec.Section(title="", children=[node]), host)
        try:
            widgets = host.findChildren(QWidget)
            assert widgets, f"Qt backend rendered no widget for {type(node).__name__}"
            for w in widgets:
                assert "unknown" not in (w.__class__.__name__.lower())
        finally:
            builder.stop()

        leaf = PanelTreeBuilder(spec.Section(title="")).render_leaf(node)
        assert leaf is not None, (
            f"Panel backend rendered nothing for {type(node).__name__}"
        )
    app.processEvents()


def test_every_editable_scene_prop_type_yields_an_editable_control():
    """No editable scene-property type may fall through to a read-only Label."""
    samples = {
        "bool": True, "float": 0.5, "color3f": (1.0, 0.0, 0.0),
        "vec3f": (1.0, 2.0, 3.0), "vec2f": (1.0, 2.0), "int": 3,
        "lens_file": "/l.usda", "texture_file": "/t.hdr",
    }
    assert set(samples) == EDITABLE_SCENE_PROP_TYPES  # keep the table honest
    for t, v in samples.items():
        node = scene_property_to_node(_prop(t, v), commit=_Recorder())
        assert not isinstance(node, spec.Label), f"{t} degraded to Label"
        assert type(node) in RENDERABLE_NODE_TYPES


def test_every_editable_graph_input_type_yields_an_editable_control():
    samples = {
        "float": 0.5, "color3": (1.0, 0.0, 0.0), "vector3": (1.0, 2.0, 3.0),
        "vector2": (1.0, 2.0), "integer": 3, "boolean": True, "filename": "/f",
    }
    assert set(samples) == EDITABLE_GRAPH_INPUT_TYPES
    for t, v in samples.items():
        node = graph_input_to_node(_Port(t, t, v), commit=_Recorder())
        assert not isinstance(node, spec.Label), f"{t} degraded to Label"
        assert type(node) in RENDERABLE_NODE_TYPES


# ── 1.1 — behavioural: node type + committed value per prop type ────


def test_bool_prop_commits_coerced_bool():
    rec = _Recorder()
    node = scene_property_to_node(_prop("bool", 1), commit=rec)
    assert isinstance(node, spec.Checkbox)
    node.setter(0)
    assert rec.last_value is False


def test_float_prop_range_from_metadata_and_commits_float():
    rec = _Recorder()
    node = scene_property_to_node(
        _prop("float", 2.0, metadata={"min": 1.0, "max": 8.0}), commit=rec)
    assert isinstance(node, spec.Slider)
    assert (node.lo, node.hi) == (1.0, 8.0)
    node.setter("3.5")
    assert rec.last_value == 3.5


def test_growable_float_uses_wide_upper_bound():
    node = scene_property_to_node(
        _prop("float", 1.0, metadata={"min": 0.0, "growable": True}),
        commit=_Recorder())
    assert node.hi >= 1.0e9


def test_color_prop_commits_rgb_tuple():
    rec = _Recorder()
    node = scene_property_to_node(_prop("color3f", (0.1, 0.2, 0.3)), commit=rec)
    assert isinstance(node, spec.Color)
    assert node.getter() == (0.1, 0.2, 0.3)
    node.setter((0.4, 0.5, 0.6))
    assert rec.last_value == (0.4, 0.5, 0.6)


@pytest.mark.parametrize("t,n", [("vec3f", 3), ("vec2f", 2)])
def test_vector_prop_components_range_and_step(t, n):
    rec = _Recorder()
    val = tuple(float(i) for i in range(n))
    node = scene_property_to_node(_prop(t, val), commit=rec)
    assert isinstance(node, spec.Vector) and node.components == n
    assert node.lo < 0 and node.hi > 0 and node.step > 0  # transform-scale span
    node.setter(val)
    assert rec.last_value == val


def test_int_prop_range_default_and_commits_int():
    rec = _Recorder()
    node = scene_property_to_node(_prop("int", 2), commit=rec)
    assert isinstance(node, spec.IntSpin)
    assert node.lo < 0 and node.hi > 0
    node.setter(5.9)
    assert rec.last_value == 5


@pytest.mark.parametrize("t", ["lens_file", "texture_file"])
def test_file_prop_commits_path_string(t):
    from pathlib import Path
    rec = _Recorder()
    node = scene_property_to_node(_prop(t, ""), commit=rec)
    assert isinstance(node, spec.FilePicker)
    node.on_pick(Path("/some/file.ext"))
    assert rec.last_value == "/some/file.ext"


# ── 1.1 — read-only + live-value behaviour ──────────────────────────


def test_non_editable_prop_is_a_readonly_label():
    node = scene_property_to_node(
        _prop("float", 3.0, editable=False), commit=_Recorder())
    assert isinstance(node, spec.Label)
    assert node.text() == "3.0000"


@pytest.mark.parametrize("t,v,expect", [
    ("rel", "/World/Mat", "→ /World/Mat"),
    ("vec3f", (1.0, 2.0, 3.0), "(1.000, 2.000, 3.000)"),
    ("asset", "tex.png", "tex.png"),
    ("token", "srgb", "srgb"),
])
def test_readonly_formatting_matches_qt(t, v, expect):
    node = scene_property_to_node(_prop(t, v, editable=False), commit=_Recorder())
    assert isinstance(node, spec.Label)
    assert node.text() == expect


def test_get_live_overrides_snapshot_for_editable_control():
    """A camera scalar the user drives updates without a graph rebuild: the
    getter reads live, not the frozen ``prop.value``."""
    prop = _prop("float", 1.0, metadata={"min": 0.0, "max": 10.0})
    live = {"v": 7.0}
    node = scene_property_to_node(
        prop, commit=_Recorder(), get_live=lambda p: live["v"])
    assert node.getter() == 7.0
    live["v"] = 9.0
    assert node.getter() == 9.0  # re-read each pull, not captured once


def test_snapshot_used_when_no_get_live():
    prop = _prop("float", 1.0, metadata={"min": 0.0, "max": 10.0})
    node = scene_property_to_node(prop, commit=_Recorder())
    assert node.getter() == 1.0
