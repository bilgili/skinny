"""Tool-handler tests: fake renderer, real queue, run manually.

Same pattern as tests/test_qt_render_session.py — no GPU, no Qt, no network.
The queue is run on a worker thread so the handlers' awaited reads behave the
way they will against a live render loop.
"""

from __future__ import annotations

import threading
import time

import pytest

from skinny.mcp_server import SceneToolError, SceneTools
from skinny.render_session import RenderCommandQueue
from skinny.scene_graph import RendererRef, SceneGraphNode, SceneGraphProperty


class FakeRenderer:
    def __init__(self, graph) -> None:
        self.scene_graph = graph
        self._material_version = 7
        self._scene_graph_version = 3
        self._usd_stage = None
        self.calls: list[tuple] = []

    def apply_material_override(self, index, key, value):
        self.calls.append(("material", index, key, value))
        self._material_version += 1

    def apply_light_override(self, light_type, index, key, value):
        self.calls.append(("light", light_type, index, key, value))
        self._material_version += 1

    def apply_instance_transform(self, path, t, r, s):
        self.calls.append(("instance_transform", path, t, r, s))
        self._material_version += 1

    def apply_node_enabled(self, path, value):
        self.calls.append(("node_enabled", path, value))


class Proxy:
    """Minimal stand-in for QtRendererProxy's request() surface."""

    def __init__(self, queue) -> None:
        self._commands = queue

    def request(self, callback):
        return self._commands.post_with_reply(callback)


def _prop(name, type_name, value, *, editable=True, **metadata):
    return SceneGraphProperty(
        name=name, display_name=name, type_name=type_name, value=value,
        editable=editable, metadata=metadata,
    )


def _node(path, type_name="Mesh", *, ref=None, properties=None, children=None):
    return SceneGraphNode(
        path=path, name=path.rstrip("/").split("/")[-1] or "/", type_name=type_name,
        children=children or [], properties=properties or [], renderer_ref=ref,
    )


def _scene():
    shader = _node("/World/mat/surface", "Shader", properties=[
        _prop("roughness", "float", 0.5, min=0.04, max=1.0),
        _prop("metalness", "float", 0.0, min=0.0, max=1.0),
    ])
    material = _node("/World/mat", "Material",
                     ref=RendererRef(kind="material", index=2), children=[shader])
    light = _node("/World/sun", "DistantLight",
                  ref=RendererRef(kind="light_dir", index=0),
                  properties=[_prop("intensity", "float", 1.0, min=0.0, max=10.0,
                                    growable=True)])
    box = _node("/World/box", "Mesh", ref=RendererRef(kind="instance", index=1),
                properties=[
                    _prop("translate", "vec3f", (0.0, 0.0, 0.0)),
                    _prop("rotate", "vec3f", (0.0, 90.0, 0.0)),
                    _prop("scale", "vec3f", (2.0, 2.0, 2.0)),
                ])
    world = _node("/World", "Xform", children=[material, light, box])
    return _node("/", "Stage", children=[world])


class Harness:
    """Runs the queue on a worker thread for the duration of a test."""

    def __init__(self) -> None:
        self.queue = RenderCommandQueue()
        self.renderer = FakeRenderer(_scene())
        self.tools = SceneTools(Proxy(self.queue), timeout=2.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.queue.run_pending(self.renderer)
            time.sleep(0.001)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.close()


# ── scene_list ───────────────────────────────────────────────────────

def test_scene_list_carries_no_property_payload(h) -> None:
    result = h.tools.scene_list("/", depth=5)
    text = repr(result)
    assert "roughness" not in text
    assert "props" not in text


def test_scene_list_respects_depth(h) -> None:
    shallow = h.tools.scene_list("/", depth=1)
    world = shallow["root"]["children"][0]
    assert "children" not in world  # depth exhausted


def test_scene_list_kind_filter_returns_only_lights(h) -> None:
    result = h.tools.scene_list("/", kind="light_dir")
    assert [n["path"] for n in result["nodes"]] == ["/World/sun"]


def test_scene_list_unknown_path_errors(h) -> None:
    with pytest.raises(SceneToolError, match="no such path"):
        h.tools.scene_list("/nope")


# ── scene_get ────────────────────────────────────────────────────────

def test_scene_get_surfaces_editable_and_bounds(h) -> None:
    node = h.tools.scene_get("/World/mat/surface")["node"]
    rough = next(p for p in node["props"] if p["name"] == "roughness")
    assert rough["editable"] is True
    assert rough["meta"]["min"] == 0.04 and rough["meta"]["max"] == 1.0


def test_scene_get_omits_children(h) -> None:
    assert "children" not in h.tools.scene_get("/World/mat")


# ── scene_set ────────────────────────────────────────────────────────

def test_material_param_on_shader_prim_applies(h) -> None:
    """The case a kind-based dispatch would have failed outright."""
    result = h.tools.scene_set("/World/mat/surface", "roughness", 0.25)

    assert result["applied"]["value"] == 0.25
    assert h.renderer.calls == [("material", 2, "roughness", 0.25)]


def test_transform_component_recomposes_from_siblings(h) -> None:
    h.tools.scene_set("/World/box", "translate", (5.0, 0.0, 0.0))

    assert h.renderer.calls == [
        ("instance_transform", "/World/box",
         (5.0, 0.0, 0.0), (0.0, 90.0, 0.0), (2.0, 2.0, 2.0)),
    ]


def test_out_of_bounds_write_is_rejected_not_clamped(h) -> None:
    with pytest.raises(SceneToolError, match="below its minimum"):
        h.tools.scene_set("/World/mat/surface", "roughness", 0.0)
    assert h.renderer.calls == []  # nothing applied


def test_growable_property_accepts_value_above_max(h) -> None:
    h.tools.scene_set("/World/sun", "intensity", 500.0)
    assert h.renderer.calls == [("light", "dir", 0, "intensity", 500.0)]


def test_unknown_property_errors_and_lists_available(h) -> None:
    with pytest.raises(SceneToolError) as excinfo:
        h.tools.scene_set("/World/mat/surface", "nope", 1.0)
    assert "roughness" in str(excinfo.value)


def test_unknown_path_errors(h) -> None:
    with pytest.raises(SceneToolError, match="no such path"):
        h.tools.scene_set("/World/ghost", "roughness", 0.5)


def test_non_editable_property_errors(h) -> None:
    h.renderer.scene_graph.children[0].properties.append(
        _prop("locked", "float", 1.0, editable=False),
    )
    with pytest.raises(SceneToolError, match="not editable"):
        h.tools.scene_set("/World", "locked", 2.0)


# ── Versions ─────────────────────────────────────────────────────────

def test_every_result_carries_both_versions(h) -> None:
    for result in (
        h.tools.scene_list("/"),
        h.tools.scene_get("/World/mat"),
        h.tools.scene_set("/World/mat/surface", "roughness", 0.3),
    ):
        assert "scene_graph_version" in result
        assert "material_version" in result


def test_property_edit_moves_material_version(h) -> None:
    """The structural counter does NOT move on a property edit -- by design."""
    before = h.tools.scene_get("/World/mat")
    h.tools.scene_set("/World/mat/surface", "roughness", 0.3)
    after = h.tools.scene_get("/World/mat")

    assert after["material_version"] != before["material_version"]
    assert after["scene_graph_version"] == before["scene_graph_version"]


def test_structural_change_moves_scene_graph_version(h) -> None:
    before = h.tools.scene_get("/World/mat")
    h.renderer._scene_graph_version += 1
    after = h.tools.scene_get("/World/mat")

    assert after["scene_graph_version"] != before["scene_graph_version"]


# ── Timeout ──────────────────────────────────────────────────────────

def test_stalled_render_thread_reports_a_timeout() -> None:
    """A wedged render thread must surface as an error, not a hang."""
    queue = RenderCommandQueue()  # never run
    tools = SceneTools(Proxy(queue), timeout=0.05)

    with pytest.raises(SceneToolError, match="did not respond"):
        tools.scene_list("/")


def test_server_module_has_no_direct_renderer_access() -> None:
    """Every renderer touch must be inside a posted closure."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "skinny" / "mcp_server.py"
    ).read_text()
    # The handlers name their closure parameter `renderer`; a bare
    # `self._proxy._renderer` or similar would be the violation.
    assert "_renderer" not in source
    assert ".renderer." not in source
