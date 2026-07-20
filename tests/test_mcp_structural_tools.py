"""Fake-renderer tests for the mcp-scene-structure tools: scene_add_model,
scene_add_primitive, scene_add_light, scene_remove, scene_save,
scene_job_status. Same harness pattern as test_mcp_tools.py -- fake renderer,
real command queue, run on a worker thread.
"""

from __future__ import annotations

import threading
import time

import pytest

from skinny.mcp_paths import resolve_roots
from skinny.mcp_server import SceneToolError, SceneTools
from skinny.render_session import RenderCommandQueue
from skinny.scene_graph import RendererRef, SceneGraphNode, SceneGraphProperty


class FakeStage:
    def __init__(self, layers=None) -> None:
        self._layers = layers or []

    def GetUsedLayers(self):
        return self._layers


class FakeRenderer:
    def __init__(self, graph, *, has_stage: bool = True, slow: bool = False) -> None:
        self.scene_graph = graph
        self._material_version = 7
        self._scene_graph_version = 3
        self._usd_stage = FakeStage() if has_stage else None
        self._usd_edit_layer = object() if has_stage else None
        self._slow = slow
        self.calls: list[tuple] = []

    def _maybe_delay(self) -> None:
        if self._slow:
            time.sleep(0.15)

    def add_model(self, usd_path, parent_prim_path="/World", name=None,
                  transform=None, validate=None):
        self._maybe_delay()
        self.calls.append(("add_model", usd_path, parent_prim_path, name,
                            transform is not None, validate is not None))
        return f"{parent_prim_path}/{name or 'Model'}"

    def add_primitive(self, prim_type, parent_prim_path="/World", name=None,
                       transform=None, color=None, roughness=None, metallic=None):
        self._maybe_delay()
        self.calls.append(("add_primitive", prim_type, parent_prim_path, name,
                            color, roughness, metallic))
        return f"{parent_prim_path}/{name or prim_type}"

    def add_light(self, light_type, parent_prim_path="/World", name=None,
                  transform=None, intensity=None, color=None):
        self._maybe_delay()
        self.calls.append(("add_light", light_type, parent_prim_path, name,
                            intensity, color))
        return f"{parent_prim_path}/{name or light_type}"

    def remove_node(self, path):
        self._maybe_delay()
        self.calls.append(("remove_node", path))

    def save_edits(self, path=None):
        self._maybe_delay()
        self.calls.append(("save_edits", path))
        return path


class Proxy:
    def __init__(self, queue) -> None:
        self._commands = queue

    def request(self, callback):
        return self._commands.post_with_reply(callback)


def _prop(name, type_name, value, *, editable=True, **metadata):
    return SceneGraphProperty(
        name=name, display_name=name, type_name=type_name, value=value,
        editable=editable, metadata=metadata,
    )


def _node(path, type_name="Xform", *, ref=None, properties=None, children=None):
    return SceneGraphNode(
        path=path, name=path.rstrip("/").split("/")[-1] or "/", type_name=type_name,
        children=children or [], properties=properties or [], renderer_ref=ref,
    )


def _scene():
    box = _node("/World/box", "Mesh", ref=RendererRef(kind="instance", index=1))
    synth = _node("/Skinny/DefaultLight", "DistantLight",
                  ref=RendererRef(kind="light_dir", index=0))
    world = _node("/World", "Xform", children=[box])
    skinny = _node("/Skinny", "Xform", children=[synth])
    return _node("/", "Stage", children=[world, skinny])


class Harness:
    def __init__(self, *, has_stage: bool = True, slow: bool = False, **tools_kwargs) -> None:
        self.queue = RenderCommandQueue()
        self.renderer = FakeRenderer(_scene(), has_stage=has_stage, slow=slow)
        self.tools = SceneTools(Proxy(self.queue), timeout=2.0, **tools_kwargs)
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


@pytest.fixture()
def roots(tmp_path):
    return resolve_roots(str(tmp_path), None)


@pytest.fixture()
def h(roots):
    harness = Harness(roots=roots)
    yield harness
    harness.close()


# ── scene_add_model ──────────────────────────────────────────────────

def test_add_model_happy_path(h, tmp_path) -> None:
    usd_path = tmp_path / "model.usda"
    usd_path.write_text("#usda 1.0\n")
    result = h.tools.scene_add_model(str(usd_path), name="Foo")

    assert result["status"] == "done"
    assert result["path"] == "/World/Foo"
    call = h.renderer.calls[0]
    assert call[0] == "add_model" and call[1] == str(usd_path) and call[5] is True  # validate given


def test_add_model_rejects_path_outside_roots(h) -> None:
    with pytest.raises(SceneToolError, match="outside the allowed roots"):
        h.tools.scene_add_model("/definitely/not/in/roots.usda")
    assert h.renderer.calls == []


def test_add_model_requires_editable_stage(roots) -> None:
    h = Harness(has_stage=False, roots=roots)
    try:
        with pytest.raises(SceneToolError, match="no editable USD stage is loaded"):
            h.tools.scene_add_model(str(next(iter(roots))) + "/x.usda")
    finally:
        h.close()


def test_add_model_translate_and_matrix_both_given_errors(h, tmp_path) -> None:
    usd_path = tmp_path / "model.usda"
    usd_path.write_text("#usda 1.0\n")
    with pytest.raises(SceneToolError, match="not both"):
        h.tools.scene_add_model(
            str(usd_path), translate=(1, 2, 3), matrix=[1, 0, 0, 0] * 4,
        )
    assert h.renderer.calls == []


def test_add_model_bad_matrix_length_errors(h, tmp_path) -> None:
    usd_path = tmp_path / "model.usda"
    usd_path.write_text("#usda 1.0\n")
    with pytest.raises(SceneToolError, match="16 numbers"):
        h.tools.scene_add_model(str(usd_path), matrix=[1, 2, 3])
    assert h.renderer.calls == []


# ── scene_add_primitive ──────────────────────────────────────────────

def test_add_primitive_happy_path(h) -> None:
    result = h.tools.scene_add_primitive("Sphere", color=(1.0, 0.0, 0.0), name="Ball")

    assert result["status"] == "done"
    assert result["path"] == "/World/Ball"
    assert h.renderer.calls == [
        ("add_primitive", "Sphere", "/World", "Ball", (1.0, 0.0, 0.0), None, None),
    ]


def test_add_primitive_bad_color_errors(h) -> None:
    with pytest.raises(SceneToolError, match="3 numbers"):
        h.tools.scene_add_primitive("Sphere", color=(1.0, 0.0))
    assert h.renderer.calls == []


def test_add_primitive_uniform_scale_scalar(h) -> None:
    result = h.tools.scene_add_primitive("Cube", scale=2.0)
    assert result["status"] == "done"


# ── scene_add_light ──────────────────────────────────────────────────

def test_add_light_happy_path(h) -> None:
    result = h.tools.scene_add_light("SphereLight", intensity=250.0, color=(1, 0.5, 0.2))

    assert result["status"] == "done"
    assert h.renderer.calls == [
        ("add_light", "SphereLight", "/World", None, 250.0, (1.0, 0.5, 0.2)),
    ]


# ── scene_remove ─────────────────────────────────────────────────────

def test_remove_happy_path(h) -> None:
    result = h.tools.scene_remove("/World/box")
    assert result["status"] == "done"
    assert h.renderer.calls == [("remove_node", "/World/box")]


def test_remove_unknown_path_is_not_found(h) -> None:
    with pytest.raises(SceneToolError, match="no such node"):
        h.tools.scene_remove("/World/ghost")
    assert h.renderer.calls == []


def test_remove_synthesized_node_is_not_deletable(h) -> None:
    with pytest.raises(SceneToolError, match="not deletable"):
        h.tools.scene_remove("/Skinny/DefaultLight")
    assert h.renderer.calls == []


def test_remove_root_is_not_deletable(h) -> None:
    with pytest.raises(SceneToolError, match="not deletable"):
        h.tools.scene_remove("/")
    assert h.renderer.calls == []


# ── scene_save ───────────────────────────────────────────────────────

def test_save_happy_path(h, tmp_path) -> None:
    dest = tmp_path / "out.usda"
    result = h.tools.scene_save(str(dest))
    assert result["status"] == "done"
    assert result["path"] == str(dest)
    assert h.renderer.calls == [("save_edits", str(dest))]


def test_save_requires_explicit_path(h) -> None:
    with pytest.raises(SceneToolError, match="requires an explicit path"):
        h.tools.scene_save("")
    assert h.renderer.calls == []


def test_save_rejects_path_outside_roots(h) -> None:
    with pytest.raises(SceneToolError, match="outside the allowed roots"):
        h.tools.scene_save("/not/in/roots/out.usda")
    assert h.renderer.calls == []


# ── job pattern ──────────────────────────────────────────────────────

def test_fast_structural_call_returns_done_inline(roots) -> None:
    h = Harness(roots=roots, structural_grace=1.0)
    try:
        result = h.tools.scene_add_primitive("Sphere")
        assert result["status"] == "done"
        assert "job_id" not in result
    finally:
        h.close()


def test_slow_structural_call_returns_pending_then_done(roots) -> None:
    h = Harness(roots=roots, slow=True, structural_grace=0.02)
    try:
        result = h.tools.scene_add_primitive("Sphere", name="Slow")
        assert result["status"] == "pending"
        job_id = result["job_id"]

        deadline = time.monotonic() + 2.0
        status = h.tools.scene_job_status(job_id)
        while status["status"] == "pending" and time.monotonic() < deadline:
            time.sleep(0.01)
            status = h.tools.scene_job_status(job_id)

        assert status["status"] == "done"
        assert status["path"] == "/World/Slow"
    finally:
        h.close()


def test_job_status_never_blocks_while_pending(roots) -> None:
    """A poll on a still-pending job must return immediately, not wait."""
    h = Harness(roots=roots, slow=True, structural_grace=0.02)
    try:
        result = h.tools.scene_add_primitive("Sphere")
        job_id = result["job_id"]

        start = time.monotonic()
        status = h.tools.scene_job_status(job_id)
        elapsed = time.monotonic() - start

        assert status["status"] == "pending"
        assert elapsed < 0.05, "scene_job_status must not block on a pending future"
    finally:
        h.close()


def test_job_status_reports_failure(roots) -> None:
    h = Harness(roots=roots, structural_grace=0.02)

    def slow_then_failing_add_primitive(*args, **kwargs):
        time.sleep(0.15)  # outlast the grace period, then fail
        raise ValueError("synthetic failure")

    h.renderer.add_primitive = slow_then_failing_add_primitive
    try:
        result = h.tools.scene_add_primitive("Sphere")
        job_id = result["job_id"]

        deadline = time.monotonic() + 2.0
        status = h.tools.scene_job_status(job_id)
        while status["status"] == "pending" and time.monotonic() < deadline:
            time.sleep(0.01)
            status = h.tools.scene_job_status(job_id)

        assert status["status"] == "failed"
        assert "synthetic failure" in status["error"]
    finally:
        h.close()


def test_job_status_unknown_id_errors(h) -> None:
    with pytest.raises(SceneToolError, match="no such job"):
        h.tools.scene_job_status("does-not-exist")


# ── asset-typed scene_set root check ────────────────────────────────

def test_scene_set_rejects_texture_path_outside_roots(roots) -> None:
    """The root check must fire before dispatch reaches the renderer at all --
    apply_dome_light_texture is deliberately left undefined on the fake so an
    AttributeError would surface loudly if the check didn't fire first."""
    h = Harness(roots=roots)
    try:
        prop = _prop("texture_file", "texture_file", "")
        node = _node("/World/dome", "DomeLight",
                      ref=RendererRef(kind="light_env", index=0), properties=[prop])
        h.renderer.scene_graph.children[0].children.append(node)  # nest under /World

        with pytest.raises(SceneToolError, match="outside the allowed roots"):
            h.tools.scene_set(node.path, "texture_file", "/not/in/roots/tex.hdr")
    finally:
        h.close()
