"""Hostless tests for the Scene-Graph proxy surface (Phase 2a).

The Scene Graph dock reads renderer-owned state (the scene-graph node tree, the
scene-graph version, the authored materials, USD stage/edit-layer presence, and
live camera params) and mutates the renderer through ~13 `apply_*` methods. Under
render-thread ownership those reads come from a worker-built `SceneStateSnapshot`
applied to the `QtRendererProxy`, and the writes are posted to the render worker.
"""
from __future__ import annotations

from skinny.ui.qt.render_session import (
    QtRendererProxy,
    RenderCommandQueue,
    build_scene_state,
)


def _proxy(queue: RenderCommandQueue) -> QtRendererProxy:
    return QtRendererProxy(
        queue, width=64, height=64, backend="metal", encoding="E0",
        sppm_glossy_roughness=None,
    )


class _Cam:
    fov = 50.0
    near = 0.1
    far = 100.0
    fstop = 2.8
    focus_distance = 3.0
    yaw = 10.0
    pitch = -5.0
    distance = 4.0
    max_distance = 50.0
    lens = None


class _Mat:
    python_module = "python_materials.foo"


class _Scene:
    materials = [_Mat()]


class _FakeRenderer:
    scene_graph = object()
    _scene_graph_version = 7
    _usd_scene = _Scene()
    scene = _Scene()
    _usd_stage = object()
    _usd_edit_layer = object()
    camera = _Cam()


def test_build_scene_state_projects_renderer_fields() -> None:
    state = build_scene_state(_FakeRenderer())

    assert state.scene_graph_version == 7
    assert state.has_usd_stage is True
    assert state.has_usd_edit_layer is True
    assert state.camera.fov == 50.0
    assert state.camera.distance == 4.0
    assert state.camera.lens is None
    assert state.usd_scene.materials[0].python_module == "python_materials.foo"


def test_apply_scene_state_exposes_reads_the_dock_uses() -> None:
    queue = RenderCommandQueue()
    proxy = _proxy(queue)
    state = build_scene_state(_FakeRenderer())

    proxy.apply_scene_state(state)

    assert proxy._scene_graph_version == 7
    assert proxy.scene_graph is not None
    assert proxy._usd_stage is not None
    assert proxy._usd_edit_layer is not None
    assert proxy.camera.fov == 50.0
    assert proxy._usd_scene.materials[0].python_module == "python_materials.foo"


def test_missing_stage_projects_as_absent() -> None:
    class _NoStage:
        scene_graph = None
        _scene_graph_version = 0
        _usd_scene = None
        scene = None
        _usd_stage = None
        _usd_edit_layer = None
        camera = None

    queue = RenderCommandQueue()
    proxy = _proxy(queue)
    proxy.apply_scene_state(build_scene_state(_NoStage()))

    assert proxy._usd_stage is None
    assert proxy._usd_edit_layer is None
    assert proxy.camera is None


def test_apply_material_override_posts_coalesced_command() -> None:
    queue = RenderCommandQueue()
    proxy = _proxy(queue)
    calls: list[tuple] = []

    class Target:
        def apply_material_override(self, index, name, value) -> None:
            calls.append((index, name, value))

    proxy.apply_material_override(2, "base_color", (1.0, 0.0, 0.0))
    proxy.apply_material_override(2, "base_color", (0.0, 1.0, 0.0))  # coalesces

    commands = queue.drain()
    assert len(commands) == 1  # last-write-wins on the same (idx,name)
    commands[0].callback(Target())
    assert calls == [(2, "base_color", (0.0, 1.0, 0.0))]


def test_add_model_returns_future_resolved_by_worker() -> None:
    queue = RenderCommandQueue()
    proxy = _proxy(queue)

    class Target:
        def add_model(self, path, parent_prim_path=None) -> str:
            return f"{parent_prim_path}/{path}"

    fut = proxy.add_model("dragon.usda", parent_prim_path="/World")
    command = queue.drain()[0]
    command.reply.set_result(command.callback(Target()))

    assert fut.result(timeout=0) == "/World/dragon.usda"
