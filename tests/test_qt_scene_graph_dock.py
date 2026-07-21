"""Render-thread-safety guard for the Scene Graph dock (Phase 2b).

The dock reads renderer state from the proxy's worker-refreshed cache and posts
every mutation to the render worker; the edits whose result it reports
(add/save/delete/texture/lens) resolve asynchronously, never blocking the GUI
thread. Source-level, mirroring `test_qt_gizmo_viewport.py`.
"""
from __future__ import annotations

from concurrent.futures import Future
import inspect

import pytest
from PySide6.QtWidgets import QApplication

pytest.importorskip("PySide6")

from skinny.ui.scene_edit_actions import SUPPORTED_LIGHT_TYPES  # noqa: E402
from skinny.ui.qt.windows import scene_graph as sg  # noqa: E402

Dock = sg.SceneGraphDock


def test_tick_refreshes_scene_state_on_the_worker() -> None:
    tick = inspect.getsource(Dock._tick)
    assert "self.renderer.refresh_scene_state()" in tick
    assert "add_done_callback" in tick


def test_result_edits_do_not_block_the_gui_thread() -> None:
    src = inspect.getsource(sg)
    # Every return-value edit is awaited off-thread via `_await`.
    for method in (
        "r.add_model(", "r.add_light(", "r.save_edits()", "self.renderer.remove_node(",
        "self.renderer.apply_camera_lens_file(",
        "self.renderer.apply_dome_light_texture(",
    ):
        assert method in src, method
    assert src.count("self._await(") >= 6
    # Every `.result()` must run inside a worker-side future done-callback
    # (`_await` and `_on_scene_state_future`), never on the GUI thread.
    worker_cbs = (
        inspect.getsource(Dock._await)
        + inspect.getsource(Dock._on_scene_state_future)
    )
    assert worker_cbs.count(".result()") == 2
    assert src.count(".result()") == 2


def test_dock_defines_the_gui_marshaller_signal() -> None:
    assert hasattr(Dock, "_run_on_gui")


class _LightRendererStub:
    scene_graph = None
    _usd_stage = object()
    _usd_edit_layer = object()
    # `has_editable_stage` now also gates on adopted scene metadata (finding #4).
    _usd_scene = object()
    _scene_graph_id = 0
    _scene_graph_version = 0

    def __init__(self):
        self.added = []

    def add_light(self, light_type, parent_prim_path):
        self.added.append((light_type, parent_prim_path))
        fut = Future()
        fut.set_result(f"{parent_prim_path}/{light_type}")
        return fut

    def refresh_scene_state(self):
        fut = Future()
        fut.set_result(None)
        return fut


def test_add_light_menu_offers_and_routes_every_supported_type() -> None:
    app = QApplication.instance() or QApplication([])
    renderer = _LightRendererStub()
    dock = Dock(renderer)
    try:
        actions = dock._add_light_btn.menu().actions()
        assert tuple(action.data() for action in actions) == SUPPORTED_LIGHT_TYPES
        for action in actions:
            action.trigger()
        app.processEvents()
        assert renderer.added == [
            (light_type, "/World") for light_type in SUPPORTED_LIGHT_TYPES
        ]
    finally:
        dock._timer.stop()
        dock.deleteLater()
        app.processEvents()


def test_add_light_requires_edit_layer_for_enablement_and_dispatch() -> None:
    app = QApplication.instance() or QApplication([])
    renderer = _LightRendererStub()
    renderer._usd_edit_layer = None
    dock = Dock(renderer)
    try:
        assert dock._add_light_btn.isEnabled() is False
        dock._on_add_light("SphereLight")
        app.processEvents()
        assert renderer.added == []

        renderer._usd_edit_layer = object()
        dock._apply_scene_state_tick(None)
        assert dock._add_light_btn.isEnabled() is True

        renderer._usd_edit_layer = None
        dock._apply_scene_state_tick(None)
        assert dock._add_light_btn.isEnabled() is False
    finally:
        dock._timer.stop()
        dock.deleteLater()
        app.processEvents()
