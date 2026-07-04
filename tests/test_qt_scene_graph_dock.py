"""Render-thread-safety guard for the Scene Graph dock (Phase 2b).

The dock reads renderer state from the proxy's worker-refreshed cache and posts
every mutation to the render worker; the edits whose result it reports
(add/save/delete/texture/lens) resolve asynchronously, never blocking the GUI
thread. Source-level, mirroring `test_qt_gizmo_viewport.py`.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

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
        "r.add_model(", "r.save_edits()", "self.renderer.remove_node(",
        "self.renderer.apply_camera_lens_file(",
        "self.renderer.apply_dome_light_texture(",
    ):
        assert method in src, method
    assert src.count("self._await(") >= 5
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
