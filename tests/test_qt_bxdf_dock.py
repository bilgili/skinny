"""Render-thread-safety guard for the BXDF Visualizer dock (Phase 3).

The lobe/BSSRDF evaluation is GPU work: it runs on the render worker and its grid
is marshalled to the GUI thread, never evaluated inline. The scene-pick callback
(invoked on the worker by `request_scene_pick`) and the material-state poll are
likewise marshalled. Source-level, mirroring `test_qt_gizmo_viewport.py`.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from skinny.ui.qt.windows import bxdf as bx  # noqa: E402

Dock = bx.BXDFDock


def test_dock_defines_marshaller_signals() -> None:
    assert hasattr(Dock, "_run_on_gui")
    assert hasattr(Dock, "_eval_ready")


def test_eval_dispatches_to_the_worker() -> None:
    src = inspect.getsource(Dock._do_eval)
    assert "self.renderer.request_bxdf_eval(" in src
    assert "self.renderer.request_bssrdf_eval(" in src
    # The grid is delivered via a worker callback that emits `_eval_ready`,
    # never rendered inline on the GUI thread inside _do_eval.
    assert "self._eval_ready.emit(" in src
    assert "on_error=cpu_fallback" in src


def test_scene_pick_result_is_marshalled_to_gui() -> None:
    src = inspect.getsource(Dock._arm_pick)
    assert "self._run_on_gui.emit(" in src
    assert "self.viewport.arm_scene_pick(worker_cb)" in src


def test_material_poll_refreshes_state_on_the_worker() -> None:
    poll = inspect.getsource(Dock._poll_material_changes)
    assert "self.renderer.refresh_scene_state()" in poll
    assert "add_done_callback" in poll
    # The old GUI-thread `id(_usd_scene)` swap check is replaced by the stable
    # projected id.
    apply = inspect.getsource(Dock._apply_state_poll)
    assert '"_usd_scene_id"' in apply
