"""Render-thread-safety guard for the Camera Debug viewport dock (Phase 5).

The DebugViewport GPU object lives on the render worker (renderer.debug_viewport);
the worker renders it each frame and emits a DebugFrame. The dock is passive — it
blits the emitted frames and posts camera/display input + lifecycle to the worker,
never touching the GPU object directly. Source-level, mirroring
`test_qt_gizmo_viewport.py` — the actual GPU render needs a real context + GPU.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from skinny.ui.qt.windows import debug_viewport as dv  # noqa: E402

Dock = dv.DebugViewportDock


def test_dock_takes_proxy_and_viewport_not_ctx_or_lock() -> None:
    params = inspect.signature(Dock.__init__).parameters
    assert list(params) == ["self", "renderer", "viewport", "parent"]
    assert "ctx" not in params
    assert "main_lock" not in params


def test_worker_helpers_are_module_level() -> None:
    for fn in ("_worker_debug_create", "_worker_debug_resize",
               "_worker_debug_destroy", "_worker_debug_drag",
               "_worker_debug_wheel", "_worker_debug_move",
               "_worker_debug_call", "_worker_debug_toggle"):
        assert callable(getattr(dv, fn)), fn


def test_input_is_posted_to_the_worker() -> None:
    for method in ("_on_drag", "_on_wheel", "_poll_wasd", "_post_call",
                   "_post_toggle"):
        body = inspect.getsource(getattr(Dock, method))
        assert "self.renderer.post(" in body, method


def test_lifecycle_posts_to_the_worker() -> None:
    for method in ("showEvent", "hideEvent", "resizeEvent", "closeEvent"):
        body = inspect.getsource(getattr(Dock, method))
        assert "self.renderer.post(" in body, method


def test_dock_blits_worker_frames_no_gpu_render_on_gui() -> None:
    src = inspect.getsource(dv)
    # The dock consumes worker-emitted frames; it must not call render_embedded
    # itself (that runs on the worker in viewport.py).
    assert "self.viewport.debug_frame_ready.connect" in src
    assert "render_embedded" not in src


def test_worker_renders_debug_frame_in_viewport_loop() -> None:
    from skinny.ui.qt import viewport as vp
    src = inspect.getsource(vp._RenderWorker._maybe_render_debug)
    assert "render_embedded" in src
    assert "self.debug_frame_ready.emit(" in src
    assert "_debug_viewport_active" in src
