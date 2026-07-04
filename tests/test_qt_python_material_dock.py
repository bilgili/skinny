"""Render-thread-safety guard for the Python Material Editor dock.

Under render-thread ownership the dock talks to the `QtRendererProxy`, never the
live renderer: the module-list read and the `MaterialReloader` reload both run on
the render worker via `renderer.request(...)`, and their results are marshalled
back through Qt signals. Source-level so it needs no QApplication, render thread,
or GPU (mirrors `test_qt_gizmo_viewport.py`).
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from skinny.ui.qt.windows import python_material_editor as pme  # noqa: E402

Dock = pme.PythonMaterialEditorDock


def test_constructor_takes_no_render_lock() -> None:
    params = inspect.signature(Dock.__init__).parameters
    assert "render_lock" not in params
    assert list(params) == ["self", "renderer", "parent"]


def test_dock_defines_cross_thread_result_signals() -> None:
    # PySide6 Signal objects are class attributes; presence is enough here.
    assert hasattr(Dock, "_modules_ready")
    assert hasattr(Dock, "_reload_ready")


def test_module_list_is_read_on_the_worker_not_the_gui_thread() -> None:
    src = inspect.getsource(pme)
    # The only module-list reads go through a worker request; a direct
    # GUI-thread `self.renderer.scene_python_modules()` must not survive.
    assert "self.renderer.scene_python_modules()" not in src
    assert "self.renderer.request(lambda r: r.scene_python_modules())" in src


def test_reload_runs_on_the_worker() -> None:
    compile_src = inspect.getsource(Dock._on_compile_clicked)
    assert "self.renderer.request(" in compile_src
    assert "MaterialReloader(r)" in compile_src
    # No synchronous GUI-thread reloader call remains.
    assert "self._reloader.reload" not in inspect.getsource(pme)
