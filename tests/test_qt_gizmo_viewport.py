"""Parity guard: the Qt render viewport must let the user grab the gizmo, not
just draw it. Asserts each mouse handler delegates to the GUI-agnostic gizmo
controller, so a future refactor can't silently regress to a display-only gizmo.

Source-level so it needs no QApplication, render thread, or GPU — only that
PySide6 imports (the module pulls it in at load).
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from skinny.ui.qt.viewport import RenderViewport  # noqa: E402


def test_press_delegates_to_gizmo_controller():
    assert "_gizmo.on_press" in inspect.getsource(RenderViewport.mousePressEvent)


def test_move_delegates_to_gizmo_controller():
    assert "_gizmo.on_move" in inspect.getsource(RenderViewport.mouseMoveEvent)


def test_release_delegates_to_gizmo_controller():
    assert "_gizmo.on_release" in inspect.getsource(RenderViewport.mouseReleaseEvent)


def test_space_key_cycles_gizmo_mode():
    src = inspect.getsource(RenderViewport.keyPressEvent)
    assert "gizmo_cycle_mode" in src
    assert "Qt.Key_Space" in src
