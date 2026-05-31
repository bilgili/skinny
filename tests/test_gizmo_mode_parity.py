"""Parity guard: every render-surface front-end binds the space key to the
transform-gizmo mode-cycle (not just draws the gizmo), and F1 keeps the HUD
toggle. Source-level so it needs no window, render thread, or GPU — a refactor
that drops the space binding from a front-end fails here.

The debug viewport is intentionally absent: it is a visualization-only window
(AABBs, grid, frustum, camera glyphs) that cannot target the transform gizmo,
so the space mode-cycle does not apply to it.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _read(rel: str) -> str:
    return (_SRC / rel).read_text(encoding="utf-8")


def test_glfw_space_cycles_gizmo_mode():
    txt = _read("app.py")
    assert "glfw.KEY_SPACE" in txt
    assert "gizmo_cycle_mode" in txt


def test_qt_space_cycles_gizmo_mode():
    txt = _read("ui/qt/viewport.py")
    assert "Qt.Key_Space" in txt
    assert "gizmo_cycle_mode" in txt


def test_f1_still_toggles_hud_in_both_frontends():
    assert "glfw.KEY_F1" in _read("app.py")
    assert "Qt.Key_F1" in _read("ui/qt/viewport.py")
    assert "show_hud" in _read("app.py")
    assert "show_hud" in _read("ui/qt/viewport.py")
