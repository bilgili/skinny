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


# ── Camera Debug key-map reconciliation (change ui-spec-scene-properties, 1.4) ──
#
# Two divergences were found across the four camera/viewport key maps and each
# is resolved here — one fixed, one recorded — and asserted so it stays that way.


def test_qt_debug_dock_binds_escape_to_close():
    """FIXED divergence: the Qt Camera Debug dock had no Escape binding while the
    GLFW debug viewport closes on Escape. The dock now closes on Escape too.
    """
    txt = _read("ui/qt/windows/debug_viewport.py")
    assert "Qt.Key_Escape" in txt
    assert "self.close()" in txt


def test_web_debug_surface_is_button_only_recorded_gap():
    """RECORDED divergence: the web (Panel) debug surface is button-only
    (Top / Left / Back / reset) with no free-camera keyboard or mouse. The
    browser viewport carries no gizmo or free-camera verb, so this is a
    deliberate gap, not a missing binding — asserted so it stays intentional.
    """
    txt = _read("ui/panel/windows.py")
    assert '"Top"' in txt and '"Left"' in txt and '"Back"' in txt
    # No Qt key handler on the browser surface (Panel has no keyPressEvent).
    assert "keyPressEvent" not in txt
