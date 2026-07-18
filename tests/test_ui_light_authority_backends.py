"""Backend checks for runtime fallback-control section visibility."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from skinny.params import STATIC_PARAMS, _set_nested
from skinny.playback import PlaybackClock
from skinny.ui.build_app_ui import build_main_ui
from tests.test_ui_spec import _StubRenderer


def _backend_renderer() -> _StubRenderer:
    renderer = _StubRenderer()
    renderer.clock = PlaybackClock(has_animation=False)
    renderer.has_usd_camera = False
    renderer._usd_controls = []
    renderer.film = SimpleNamespace(iso=100.0, exposure_time=1.0)
    for param in STATIC_PARAMS:
        if param.path.startswith("mtlx."):
            continue
        value = float(param.lo) if param.kind == "continuous" else 0
        _set_nested(renderer, param.path, value)
        if (
            param.kind == "discrete"
            and param.choice_source is not None
            and not hasattr(renderer, param.choice_source)
        ):
            setattr(
                renderer,
                param.choice_source,
                [SimpleNamespace(name="Default")],
            )
    return renderer


def test_panel_removes_fallback_section_headings_and_bodies():
    pytest.importorskip("panel")
    from skinny.ui.panel.backend import PanelTreeBuilder

    renderer = _backend_renderer()
    builder = PanelTreeBuilder(build_main_ui(renderer))
    assert "IBL" in builder.layout._names
    assert "Direct Light" in builder.layout._names

    renderer.uses_default_lights = False
    builder._tick()
    assert "IBL" not in builder.layout._names
    assert "Direct Light" not in builder.layout._names

    renderer.uses_default_lights = True
    builder._tick()
    assert "IBL" in builder.layout._names
    assert "Direct Light" in builder.layout._names


def test_qt_removes_fallback_section_headings_and_bodies():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget
    from skinny.ui.qt.backend import QtTreeBuilder

    app = QApplication.instance() or QApplication([])
    renderer = _backend_renderer()
    parent = QWidget()
    builder = QtTreeBuilder(build_main_ui(renderer), parent)
    try:
        assert not builder._sections["IBL"].isHidden()
        assert not builder._sections["Direct Light"].isHidden()

        renderer.uses_default_lights = False
        builder._tick()
        assert builder._sections["IBL"].isHidden()
        assert builder._sections["Direct Light"].isHidden()

        renderer.uses_default_lights = True
        builder._tick()
        assert not builder._sections["IBL"].isHidden()
        assert not builder._sections["Direct Light"].isHidden()
    finally:
        builder._timer.stop()
        parent.deleteLater()
        app.processEvents()
