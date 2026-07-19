"""Focused wiring checks for Panel scene-graph light creation."""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest

pn = pytest.importorskip("panel")

from skinny.ui.panel import windows  # noqa: E402
from skinny.ui.scene_edit_actions import SUPPORTED_LIGHT_TYPES  # noqa: E402


class _Renderer:
    scene_graph = None
    _scene_graph_version = 0
    _usd_stage = object()
    _usd_edit_layer = object()

    def __init__(self, failure: Exception | None = None):
        self.failure = failure
        self.added = []

    def add_light(self, light_type, parent_prim_path):
        if self.failure is not None:
            raise self.failure
        self.added.append((light_type, parent_prim_path))
        return f"{parent_prim_path}/{light_type}"


def _pane(monkeypatch, renderer):
    monkeypatch.setattr(
        pn.state, "add_periodic_callback", lambda *args, **kwargs: None,
    )
    session = SimpleNamespace(renderer=renderer, _lock=Lock())
    return windows.build_scene_graph_pane(session, lambda: None)


def test_panel_scene_graph_menu_routes_every_supported_light(monkeypatch):
    renderer = _Renderer()
    pane = _pane(monkeypatch, renderer)
    menu = pane.select(pn.widgets.MenuButton)[0]

    assert tuple(value for _label, value in menu.items) == SUPPORTED_LIGHT_TYPES
    for light_type in SUPPORTED_LIGHT_TYPES:
        menu.clicked = light_type

    assert renderer.added == [
        (light_type, "/World") for light_type in SUPPORTED_LIGHT_TYPES
    ]


def test_panel_add_light_requires_edit_layer(monkeypatch):
    renderer = _Renderer()
    renderer._usd_edit_layer = None
    pane = _pane(monkeypatch, renderer)
    menu = pane.select(pn.widgets.MenuButton)[0]
    alert = pane.select(pn.pane.Alert)[0]

    assert menu.disabled is True
    menu.clicked = "SphereLight"
    assert renderer.added == []
    assert alert.visible is True
    assert "editable USD scene" in alert.object


def test_panel_surfaces_arbitrary_resync_failure(monkeypatch):
    renderer = _Renderer(OSError("synthetic loader failure"))
    pane = _pane(monkeypatch, renderer)
    menu = pane.select(pn.widgets.MenuButton)[0]
    alert = pane.select(pn.pane.Alert)[0]

    menu.clicked = "SphereLight"

    assert alert.visible is True
    assert alert.alert_type == "danger"
    assert "synthetic loader failure" in alert.object
