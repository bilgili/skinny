"""Unit tests for the GUI-agnostic rotate-gizmo mouse interaction.

Pure: no Vulkan, no display, no QApplication. A stub renderer records the
gizmo API calls so the press/move/release precedence and drag lifecycle can be
asserted directly.
"""

from __future__ import annotations

from skinny.ui.gizmo_input import (
    GizmoMouseController,
    PRESS_AUTOFOCUS,
    PRESS_CAMERA,
    PRESS_GIZMO,
    PRESS_PICK,
    PRESS_ZOOM,
    gizmo_press_action,
)


class _StubGizmo:
    def __init__(self) -> None:
        self.hover = "<unset>"

    def set_hover(self, axis) -> None:
        self.hover = axis


class _StubRenderer:
    """Records gizmo_* calls. ``hit_axis`` is what hit-test returns; set
    ``begin_ok=False`` to simulate a begin_drag that fails the race."""

    def __init__(self, hit_axis=None, begin_ok=True) -> None:
        self._hit_axis = hit_axis
        self._begin_ok = begin_ok
        self.gizmo = _StubGizmo()
        self.calls: list[tuple] = []

    def gizmo_hit_test(self, x, y):
        self.calls.append(("hit_test", x, y))
        return self._hit_axis

    def gizmo_begin_drag(self, axis, x, y):
        self.calls.append(("begin_drag", axis, x, y))
        return self._begin_ok

    def gizmo_update_drag(self, x, y):
        self.calls.append(("update_drag", x, y))
        return True

    def gizmo_end_drag(self):
        self.calls.append(("end_drag",))

    def _names(self) -> list[str]:
        return [c[0] for c in self.calls]


class TestGizmoPressAction:
    def test_precedence_order(self):
        # Each armed mode wins over everything below it, even with a ring hit.
        assert gizmo_press_action(
            shift=True, pick_armed=True, zoom_arming=True, ring_hit=True,
        ) == PRESS_AUTOFOCUS
        assert gizmo_press_action(
            shift=False, pick_armed=True, zoom_arming=True, ring_hit=True,
        ) == PRESS_PICK
        assert gizmo_press_action(
            shift=False, pick_armed=False, zoom_arming=True, ring_hit=True,
        ) == PRESS_ZOOM

    def test_gizmo_wins_over_camera_on_ring_hit(self):
        assert gizmo_press_action(
            shift=False, pick_armed=False, zoom_arming=False, ring_hit=True,
        ) == PRESS_GIZMO

    def test_camera_on_miss(self):
        assert gizmo_press_action(
            shift=False, pick_armed=False, zoom_arming=False, ring_hit=False,
        ) == PRESS_CAMERA


class TestControllerPress:
    def test_ring_hit_begins_drag(self):
        r = _StubRenderer(hit_axis="x")
        c = GizmoMouseController()
        action = c.on_press(
            r, (10.0, 20.0), shift=False, pick_armed=False, zoom_arming=False,
        )
        assert action == PRESS_GIZMO
        assert c.is_dragging
        assert ("begin_drag", "x", 10.0, 20.0) in r.calls

    def test_miss_is_camera_and_no_begin(self):
        r = _StubRenderer(hit_axis=None)
        c = GizmoMouseController()
        action = c.on_press(
            r, (10.0, 20.0), shift=False, pick_armed=False, zoom_arming=False,
        )
        assert action == PRESS_CAMERA
        assert not c.is_dragging
        assert "begin_drag" not in r._names()

    def test_begin_drag_failure_degrades_to_camera(self):
        r = _StubRenderer(hit_axis="y", begin_ok=False)
        c = GizmoMouseController()
        action = c.on_press(
            r, (1.0, 2.0), shift=False, pick_armed=False, zoom_arming=False,
        )
        assert action == PRESS_CAMERA
        assert not c.is_dragging

    def test_modal_modes_never_touch_gizmo(self):
        for kwargs in (
            dict(shift=True, pick_armed=False, zoom_arming=False),
            dict(shift=False, pick_armed=True, zoom_arming=False),
            dict(shift=False, pick_armed=False, zoom_arming=True),
        ):
            r = _StubRenderer(hit_axis="x")
            c = GizmoMouseController()
            action = c.on_press(r, (5.0, 5.0), **kwargs)
            assert action in (PRESS_AUTOFOCUS, PRESS_PICK, PRESS_ZOOM)
            assert r.calls == []  # no hit-test, no begin_drag
            assert not c.is_dragging

    def test_mapped_none_is_camera(self):
        r = _StubRenderer(hit_axis="x")
        c = GizmoMouseController()
        action = c.on_press(
            r, None, shift=False, pick_armed=False, zoom_arming=False,
        )
        assert action == PRESS_CAMERA
        assert r.calls == []


class TestControllerMove:
    def test_drag_updates_and_consumes(self):
        r = _StubRenderer(hit_axis="x")
        c = GizmoMouseController()
        c.on_press(r, (0.0, 0.0), shift=False, pick_armed=False, zoom_arming=False)
        r.calls.clear()
        consumed = c.on_move(
            r, (3.0, 4.0), any_button_down=False, zoom_dragging=False,
        )
        assert consumed is True
        assert ("update_drag", 3.0, 4.0) in r.calls
        assert "set_hover" not in r._names()

    def test_idle_move_sets_hover(self):
        r = _StubRenderer(hit_axis="z")
        c = GizmoMouseController()
        consumed = c.on_move(
            r, (7.0, 8.0), any_button_down=False, zoom_dragging=False,
        )
        assert consumed is False
        assert r.gizmo.hover == "z"

    def test_idle_move_none_clears_hover(self):
        r = _StubRenderer(hit_axis="z")
        c = GizmoMouseController()
        c.on_move(r, None, any_button_down=False, zoom_dragging=False)
        assert r.gizmo.hover is None

    def test_button_down_suppresses_hover(self):
        r = _StubRenderer(hit_axis="z")
        c = GizmoMouseController()
        c.on_move(r, (1.0, 1.0), any_button_down=True, zoom_dragging=False)
        assert r.gizmo.hover == "<unset>"  # set_hover never called

    def test_zoom_drag_suppresses_hover(self):
        r = _StubRenderer(hit_axis="z")
        c = GizmoMouseController()
        c.on_move(r, (1.0, 1.0), any_button_down=False, zoom_dragging=True)
        assert r.gizmo.hover == "<unset>"


class TestControllerRelease:
    def test_release_ends_active_drag(self):
        r = _StubRenderer(hit_axis="x")
        c = GizmoMouseController()
        c.on_press(r, (0.0, 0.0), shift=False, pick_armed=False, zoom_arming=False)
        ended = c.on_release(r)
        assert ended is True
        assert not c.is_dragging
        assert ("end_drag",) in r.calls

    def test_release_without_drag_is_noop(self):
        r = _StubRenderer(hit_axis=None)
        c = GizmoMouseController()
        ended = c.on_release(r)
        assert ended is False
        assert "end_drag" not in r._names()
