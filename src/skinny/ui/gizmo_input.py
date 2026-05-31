"""GUI-agnostic rotate-gizmo mouse interaction.

Kept free of any widget-toolkit import so the press/move/release decision logic
can be unit-tested with a stub renderer — no ``QApplication``, no display. A
render-surface front-end maps its widget coordinates to render pixels, holds its
render lock, and delegates to :class:`GizmoMouseController`; centralising the
logic here means a front-end that can target the gizmo also lets the user grab it
(parity), and the precedence relative to the modal viewport modes lives in one
tested place.

The ``renderer`` argument is duck-typed: it must expose
``gizmo_hit_test(x, y) -> axis | None``, ``gizmo_begin_drag(axis, x, y) -> bool``,
``gizmo_update_drag(x, y)``, ``gizmo_end_drag()``, and a ``gizmo`` object with
``set_hover(axis | None)``. Coordinates are render pixels (the caller maps from
widget space and passes ``None`` when the cursor is outside the rendered image).
"""

from __future__ import annotations

from typing import Optional

# Action tags returned by the press dispatch, in precedence order.
PRESS_AUTOFOCUS = "autofocus"
PRESS_PICK = "pick"
PRESS_ZOOM = "zoom"
PRESS_GIZMO = "gizmo"
PRESS_CAMERA = "camera"

Pixel = Optional[tuple[float, float]]


def gizmo_press_action(
    *, shift: bool, pick_armed: bool, zoom_arming: bool, ring_hit: bool,
) -> str:
    """Which interaction a left-press triggers, in fixed precedence:
    shift-autofocus → scene-pick → zoom-rect → gizmo → camera.

    The gizmo only wins when no modal mode is armed and a ring was hit; anything
    else falls through to camera control.
    """
    if shift:
        return PRESS_AUTOFOCUS
    if pick_armed:
        return PRESS_PICK
    if zoom_arming:
        return PRESS_ZOOM
    if ring_hit:
        return PRESS_GIZMO
    return PRESS_CAMERA


class GizmoMouseController:
    """Tracks an in-progress gizmo drag and drives the renderer gizmo API.

    The Qt viewport (and any other render-surface front-end) delegates its
    press/move/release handling here so the gizmo is never display-only.
    """

    def __init__(self) -> None:
        self._dragging = False

    @property
    def is_dragging(self) -> bool:
        return self._dragging

    def on_press(
        self, renderer, mapped: Pixel,
        *, shift: bool, pick_armed: bool, zoom_arming: bool,
    ) -> str:
        """Resolve a left-press. Returns the :func:`gizmo_press_action` verdict.

        On ``"gizmo"`` a drag is begun; if ``gizmo_begin_drag`` reports failure
        (e.g. the target raced away) the press degrades to ``"camera"``. The
        gizmo is hit-tested only when no modal mode is armed, mirroring the
        precedence — so a shift/pick/zoom press never disturbs gizmo state.
        """
        axis = None
        if not (shift or pick_armed or zoom_arming) and mapped is not None:
            axis = renderer.gizmo_hit_test(mapped[0], mapped[1])
        action = gizmo_press_action(
            shift=shift, pick_armed=pick_armed, zoom_arming=zoom_arming,
            ring_hit=axis is not None,
        )
        if action == PRESS_GIZMO:
            if renderer.gizmo_begin_drag(axis, mapped[0], mapped[1]):
                self._dragging = True
            else:
                return PRESS_CAMERA
        return action

    def on_move(
        self, renderer, mapped: Pixel,
        *, any_button_down: bool, zoom_dragging: bool,
    ) -> bool:
        """Handle a mouse move. Returns ``True`` when the gizmo consumed it (a
        drag is in progress), so the caller skips camera control.

        While idle (no button held, no zoom drag) the ring under the cursor is
        highlighted via ``set_hover``; a ``None`` mapping clears the highlight.
        """
        if self._dragging:
            if mapped is not None:
                renderer.gizmo_update_drag(mapped[0], mapped[1])
            return True
        if not any_button_down and not zoom_dragging:
            axis = (
                renderer.gizmo_hit_test(mapped[0], mapped[1])
                if mapped is not None else None
            )
            renderer.gizmo.set_hover(axis)
        return False

    def on_release(self, renderer) -> bool:
        """End an in-progress drag. Returns ``True`` if a drag was ended."""
        if self._dragging:
            renderer.gizmo_end_drag()
            self._dragging = False
            return True
        return False
