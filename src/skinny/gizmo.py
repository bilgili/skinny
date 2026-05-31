"""Transform gizmo — CPU-side hit-testing and drag math.

A `TransformGizmo` tracks a single selected mesh instance and exposes one of
four manipulators around the instance's pivot:

- ``ROTATE_WORLD`` / ``ROTATE_LOCAL`` — three orthogonal rings.
- ``TRANSLATE_WORLD`` / ``TRANSLATE_LOCAL`` — three axis arrows.

The *world* variants align to the canonical X/Y/Z world axes; the *local*
variants align to the targeted instance's current orientation. The active mode
is a single ordered enum cycled by the space key (``(index + 1) % 4``, grouped
by type). A ``W``/``L`` glyph is drawn above the pivot to hint the coordinate
space; the gizmo shape (rings vs arrows) hints the type.

The renderer rebuilds the line list each frame from ``build_segments`` and
uploads it to GPU binding 22; ``main_pass.slang`` draws each segment as an
anti-aliased line over the final tonemapped image.

Coordinate conventions:
- Pixel coordinates use a top-left origin (0..width-1, 0..height-1) to
  match the GPU's ``RWTexture2D`` indexing.
- View / proj matrices are stored row-vector / math-transposed (same
  convention as renderer.py's `_perspective` and `_look_at`), so a clip-
  space coordinate is computed as ``world_h @ view @ proj``.
- Instance transforms are row-vector convention (see scene_graph.py): a
  local direction ``d`` maps to world as ``d @ M[:3, :3]``, so the world-space
  directions of the instance's local X/Y/Z axes are the (normalized) rows of
  the upper 3x3 — this is the gizmo's *local basis*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


GIZMO_SEGMENT_CAPACITY = 256        # ≥ 3 axes × 64 segments + glyph + arrow stubs
GIZMO_SEGMENT_STRIDE = 32           # 2×float2 + float3 + float = 8 floats
GIZMO_RING_SEGMENTS = 64            # discretization per axis ring
GIZMO_RADIUS_FRACTION = 0.18        # ring radius / arrow length as fraction of viewport min dim
GIZMO_LINE_HALF_WIDTH = 1.5         # pixels
GIZMO_HIT_TOLERANCE = 6.0           # pixels — generous so the handle is easy to grab

# Translate-arrow head, in screen pixels.
GIZMO_ARROWHEAD_PX = 11.0
GIZMO_ARROWHEAD_HALF_ANGLE = math.radians(24.0)

# World/local glyph hint, in screen pixels, anchored above the projected pivot.
GIZMO_GLYPH_OFFSET_PX = 26.0        # gap between the pivot and the glyph's base
GIZMO_GLYPH_WIDTH_PX = 13.0
GIZMO_GLYPH_HEIGHT_PX = 17.0
GIZMO_GLYPH_COLOR = (0.92, 0.92, 0.92)

AXIS_COLORS = {
    "x": (0.95, 0.30, 0.30),
    "y": (0.30, 0.85, 0.30),
    "z": (0.30, 0.55, 0.95),
}
AXIS_HIGHLIGHT = {
    "x": (1.0, 0.85, 0.20),
    "y": (1.0, 0.85, 0.20),
    "z": (1.0, 0.85, 0.20),
}

_AXES = ("x", "y", "z")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class GizmoMode(IntEnum):
    """The four transform-gizmo modes, ordered grouped by type so a space
    press flips world↔local first, then switches the type."""

    ROTATE_WORLD = 0
    ROTATE_LOCAL = 1
    TRANSLATE_WORLD = 2
    TRANSLATE_LOCAL = 3


def is_translate(mode: GizmoMode) -> bool:
    return int(mode) >= int(GizmoMode.TRANSLATE_WORLD)


def is_local(mode: GizmoMode) -> bool:
    return int(mode) % 2 == 1


@dataclass
class GizmoSegment:
    ax: float
    ay: float
    bx: float
    by: float
    r: float
    g: float
    b: float
    width: float


def _project_to_pixel(
    p_world: np.ndarray, view: np.ndarray, proj: np.ndarray,
    width: int, height: int,
) -> Optional[tuple[float, float, float]]:
    """Project a world-space point to pixel space. Returns ``(px, py, depth)``
    or ``None`` if the point is behind the near plane.
    """
    h = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    clip = h @ view.astype(np.float64) @ proj.astype(np.float64)
    w = float(clip[3])
    if w < 1e-4:
        return None
    ndc_x = clip[0] / w
    ndc_y = clip[1] / w
    px = (ndc_x * 0.5 + 0.5) * float(width)
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * float(height)
    return px, py, w


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Column-vector rotation matrix about a (non-zero) world-space axis."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / n
    c = math.cos(theta)
    s = math.sin(theta)
    k = np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * k + (1.0 - c) * (k @ k)


class TransformGizmo:
    """Per-instance transform gizmo: rotate rings or translate arrows, in
    world or local space, around a pivot."""

    def __init__(self) -> None:
        self.target_index: int = -1
        self.pivot_world: np.ndarray = np.zeros(3, dtype=np.float32)
        self.mode: GizmoMode = GizmoMode.ROTATE_WORLD
        # Local basis: rows are the world-space directions of the instance's
        # local X/Y/Z axes (identity until a target with a rotation is set).
        self.target_basis: np.ndarray = np.eye(3, dtype=np.float64)

        # Active drag state
        self._drag_axis: Optional[str] = None
        self._drag_start_screen_angle: float = 0.0
        self._drag_start_mouse: tuple[float, float] = (0.0, 0.0)
        self._drag_start_translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._drag_start_rotate: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._drag_start_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._drag_start_pivot: np.ndarray = np.zeros(3, dtype=np.float64)
        self._drag_basis: np.ndarray = np.eye(3, dtype=np.float64)

        # Hover for visual highlight (set during hit-test before drag)
        self._hover_axis: Optional[str] = None

    # ── Selection ────────────────────────────────────────────────

    def set_target(
        self, index: int, pivot_world: np.ndarray,
        basis: Optional[np.ndarray] = None,
    ) -> None:
        if index < 0:
            self.clear_target()
            return
        self.target_index = int(index)
        self.pivot_world = np.asarray(pivot_world, dtype=np.float32).reshape(3)
        if basis is not None:
            self.target_basis = np.asarray(basis, dtype=np.float64).reshape(3, 3)

    def clear_target(self) -> None:
        self.target_index = -1
        self._drag_axis = None
        self._hover_axis = None

    @property
    def has_target(self) -> bool:
        return self.target_index >= 0

    @property
    def is_dragging(self) -> bool:
        return self._drag_axis is not None

    # ── Mode ─────────────────────────────────────────────────────

    def cycle_mode(self) -> GizmoMode:
        """Advance to the next mode (``(index + 1) % 4``). No-op while a drag
        is in progress, so a stray space press can't change manipulators
        mid-edit."""
        if self.is_dragging:
            return self.mode
        self.mode = GizmoMode((int(self.mode) + 1) % 4)
        return self.mode

    # ── Basis ────────────────────────────────────────────────────

    def _axis_dirs(self, *, drag: bool = False) -> np.ndarray:
        """3x3 of unit world-space directions for axes X/Y/Z in the active
        space: identity (world modes) or the (frozen, during a drag) local
        basis (local modes)."""
        if not is_local(self.mode):
            return np.eye(3, dtype=np.float64)
        b = self._drag_basis if drag else self.target_basis
        out = np.array(b, dtype=np.float64).reshape(3, 3).copy()
        for i in range(3):
            n = float(np.linalg.norm(out[i]))
            out[i] = out[i] / n if n > 1e-12 else np.eye(3)[i]
        return out

    def _axis_dir(self, axis: str, *, drag: bool = False) -> np.ndarray:
        return self._axis_dirs(drag=drag)[_AXIS_INDEX[axis]]

    def _ring_plane(self, axis: str) -> tuple[np.ndarray, np.ndarray]:
        """Two orthonormal in-plane basis vectors for ``axis``'s ring."""
        d = self._axis_dirs()
        if axis == "x":
            return d[1], d[2]
        if axis == "y":
            return d[0], d[2]
        return d[0], d[1]

    # ── Geometry ─────────────────────────────────────────────────

    def _handle_size_world(
        self, view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> float:
        """World-space radius / arrow length that projects to roughly
        ``GIZMO_RADIUS_FRACTION * min(width, height)`` pixels at the pivot."""
        pivot = self.pivot_world.astype(np.float64)
        center = _project_to_pixel(pivot, view, proj, width, height)
        if center is None:
            return 1.0
        target_px = GIZMO_RADIUS_FRACTION * min(width, height)
        offset = pivot + np.array([1.0, 0.0, 0.0], dtype=np.float64)
        proj_off = _project_to_pixel(offset, view, proj, width, height)
        if proj_off is None:
            return 1.0
        dx = proj_off[0] - center[0]
        dy = proj_off[1] - center[1]
        px_per_unit = math.hypot(dx, dy)
        if px_per_unit < 1e-3:
            return 1.0
        return float(target_px / px_per_unit)

    def _ring_pixels(
        self, axis: str, radius_world: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> list[tuple[float, float]]:
        u, v = self._ring_plane(axis)
        pivot = self.pivot_world.astype(np.float64)
        out: list[tuple[float, float]] = []
        for i in range(GIZMO_RING_SEGMENTS):
            theta = 2.0 * math.pi * i / GIZMO_RING_SEGMENTS
            p = pivot + radius_world * (math.cos(theta) * u + math.sin(theta) * v)
            proj_p = _project_to_pixel(p, view, proj, width, height)
            if proj_p is None:
                out.append((-1.0e6, -1.0e6))
            else:
                out.append((proj_p[0], proj_p[1]))
        return out

    def _arrow_segments(
        self, axis: str, length_world: float, color: tuple,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> list[GizmoSegment]:
        pivot = self.pivot_world.astype(np.float64)
        tip_w = pivot + length_world * self._axis_dir(axis)
        base_px = _project_to_pixel(pivot, view, proj, width, height)
        tip_px = _project_to_pixel(tip_w, view, proj, width, height)
        if base_px is None or tip_px is None:
            return []
        segs = [GizmoSegment(
            ax=base_px[0], ay=base_px[1], bx=tip_px[0], by=tip_px[1],
            r=color[0], g=color[1], b=color[2], width=GIZMO_LINE_HALF_WIDTH,
        )]
        dx = tip_px[0] - base_px[0]
        dy = tip_px[1] - base_px[1]
        dlen = math.hypot(dx, dy)
        if dlen > 1e-3:
            # Barbs point back from the tip, ±half-angle off the reversed shaft.
            ux, uy = -dx / dlen, -dy / dlen
            ca = math.cos(GIZMO_ARROWHEAD_HALF_ANGLE)
            sa = math.sin(GIZMO_ARROWHEAD_HALF_ANGLE)
            for s in (1.0, -1.0):
                bx = ux * ca - uy * (s * sa)
                by = ux * (s * sa) + uy * ca
                segs.append(GizmoSegment(
                    ax=tip_px[0], ay=tip_px[1],
                    bx=tip_px[0] + GIZMO_ARROWHEAD_PX * bx,
                    by=tip_px[1] + GIZMO_ARROWHEAD_PX * by,
                    r=color[0], g=color[1], b=color[2], width=GIZMO_LINE_HALF_WIDTH,
                ))
        return segs

    def _glyph_segments(
        self, letter: str,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> list[GizmoSegment]:
        center = _project_to_pixel(self.pivot_world, view, proj, width, height)
        if center is None:
            return []
        cx = center[0]
        bottom = center[1] - GIZMO_GLYPH_OFFSET_PX   # above the pivot (smaller y)
        top = bottom - GIZMO_GLYPH_HEIGHT_PX
        half_w = GIZMO_GLYPH_WIDTH_PX * 0.5
        left, right = cx - half_w, cx + half_w
        if letter == "L":
            pts = [(left, top), (left, bottom), (right, bottom)]
        else:  # "W"
            pts = [
                (left, top),
                (left + 0.25 * GIZMO_GLYPH_WIDTH_PX, bottom),
                (cx, top + 0.45 * GIZMO_GLYPH_HEIGHT_PX),
                (right - 0.25 * GIZMO_GLYPH_WIDTH_PX, bottom),
                (right, top),
            ]
        c = GIZMO_GLYPH_COLOR
        segs: list[GizmoSegment] = []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            segs.append(GizmoSegment(
                ax=a[0], ay=a[1], bx=b[0], by=b[1],
                r=c[0], g=c[1], b=c[2], width=GIZMO_LINE_HALF_WIDTH,
            ))
        return segs

    def build_segments(
        self, view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> list[GizmoSegment]:
        if not self.has_target:
            return []
        size = self._handle_size_world(view, proj, width, height)
        active = self._drag_axis or self._hover_axis
        segs: list[GizmoSegment] = []
        if is_translate(self.mode):
            for axis in _AXES:
                color = AXIS_HIGHLIGHT[axis] if axis == active else AXIS_COLORS[axis]
                segs.extend(self._arrow_segments(
                    axis, size, color, view, proj, width, height,
                ))
        else:
            for axis in _AXES:
                color = AXIS_HIGHLIGHT[axis] if axis == active else AXIS_COLORS[axis]
                pts = self._ring_pixels(axis, size, view, proj, width, height)
                for i in range(len(pts)):
                    a = pts[i]
                    b = pts[(i + 1) % len(pts)]
                    if a[0] < -1e5 or b[0] < -1e5:
                        continue
                    segs.append(GizmoSegment(
                        ax=a[0], ay=a[1], bx=b[0], by=b[1],
                        r=color[0], g=color[1], b=color[2],
                        width=GIZMO_LINE_HALF_WIDTH,
                    ))
        letter = "L" if is_local(self.mode) else "W"
        segs.extend(self._glyph_segments(letter, view, proj, width, height))
        return segs[:GIZMO_SEGMENT_CAPACITY]

    # ── Hit test ─────────────────────────────────────────────────

    def hit_test(
        self, mouse_x: float, mouse_y: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> Optional[str]:
        if not self.has_target:
            return None
        size = self._handle_size_world(view, proj, width, height)
        best_axis: Optional[str] = None
        best_dist = GIZMO_HIT_TOLERANCE
        if is_translate(self.mode):
            pivot = self.pivot_world.astype(np.float64)
            base_px = _project_to_pixel(pivot, view, proj, width, height)
            if base_px is None:
                return None
            for axis in _AXES:
                tip_w = pivot + size * self._axis_dir(axis)
                tip_px = _project_to_pixel(tip_w, view, proj, width, height)
                if tip_px is None:
                    continue
                d = _point_to_segment_distance(
                    mouse_x, mouse_y, base_px[0], base_px[1], tip_px[0], tip_px[1],
                )
                if d < best_dist:
                    best_dist = d
                    best_axis = axis
            return best_axis
        for axis in _AXES:
            pts = self._ring_pixels(axis, size, view, proj, width, height)
            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]
                if a[0] < -1e5 or b[0] < -1e5:
                    continue
                d = _point_to_segment_distance(
                    mouse_x, mouse_y, a[0], a[1], b[0], b[1],
                )
                if d < best_dist:
                    best_dist = d
                    best_axis = axis
        return best_axis

    def set_hover(self, axis: Optional[str]) -> None:
        self._hover_axis = axis

    # ── Drag ─────────────────────────────────────────────────────

    def begin_drag(
        self, axis: str,
        mouse_x: float, mouse_y: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
        instance_transform: np.ndarray,
    ) -> None:
        from skinny.scene_graph import decompose_trs_matrix
        self._drag_axis = axis
        t, r, s = decompose_trs_matrix(instance_transform)
        self._drag_start_translate = t
        self._drag_start_rotate = r
        self._drag_start_scale = s
        self._drag_start_mouse = (mouse_x, mouse_y)
        # Freeze the pivot and basis at drag start: translation moves the pivot
        # and a local rotation reorients the basis, so a live read mid-drag
        # would create a feedback loop.
        self._drag_start_pivot = self.pivot_world.astype(np.float64).copy()
        self._drag_basis = np.array(self.target_basis, dtype=np.float64).copy()
        self._drag_start_screen_angle = self._screen_angle(
            mouse_x, mouse_y, self._drag_start_pivot, view, proj, width, height,
        )

    def update_drag(
        self,
        mouse_x: float, mouse_y: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> Optional[tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]]:
        if self._drag_axis is None:
            return None
        if is_translate(self.mode):
            return self._update_translate(mouse_x, mouse_y, view, proj, width, height)
        return self._update_rotate(mouse_x, mouse_y, view, proj, width, height)

    def _update_rotate(
        self, mouse_x: float, mouse_y: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ):
        cur = self._screen_angle(
            mouse_x, mouse_y, self._drag_start_pivot, view, proj, width, height,
        )
        delta = cur - self._drag_start_screen_angle
        # Wrap into (-π, π] so a tiny drag near the seam doesn't snap by 2π.
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        # Delta rotation about the world-space axis vector n (canonical for
        # world modes, the frozen local axis for local modes).
        n = self._axis_dir(self._drag_axis, drag=True)
        euler = self._compose_rotation(self._drag_start_rotate, n, delta)
        return self._drag_start_translate, euler, self._drag_start_scale

    def _compose_rotation(
        self, start_rotate: tuple[float, float, float],
        n: np.ndarray, delta: float,
    ) -> tuple[float, float, float]:
        from skinny.scene_graph import compose_trs_matrix, decompose_trs_matrix
        r_row = compose_trs_matrix(
            (0.0, 0.0, 0.0), start_rotate, (1.0, 1.0, 1.0),
        )[:3, :3].astype(np.float64)
        # Row-vector rotation → column-vector form, premultiply the world-space
        # delta (D · R), then back to row-vector for decomposition.
        rc = r_row.T
        rc_new = _rodrigues(n, delta) @ rc
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = rc_new.T
        _, euler, _ = decompose_trs_matrix(m)
        return euler

    def _update_translate(
        self, mouse_x: float, mouse_y: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ):
        base = self._drag_start_pivot
        d = self._axis_dir(self._drag_axis, drag=True)
        base_px = _project_to_pixel(base, view, proj, width, height)
        off_px = _project_to_pixel(base + d, view, proj, width, height)
        if base_px is None or off_px is None:
            return self._drag_start_translate, self._drag_start_rotate, self._drag_start_scale
        sdx = off_px[0] - base_px[0]
        sdy = off_px[1] - base_px[1]
        px_per_world = math.hypot(sdx, sdy)
        if px_per_world < 1e-4:
            return self._drag_start_translate, self._drag_start_rotate, self._drag_start_scale
        ux, uy = sdx / px_per_world, sdy / px_per_world
        along_px = (mouse_x - self._drag_start_mouse[0]) * ux + (
            mouse_y - self._drag_start_mouse[1]
        ) * uy
        world_units = along_px / px_per_world
        t0 = self._drag_start_translate
        t_new = (
            float(t0[0] + d[0] * world_units),
            float(t0[1] + d[1] * world_units),
            float(t0[2] + d[2] * world_units),
        )
        return t_new, self._drag_start_rotate, self._drag_start_scale

    def end_drag(self) -> None:
        self._drag_axis = None

    def _screen_angle(
        self,
        mouse_x: float, mouse_y: float,
        pivot: np.ndarray,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> float:
        """Angle (radians) from the pivot to the mouse in screen space."""
        center = _project_to_pixel(pivot, view, proj, width, height)
        if center is None:
            return 0.0
        dx = mouse_x - center[0]
        dy = mouse_y - center[1]
        return math.atan2(dy, dx)


def _point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + abx * t
    cy = ay + aby * t
    return math.hypot(px - cx, py - cy)
