"""Rotate gizmo — CPU-side hit-testing and drag math.

A `RotateGizmo` tracks a single selected mesh instance and exposes three
orthogonal screen-space rings (X/Y/Z, world-axis aligned) around the
instance's pivot. The renderer rebuilds the line list each frame from
``build_segments`` and uploads it to GPU binding 22; ``main_pass.slang``
draws each segment as an anti-aliased line over the final tonemapped
image.

Coordinate conventions:
- Pixel coordinates use a top-left origin (0..width-1, 0..height-1) to
  match the GPU's ``RWTexture2D`` indexing.
- View / proj matrices are stored row-vector / math-transposed (same
  convention as renderer.py's `_perspective` and `_look_at`), so a clip-
  space coordinate is computed as ``world_h @ view @ proj``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


GIZMO_SEGMENT_CAPACITY = 256        # ≥ 3 axes × 64 segments + 8 axis stubs
GIZMO_SEGMENT_STRIDE = 32           # 2×float2 + float3 + float = 8 floats
GIZMO_RING_SEGMENTS = 64            # discretization per axis ring
GIZMO_RADIUS_FRACTION = 0.18        # ring radius as fraction of viewport min dim
GIZMO_LINE_HALF_WIDTH = 1.5         # pixels
GIZMO_HIT_TOLERANCE = 6.0           # pixels — generous so screens-space ring is easy to grab

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


class RotateGizmo:
    """Per-instance rotate gizmo: three world-axis rings around a pivot."""

    def __init__(self) -> None:
        self.target_index: int = -1
        self.pivot_world: np.ndarray = np.zeros(3, dtype=np.float32)

        # Active drag state
        self._drag_axis: Optional[str] = None
        self._drag_start_screen_angle: float = 0.0
        self._drag_start_translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._drag_start_rotate: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._drag_start_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._drag_start_transform: Optional[np.ndarray] = None

        # Hover for visual highlight (set during hit-test before drag)
        self._hover_axis: Optional[str] = None

    # ── Selection ────────────────────────────────────────────────

    def set_target(self, index: int, pivot_world: np.ndarray) -> None:
        if index < 0:
            self.clear_target()
            return
        self.target_index = int(index)
        self.pivot_world = np.asarray(pivot_world, dtype=np.float32).reshape(3)

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

    # ── Geometry ─────────────────────────────────────────────────

    def _ring_radius_world(
        self, view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> float:
        """Pick a world-space radius that projects to roughly
        ``GIZMO_RADIUS_FRACTION * min(width, height)`` pixels at the pivot.
        """
        pivot = self.pivot_world.astype(np.float64)
        center = _project_to_pixel(pivot, view, proj, width, height)
        if center is None:
            return 1.0
        target_px = GIZMO_RADIUS_FRACTION * min(width, height)
        # Try a unit-X offset, measure pixel distance, scale to target.
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

    def _axis_basis(self, axis: str) -> tuple[np.ndarray, np.ndarray]:
        """Two orthonormal basis vectors spanning the ring plane for ``axis``."""
        if axis == "x":
            return (np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]))
        if axis == "y":
            return (np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]))
        return (np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]))

    def _ring_pixels(
        self, axis: str, radius_world: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> list[tuple[float, float]]:
        u, v = self._axis_basis(axis)
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

    def build_segments(
        self, view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> list[GizmoSegment]:
        if not self.has_target:
            return []
        radius = self._ring_radius_world(view, proj, width, height)
        segs: list[GizmoSegment] = []
        active = self._drag_axis or self._hover_axis
        for axis in ("x", "y", "z"):
            color = AXIS_HIGHLIGHT[axis] if axis == active else AXIS_COLORS[axis]
            pts = self._ring_pixels(axis, radius, view, proj, width, height)
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
        return segs[:GIZMO_SEGMENT_CAPACITY]

    # ── Hit test ─────────────────────────────────────────────────

    def hit_test(
        self, mouse_x: float, mouse_y: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> Optional[str]:
        if not self.has_target:
            return None
        radius = self._ring_radius_world(view, proj, width, height)
        best_axis: Optional[str] = None
        best_dist = GIZMO_HIT_TOLERANCE
        for axis in ("x", "y", "z"):
            pts = self._ring_pixels(axis, radius, view, proj, width, height)
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
        self._drag_start_transform = np.asarray(
            instance_transform, dtype=np.float32,
        ).copy()
        self._drag_start_screen_angle = self._screen_angle(
            mouse_x, mouse_y, view, proj, width, height,
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
        cur = self._screen_angle(
            mouse_x, mouse_y, view, proj, width, height,
        )
        delta = cur - self._drag_start_screen_angle
        # Wrap into (-π, π] so a tiny drag near the seam doesn't snap by 2π.
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        delta_deg = math.degrees(delta)
        rx, ry, rz = self._drag_start_rotate
        if self._drag_axis == "x":
            rx += delta_deg
        elif self._drag_axis == "y":
            ry += delta_deg
        else:
            rz += delta_deg
        return self._drag_start_translate, (rx, ry, rz), self._drag_start_scale

    def end_drag(self) -> None:
        self._drag_axis = None
        self._drag_start_transform = None

    def _screen_angle(
        self,
        mouse_x: float, mouse_y: float,
        view: np.ndarray, proj: np.ndarray,
        width: int, height: int,
    ) -> float:
        """Angle (radians) from the pivot to the mouse in screen space."""
        center = _project_to_pixel(
            self.pivot_world, view, proj, width, height,
        )
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
