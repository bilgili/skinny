"""Geometry emitters for the confirming-scene suite (change confirming-test-scenes).

Everything is a pbrt ``trianglemesh`` so the SAME triangles flow to pbrt (the
reference renderer) and, via ``import_pbrt``, to skinny's USD — no analytic-vs-
tessellated silhouette mismatch. Keep the tessellation modest: these are 128×128
discriminating scenes, not beauty renders.
"""

from __future__ import annotations

import math


def uv_sphere(cx: float, cy: float, cz: float, r: float,
              rings: int = 24, segs: int = 48) -> tuple[list[tuple], list[int]]:
    """A UV sphere centered at (cx,cy,cz). Returns (points, tri-indices)."""
    pts: list[tuple] = []
    for i in range(rings + 1):
        theta = (i / rings) * math.pi
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(segs + 1):
            phi = (j / segs) * 2.0 * math.pi
            pts.append((cx + r * st * math.cos(phi),
                        cy + r * ct,
                        cz + r * st * math.sin(phi)))
    idx: list[int] = []
    row = segs + 1
    for i in range(rings):
        for j in range(segs):
            a = i * row + j
            b = (i + 1) * row + j
            c = (i + 1) * row + (j + 1)
            d = i * row + (j + 1)
            idx += [a, b, c, a, c, d]
    return pts, idx


def quad(p0: tuple, p1: tuple, p2: tuple, p3: tuple) -> tuple[list[tuple], list[int]]:
    """A single quad (two triangles) with winding p0→p1→p2→p3."""
    return [p0, p1, p2, p3], [0, 1, 2, 0, 2, 3]


def ground(size: float = 4.0, y: float = 0.0) -> tuple[list[tuple], list[int]]:
    """A large floor quad in the XZ plane, upward normal."""
    return quad((-size, y, size), (size, y, size), (size, y, -size), (-size, y, -size))


def ceiling_light(half: float = 1.0, y: float = 4.0) -> tuple[list[tuple], list[int]]:
    """A downward-facing emissive quad above the scene (normal -Y)."""
    return quad((-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half))


def fmt_points(pts: list[tuple]) -> str:
    return " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in pts)


def fmt_indices(idx: list[int]) -> str:
    return " ".join(str(i) for i in idx)


def trianglemesh(pts: list[tuple], idx: list[int], uv: list[tuple] | None = None) -> str:
    """A pbrt ``Shape "trianglemesh"`` statement body (P + indices [+ uv])."""
    s = (f'Shape "trianglemesh" "point3 P" [ {fmt_points(pts)} ] '
         f'"integer indices" [ {fmt_indices(idx)} ]')
    if uv is not None:
        uvs = " ".join(f"{u:.5f} {v:.5f}" for u, v in uv)
        s += f' "point2 uv" [ {uvs} ]'
    return s
