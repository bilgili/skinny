"""Host (numpy) reference for the Metal Camera Debug compute rasteriser
(change metal-tool-dock-render P2, task 3.6).

The Metal backend has no graphics pipeline, so `DebugViewport` on Metal
scan-converts the debug line/triangle vertex streams in a compute kernel
(`shaders/debug_raster.slang`). This module is the *authoritative* CPU mirror of
that kernel's math: the same vertex-transform, the same DDA line rasteriser, the
same NDC→pixel mapping and RGBA8 packing. The MSL kernel is written to match this
byte-for-byte on the covered cases so the rasteriser is checkable **without a
GPU** (task 3.6) and the GPU output can be diffed against it (task 3.8).

Vertex stream layout (shared with the Vulkan path, `debug_viewport._VERTEX_FLOATS`):
7 floats per vertex — `x y z r g b a`. Lines are a vertex list (2 per line);
triangles a vertex list (3 per triangle, added in a later phase). A vertex whose
`a > 1.5` is a screen-space HUD sentinel: its `xy` is already NDC and bypasses
`viewProj` (mirrors `debug_line.slang`).

Conventions (self-consistent GPU↔numpy; NOT pixel-identical to the Vulkan
rasteriser — the two backends are independent):
- `viewProj` is the math-form 4x4 (`proj_math @ view_math`), row-major; clip =
  `viewProj @ [x, y, z, 1]`.
- NDC→pixel: `px = (ndc.x*0.5 + 0.5) * W`, `py = (0.5 - ndc.y*0.5) * H`
  (row 0 = top, +y_ndc up).
- A line is dropped if either endpoint has `clip.w <= 0` (behind the eye) — no
  near-plane clipping in this first phase (task 3.1).
- Background clear = (13, 13, 18, 255) = round(255 * (0.05, 0.05, 0.07, 1.0)),
  matching the Vulkan render-pass clear.
"""

from __future__ import annotations

import numpy as np

VERTEX_FLOATS = 7
CLEAR_RGBA8 = (13, 13, 18, 255)  # round(255 * (0.05, 0.05, 0.07, 1.0))
_W_EPS = 1e-6


def _to_rgba8(c: np.ndarray) -> tuple[int, int, int, int]:
    """Clamp a float4 color to an RGBA8 tuple (matches the MSL pack)."""
    v = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return (
        int(v[0] * 255.0 + 0.5),
        int(v[1] * 255.0 + 0.5),
        int(v[2] * 255.0 + 0.5),
        int(v[3] * 255.0 + 0.5),
    )


def project_vertex(vert: np.ndarray, view_proj: np.ndarray, width: int, height: int):
    """Project one 7-float vertex → (px, py, depth, visible). `px/py` are float
    pixel coords, `depth` is NDC z, `visible` is False when behind the eye.

    A HUD sentinel vertex (`a > 1.5`) bypasses `view_proj`: its `xy` is NDC."""
    pos = np.asarray(vert[:3], np.float64)
    alpha = float(vert[6])
    if alpha > 1.5:
        ndc_x, ndc_y, depth = pos[0], pos[1], 0.0
    else:
        clip = view_proj @ np.array([pos[0], pos[1], pos[2], 1.0], np.float64)
        if clip[3] <= _W_EPS:
            return 0.0, 0.0, 0.0, False
        ndc_x = clip[0] / clip[3]
        ndc_y = clip[1] / clip[3]
        depth = clip[2] / clip[3]
    px = (ndc_x * 0.5 + 0.5) * width
    py = (0.5 - ndc_y * 0.5) * height
    return px, py, depth, True


def _plot(img: np.ndarray, x: int, y: int, rgba: tuple[int, int, int, int]) -> None:
    h, w, _ = img.shape
    if 0 <= x < w and 0 <= y < h:
        img[y, x, 0] = rgba[0]
        img[y, x, 1] = rgba[1]
        img[y, x, 2] = rgba[2]
        img[y, x, 3] = rgba[3]


def raster_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                rgba: tuple[int, int, int, int], max_steps: int = 1 << 16) -> None:
    """DDA-rasterise one line into `img` (H, W, 4) uint8. Bounded by `max_steps`
    (the MSL kernel caps identically so no unbounded per-line loop under the
    macOS GPU watchdog)."""
    dx = x1 - x0
    dy = y1 - y0
    steps = int(np.ceil(max(abs(dx), abs(dy))))
    if steps <= 0:
        _plot(img, int(np.floor(x0 + 0.5)), int(np.floor(y0 + 0.5)), rgba)
        return
    steps = min(steps, max_steps)
    inv = 1.0 / steps
    for i in range(steps + 1):
        t = i * inv
        x = int(np.floor(x0 + dx * t + 0.5))
        y = int(np.floor(y0 + dy * t + 0.5))
        _plot(img, x, y, rgba)


def rasterise_lines(line_floats, view_proj: np.ndarray,
                    width: int, height: int) -> np.ndarray:
    """Reference rasterisation of a line-vertex stream → (H, W, 4) uint8 RGBA.

    `line_floats` is a flat float sequence (`VERTEX_FLOATS` per vertex, 2 per
    line). Returns the cleared-then-drawn image (row 0 = top)."""
    verts = np.asarray(line_floats, np.float64).reshape(-1, VERTEX_FLOATS)
    img = np.empty((height, width, 4), np.uint8)
    img[..., :] = CLEAR_RGBA8
    n_lines = verts.shape[0] // 2
    for li in range(n_lines):
        a = verts[2 * li]
        b = verts[2 * li + 1]
        ax, ay, _az, av = project_vertex(a, view_proj, width, height)
        bx, by, _bz, bv = project_vertex(b, view_proj, width, height)
        if not (av and bv):
            continue
        # Line color = endpoint A's rgb (the Vulkan pipeline interpolates, but
        # debug lines are single-colored; A's color is authoritative here).
        rgba = _to_rgba8(a[3:7]) if av else _to_rgba8(b[3:7])
        raster_line(img, ax, ay, bx, by, rgba)
    return img
