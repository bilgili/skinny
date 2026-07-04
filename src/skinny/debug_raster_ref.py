"""Host (numpy) reference for the Metal Camera Debug compute rasteriser
(change metal-tool-dock-render P2, tasks 3.1–3.6).

The Metal backend has no graphics pipeline, so `DebugViewport` on Metal
scan-converts the debug line/triangle vertex streams in a compute kernel
(`shaders/debug_raster.slang`). This module is the *authoritative* CPU mirror of
that kernel's math: vertex transform, DDA line raster, edge-function triangle
raster, packed-depth ordering, alpha blend, NDC→pixel mapping, RGBA8 packing. The
MSL kernel matches this so the rasteriser is checkable **without a GPU** (task
3.6) and the GPU output is diffable against it (task 3.8).

Vertex stream layout (shared with the Vulkan path, `debug_viewport._VERTEX_FLOATS`):
7 floats per vertex — `x y z r g b a`. Lines are a vertex list (2 per line);
triangles a vertex list (3 per triangle). A vertex whose `a > 1.5` is a
screen-space HUD sentinel: its `xy` is already NDC and bypasses `viewProj`
(mirrors `debug_line.slang`).

Ordering (design D4, mirrors the Vulkan two-pipeline split):
- **Lines are opaque** — depth-tested and depth-writing.
- **Triangles blend** (src-alpha over) — depth-tested against the opaque line
  depth, but do NOT write depth, and are drawn after the lines. Overlapping
  transparent triangles compose in arbitrary order (matches the unsorted Vulkan
  blend; the GPU kernel races there too).

Conventions (self-consistent GPU↔numpy; NOT pixel-identical to the Vulkan
rasteriser — the two backends are independent):
- `viewProj` is the math-form 4x4 (`proj_math @ view_math`), row-major; clip =
  `viewProj @ [x, y, z, 1]`.
- NDC→pixel: `px = (ndc.x*0.5 + 0.5) * W`, `py = (0.5 - ndc.y*0.5) * H`
  (row 0 = top, +y_ndc up).
- Depth key: `d01 = clip.z/clip.w * 0.5 + 0.5` clamped to [0,1] (near→0 wins),
  quantised to 24 bits (`pack_depth`) and stored as `depth24 << 8 | (line_idx &
  0xFF)`. atomic_min orders primarily by depth (nearest wins) and, at equal
  depth, by the low-byte line tag (smallest line index wins) — a deterministic
  tie-break that matches the Vulkan `VK_COMPARE_OP_LESS` "earlier draw wins".
  HUD sentinels use `d01 = 0` so they stay on top.
- Triangles are depth-tested strictly against the winning **depth** (high 24
  bits), occluding on equal depth — same as the Vulkan transparent pipeline's
  `VK_COMPARE_OP_LESS` (an equal-depth fill does not blend over its outline).
- A primitive is dropped if any vertex has `clip.w <= 0` (behind the eye) — no
  near-plane clipping in these phases.
- Background clear = (13, 13, 18, 255) = round(255 * (0.05, 0.05, 0.07, 1.0)).
"""

from __future__ import annotations

import numpy as np

VERTEX_FLOATS = 7
CLEAR_RGBA8 = (13, 13, 18, 255)  # round(255 * (0.05, 0.05, 0.07, 1.0))
DEPTH_CLEAR = 0xFFFFFFFF  # depth24 0xFFFFFF (far) | tag 0xFF
DEPTH_SCALE = 16777215.0  # 0xFFFFFF — 24-bit depth (low byte reserved for the tag)
_W_EPS = 1e-6


def _to_rgba8(c) -> tuple[int, int, int, int]:
    v = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return (int(v[0] * 255.0 + 0.5), int(v[1] * 255.0 + 0.5),
            int(v[2] * 255.0 + 0.5), int(v[3] * 255.0 + 0.5))


def pack_depth(d01: float) -> int:
    """24-bit depth quant (near→0). The full depth key is `pack_depth(d) << 8 |
    (line_tag & 0xFF)`; triangles compare against this 24-bit depth alone."""
    return int(min(max(d01, 0.0), 1.0) * DEPTH_SCALE) & 0xFFFFFF


def project_vertex(vert, view_proj: np.ndarray, width: int, height: int):
    """Project one 7-float vertex → (px, py, depth01, visible). `px/py` are float
    pixel coords, `depth01` is the [0,1] depth (0 = near / on top), `visible` is
    False when behind the eye. A HUD sentinel (`a > 1.5`) bypasses `view_proj`:
    its `xy` is NDC and its depth is 0 (always in front)."""
    pos = np.asarray(vert[:3], np.float64)
    alpha = float(vert[6])
    if alpha > 1.5:
        ndc_x, ndc_y, d01 = pos[0], pos[1], 0.0
    else:
        clip = view_proj @ np.array([pos[0], pos[1], pos[2], 1.0], np.float64)
        if clip[3] <= _W_EPS:
            return 0.0, 0.0, 0.0, False
        ndc_x = clip[0] / clip[3]
        ndc_y = clip[1] / clip[3]
        d01 = np.clip(clip[2] / clip[3] * 0.5 + 0.5, 0.0, 1.0)
    px = (ndc_x * 0.5 + 0.5) * width
    py = (0.5 - ndc_y * 0.5) * height
    return px, py, d01, True


def _raster_line_depth(img, depth, x0, y0, d0, x1, y1, d1, rgba, tag: int,
                       max_steps: int = 1 << 16) -> None:
    """DDA line, opaque, depth-tested + writing the `depth24<<8 | tag` key.
    Bounded by `max_steps` (the MSL kernel caps identically for the watchdog)."""
    h, w, _ = img.shape
    dx = x1 - x0
    dy = y1 - y0
    dd = d1 - d0
    lo = tag & 0xFF
    steps = int(np.ceil(max(abs(dx), abs(dy))))
    steps = max(steps, 0)
    n = min(steps, max_steps)
    inv = 1.0 / n if n > 0 else 0.0
    for i in range(n + 1):
        t = i * inv
        x = int(np.floor(x0 + dx * t + 0.5))
        y = int(np.floor(y0 + dy * t + 0.5))
        if not (0 <= x < w and 0 <= y < h):
            continue
        key = (pack_depth(d0 + dd * t) << 8) | lo
        if key < int(depth[y, x]):
            depth[y, x] = key
            img[y, x, 0], img[y, x, 1], img[y, x, 2], img[y, x, 3] = rgba


def _blend_tri(img, depth, p0, p1, p2, color) -> None:
    """Edge-function triangle raster, alpha-blended (src-alpha over), depth-tested
    against `depth` but NOT depth-writing. `pN` = (px, py, d01); `color` is float4."""
    h, w, _ = img.shape
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = p0, p1, p2
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-9:
        return  # degenerate
    inv_area = 1.0 / area
    min_x = max(int(np.floor(min(x0, x1, x2))), 0)
    max_x = min(int(np.ceil(max(x0, x1, x2))), w - 1)
    min_y = max(int(np.floor(min(y0, y1, y2))), 0)
    max_y = min(int(np.ceil(max(y0, y1, y2))), h - 1)
    src = np.clip(np.asarray(color[:3], np.float64), 0.0, 1.0)
    alpha = float(np.clip(color[3], 0.0, 1.0))
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            py = y + 0.5
            w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv_area
            w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv_area
            w2 = 1.0 - w0 - w1
            if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                continue  # outside (cullMode NONE → inv_area sign normalises winding)
            d01 = w0 * z0 + w1 * z1 + w2 * z2
            # Strict depth test against the opaque line depth (high 24 bits),
            # occluding on equal depth — Vulkan VK_COMPARE_OP_LESS.
            if pack_depth(d01) >= (int(depth[y, x]) >> 8):
                continue
            dst = img[y, x, :3].astype(np.float64) / 255.0
            out = np.clip(src * alpha + dst * (1.0 - alpha), 0.0, 1.0)
            img[y, x, 0] = int(out[0] * 255.0 + 0.5)
            img[y, x, 1] = int(out[1] * 255.0 + 0.5)
            img[y, x, 2] = int(out[2] * 255.0 + 0.5)


def rasterise(line_floats, tri_floats, view_proj: np.ndarray,
              width: int, height: int) -> np.ndarray:
    """Full reference pipeline → (H, W, 4) uint8 RGBA (row 0 = top): clear, then
    opaque depth-tested lines, then alpha-blended depth-tested triangles."""
    img = np.empty((height, width, 4), np.uint8)
    img[..., :] = CLEAR_RGBA8
    depth = np.full((height, width), DEPTH_CLEAR, np.uint32)

    lv = np.asarray(line_floats, np.float64).reshape(-1, VERTEX_FLOATS) \
        if len(line_floats) else np.zeros((0, VERTEX_FLOATS))
    for li in range(lv.shape[0] // 2):
        a, b = lv[2 * li], lv[2 * li + 1]
        ax, ay, ad, av = project_vertex(a, view_proj, width, height)
        bx, by, bd, bv = project_vertex(b, view_proj, width, height)
        if not (av and bv):
            continue
        _raster_line_depth(img, depth, ax, ay, ad, bx, by, bd,
                           _to_rgba8(a[3:7]), li)

    tv = np.asarray(tri_floats, np.float64).reshape(-1, VERTEX_FLOATS) \
        if len(tri_floats) else np.zeros((0, VERTEX_FLOATS))
    for ti in range(tv.shape[0] // 3):
        a, b, c = tv[3 * ti], tv[3 * ti + 1], tv[3 * ti + 2]
        ax, ay, ad, av = project_vertex(a, view_proj, width, height)
        bx, by, bd, bv = project_vertex(b, view_proj, width, height)
        cx, cy, cd, cv = project_vertex(c, view_proj, width, height)
        if not (av and bv and cv):
            continue
        _blend_tri(img, depth, (ax, ay, ad), (bx, by, bd), (cx, cy, cd), a[3:7])
    return img


def rasterise_lines(line_floats, view_proj: np.ndarray,
                    width: int, height: int) -> np.ndarray:
    """Lines-only convenience (phase 1) — delegates to :func:`rasterise`."""
    return rasterise(line_floats, [], view_proj, width, height)
