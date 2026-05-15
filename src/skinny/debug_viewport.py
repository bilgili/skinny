"""Camera-debug viewport — second GLFW window that rasterises wireframe
visualisations of the render camera, its lens elements, the per-instance
world-space AABBs (or full mesh wireframes), a ground grid, and a small
camera-body glyph at the render camera's pose.

Decoupled from the main compute renderer: shares the same VulkanContext
device/queue but owns its own surface, swapchain, depth buffer, render
pass, line-list graphics pipeline, vertex buffer, and per-frame sync
primitives. Toggle on/off via ``DebugViewport.toggle()``; the window is
kept alive after first open and shown/hidden rather than rebuilt.

Geometry is regenerated each frame from the live ``Renderer`` state — no
caches — so any change to the rendered camera's pose, FOV, or lens stack
is reflected immediately. All primitives go through one line-list
pipeline; the only per-mode switch is mesh wireframe vs per-instance
AABB (the rest — frustum, lens rings, camera glyph, ground grid — are
always drawn).
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import vulkan as vk

from skinny.renderer import FreeCamera, OrbitCamera, _look_at, _perspective
from skinny.vk_context import VulkanContext


# Vertex stream: float3 pos + float4 color (rgba), interleaved.
_VERTEX_STRIDE_BYTES = 28
_VERTEX_FLOATS = 7  # 3 pos + 4 color
# 4 MiB / 28 B = ~149k vertices for the line stream.
_LINE_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_LINE_VERTICES = _LINE_BUFFER_BYTES // _VERTEX_STRIDE_BYTES
# 1 MiB / 28 B = ~37k vertices for the triangle stream.
_TRI_BUFFER_BYTES = 1 * 1024 * 1024
_MAX_TRI_VERTICES = _TRI_BUFFER_BYTES // _VERTEX_STRIDE_BYTES

_DEFAULT_WIDTH = 960
_DEFAULT_HEIGHT = 720

_DEPTH_FORMAT = vk.VK_FORMAT_D32_SFLOAT

# Line colours (linear RGB; gamma is left to the swapchain when SRGB is
# selected, otherwise these are the literal output values).
_COL_FRUSTUM = (1.00, 0.85, 0.20)   # warm yellow
_COL_AABB    = (0.30, 0.95, 0.30)   # green
_COL_LENS    = (0.40, 0.80, 1.00)   # cyan
_COL_GLYPH   = (1.00, 0.45, 0.55)   # pink/red — points away from cam
_COL_GLYPH_UP = (1.00, 0.95, 0.95)
_COL_GRID_MAJOR = (0.55, 0.55, 0.55)
_COL_GRID_MINOR = (0.25, 0.25, 0.25)
_COL_AXIS_X = (0.95, 0.30, 0.30)
_COL_AXIS_Y = (0.30, 0.95, 0.30)
_COL_AXIS_Z = (0.30, 0.45, 0.95)
_COL_WIRE = (0.75, 0.78, 0.85)
_COL_FOCUS_PLANE = (0.85, 0.85, 1.00, 0.10)  # bluish, mostly transparent
_COL_FOCUS_OUTLINE = (0.85, 0.85, 1.00)
_COL_RENDER_AREA_OUTLINE = (1.00, 0.30, 0.30)
_COL_DOF_PLANE = (1.00, 0.55, 0.10, 0.10)
_COL_DOF_OUTLINE = (1.00, 0.65, 0.20)
_COL_DOF_DOT = (1.00, 0.55, 0.10, 1.0)
_COL_HUD_TEXT = (0.95, 0.95, 0.95)


# ─── Bitmap font (5×7) for the screen-space HUD ───────────────────────
#
# Each glyph is a 35-char string (5 cols × 7 rows, top→bottom, left→right).
# 'X' = lit cell, anything else = empty. Each lit cell is rasterised as a
# single 1-pixel horizontal stroke by ``_gen_text_line``. The fragment
# shader detects the color.a > 1.5 sentinel and bypasses the camera matrix
# so vertices are interpreted as NDC directly.

_FONT_W = 5
_FONT_H = 7

_FONT: dict[str, str] = {
    " ": "...................................",
    "A": ".XXX." "X...X" "X...X" "XXXXX" "X...X" "X...X" "X...X",
    "B": "XXXX." "X...X" "X...X" "XXXX." "X...X" "X...X" "XXXX.",
    "C": ".XXXX" "X...." "X...." "X...." "X...." "X...." ".XXXX",
    "D": "XXXX." "X...X" "X...X" "X...X" "X...X" "X...X" "XXXX.",
    "E": "XXXXX" "X...." "X...." "XXXX." "X...." "X...." "XXXXX",
    "F": "XXXXX" "X...." "X...." "XXXX." "X...." "X...." "X....",
    "G": ".XXXX" "X...." "X...." "X..XX" "X...X" "X...X" ".XXXX",
    "H": "X...X" "X...X" "X...X" "XXXXX" "X...X" "X...X" "X...X",
    "I": "XXXXX" "..X.." "..X.." "..X.." "..X.." "..X.." "XXXXX",
    "J": "....X" "....X" "....X" "....X" "X...X" "X...X" ".XXX.",
    "K": "X...X" "X..X." "X.X.." "XX..." "X.X.." "X..X." "X...X",
    "L": "X...." "X...." "X...." "X...." "X...." "X...." "XXXXX",
    "M": "X...X" "XX.XX" "X.X.X" "X...X" "X...X" "X...X" "X...X",
    "N": "X...X" "XX..X" "X.X.X" "X..XX" "X...X" "X...X" "X...X",
    "O": ".XXX." "X...X" "X...X" "X...X" "X...X" "X...X" ".XXX.",
    "P": "XXXX." "X...X" "X...X" "XXXX." "X...." "X...." "X....",
    "Q": ".XXX." "X...X" "X...X" "X...X" "X.X.X" "X..X." ".XX.X",
    "R": "XXXX." "X...X" "X...X" "XXXX." "X.X.." "X..X." "X...X",
    "S": ".XXXX" "X...." "X...." ".XXX." "....X" "....X" "XXXX.",
    "T": "XXXXX" "..X.." "..X.." "..X.." "..X.." "..X.." "..X..",
    "U": "X...X" "X...X" "X...X" "X...X" "X...X" "X...X" ".XXX.",
    "V": "X...X" "X...X" "X...X" "X...X" "X...X" ".X.X." "..X..",
    "W": "X...X" "X...X" "X...X" "X.X.X" "X.X.X" "X.X.X" ".X.X.",
    "X": "X...X" "X...X" ".X.X." "..X.." ".X.X." "X...X" "X...X",
    "Y": "X...X" "X...X" ".X.X." "..X.." "..X.." "..X.." "..X..",
    "Z": "XXXXX" "....X" "...X." "..X.." ".X..." "X...." "XXXXX",
    "0": ".XXX." "X...X" "X..XX" "X.X.X" "XX..X" "X...X" ".XXX.",
    "1": "..X.." ".XX.." "..X.." "..X.." "..X.." "..X.." "XXXXX",
    "2": ".XXX." "X...X" "....X" "...X." "..X.." ".X..." "XXXXX",
    "3": ".XXX." "X...X" "....X" "..XX." "....X" "X...X" ".XXX.",
    "4": "...X." "..XX." ".X.X." "X..X." "XXXXX" "...X." "...X.",
    "5": "XXXXX" "X...." "X...." "XXXX." "....X" "X...X" ".XXX.",
    "6": "..XX." ".X..." "X...." "XXXX." "X...X" "X...X" ".XXX.",
    "7": "XXXXX" "....X" "...X." "..X.." "..X.." "..X.." "..X..",
    "8": ".XXX." "X...X" "X...X" ".XXX." "X...X" "X...X" ".XXX.",
    "9": ".XXX." "X...X" "X...X" ".XXXX" "....X" "...X." ".XX..",
    ":": "....." "..X.." "..X.." "....." "..X.." "..X.." ".....",
    ".": "....." "....." "....." "....." "....." "..X.." "..X..",
    ",": "....." "....." "....." "....." "..X.." "..X.." ".X...",
    "/": "....X" "....X" "...X." "..X.." ".X..." "X...." "X....",
    "-": "....." "....." "....." "XXXXX" "....." "....." ".....",
    "+": "....." "..X.." "..X.." "XXXXX" "..X.." "..X.." ".....",
    "(": "...X." "..X.." ".X..." ".X..." ".X..." "..X.." "...X.",
    ")": ".X..." "..X.." "...X." "...X." "...X." "..X.." ".X...",
    "[": "..XX." "..X.." "..X.." "..X.." "..X.." "..X.." "..XX.",
    "]": ".XX.." "..X.." "..X.." "..X.." "..X.." "..X.." ".XX..",
    "=": "....." "....." "XXXXX" "....." "XXXXX" "....." ".....",
    "<": "....." "...X." "..X.." ".X..." "..X.." "...X." ".....",
    ">": "....." ".X..." "..X.." "...X." "..X.." ".X..." ".....",
    "?": ".XXX." "X...X" "....X" "...X." "..X.." "....." "..X..",
    "!": "..X.." "..X.." "..X.." "..X.." "..X.." "....." "..X..",
}


def _emit_screen_line(out: list, x0: float, y0: float, x1: float, y1: float,
                      color3) -> None:
    """Append a screen-space NDC line. Alpha=2.0 sentinel triggers the
    debug_line vertex shader's no-transform branch so xy lands directly
    in clip space.
    """
    r, g, b = float(color3[0]), float(color3[1]), float(color3[2])
    out.extend([
        float(x0), float(y0), 0.0, r, g, b, 2.0,
        float(x1), float(y1), 0.0, r, g, b, 2.0,
    ])


def _gen_text_line(out: list, text: str, px: int, py: int,
                   width_px: int, height_px: int,
                   scale: int = 2,
                   color=_COL_HUD_TEXT) -> None:
    """Rasterise ``text`` starting at pixel ``(px, py)`` (top-left origin
    of the first glyph) into the screen-space line stream.

    Each lit font cell becomes a 1-pixel horizontal stroke ``scale`` long.
    """
    inv_w = 2.0 / max(width_px, 1)
    inv_h = 2.0 / max(height_px, 1)
    char_pitch = (_FONT_W + 1) * scale
    for ci, ch in enumerate(text.upper()):
        glyph = _FONT.get(ch)
        if glyph is None:
            glyph = _FONT[" "]
        ox = px + ci * char_pitch
        for row in range(_FONT_H):
            for col in range(_FONT_W):
                if glyph[row * _FONT_W + col] != "X":
                    continue
                x0 = ox + col * scale
                y0 = py + row * scale
                x1 = x0 + scale
                ndc_x0 = x0 * inv_w - 1.0
                ndc_x1 = x1 * inv_w - 1.0
                # Stack ``scale`` horizontal strokes so each lit cell fills
                # solidly even at integer pixel scales.
                for sy in range(scale):
                    ndc_y = (y0 + sy) * inv_h - 1.0
                    _emit_screen_line(out, ndc_x0, ndc_y, ndc_x1, ndc_y, color)


def _gen_hud_overlay(out: list, lines: list[str],
                     width_px: int, height_px: int,
                     scale: int = 2,
                     margin_px: int = 8,
                     line_gap_px: int = 2,
                     color=_COL_HUD_TEXT) -> None:
    """Emit a top-left aligned HUD block listing keyboard shortcuts."""
    if not lines:
        return
    line_height = _FONT_H * scale + line_gap_px
    py = margin_px
    for text in lines:
        _gen_text_line(
            out, text, margin_px, py,
            width_px, height_px, scale=scale, color=color,
        )
        py += line_height


# ─── Slang → SPIR-V helper ────────────────────────────────────────────


def _compile_slang_stage(
    shader_dir: Path, src_module: str, entry: str, stage: str, out_path: Path,
) -> Path:
    slangc = shutil.which("slangc")
    if slangc is None:
        raise RuntimeError("slangc not found on PATH — install the Slang compiler")
    src = shader_dir / f"{src_module}.slang"
    cmd = [
        slangc,
        str(src),
        "-target", "spirv",
        "-entry", entry,
        "-stage", stage,
        "-o", str(out_path),
        "-I", str(shader_dir),
        "-fvk-use-scalar-layout",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Slang compile failed for {src_module}.slang ({stage}/{entry}):\n"
            f"{result.stderr}"
        )
    return out_path


def _find_memory_type(type_filter: int, properties: int, mem_props) -> int:
    for i in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << i)) and (
            mem_props.memoryTypes[i].propertyFlags & properties
        ) == properties:
            return i
    raise RuntimeError("Failed to find suitable memory type")


# ─── Geometry generators ──────────────────────────────────────────────


def _color4(c) -> tuple[float, float, float, float]:
    """Coerce 3- or 4-tuple colors to RGBA floats (alpha defaults to 1)."""
    if len(c) == 4:
        return float(c[0]), float(c[1]), float(c[2]), float(c[3])
    return float(c[0]), float(c[1]), float(c[2]), 1.0


def _emit_line(out: list, a, color_a, b, color_b) -> None:
    ca = _color4(color_a)
    cb = _color4(color_b)
    out.extend([
        float(a[0]), float(a[1]), float(a[2]), ca[0], ca[1], ca[2], ca[3],
        float(b[0]), float(b[1]), float(b[2]), cb[0], cb[1], cb[2], cb[3],
    ])


def _emit_tri(out: list, a, b, c, color) -> None:
    col = _color4(color)
    for v in (a, b, c):
        out.extend([
            float(v[0]), float(v[1]), float(v[2]),
            col[0], col[1], col[2], col[3],
        ])


def _gen_grid(out: list, half_extent: float = 5.0, spacing: float = 0.5,
              y: float = 0.0) -> None:
    """XZ-plane grid centred on origin. Major lines every 5 cells.

    Caller passes the line accumulator; this function appends.
    """
    n = int(np.ceil(half_extent / spacing))
    for i in range(-n, n + 1):
        offset = i * spacing
        is_axis = (i == 0)
        is_major = (i % 5 == 0)
        # X-aligned line (constant z = offset)
        col = _COL_AXIS_X if is_axis else (_COL_GRID_MAJOR if is_major else _COL_GRID_MINOR)
        if is_axis:
            # Split into ±x halves so we can colour the +X half red and
            # the −X half neutral; cleaner orientation cue.
            _emit_line(out,
                       (-half_extent, y, offset), _COL_GRID_MAJOR,
                       (0.0, y, offset), _COL_GRID_MAJOR)
            _emit_line(out,
                       (0.0, y, offset), _COL_AXIS_X,
                       (half_extent, y, offset), _COL_AXIS_X)
        else:
            _emit_line(out,
                       (-half_extent, y, offset), col,
                       (half_extent, y, offset), col)
        # Z-aligned line (constant x = offset)
        col = _COL_AXIS_Z if is_axis else (_COL_GRID_MAJOR if is_major else _COL_GRID_MINOR)
        if is_axis:
            _emit_line(out,
                       (offset, y, -half_extent), _COL_GRID_MAJOR,
                       (offset, y, 0.0), _COL_GRID_MAJOR)
            _emit_line(out,
                       (offset, y, 0.0), _COL_AXIS_Z,
                       (offset, y, half_extent), _COL_AXIS_Z)
        else:
            _emit_line(out,
                       (offset, y, -half_extent), col,
                       (offset, y, half_extent), col)
    # Y axis stub for orientation
    _emit_line(out, (0.0, y, 0.0), _COL_AXIS_Y, (0.0, y + 1.0, 0.0), _COL_AXIS_Y)


def _gen_aabb_box(out: list, amin, amax, color=_COL_AABB) -> None:
    x0, y0, z0 = float(amin[0]), float(amin[1]), float(amin[2])
    x1, y1, z1 = float(amax[0]), float(amax[1]), float(amax[2])
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom-z face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top-z face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connectors
    ]
    for a, b in edges:
        _emit_line(out, corners[a], color, corners[b], color)


def _gen_frustum(out: list, view_math: np.ndarray, proj_math: np.ndarray,
                 color=_COL_FRUSTUM) -> None:
    """Wireframe pyramid built by un-projecting NDC corners.

    ``view_math`` and ``proj_math`` are the math-form 4x4 matrices
    (i.e. ``renderer._look_at`` and ``renderer._perspective`` storage
    transposed back). Vulkan NDC: x∈[-1,1], y∈[-1,1], z∈[0,1].
    With the renderer's ``_perspective``, near maps to z_ndc=0 and far
    to z_ndc=1.
    """
    view_inv = np.linalg.inv(view_math)
    proj_inv = np.linalg.inv(proj_math)
    near_corners = []
    far_corners = []
    for sx, sy in [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]:
        ndc_near = np.array([sx, sy, 0.0, 1.0], dtype=np.float64)
        ndc_far  = np.array([sx, sy, 1.0, 1.0], dtype=np.float64)
        cn = proj_inv @ ndc_near
        cf = proj_inv @ ndc_far
        cn /= cn[3] if cn[3] != 0 else 1.0
        cf /= cf[3] if cf[3] != 0 else 1.0
        wn = view_inv @ cn
        wf = view_inv @ cf
        near_corners.append(wn[:3])
        far_corners.append(wf[:3])
    # Near rectangle
    for i in range(4):
        _emit_line(out, near_corners[i], color, near_corners[(i + 1) % 4], color)
    # Far rectangle
    for i in range(4):
        _emit_line(out, far_corners[i], color, far_corners[(i + 1) % 4], color)
    # Connectors
    for i in range(4):
        _emit_line(out, near_corners[i], color, far_corners[i], color)
    # Apex spokes (camera origin → near corners) for visual anchor
    cam_origin = view_inv @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    apex = cam_origin[:3] / (cam_origin[3] if cam_origin[3] != 0 else 1.0)
    for nc in near_corners:
        _emit_line(out, apex, color, nc, color)


def _gen_camera_glyph(out: list, position: np.ndarray, forward: np.ndarray,
                      up: np.ndarray, size: float = 0.15) -> None:
    """Tiny wireframe box + forward arrow at the render camera's pose."""
    f = forward / max(float(np.linalg.norm(forward)), 1e-6)
    r = np.cross(f, up)
    r = r / max(float(np.linalg.norm(r)), 1e-6)
    u = np.cross(r, f)

    s = size
    body_d = s * 1.2  # depth (along -forward, since lens points along +forward)
    body_w = s * 0.8
    body_h = s * 0.6

    # 8 corners of the body box centred just behind the camera origin
    centre = np.asarray(position, np.float32) - f * (body_d * 0.5)
    cs = []
    for dx, dy, dz in [
        (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
        (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
    ]:
        cs.append(centre + r * (dx * body_w * 0.5)
                         + u * (dy * body_h * 0.5)
                         + f * (dz * body_d * 0.5))
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        _emit_line(out, cs[a], _COL_GLYPH, cs[b], _COL_GLYPH)

    # Lens stub: small cone forward
    tip = position + f * (s * 1.2)
    rim = []
    sides = 12
    for i in range(sides):
        t = (2.0 * np.pi * i) / sides
        rim.append(position + r * (np.cos(t) * s * 0.4)
                            + u * (np.sin(t) * s * 0.4)
                            + f * (s * 0.1))
    for i in range(sides):
        _emit_line(out, rim[i], _COL_GLYPH, rim[(i + 1) % sides], _COL_GLYPH)
        _emit_line(out, rim[i], _COL_GLYPH, tip, _COL_GLYPH)

    # Up indicator
    _emit_line(out, position, _COL_GLYPH_UP, position + u * (s * 1.5), _COL_GLYPH_UP)


def _gen_lens_rings(out: list, position: np.ndarray, forward: np.ndarray,
                    up: np.ndarray, lens, mm_per_unit: float = 1000.0,
                    color=_COL_LENS) -> None:
    """Stack of rings along the optical axis representing each enabled
    LensElement's clear aperture and axial spacing.

    ``mm_per_unit`` converts the lens's intrinsic mm units to scene units
    (1000 = 1 unit ≈ 1 metre). Caller can pass ``Scene.mm_per_unit``.
    """
    if lens is None:
        return
    elements = lens.active_elements
    if not elements:
        return
    f = forward / max(float(np.linalg.norm(forward)), 1e-6)
    r = np.cross(f, up)
    r = r / max(float(np.linalg.norm(r)), 1e-6)
    u = np.cross(r, f)

    # Front element sits at the camera origin; subsequent elements step
    # backward along −forward by their predecessor's thickness.
    z_mm = 0.0
    sides = 24
    for i, e in enumerate(elements):
        radius_units = (e.aperture_mm * 0.5) / max(mm_per_unit, 1e-6)
        centre = position - f * (z_mm / max(mm_per_unit, 1e-6))
        ring = []
        for k in range(sides):
            t = (2.0 * np.pi * k) / sides
            ring.append(centre + r * (np.cos(t) * radius_units)
                               + u * (np.sin(t) * radius_units))
        ring_color = (1.0, 0.5, 0.2) if e.is_aperture_stop else color
        for k in range(sides):
            _emit_line(out, ring[k], ring_color, ring[(k + 1) % sides], ring_color)
        # Axial connector to the next ring
        if i + 1 < len(elements):
            next_z_mm = z_mm + e.thickness_mm
            next_centre = position - f * (next_z_mm / max(mm_per_unit, 1e-6))
            _emit_line(out, centre, color, next_centre, color)
        z_mm += e.thickness_mm


def _gen_focus_plane(line_out: list, tri_out: list, position: np.ndarray,
                     forward: np.ndarray, up: np.ndarray,
                     focus_distance: float, half_size: float = 25.0,
                     fill_color=_COL_FOCUS_PLANE,
                     outline_color=_COL_FOCUS_OUTLINE,
                     fov_deg: float | None = None,
                     aspect: float | None = None) -> None:
    """Two-triangle quad perpendicular to ``forward`` at ``focus_distance``.

    When ``fov_deg`` and ``aspect`` are supplied, the quad is bounded by
    the camera frustum at that distance (matching the FOV-corner dots).
    Otherwise ``half_size`` controls a square extent.
    """
    if focus_distance <= 0.0:
        return
    f = forward / max(float(np.linalg.norm(forward)), 1e-6)
    r = np.cross(f, up)
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-6:
        # forward parallel to world-up — pick an arbitrary right axis.
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        r = r / r_norm
    u = np.cross(r, f)

    centre = np.asarray(position, np.float32) + f * float(focus_distance)
    if fov_deg is not None and aspect is not None:
        half_h = float(focus_distance) * float(np.tan(np.radians(float(fov_deg)) * 0.5))
        half_w = half_h * float(aspect)
    else:
        half_w = half_h = float(half_size)
    p00 = centre + r * (-half_w) + u * (-half_h)
    p10 = centre + r * (+half_w) + u * (-half_h)
    p11 = centre + r * (+half_w) + u * (+half_h)
    p01 = centre + r * (-half_w) + u * (+half_h)
    _emit_tri(tri_out, p00, p10, p11, fill_color)
    _emit_tri(tri_out, p00, p11, p01, fill_color)
    _emit_line(line_out, p00, outline_color, p10, outline_color)
    _emit_line(line_out, p10, outline_color, p11, outline_color)
    _emit_line(line_out, p11, outline_color, p01, outline_color)
    _emit_line(line_out, p01, outline_color, p00, outline_color)


def _gen_screen_rect_at_distance(
    line_out: list, tri_out: list, position: np.ndarray,
    forward: np.ndarray, up: np.ndarray,
    distance: float, fov_deg: float, aspect: float,
    fill_color, outline_color,
) -> None:
    """Back-projected screen rectangle (frustum × distance) as filled
    quad + 4-edge outline.

    Corners coincide with the FOV-corner dots produced by
    ``_gen_fov_corner_dots`` so the rendered area visualisation aligns
    with the yellow markers regardless of camera orientation. Used to
    show the constant-size rendered region at the focus plane;
    fstop only modulates ``fill_color``'s alpha, not the rectangle's
    extent.
    """
    if distance <= 0.0:
        return
    f = forward / max(float(np.linalg.norm(forward)), 1e-6)
    r = np.cross(f, up)
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-6:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        r = r / r_norm
    u = np.cross(r, f)

    half_h = float(distance) * float(np.tan(np.radians(fov_deg) * 0.5))
    half_w = half_h * float(aspect)
    centre = np.asarray(position, np.float32) + f * float(distance)
    p00 = centre + r * (-half_w) + u * (-half_h)
    p10 = centre + r * (+half_w) + u * (-half_h)
    p11 = centre + r * (+half_w) + u * (+half_h)
    p01 = centre + r * (-half_w) + u * (+half_h)
    _emit_tri(tri_out, p00, p10, p11, fill_color)
    _emit_tri(tri_out, p00, p11, p01, fill_color)
    _emit_line(line_out, p00, outline_color, p10, outline_color)
    _emit_line(line_out, p10, outline_color, p11, outline_color)
    _emit_line(line_out, p11, outline_color, p01, outline_color)
    _emit_line(line_out, p01, outline_color, p00, outline_color)


def _fstop_alpha(fstop: float) -> float:
    """Map fstop (typ. 1.0–22.0) to a fill alpha in [0.10, 0.65].

    Bigger fstop = more saturated red overlay. Cleared to a hard floor at
    0.10 so the overlay is visible even at f/1.
    """
    f = max(float(fstop), 0.0)
    norm = (f - 1.0) / 21.0
    norm = max(0.0, min(1.0, norm))
    return 0.10 + 0.55 * norm


def _gen_fov_corner_dots(tri_out: list, position: np.ndarray,
                         forward: np.ndarray, up: np.ndarray,
                         focus_distance: float, fov_deg: float, aspect: float,
                         radius_world: float, segments: int = 28,
                         color=(1.0, 0.95, 0.15, 1.0)) -> None:
    """Yellow filled disks at the 4 frustum-edge × focus-plane intersections."""
    if focus_distance <= 0.0 or radius_world <= 0.0:
        return
    f = forward / max(float(np.linalg.norm(forward)), 1e-6)
    r = np.cross(f, up)
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-6:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        r = r / r_norm
    u = np.cross(r, f)
    half_h = float(focus_distance) * float(np.tan(np.radians(fov_deg) * 0.5))
    half_w = half_h * float(aspect)
    plane_centre = np.asarray(position, np.float32) + f * float(focus_distance)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            centre = plane_centre + r * (sx * half_w) + u * (sy * half_h)
            rim = []
            for k in range(segments):
                t = (2.0 * np.pi * k) / segments
                rim.append(centre + r * (np.cos(t) * radius_world)
                                  + u * (np.sin(t) * radius_world))
            for k in range(segments):
                _emit_tri(tri_out, centre, rim[k], rim[(k + 1) % segments], color)


def _gen_mesh_wireframe(out: list, mesh, transform: np.ndarray,
                        color=_COL_WIRE, max_vertices_remaining: int = 0) -> int:
    """Append triangle-edge line list for ``mesh`` transformed by ``transform``.

    Returns the number of *vertices* (not segments) appended. Stops early
    if ``max_vertices_remaining`` is reached so the caller can budget the
    vertex stream across multiple meshes.
    """
    if mesh is None or mesh.num_triangles == 0 or mesh.num_vertices == 0:
        return 0
    # Vertex stride: per skinny.mesh, vertices are 32 bytes (pos3 + pad +
    # normal3 + pad + uv2 — see scene.py loader). Read the first 12 bytes
    # of each 32-byte stride for position.
    vbytes = mesh.vertex_bytes
    ibytes = mesh.index_bytes
    if not vbytes or not ibytes:
        return 0
    stride = 32
    if mesh.num_vertices * stride != len(vbytes):
        # Fallback to compact 12-byte (pos-only) layout if encountered.
        if mesh.num_vertices * 12 == len(vbytes):
            stride = 12
        else:
            return 0
    positions = np.frombuffer(vbytes, dtype=np.float32)
    if stride == 32:
        positions = positions.reshape((-1, 8))[:, :3]
    else:
        positions = positions.reshape((-1, 3))
    indices = np.frombuffer(ibytes, dtype=np.uint32).reshape((-1, 3))

    # World-transform all vertices once (USD/row-vector convention).
    homo = np.concatenate(
        [positions.astype(np.float32), np.ones((positions.shape[0], 1), np.float32)],
        axis=1,
    )
    world = (homo @ transform)[:, :3]

    appended = 0
    for tri in indices:
        if max_vertices_remaining and appended + 6 > max_vertices_remaining:
            break
        a, b, c = world[tri[0]], world[tri[1]], world[tri[2]]
        _emit_line(out, a, color, b, color)
        _emit_line(out, b, color, c, color)
        _emit_line(out, c, color, a, color)
        appended += 6
    return appended


# ─── DebugViewport ────────────────────────────────────────────────────


class DebugViewport:
    """Owning class for the secondary debug render window."""

    def __init__(
        self, vk_ctx: VulkanContext, shader_dir: Path,
        width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT,
        *, embedded: bool = False,
    ) -> None:
        self._vk_ctx = vk_ctx
        self._shader_dir = shader_dir
        self._width = int(width)
        self._height = int(height)
        # ``embedded`` skips GLFW window + surface + swapchain and instead
        # renders to a single offscreen color image; ``render_embedded()``
        # copies that image into a host-visible staging buffer and returns
        # raw RGBA8 bytes for a Qt blit.
        self._embedded = bool(embedded)

        self._open = False
        self._window = None
        self._surface = None
        self._swapchain = None
        self._swapchain_format = (
            vk.VK_FORMAT_R8G8B8A8_UNORM if self._embedded
            else vk.VK_FORMAT_B8G8R8A8_UNORM
        )
        self._swapchain_images: list = []
        self._swapchain_views: list = []
        # Embedded-mode targets.
        self._offscreen_image = None
        self._offscreen_memory = None
        self._offscreen_view = None
        self._readback_buffer = None
        self._readback_memory = None
        self._readback_mapped = None
        self._readback_size = 0
        self._depth_image = None
        self._depth_memory = None
        self._depth_view = None
        self._render_pass = None
        self._framebuffers: list = []
        self._pipeline_layout = None
        self._descriptor_set_layout = None
        self._descriptor_pool = None
        self._descriptor_set = None
        self._pipeline_lines = None
        self._pipeline_tris = None
        self._vert_module = None
        self._frag_module = None
        self._vbo = None
        self._vbo_memory = None
        self._vbo_mapped = None
        self._tri_vbo = None
        self._tri_vbo_memory = None
        self._tri_vbo_mapped = None
        self._ubo_buffer = None
        self._ubo_memory = None
        self._cmd_buffer = None
        self._image_available_sem = None
        self._render_finished_sem = None
        self._in_flight_fence = None
        self._needs_resize = False

        # Cameras for the debug viewport itself.
        self.orbit_camera = OrbitCamera()
        self.orbit_camera.distance = 6.0
        self.orbit_camera.fov = 50.0
        self.free_camera = FreeCamera()
        self.camera_mode: str = "orbit"

        # Display options
        self.show_mesh_wires: bool = False  # default: AABBs only
        self.show_grid: bool = True
        self.show_frustum: bool = True
        self.show_lens: bool = True
        self.show_glyph: bool = True
        self.show_focus_plane: bool = True
        self.show_render_area: bool = True
        self.show_dof_planes: bool = True
        self.ortho_mode: bool = False
        self.show_hud: bool = True

        # Input state for window-local callbacks
        self._left_down = False
        self._right_down = False
        self._last_mx = 0.0
        self._last_my = 0.0
        self._key_state: dict[int, bool] = {}

    # ── Public API ───────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._open

    def toggle(self) -> None:
        if self._open:
            self.close()
        else:
            self.open()

    def open(self) -> None:
        if self._open:
            return
        if self._embedded:
            if self._cmd_buffer is None:
                self._build_embedded_resources()
        elif self._window is None:
            self._build_window_and_resources()
        else:
            import glfw
            glfw.show_window(self._window)
        self._open = True

    def close(self) -> None:
        if not self._open:
            return
        if not self._embedded and self._window is not None:
            import glfw
            glfw.hide_window(self._window)
        self._open = False

    def update(self, dt: float) -> None:
        """Advance debug-cam free-fly translation. Called per frame regardless
        of open state to keep state fresh for the moment of toggle on.
        Embedded mode: caller drives ``move_free_camera`` from Qt key
        events instead — this poll is GLFW-only.
        """
        if not self._open or self._embedded:
            return
        if self.camera_mode != "free":
            return
        import glfw
        w = self._window
        if w is None:
            return
        f = (glfw.get_key(w, glfw.KEY_W) == glfw.PRESS) - (glfw.get_key(w, glfw.KEY_S) == glfw.PRESS)
        r = (glfw.get_key(w, glfw.KEY_D) == glfw.PRESS) - (glfw.get_key(w, glfw.KEY_A) == glfw.PRESS)
        u = (glfw.get_key(w, glfw.KEY_E) == glfw.PRESS) - (glfw.get_key(w, glfw.KEY_Q) == glfw.PRESS)
        if f or r or u:
            self.free_camera.move(float(f), float(r), float(u), dt)

    def render(self, renderer) -> None:
        """Render one frame using state pulled from ``renderer`` (the main
        compute Renderer). No-op when closed. GLFW-windowed path only —
        embedded mode uses ``render_embedded``.
        """
        if not self._open or self._embedded or self._window is None:
            return
        import glfw
        if glfw.window_should_close(self._window):
            self.close()
            return
        if self._needs_resize:
            self._recreate_swapchain()
            self._needs_resize = False

        self._draw_frame(renderer)

    def render_embedded(self, renderer) -> bytes | None:
        """Embedded mode: render one frame to the offscreen image, copy
        into the host-visible staging buffer, return raw RGBA8 bytes.

        Returns ``None`` when closed or before resources are built.
        """
        if not self._open or not self._embedded:
            return None
        if self._cmd_buffer is None:
            return None
        return self._draw_frame_embedded(renderer)

    def resize_embedded(self, width: int, height: int) -> None:
        """Embedded mode: rebuild offscreen target + depth + staging when
        the host widget changes size.
        """
        if not self._embedded:
            return
        w = max(int(width), 1)
        h = max(int(height), 1)
        if (w, h) == (self._width, self._height):
            return
        self._width, self._height = w, h
        if self._cmd_buffer is None:
            return
        ctx = self._vk_ctx
        vk.vkDeviceWaitIdle(ctx.device)
        self._destroy_offscreen_objects()
        self._create_offscreen_target()
        self._create_depth_buffer()
        self._create_framebuffers()

    def destroy(self) -> None:
        """Tear down all Vulkan resources (and the GLFW window, if any)."""
        if not self._embedded and self._window is None:
            return
        if self._embedded and self._cmd_buffer is None:
            return
        ctx = self._vk_ctx
        vk.vkDeviceWaitIdle(ctx.device)
        if self._embedded:
            self._destroy_offscreen_objects()
        else:
            self._destroy_swapchain_objects()
        if self._pipeline_lines is not None:
            vk.vkDestroyPipeline(ctx.device, self._pipeline_lines, None)
        if self._pipeline_tris is not None:
            vk.vkDestroyPipeline(ctx.device, self._pipeline_tris, None)
        if self._pipeline_layout is not None:
            vk.vkDestroyPipelineLayout(ctx.device, self._pipeline_layout, None)
        if self._descriptor_pool is not None:
            vk.vkDestroyDescriptorPool(ctx.device, self._descriptor_pool, None)
        if self._descriptor_set_layout is not None:
            vk.vkDestroyDescriptorSetLayout(ctx.device, self._descriptor_set_layout, None)
        if self._vert_module is not None:
            vk.vkDestroyShaderModule(ctx.device, self._vert_module, None)
        if self._frag_module is not None:
            vk.vkDestroyShaderModule(ctx.device, self._frag_module, None)
        if self._vbo_memory is not None and self._vbo_mapped is not None:
            vk.vkUnmapMemory(ctx.device, self._vbo_memory)
            self._vbo_mapped = None
        if self._vbo is not None:
            vk.vkDestroyBuffer(ctx.device, self._vbo, None)
        if self._vbo_memory is not None:
            vk.vkFreeMemory(ctx.device, self._vbo_memory, None)
        if self._tri_vbo_memory is not None and self._tri_vbo_mapped is not None:
            vk.vkUnmapMemory(ctx.device, self._tri_vbo_memory)
            self._tri_vbo_mapped = None
        if self._tri_vbo is not None:
            vk.vkDestroyBuffer(ctx.device, self._tri_vbo, None)
        if self._tri_vbo_memory is not None:
            vk.vkFreeMemory(ctx.device, self._tri_vbo_memory, None)
        if self._ubo_memory is not None and getattr(self, "_ubo_mapped", None) is not None:
            vk.vkUnmapMemory(ctx.device, self._ubo_memory)
            self._ubo_mapped = None
        if self._ubo_buffer is not None:
            vk.vkDestroyBuffer(ctx.device, self._ubo_buffer, None)
        if self._ubo_memory is not None:
            vk.vkFreeMemory(ctx.device, self._ubo_memory, None)
        if self._image_available_sem is not None:
            vk.vkDestroySemaphore(ctx.device, self._image_available_sem, None)
            self._image_available_sem = None
        if self._render_finished_sem is not None:
            vk.vkDestroySemaphore(ctx.device, self._render_finished_sem, None)
            self._render_finished_sem = None
        if self._in_flight_fence is not None:
            vk.vkDestroyFence(ctx.device, self._in_flight_fence, None)
            self._in_flight_fence = None
        if self._render_pass is not None:
            vk.vkDestroyRenderPass(ctx.device, self._render_pass, None)
            self._render_pass = None
        if self._surface is not None:
            ctx._vkDestroySurfaceKHR(ctx.instance, self._surface, None)
            self._surface = None

        if not self._embedded and self._window is not None:
            import glfw
            glfw.destroy_window(self._window)
            self._window = None
        self._cmd_buffer = None

    # ── Window & resource setup ──────────────────────────────────

    def _build_embedded_resources(self) -> None:
        """Build all GPU resources for offscreen rendering. No GLFW, no
        surface, no swapchain.
        """
        self._verify_embedded_graphics_capable()
        self._create_offscreen_target()
        self._create_render_pass()
        self._create_depth_buffer()
        self._create_framebuffers()
        self._create_descriptor_layout_and_pool()
        self._create_pipeline()
        self._create_buffers()
        self._allocate_command_buffer()
        self._create_sync_primitives()
        self._update_descriptor_set()

    def _verify_embedded_graphics_capable(self) -> None:
        """Embedded mode only needs the compute queue family to support
        graphics (no present requirement). Most modern GPUs expose one
        unified graphics+compute queue; only obscure compute-only setups
        hit this path.
        """
        ctx = self._vk_ctx
        family_idx = ctx.queue_family_indices["compute"]
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device)
        flags = families[family_idx].queueFlags
        if not (flags & vk.VK_QUEUE_GRAPHICS_BIT):
            raise RuntimeError(
                "Compute queue family lacks VK_QUEUE_GRAPHICS_BIT — embedded "
                "debug viewport needs a graphics-capable queue."
            )

    def _create_offscreen_target(self) -> None:
        """Single offscreen color image + readback staging buffer.

        The color image is sized to ``(self._width, self._height)`` with
        format ``VK_FORMAT_R8G8B8A8_UNORM`` (matches Qt's
        ``QImage::Format_RGBA8888``), usage ``COLOR_ATTACHMENT |
        TRANSFER_SRC``, device-local. Staging buffer is host-visible
        coherent, sized ``width * height * 4``.
        """
        ctx = self._vk_ctx
        img_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=self._swapchain_format,
            extent=vk.VkExtent3D(width=self._width, height=self._height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=(
                vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            ),
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self._offscreen_image = vk.vkCreateImage(ctx.device, img_info, None)
        reqs = vk.vkGetImageMemoryRequirements(ctx.device, self._offscreen_image)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        mtype = _find_memory_type(
            reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props,
        )
        self._offscreen_memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=reqs.size, memoryTypeIndex=mtype,
            ),
            None,
        )
        vk.vkBindImageMemory(
            ctx.device, self._offscreen_image, self._offscreen_memory, 0,
        )
        view_info = vk.VkImageViewCreateInfo(
            image=self._offscreen_image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=self._swapchain_format,
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        self._offscreen_view = vk.vkCreateImageView(ctx.device, view_info, None)

        # Staging buffer for image → host readback.
        self._readback_size = self._width * self._height * 4
        buf_info = vk.VkBufferCreateInfo(
            size=self._readback_size,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self._readback_buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)
        buf_reqs = vk.vkGetBufferMemoryRequirements(
            ctx.device, self._readback_buffer,
        )
        buf_type = _find_memory_type(
            buf_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        self._readback_memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=buf_reqs.size, memoryTypeIndex=buf_type,
            ),
            None,
        )
        vk.vkBindBufferMemory(
            ctx.device, self._readback_buffer, self._readback_memory, 0,
        )
        self._readback_mapped = vk.vkMapMemory(
            ctx.device, self._readback_memory, 0, self._readback_size, 0,
        )

    def _build_window_and_resources(self) -> None:
        import glfw
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self._window = glfw.create_window(
            self._width, self._height, "Skinny — Camera Debug", None, None,
        )
        if not self._window:
            raise RuntimeError("Failed to create debug-viewport GLFW window")
        glfw.set_framebuffer_size_callback(self._window, self._on_framebuffer_resize)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_mouse_move)
        glfw.set_scroll_callback(self._window, self._on_scroll)
        glfw.set_key_callback(self._window, self._on_key)

        self._surface = self._create_surface(self._window)
        self._verify_graphics_capable()
        self._create_swapchain()
        self._create_render_pass()
        self._create_depth_buffer()
        self._create_framebuffers()
        self._create_descriptor_layout_and_pool()
        self._create_pipeline()
        self._create_buffers()
        self._allocate_command_buffer()
        self._create_sync_primitives()
        self._update_descriptor_set()

    def _create_surface(self, window):
        import glfw
        ctx = self._vk_ctx
        instance_handle = int(vk.ffi.cast("uintptr_t", ctx.instance))
        surface = ctypes.c_void_p(0)
        result = glfw.create_window_surface(
            instance_handle, window, None, ctypes.byref(surface),
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create debug viewport surface: {result}")
        return vk.ffi.cast("VkSurfaceKHR", surface.value)

    def _verify_graphics_capable(self) -> None:
        ctx = self._vk_ctx
        family_idx = ctx.queue_family_indices["compute"]
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device)
        flags = families[family_idx].queueFlags
        if not (flags & vk.VK_QUEUE_GRAPHICS_BIT):
            raise RuntimeError(
                "Selected queue family lacks VK_QUEUE_GRAPHICS_BIT — debug "
                "viewport needs a graphics-capable queue.",
            )
        if not ctx._vkGetPhysicalDeviceSurfaceSupportKHR(
            ctx.physical_device, family_idx, self._surface,
        ):
            raise RuntimeError(
                "Compute/graphics queue does not present to the debug viewport surface.",
            )

    # ── Swapchain ────────────────────────────────────────────────

    def _create_swapchain(self) -> None:
        ctx = self._vk_ctx
        capabilities = ctx._vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            ctx.physical_device, self._surface,
        )
        formats = ctx._vkGetPhysicalDeviceSurfaceFormatsKHR(
            ctx.physical_device, self._surface,
        )
        present_modes = ctx._vkGetPhysicalDeviceSurfacePresentModesKHR(
            ctx.physical_device, self._surface,
        )
        chosen_format = formats[0]
        for fmt in formats:
            if (
                fmt.format == vk.VK_FORMAT_B8G8R8A8_UNORM
                and fmt.colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                chosen_format = fmt
                break
        chosen_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if vk.VK_PRESENT_MODE_MAILBOX_KHR in present_modes:
            chosen_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR

        # Use the current framebuffer extent rather than the user's hint
        # so we honour the OS-imposed surface size on creation.
        cur_extent = capabilities.currentExtent
        if cur_extent.width != 0xFFFFFFFF:
            w, h = int(cur_extent.width), int(cur_extent.height)
        else:
            w, h = self._width, self._height
        self._width, self._height = max(w, 1), max(h, 1)
        extent = vk.VkExtent2D(width=self._width, height=self._height)

        image_count = capabilities.minImageCount + 1
        if capabilities.maxImageCount > 0:
            image_count = min(image_count, capabilities.maxImageCount)

        create_info = vk.VkSwapchainCreateInfoKHR(
            surface=self._surface,
            minImageCount=image_count,
            imageFormat=chosen_format.format,
            imageColorSpace=chosen_format.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            preTransform=capabilities.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=chosen_mode,
            clipped=vk.VK_TRUE,
        )
        self._swapchain = ctx._vkCreateSwapchainKHR(ctx.device, create_info, None)
        self._swapchain_format = chosen_format.format
        self._swapchain_images = list(
            ctx._vkGetSwapchainImagesKHR(ctx.device, self._swapchain),
        )
        self._swapchain_views = []
        for img in self._swapchain_images:
            view_info = vk.VkImageViewCreateInfo(
                image=img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self._swapchain_format,
                components=vk.VkComponentMapping(
                    r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1,
                ),
            )
            self._swapchain_views.append(
                vk.vkCreateImageView(ctx.device, view_info, None),
            )

    # ── Depth ────────────────────────────────────────────────────

    def _create_depth_buffer(self) -> None:
        ctx = self._vk_ctx
        img_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=_DEPTH_FORMAT,
            extent=vk.VkExtent3D(width=self._width, height=self._height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self._depth_image = vk.vkCreateImage(ctx.device, img_info, None)
        reqs = vk.vkGetImageMemoryRequirements(ctx.device, self._depth_image)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        mem_type = _find_memory_type(
            reqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props,
        )
        self._depth_memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=reqs.size, memoryTypeIndex=mem_type,
            ),
            None,
        )
        vk.vkBindImageMemory(ctx.device, self._depth_image, self._depth_memory, 0)

        view_info = vk.VkImageViewCreateInfo(
            image=self._depth_image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=_DEPTH_FORMAT,
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        self._depth_view = vk.vkCreateImageView(ctx.device, view_info, None)

    # ── Render pass & framebuffers ───────────────────────────────

    def _create_render_pass(self) -> None:
        ctx = self._vk_ctx
        # Embedded mode: render pass leaves the color image in
        # TRANSFER_SRC so the subsequent ``vkCmdCopyImageToBuffer`` can
        # read it without an extra barrier. Windowed mode hands the image
        # to the presentation engine.
        final_color_layout = (
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL if self._embedded
            else vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        )
        color_attachment = vk.VkAttachmentDescription(
            format=self._swapchain_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=final_color_layout,
        )
        depth_attachment = vk.VkAttachmentDescription(
            format=_DEPTH_FORMAT,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        color_ref = vk.VkAttachmentReference(
            attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        depth_ref = vk.VkAttachmentReference(
            attachment=1, layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_ref],
            pDepthStencilAttachment=depth_ref,
        )
        dep = vk.VkSubpassDependency(
            srcSubpass=vk.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=(
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
            ),
            dstStageMask=(
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
            ),
            srcAccessMask=0,
            dstAccessMask=(
                vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            ),
        )
        rp_info = vk.VkRenderPassCreateInfo(
            attachmentCount=2,
            pAttachments=[color_attachment, depth_attachment],
            subpassCount=1,
            pSubpasses=[subpass],
            dependencyCount=1,
            pDependencies=[dep],
        )
        self._render_pass = vk.vkCreateRenderPass(ctx.device, rp_info, None)

    def _create_framebuffers(self) -> None:
        ctx = self._vk_ctx
        self._framebuffers = []
        views = (
            [self._offscreen_view] if self._embedded
            else self._swapchain_views
        )
        for view in views:
            fb_info = vk.VkFramebufferCreateInfo(
                renderPass=self._render_pass,
                attachmentCount=2,
                pAttachments=[view, self._depth_view],
                width=self._width,
                height=self._height,
                layers=1,
            )
            self._framebuffers.append(
                vk.vkCreateFramebuffer(ctx.device, fb_info, None),
            )

    # ── Descriptor & pipeline ────────────────────────────────────

    def _create_descriptor_layout_and_pool(self) -> None:
        ctx = self._vk_ctx
        binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
        )
        self._descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device,
            vk.VkDescriptorSetLayoutCreateInfo(bindingCount=1, pBindings=[binding]),
            None,
        )
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1,
        )
        self._descriptor_pool = vk.vkCreateDescriptorPool(
            ctx.device,
            vk.VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size],
            ),
            None,
        )
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=self._descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self._descriptor_set_layout],
        )
        self._descriptor_set = vk.vkAllocateDescriptorSets(ctx.device, alloc_info)[0]

    def _create_pipeline(self) -> None:
        ctx = self._vk_ctx
        out_dir = self._shader_dir
        vert_spv = _compile_slang_stage(
            out_dir, "debug_line", "vs_main", "vertex",
            out_dir / "debug_line.vs.spv",
        )
        frag_spv = _compile_slang_stage(
            out_dir, "debug_line", "fs_main", "fragment",
            out_dir / "debug_line.fs.spv",
        )
        vbytes = vert_spv.read_bytes()
        fbytes = frag_spv.read_bytes()
        self._vert_module = vk.vkCreateShaderModule(
            ctx.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(vbytes), pCode=vbytes),
            None,
        )
        self._frag_module = vk.vkCreateShaderModule(
            ctx.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(fbytes), pCode=fbytes),
            None,
        )

        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=self._vert_module, pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=self._frag_module, pName="main",
            ),
        ]
        binding_desc = vk.VkVertexInputBindingDescription(
            binding=0, stride=_VERTEX_STRIDE_BYTES,
            inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        )
        attrs = [
            vk.VkVertexInputAttributeDescription(
                location=0, binding=0,
                format=vk.VK_FORMAT_R32G32B32_SFLOAT, offset=0,
            ),
            vk.VkVertexInputAttributeDescription(
                location=1, binding=0,
                format=vk.VK_FORMAT_R32G32B32A32_SFLOAT, offset=12,
            ),
        ]
        vinput = vk.VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[binding_desc],
            vertexAttributeDescriptionCount=2,
            pVertexAttributeDescriptions=attrs,
        )
        ia_lines = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            primitiveRestartEnable=vk.VK_FALSE,
        )
        ia_tris = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE,
        )
        viewport_state = vk.VkPipelineViewportStateCreateInfo(
            viewportCount=1, scissorCount=1,
        )
        raster = vk.VkPipelineRasterizationStateCreateInfo(
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            cullMode=vk.VK_CULL_MODE_NONE,
            frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE,
            lineWidth=1.0,
        )
        ms = vk.VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
            sampleShadingEnable=vk.VK_FALSE,
        )
        ds_opaque = vk.VkPipelineDepthStencilStateCreateInfo(
            depthTestEnable=vk.VK_TRUE,
            depthWriteEnable=vk.VK_TRUE,
            depthCompareOp=vk.VK_COMPARE_OP_LESS,
            depthBoundsTestEnable=vk.VK_FALSE,
            stencilTestEnable=vk.VK_FALSE,
        )
        ds_blend = vk.VkPipelineDepthStencilStateCreateInfo(
            depthTestEnable=vk.VK_TRUE,
            depthWriteEnable=vk.VK_FALSE,  # transparent — read but don't write
            depthCompareOp=vk.VK_COMPARE_OP_LESS,
            depthBoundsTestEnable=vk.VK_FALSE,
            stencilTestEnable=vk.VK_FALSE,
        )
        color_mask = (
            vk.VK_COLOR_COMPONENT_R_BIT
            | vk.VK_COLOR_COMPONENT_G_BIT
            | vk.VK_COLOR_COMPONENT_B_BIT
            | vk.VK_COLOR_COMPONENT_A_BIT
        )
        cb_attachment_opaque = vk.VkPipelineColorBlendAttachmentState(
            blendEnable=vk.VK_FALSE,
            colorWriteMask=color_mask,
        )
        cb_opaque = vk.VkPipelineColorBlendStateCreateInfo(
            logicOpEnable=vk.VK_FALSE,
            attachmentCount=1, pAttachments=[cb_attachment_opaque],
        )
        cb_attachment_blend = vk.VkPipelineColorBlendAttachmentState(
            blendEnable=vk.VK_TRUE,
            srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp=vk.VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            alphaBlendOp=vk.VK_BLEND_OP_ADD,
            colorWriteMask=color_mask,
        )
        cb_blend = vk.VkPipelineColorBlendStateCreateInfo(
            logicOpEnable=vk.VK_FALSE,
            attachmentCount=1, pAttachments=[cb_attachment_blend],
        )
        dynamic = vk.VkPipelineDynamicStateCreateInfo(
            dynamicStateCount=2,
            pDynamicStates=[vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR],
        )

        self._pipeline_layout = vk.vkCreatePipelineLayout(
            ctx.device,
            vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=1, pSetLayouts=[self._descriptor_set_layout],
            ),
            None,
        )

        line_info = vk.VkGraphicsPipelineCreateInfo(
            stageCount=2, pStages=stages,
            pVertexInputState=vinput,
            pInputAssemblyState=ia_lines,
            pViewportState=viewport_state,
            pRasterizationState=raster,
            pMultisampleState=ms,
            pDepthStencilState=ds_opaque,
            pColorBlendState=cb_opaque,
            pDynamicState=dynamic,
            layout=self._pipeline_layout,
            renderPass=self._render_pass,
            subpass=0,
        )
        tri_info = vk.VkGraphicsPipelineCreateInfo(
            stageCount=2, pStages=stages,
            pVertexInputState=vinput,
            pInputAssemblyState=ia_tris,
            pViewportState=viewport_state,
            pRasterizationState=raster,
            pMultisampleState=ms,
            pDepthStencilState=ds_blend,
            pColorBlendState=cb_blend,
            pDynamicState=dynamic,
            layout=self._pipeline_layout,
            renderPass=self._render_pass,
            subpass=0,
        )
        pipelines = vk.vkCreateGraphicsPipelines(
            ctx.device, vk.VK_NULL_HANDLE, 2, [line_info, tri_info], None,
        )
        self._pipeline_lines = pipelines[0]
        self._pipeline_tris = pipelines[1]

    # ── Buffers (vertex stream + UBO) ────────────────────────────

    def _create_buffers(self) -> None:
        ctx = self._vk_ctx
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)

        # Line VBO — host-visible coherent, persistently mapped.
        self._vbo, self._vbo_memory, self._vbo_mapped = self._make_host_vbo(
            mem_props, _LINE_BUFFER_BYTES,
        )
        # Triangle VBO (smaller — only the focus plane + iris disk today).
        self._tri_vbo, self._tri_vbo_memory, self._tri_vbo_mapped = (
            self._make_host_vbo(mem_props, _TRI_BUFFER_BYTES)
        )

        # UBO — host-visible coherent (only 64 bytes for the view*proj).
        ubo_size = 64
        ubo_info = vk.VkBufferCreateInfo(
            size=ubo_size,
            usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self._ubo_buffer = vk.vkCreateBuffer(ctx.device, ubo_info, None)
        ubo_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self._ubo_buffer)
        ubo_type = _find_memory_type(
            ubo_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        self._ubo_memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=ubo_reqs.size, memoryTypeIndex=ubo_type,
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, self._ubo_buffer, self._ubo_memory, 0)
        self._ubo_size = ubo_size
        self._ubo_mapped = vk.vkMapMemory(
            ctx.device, self._ubo_memory, 0, ubo_size, 0,
        )

    def _make_host_vbo(self, mem_props, size_bytes: int):
        ctx = self._vk_ctx
        info = vk.VkBufferCreateInfo(
            size=size_bytes,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(ctx.device, info, None)
        reqs = vk.vkGetBufferMemoryRequirements(ctx.device, buf)
        mtype = _find_memory_type(
            reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        mem = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=reqs.size, memoryTypeIndex=mtype,
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, buf, mem, 0)
        mapped = vk.vkMapMemory(ctx.device, mem, 0, size_bytes, 0)
        return buf, mem, mapped

    def _allocate_command_buffer(self) -> None:
        ctx = self._vk_ctx
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self._cmd_buffer = vk.vkAllocateCommandBuffers(ctx.device, alloc_info)[0]

    def _create_sync_primitives(self) -> None:
        ctx = self._vk_ctx
        sem_info = vk.VkSemaphoreCreateInfo()
        if not self._embedded:
            self._image_available_sem = vk.vkCreateSemaphore(ctx.device, sem_info, None)
            self._render_finished_sem = vk.vkCreateSemaphore(ctx.device, sem_info, None)
        fence_info = vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT)
        self._in_flight_fence = vk.vkCreateFence(ctx.device, fence_info, None)

    def _update_descriptor_set(self) -> None:
        ctx = self._vk_ctx
        buffer_info = vk.VkDescriptorBufferInfo(
            buffer=self._ubo_buffer, offset=0, range=self._ubo_size,
        )
        write = vk.VkWriteDescriptorSet(
            dstSet=self._descriptor_set,
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=[buffer_info],
        )
        vk.vkUpdateDescriptorSets(ctx.device, 1, [write], 0, None)

    # ── Per-frame draw ───────────────────────────────────────────

    def _draw_frame(self, renderer) -> None:
        ctx = self._vk_ctx

        vk.vkWaitForFences(ctx.device, 1, [self._in_flight_fence], vk.VK_TRUE, 2_000_000_000)

        try:
            image_index = ctx.vkAcquireNextImageKHR(
                ctx.device, self._swapchain, 2_000_000_000,
                self._image_available_sem, vk.VK_NULL_HANDLE,
            )
        except vk.VkErrorOutOfDateKhr:
            self._needs_resize = True
            return
        except vk.VkSuboptimalKhr:
            self._needs_resize = True
            return

        vk.vkResetFences(ctx.device, 1, [self._in_flight_fence])

        # Build vertex streams + UBO from current renderer state.
        line_count, tri_count = self._build_geometry(renderer)
        self._upload_camera_ubo()

        # Record cmd buffer.
        cb = self._cmd_buffer
        vk.vkResetCommandBuffer(cb, 0)
        vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))
        clear_color = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=[0.05, 0.05, 0.07, 1.0]),
        )
        clear_depth = vk.VkClearValue(
            depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0),
        )
        rp_begin = vk.VkRenderPassBeginInfo(
            renderPass=self._render_pass,
            framebuffer=self._framebuffers[image_index],
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(width=self._width, height=self._height),
            ),
            clearValueCount=2,
            pClearValues=[clear_color, clear_depth],
        )
        vk.vkCmdBeginRenderPass(cb, rp_begin, vk.VK_SUBPASS_CONTENTS_INLINE)

        viewport = vk.VkViewport(
            x=0.0, y=0.0,
            width=float(self._width), height=float(self._height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=vk.VkExtent2D(width=self._width, height=self._height),
        )
        vk.vkCmdSetViewport(cb, 0, 1, [viewport])
        vk.vkCmdSetScissor(cb, 0, 1, [scissor])
        vk.vkCmdBindDescriptorSets(
            cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._pipeline_layout, 0, 1, [self._descriptor_set], 0, None,
        )
        if line_count > 0:
            vk.vkCmdBindPipeline(
                cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._pipeline_lines,
            )
            vk.vkCmdBindVertexBuffers(cb, 0, 1, [self._vbo], [0])
            vk.vkCmdDraw(cb, line_count, 1, 0, 0)
        if tri_count > 0:
            vk.vkCmdBindPipeline(
                cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._pipeline_tris,
            )
            vk.vkCmdBindVertexBuffers(cb, 0, 1, [self._tri_vbo], [0])
            vk.vkCmdDraw(cb, tri_count, 1, 0, 0)

        vk.vkCmdEndRenderPass(cb)
        vk.vkEndCommandBuffer(cb)

        wait_stage = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        submit = vk.VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self._image_available_sem],
            pWaitDstStageMask=[wait_stage],
            commandBufferCount=1,
            pCommandBuffers=[cb],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self._render_finished_sem],
        )
        vk.vkQueueSubmit(ctx.compute_queue, 1, [submit], self._in_flight_fence)

        present_info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self._render_finished_sem],
            swapchainCount=1,
            pSwapchains=[self._swapchain],
            pImageIndices=[image_index],
        )
        try:
            ctx.vkQueuePresentKHR(ctx.compute_queue, present_info)
        except (vk.VkErrorOutOfDateKhr, vk.VkSuboptimalKhr):
            self._needs_resize = True

    def _draw_frame_embedded(self, renderer) -> bytes:
        """Embedded: render → copy → return RGBA8 bytes. Synchronous."""
        ctx = self._vk_ctx
        vk.vkWaitForFences(
            ctx.device, 1, [self._in_flight_fence], vk.VK_TRUE, 2_000_000_000,
        )
        vk.vkResetFences(ctx.device, 1, [self._in_flight_fence])

        line_count, tri_count = self._build_geometry(renderer)
        self._upload_camera_ubo()

        cb = self._cmd_buffer
        vk.vkResetCommandBuffer(cb, 0)
        vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))
        clear_color = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=[0.05, 0.05, 0.07, 1.0]),
        )
        clear_depth = vk.VkClearValue(
            depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0),
        )
        rp_begin = vk.VkRenderPassBeginInfo(
            renderPass=self._render_pass,
            framebuffer=self._framebuffers[0],
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(width=self._width, height=self._height),
            ),
            clearValueCount=2,
            pClearValues=[clear_color, clear_depth],
        )
        vk.vkCmdBeginRenderPass(cb, rp_begin, vk.VK_SUBPASS_CONTENTS_INLINE)

        viewport = vk.VkViewport(
            x=0.0, y=0.0,
            width=float(self._width), height=float(self._height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=vk.VkExtent2D(width=self._width, height=self._height),
        )
        vk.vkCmdSetViewport(cb, 0, 1, [viewport])
        vk.vkCmdSetScissor(cb, 0, 1, [scissor])
        vk.vkCmdBindDescriptorSets(
            cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._pipeline_layout, 0, 1, [self._descriptor_set], 0, None,
        )
        if line_count > 0:
            vk.vkCmdBindPipeline(
                cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._pipeline_lines,
            )
            vk.vkCmdBindVertexBuffers(cb, 0, 1, [self._vbo], [0])
            vk.vkCmdDraw(cb, line_count, 1, 0, 0)
        if tri_count > 0:
            vk.vkCmdBindPipeline(
                cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._pipeline_tris,
            )
            vk.vkCmdBindVertexBuffers(cb, 0, 1, [self._tri_vbo], [0])
            vk.vkCmdDraw(cb, tri_count, 1, 0, 0)

        vk.vkCmdEndRenderPass(cb)

        # Image is now in TRANSFER_SRC_OPTIMAL (render pass final layout).
        # Copy into the staging buffer for host readback.
        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0, bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(
                width=self._width, height=self._height, depth=1,
            ),
        )
        vk.vkCmdCopyImageToBuffer(
            cb, self._offscreen_image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            self._readback_buffer, 1, [region],
        )

        # Ensure host reads see the copy.
        host_barrier = vk.VkBufferMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=self._readback_buffer,
            offset=0, size=self._readback_size,
        )
        vk.vkCmdPipelineBarrier(
            cb,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, None, 1, [host_barrier], 0, None,
        )

        vk.vkEndCommandBuffer(cb)

        submit = vk.VkSubmitInfo(
            commandBufferCount=1, pCommandBuffers=[cb],
        )
        vk.vkQueueSubmit(ctx.compute_queue, 1, [submit], self._in_flight_fence)
        vk.vkWaitForFences(
            ctx.device, 1, [self._in_flight_fence], vk.VK_TRUE, 2_000_000_000,
        )

        # Read pixels from the host-mapped staging buffer.
        import cffi
        ffi = cffi.FFI()
        out = bytearray(self._readback_size)
        ffi.memmove(out, self._readback_mapped, self._readback_size)
        return bytes(out)

    def _destroy_offscreen_objects(self) -> None:
        ctx = self._vk_ctx
        for fb in self._framebuffers:
            vk.vkDestroyFramebuffer(ctx.device, fb, None)
        self._framebuffers = []
        if self._depth_view is not None:
            vk.vkDestroyImageView(ctx.device, self._depth_view, None)
            self._depth_view = None
        if self._depth_image is not None:
            vk.vkDestroyImage(ctx.device, self._depth_image, None)
            self._depth_image = None
        if self._depth_memory is not None:
            vk.vkFreeMemory(ctx.device, self._depth_memory, None)
            self._depth_memory = None
        if self._offscreen_view is not None:
            vk.vkDestroyImageView(ctx.device, self._offscreen_view, None)
            self._offscreen_view = None
        if self._offscreen_image is not None:
            vk.vkDestroyImage(ctx.device, self._offscreen_image, None)
            self._offscreen_image = None
        if self._offscreen_memory is not None:
            vk.vkFreeMemory(ctx.device, self._offscreen_memory, None)
            self._offscreen_memory = None
        if self._readback_memory is not None and self._readback_mapped is not None:
            vk.vkUnmapMemory(ctx.device, self._readback_memory)
            self._readback_mapped = None
        if self._readback_buffer is not None:
            vk.vkDestroyBuffer(ctx.device, self._readback_buffer, None)
            self._readback_buffer = None
        if self._readback_memory is not None:
            vk.vkFreeMemory(ctx.device, self._readback_memory, None)
            self._readback_memory = None

    # ── Geometry build & UBO upload ──────────────────────────────

    def _build_geometry(self, renderer) -> tuple[int, int]:
        line_floats: list = []
        tri_floats: list = []

        # Screen-space HUD first so it never gets truncated by a heavy
        # mesh-wireframe pass filling the line-vertex budget.
        if self.show_hud:
            _gen_hud_overlay(
                line_floats, self._hud_lines(),
                self._width, self._height,
            )

        # Ground grid
        if self.show_grid:
            _gen_grid(line_floats)

        # Render-camera state for frustum + glyph + lens + focus plane
        rcam = renderer.camera
        aspect = self._render_aspect_for(renderer)
        view_storage = rcam.view_matrix()
        proj_storage = rcam.projection_matrix(aspect)
        # _look_at and _perspective return the math matrices already
        # transposed for column-major GPU upload, so transpose them back
        # before doing CPU-side math.
        view_math = view_storage.T.astype(np.float64)
        proj_math = proj_storage.T.astype(np.float64)

        if self.show_frustum:
            _gen_frustum(line_floats, view_math, proj_math)

        forward, world_up = self._render_cam_basis(rcam)

        if self.show_glyph:
            _gen_camera_glyph(line_floats, np.asarray(rcam.position, np.float32),
                              forward, world_up)

        mm_per_unit = 1000.0
        usd = getattr(renderer, "_usd_scene", None)
        if usd is not None and getattr(usd, "mm_per_unit", None):
            mm_per_unit = float(usd.mm_per_unit)

        if self.show_lens and getattr(rcam, "lens", None) is not None:
            _gen_lens_rings(line_floats, np.asarray(rcam.position, np.float32),
                            forward, world_up, rcam.lens, mm_per_unit)

        # Focus plane + back-projected screen-rect overlay
        focus_distance = self._resolve_focus_distance(rcam)
        if self.show_focus_plane and focus_distance > 0.0:
            _gen_focus_plane(
                line_floats, tri_floats,
                np.asarray(rcam.position, np.float32),
                forward, world_up, focus_distance,
                fov_deg=float(rcam.fov), aspect=float(aspect),
            )
        if self.show_render_area and focus_distance > 0.0:
            fstop = float(getattr(rcam, "fstop", 0.0))
            if fstop > 0.0:
                alpha = _fstop_alpha(fstop)
                fill = (1.0, 0.10, 0.10, alpha)
                _gen_screen_rect_at_distance(
                    line_floats, tri_floats,
                    np.asarray(rcam.position, np.float32),
                    forward, world_up, focus_distance,
                    float(rcam.fov), float(aspect),
                    fill, _COL_RENDER_AREA_OUTLINE,
                )

        # Yellow corner dots: where frustum edges hit focus plane.
        if self.show_focus_plane and self.show_frustum and focus_distance > 0.0:
            dot_radius = max(0.025, 0.025 * focus_distance)
            _gen_fov_corner_dots(
                tri_floats,
                np.asarray(rcam.position, np.float32),
                forward, world_up, focus_distance,
                float(rcam.fov), float(aspect),
                dot_radius,
            )

        # Orange DOF near/far planes + frustum-corner dots at each.
        if self.show_dof_planes and focus_distance > 0.0:
            dof = self._dof_distances_units(rcam, focus_distance, mm_per_unit)
            if dof is not None:
                near_d, far_d = dof
                cam_pos = np.asarray(rcam.position, np.float32)
                for d in (near_d, far_d):
                    _gen_focus_plane(
                        line_floats, tri_floats,
                        cam_pos, forward, world_up, d,
                        fill_color=_COL_DOF_PLANE,
                        outline_color=_COL_DOF_OUTLINE,
                        fov_deg=float(rcam.fov), aspect=float(aspect),
                    )
                    if self.show_frustum:
                        _gen_fov_corner_dots(
                            tri_floats, cam_pos, forward, world_up, d,
                            float(rcam.fov), float(aspect),
                            max(0.022, 0.022 * d),
                            color=_COL_DOF_DOT,
                        )

        # Per-instance: AABB or full mesh wireframe (one or the other).
        usd_scene = getattr(renderer, "_usd_scene", None)
        instances = []
        if usd_scene is not None and getattr(usd_scene, "instances", None):
            instances = [inst for inst in usd_scene.instances if inst.enabled]

        if self.show_mesh_wires:
            for inst in instances:
                used = len(line_floats) // _VERTEX_FLOATS
                remaining = max(0, _MAX_LINE_VERTICES - used)
                if remaining < 6:
                    break
                _gen_mesh_wireframe(
                    line_floats, inst.mesh, inst.transform,
                    max_vertices_remaining=remaining,
                )
        else:
            for inst in instances:
                wmin, wmax = inst.world_bounds()
                _gen_aabb_box(line_floats, wmin, wmax)

        line_count = self._upload_stream(
            line_floats, self._vbo_mapped, _MAX_LINE_VERTICES, even=True,
        )
        tri_count = self._upload_stream(
            tri_floats, self._tri_vbo_mapped, _MAX_TRI_VERTICES, even=False,
            tri_align=True,
        )
        return line_count, tri_count

    def _hud_lines(self) -> list[str]:
        """Keyboard shortcuts for the debug viewport, drawn top-left."""
        cam_str = "ORBIT" if self.camera_mode == "orbit" else "FREE"
        proj_str = "ORTHO" if self.ortho_mode else "PERSP"
        return [
            f"DEBUG VIEWPORT  CAM:{cam_str}  PROJ:{proj_str}",
            "C CAM   F RESET   O ORTHO/PERSP",
            "T TOP   L LEFT    B BACK",
            "G GRID  M WIRES   D DOF",
            "P FOCUS PLANE     I RENDER AREA",
            "WASD MOVE  QE UP/DN  LMB ORBIT/LOOK",
            "SPACE HUD   ESC CLOSE",
        ]

    @staticmethod
    def _render_cam_basis(rcam) -> tuple[np.ndarray, np.ndarray]:
        forward = (rcam.target - rcam.position) if hasattr(rcam, "target") else rcam.forward()
        f_norm = float(np.linalg.norm(forward))
        if f_norm > 1e-6:
            forward = forward / f_norm
        else:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray(forward, np.float32), world_up

    @staticmethod
    def _resolve_focus_distance(rcam) -> float:
        focus = float(getattr(rcam, "focus_distance", 0.0))
        if focus > 1e-3:
            return focus
        # Pinhole / unauthored: orbit-cam falls back to its orbit distance,
        # free-cam has no implicit fallback (return 0 to suppress overlay).
        return float(getattr(rcam, "distance", 0.0))

    @staticmethod
    def _dof_distances_units(rcam, focus_distance_units: float, mm_per_unit: float,
                             far_cap_units: float = 200.0):
        """Thin-lens DOF: returns (near_units, far_units) or None.

        Uses standard formula:
            H = f²/(N·c) + f
            s_near = s·H / (H + (s − f))
            s_far  = s·H / (H − (s − f))     (capped when denom ≤ 0)
        with c = sensor_height_mm / 1500 (35mm-style CoC rule of thumb).
        All intermediate quantities are in mm; result converts back to
        scene units via ``mm_per_unit``.
        """
        fstop = float(getattr(rcam, "fstop", 0.0))
        focal_mm = float(getattr(rcam, "focal_length_mm", 0.0))
        sensor_h_mm = float(getattr(rcam, "vertical_aperture_mm", 24.0))
        if fstop <= 0.0 or focal_mm <= 0.0 or focus_distance_units <= 0.0:
            return None
        coc_mm = max(sensor_h_mm / 1500.0, 1e-4)
        s_mm = focus_distance_units * max(mm_per_unit, 1e-6)
        H_mm = (focal_mm * focal_mm) / (fstop * coc_mm) + focal_mm
        near_mm = s_mm * H_mm / (H_mm + (s_mm - focal_mm))
        denom = H_mm - (s_mm - focal_mm)
        if denom <= 1e-3:
            far_mm = far_cap_units * max(mm_per_unit, 1e-6)
        else:
            far_mm = s_mm * H_mm / denom
        near_units = max(near_mm / max(mm_per_unit, 1e-6), 1e-3)
        far_units = min(far_mm / max(mm_per_unit, 1e-6), far_cap_units)
        if far_units <= near_units:
            return None
        return near_units, far_units

    @staticmethod
    def _upload_stream(floats: list, mapped_ptr, max_vertices: int,
                       even: bool, tri_align: bool = False) -> int:
        if not floats or mapped_ptr is None:
            return 0
        arr = np.asarray(floats, dtype=np.float32)
        vert_count = arr.size // _VERTEX_FLOATS
        if vert_count > max_vertices:
            vert_count = max_vertices
            if even:
                vert_count &= ~1
        if tri_align:
            vert_count -= vert_count % 3
        if vert_count <= 0:
            return 0
        arr = arr[: vert_count * _VERTEX_FLOATS]
        buf_bytes = arr.tobytes()
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(mapped_ptr, buf_bytes, len(buf_bytes))
        return vert_count

    def _render_aspect_for(self, renderer) -> float:
        # Use the main renderer's render-target aspect (what the user
        # sees in the central viewport) rather than the debug viewport's
        # own surface — the frustum/lens overlays would otherwise show
        # the wrong aspect when the debug dock isn't 16:9.
        return float(renderer.width) / max(1.0, float(renderer.height))

    def _upload_camera_ubo(self) -> None:
        cam = self.orbit_camera if self.camera_mode == "orbit" else self.free_camera
        aspect = float(self._width) / max(1.0, float(self._height))
        # view_storage / proj_storage are the math matrices already
        # transposed for column-major GPU upload. row-major bytes of
        # (view_storage @ proj_storage) decode column-major as
        # proj_math @ view_math, which is what the shader's mul()
        # consumes.
        view_storage = cam.view_matrix()
        proj_storage = self._projection_storage(cam, aspect)
        upload = (view_storage @ proj_storage).astype(np.float32, copy=False)
        data = upload.tobytes(order="C")
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(self._ubo_mapped, data, len(data))

    def _projection_storage(self, cam, aspect: float) -> np.ndarray:
        """Return the (transposed-for-GPU) 4x4 projection matrix.

        Perspective is delegated to the camera. Ortho is built here using
        the orbit distance (or move_speed × heuristic for the free cam)
        as the vertical extent so framing roughly matches the perspective
        view at toggle time.
        """
        if not self.ortho_mode:
            p = cam.projection_matrix(aspect).copy()
            # Vulkan clip-space Y points down; renderer._perspective is
            # GL-style (Y up). Negate P[1,1] so world renders right-side
            # up in the rasterised debug viewport. HUD overlay vertices
            # bypass this matrix and are authored directly in Vulkan NDC.
            p[1, 1] = -p[1, 1]
            return p

        if hasattr(cam, "distance"):
            ref = float(cam.distance)
        else:
            ref = max(float(getattr(cam, "move_speed", 1.5)) * 4.0, 1.0)
        half_h = ref * float(np.tan(np.radians(float(cam.fov)) * 0.5))
        half_w = half_h * float(aspect)
        near = float(cam.near)
        far = float(cam.far)
        p = np.zeros((4, 4), dtype=np.float32)
        # Numpy storage = math P^T row-major (matches _perspective).
        p[0, 0] = 1.0 / max(half_w, 1e-6)
        # Negate Y for Vulkan clip-space (see perspective branch above).
        p[1, 1] = -1.0 / max(half_h, 1e-6)
        p[2, 2] = 1.0 / (near - far)
        p[3, 2] = near / (near - far)
        p[3, 3] = 1.0
        return p

    # ── Resize / teardown helpers ────────────────────────────────

    def _recreate_swapchain(self) -> None:
        ctx = self._vk_ctx
        import glfw
        # Wait until window has nonzero size (e.g. user finished minimising).
        w, h = glfw.get_framebuffer_size(self._window)
        while w == 0 or h == 0:
            glfw.wait_events()
            w, h = glfw.get_framebuffer_size(self._window)
        self._width, self._height = int(w), int(h)
        vk.vkDeviceWaitIdle(ctx.device)
        self._destroy_swapchain_objects()
        self._create_swapchain()
        self._create_depth_buffer()
        self._create_framebuffers()

    def _destroy_swapchain_objects(self) -> None:
        ctx = self._vk_ctx
        for fb in self._framebuffers:
            vk.vkDestroyFramebuffer(ctx.device, fb, None)
        self._framebuffers = []
        if self._depth_view is not None:
            vk.vkDestroyImageView(ctx.device, self._depth_view, None)
            self._depth_view = None
        if self._depth_image is not None:
            vk.vkDestroyImage(ctx.device, self._depth_image, None)
            self._depth_image = None
        if self._depth_memory is not None:
            vk.vkFreeMemory(ctx.device, self._depth_memory, None)
            self._depth_memory = None
        for view in self._swapchain_views:
            vk.vkDestroyImageView(ctx.device, view, None)
        self._swapchain_views = []
        self._swapchain_images = []
        if self._swapchain is not None:
            ctx._vkDestroySwapchainKHR(ctx.device, self._swapchain, None)
            self._swapchain = None

    # ── GLFW callbacks ───────────────────────────────────────────

    def _on_framebuffer_resize(self, _win, _w, _h) -> None:
        self._needs_resize = True

    def _on_mouse_button(self, _win, button, action, _mods) -> None:
        import glfw
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_down = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_down = (action == glfw.PRESS)
        if action == glfw.PRESS:
            self._last_mx, self._last_my = glfw.get_cursor_pos(self._window)

    def _on_mouse_move(self, _win, mx, my) -> None:
        dx = float(mx) - self._last_mx
        dy = float(my) - self._last_my
        self._last_mx = float(mx)
        self._last_my = float(my)
        if self.camera_mode == "orbit":
            if self._left_down:
                self.orbit_camera.orbit(dx, dy)
            elif self._right_down:
                self.orbit_camera.pan(dx, dy)
        else:
            if self._left_down:
                self.free_camera.look(dx, dy)

    def _on_scroll(self, _win, _xoff, yoff) -> None:
        cam = self.orbit_camera if self.camera_mode == "orbit" else self.free_camera
        cam.zoom(float(yoff))

    def _on_key(self, _win, key, _scancode, action, _mods) -> None:
        import glfw
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_C:
            self._toggle_cam_mode()
        elif key == glfw.KEY_F:
            self._reset_debug_camera()
        elif key == glfw.KEY_M:
            self.show_mesh_wires = not self.show_mesh_wires
            print(f"[Debug viewport] mesh wires: {'on' if self.show_mesh_wires else 'off'} "
                  f"(AABBs {'off' if self.show_mesh_wires else 'on'})")
        elif key == glfw.KEY_G:
            self.show_grid = not self.show_grid
        elif key == glfw.KEY_P:
            self.show_focus_plane = not self.show_focus_plane
            print(f"[Debug viewport] focus plane: {'on' if self.show_focus_plane else 'off'}")
        elif key == glfw.KEY_I:
            self.show_render_area = not self.show_render_area
            print(f"[Debug viewport] render area: {'on' if self.show_render_area else 'off'}")
        elif key == glfw.KEY_O:
            self.ortho_mode = not self.ortho_mode
            print(f"[Debug viewport] projection: {'ortho' if self.ortho_mode else 'perspective'}")
        elif key == glfw.KEY_D:
            self.show_dof_planes = not self.show_dof_planes
            print(f"[Debug viewport] DOF planes: {'on' if self.show_dof_planes else 'off'}")
        elif key == glfw.KEY_T:
            self.view_top()
        elif key == glfw.KEY_B:
            self.view_back()
        elif key == glfw.KEY_L:
            self.view_left()
        elif key == glfw.KEY_SPACE:
            self.show_hud = not self.show_hud
            print(f"[Debug viewport] HUD: {'on' if self.show_hud else 'off'}")
        elif key == glfw.KEY_ESCAPE:
            self.close()

    def _toggle_cam_mode(self) -> None:
        if self.camera_mode == "orbit":
            o = self.orbit_camera
            self.free_camera.position = o.position.astype(np.float32).copy()
            self.free_camera.yaw = -o.yaw
            self.free_camera.pitch = -o.pitch
            self.camera_mode = "free"
        else:
            self.camera_mode = "orbit"
        print(f"[Debug viewport] camera mode: {self.camera_mode}")

    def _reset_debug_camera(self) -> None:
        self.orbit_camera = OrbitCamera()
        self.orbit_camera.distance = 6.0
        self.orbit_camera.fov = 50.0
        self.free_camera = FreeCamera()
        self.camera_mode = "orbit"

    # ── View shortcuts ───────────────────────────────────────────

    def view_top(self) -> None:
        self._snap_orbit(yaw=0.0, pitch=float(np.pi / 2.0 - 0.02))

    def view_left(self) -> None:
        self._snap_orbit(yaw=float(-np.pi / 2.0), pitch=0.0)

    def view_back(self) -> None:
        self._snap_orbit(yaw=float(np.pi), pitch=0.0)

    def _snap_orbit(self, yaw: float, pitch: float) -> None:
        """Switch to orbit mode and snap target/distance to fit the active scene."""
        self.camera_mode = "orbit"
        cam = self.orbit_camera
        cam.yaw = yaw
        cam.pitch = pitch
        bounds = self._scene_bounds_or_none()
        if bounds is not None:
            wmin, wmax = bounds
            centre = (wmin + wmax) * 0.5
            extent = float(np.linalg.norm(wmax - wmin))
            cam.target = centre.astype(np.float32)
            # Frame the bounding sphere given the orbit FOV.
            half_fov = float(np.radians(cam.fov)) * 0.5
            radius = max(extent * 0.5, 0.5)
            cam.distance = float(np.clip(
                radius / max(np.tan(half_fov), 1e-3) * 1.4, 0.5, 50.0,
            ))

    def _scene_bounds_or_none(self):
        renderer = getattr(self, "_renderer", None)
        if renderer is None:
            return None
        usd = getattr(renderer, "_usd_scene", None)
        if usd is None:
            return None
        try:
            return usd.world_bounds()
        except Exception:
            return None

    def attach_renderer(self, renderer) -> None:
        """Stash the main Renderer so view shortcuts can fit-to-bounds."""
        self._renderer = renderer
