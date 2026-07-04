"""Hostless parity tests for the Camera Debug rasteriser reference
(change metal-tool-dock-render P2, task 3.6).

`skinny.debug_raster_ref` is the CPU mirror of `shaders/debug_raster.slang`. These
tests pin its transform + DDA + packing behaviour so the MSL kernel (diffed
against this reference on the GPU, task 3.8) has a checkable spec. No GPU.
"""

from __future__ import annotations

import numpy as np

from skinny.debug_raster_ref import (
    CLEAR_RGBA8,
    VERTEX_FLOATS,
    project_vertex,
    rasterise_lines,
)

_IDENTITY = np.eye(4, dtype=np.float64)


def _line(a_xyz, a_rgba, b_xyz, b_rgba):
    return [*a_xyz, *a_rgba, *b_xyz, *b_rgba]


def test_empty_stream_is_cleared():
    img = rasterise_lines([], _IDENTITY, 8, 8)
    assert img.shape == (8, 8, 4)
    assert np.all(img.reshape(-1, 4) == np.array(CLEAR_RGBA8, np.uint8))


def test_vertex_floats_layout():
    assert VERTEX_FLOATS == 7  # 3 pos + 4 rgba — must match debug_viewport


def test_projected_center_line_horizontal():
    # Identity viewProj: (x,y,0) → clip (x,y,0,1) → ndc (x,y). A line along x at
    # y=0 maps to a horizontal run through the vertical center row.
    W = H = 16
    verts = _line((-0.5, 0.0, 0.0), (1, 0, 0, 1), (0.5, 0.0, 0.0), (1, 0, 0, 1))
    img = rasterise_lines(verts, _IDENTITY, W, H)
    row = H // 2  # py = (0.5 - 0)*H = H/2
    red = np.array([255, 0, 0, 255], np.uint8)
    lit = np.where(np.all(img[row] == red, axis=-1))[0]
    assert lit.size >= 2, "expected a red horizontal run at the center row"
    # Centered around x = W/2 (ndc 0 → px W/2).
    assert lit.min() < W // 2 < lit.max()


def test_hud_sentinel_bypasses_viewproj():
    # a > 1.5 → xy is NDC directly, regardless of viewProj. A degenerate
    # viewProj (all zeros) would drop any projected vertex, so a drawn line
    # proves the bypass.
    W = H = 16
    verts = _line((-0.8, 0.0, 0.0), (0, 1, 0, 2.0), (0.8, 0.0, 0.0), (0, 1, 0, 2.0))
    img = rasterise_lines(verts, np.zeros((4, 4)), W, H)
    row = H // 2
    green = np.array([0, 255, 0, 255], np.uint8)
    lit = np.where(np.all(img[row] == green, axis=-1))[0]
    assert lit.size >= 2, "HUD sentinel line should draw despite zero viewProj"


def test_behind_eye_line_dropped():
    # viewProj whose w-row = z: a vertex at z=-1 has clip.w = -1 (behind eye)
    # and the whole line is skipped.
    vp = _IDENTITY.copy()
    vp[3] = [0.0, 0.0, 1.0, 0.0]  # w = z
    verts = _line((0.0, 0.0, -1.0), (1, 1, 1, 1), (0.2, 0.0, 2.0), (1, 1, 1, 1))
    img = rasterise_lines(verts, vp, 16, 16)
    assert np.all(img.reshape(-1, 4) == np.array(CLEAR_RGBA8, np.uint8)), \
        "a line with a behind-eye endpoint must be dropped"


def test_project_vertex_hud_and_projected():
    # HUD sentinel: NDC (0,0) → pixel center; depth 0; visible.
    px, py, depth, vis = project_vertex(
        np.array([0.0, 0.0, 0.0, 0, 0, 0, 2.0]), _IDENTITY, 100, 40)
    assert vis and depth == 0.0
    assert px == 50.0 and py == 20.0
    # Projected origin under identity → same pixel center, w=1.
    px2, py2, _d, vis2 = project_vertex(
        np.array([0.0, 0.0, 0.0, 1, 1, 1, 1.0]), _IDENTITY, 100, 40)
    assert vis2 and px2 == 50.0 and py2 == 20.0


def test_color_clamped_on_pack():
    # Over-range color clamps to 255 (matches the MSL clamp*255+0.5 pack).
    verts = _line((-0.5, 0.0, 0.0), (2.0, 2.0, 2.0, 1), (0.5, 0.0, 0.0), (2.0, 2.0, 2.0, 1))
    img = rasterise_lines(verts, _IDENTITY, 16, 16)
    row = 8
    white = np.array([255, 255, 255, 255], np.uint8)
    assert np.any(np.all(img[row] == white, axis=-1))
