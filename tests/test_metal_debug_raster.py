"""Metal Camera Debug rasteriser GPU cross-check
(change metal-tool-dock-render P2, tasks 3.1 / 3.8).

Dispatches `DebugRasterMetal` (debug_raster.slang) on a real MetalContext and
diffs the packed-RGBA8 output against the hostless numpy reference
(`skinny.debug_raster_ref.rasterise_lines`) for the same line stream — the kernel
is written to mirror the reference, so a projected line + a HUD sentinel line
must land on the same pixels. Confirms the shader compiles to MSL, the compute
kernels bind their vars, and the DDA/transform/pack math matches.

DANGER — cold-compiles debug_raster.slang (MTLCompilerService RAM spike). Run
ONLY through scripts/guarded_metal.sh with the gate set:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_DEBUG_RASTER=1 PYTHONPATH=$PWD/src TIMEOUT_S=300 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest tests/test_metal_debug_raster.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from skinny.backend_select import metal_available
from skinny.debug_raster_ref import CLEAR_RGBA8, rasterise

pytest.importorskip("slangpy")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _PROJECT_ROOT / "src" / "skinny" / "shaders"
_GATE = "RUN_METAL_DEBUG_RASTER"


def _lines():
    """A near red line + a farther blue line (coincident row → depth test) + a
    HUD-sentinel green vertical line."""
    return [
        # near red line (z=-0.4 → nearer) along x at y=0 → center row
        -0.6, 0.0, -0.4, 1.0, 0.0, 0.0, 1.0,
         0.6, 0.0, -0.4, 1.0, 0.0, 0.0, 1.0,
        # farther blue line (z=+0.4) same row — must be occluded by the red one
        -0.6, 0.0, 0.4, 0.0, 0.0, 1.0, 1.0,
         0.6, 0.0, 0.4, 0.0, 0.0, 1.0, 1.0,
        # HUD sentinel (a=2) green line along y at x=0 → center column
         0.0, -0.6, 0.0, 0.0, 1.0, 0.0, 2.0,
         0.0,  0.6, 0.0, 0.0, 1.0, 0.0, 2.0,
    ]


def _tris():
    """One half-alpha triangle at z=0 (in front of nothing → blends over bg)."""
    return [
        -0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 0.5,
         0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 0.5,
         0.0,  0.5, 0.0, 1.0, 1.0, 1.0, 0.5,
    ]


@pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=f"cold-compiles debug_raster.slang; set {_GATE}=1 under scripts/guarded_metal.sh",
)
def test_metal_debug_raster_matches_reference():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    try:
        from skinny.metal_compute import DebugRasterMetal
        from skinny.metal_context import MetalContext
    except OSError as exc:
        pytest.skip(f"import needs the Vulkan SDK on the dylib path: {exc}")

    W = H = 32
    vp = np.eye(4, dtype=np.float64)
    lines, tris = _lines(), _tris()

    ref = rasterise(lines, tris, vp, W, H)  # (H, W, 4) uint8

    ctx = MetalContext(window=None, width=64, height=64)
    raster = None
    try:
        raster = DebugRasterMetal(ctx, _SHADER_DIR)
        raw = raster.render(lines, tris, vp, W, H)
        assert len(raw) == W * H * 4
        gpu = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 4)

        clear = np.array(CLEAR_RGBA8, np.uint8)
        # Something was drawn (not an all-clear frame).
        assert np.any(np.any(gpu != clear, axis=-1)), "GPU frame is entirely background"
        # Corners are background (nothing drawn there in either image).
        assert np.all(gpu[0, 0] == clear) and np.all(gpu[H - 1, W - 1] == clear)

        # Per-channel ±1 agreement with the reference (fp32 GPU vs fp64 host
        # rounding at line/edge boundaries + blended texels).
        close = np.all(np.abs(gpu.astype(int) - ref.astype(int)) <= 1, axis=-1)
        frac = float(np.mean(close))
        assert frac >= 0.98, f"GPU vs reference within-1 agreement {frac:.4f}"

        # Depth test held: the farther blue line was occluded by the near red one.
        blue = np.array([0, 0, 255, 255], np.uint8)
        assert not np.any(np.all(gpu == blue, axis=-1)), "occluded blue line leaked"

        # The transparent triangle blended (a half-white texel exists).
        assert np.any((gpu[..., 0] > 120) & (gpu[..., 0] < 200)), "no blended tri texel"
    finally:
        if raster is not None:
            raster.destroy()
        ctx.destroy()
