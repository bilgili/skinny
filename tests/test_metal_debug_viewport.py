"""Headless Metal Camera Debug viewport render (change metal-tool-dock-render P2,
tasks 3.7 / 3.8).

Drives the full embedded path on a real MetalContext: `DebugViewport(embedded=
True).open()` builds the compute rasteriser instead of the Vulkan graphics
pipeline, and `render_embedded(renderer)` generates the CPU vertex streams
(grid + frustum + glyph + focus plane), dispatches `DebugRasterMetal`, and returns
an RGBA8 frame — the same shape the worker `DebugFrame` path consumes. Confirms
3.7 (the DebugViewport Metal branch) end to end without a GLFW/Qt window.

A lightweight stub stands in for the main Renderer (the debug streams only read
`camera` / `width` / `height` / `_usd_scene`), so this does NOT cold-compile the
megakernel — only debug_raster.slang.

DANGER — compiles debug_raster.slang (MTLCompilerService). Run through
scripts/guarded_metal.sh with the gate set:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_DEBUG_RASTER=1 PYTHONPATH=$PWD/src TIMEOUT_S=300 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest tests/test_metal_debug_viewport.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from skinny.backend_select import metal_available
from skinny.debug_raster_ref import CLEAR_RGBA8

pytest.importorskip("slangpy")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _PROJECT_ROOT / "src" / "skinny" / "shaders"
_GATE = "RUN_METAL_DEBUG_RASTER"


@pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=f"compiles debug_raster.slang; set {_GATE}=1 under scripts/guarded_metal.sh",
)
def test_metal_debug_viewport_embedded_render():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    try:
        from skinny.debug_viewport import DebugViewport
        from skinny.metal_context import MetalContext
        from skinny.renderer import OrbitCamera
    except OSError as exc:
        pytest.skip(f"import needs the Vulkan SDK on the dylib path: {exc}")

    class _StubRenderer:
        def __init__(self):
            self.camera = OrbitCamera()
            self.camera.distance = 6.0
            self.width = 256
            self.height = 256
            self._usd_scene = None

    W = H = 240
    ctx = MetalContext(window=None, width=64, height=64)
    dv = None
    try:
        dv = DebugViewport(vk_ctx=ctx, shader_dir=_SHADER_DIR,
                           width=W, height=H, embedded=True)
        assert dv.is_metal, "expected the Metal branch"
        dv.open()
        assert dv.is_open

        pixels = dv.render_embedded(_StubRenderer())
        assert pixels is not None, "Metal debug render returned None"
        assert len(pixels) == W * H * 4
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(H, W, 4)

        clear = np.array(CLEAR_RGBA8, np.uint8)
        lit = int(np.sum(np.any(img != clear, axis=-1)))
        # The grid + HUD + frustum draw a substantial number of pixels.
        assert lit > 200, f"debug frame nearly empty ({lit} lit px)"
        assert np.isfinite(img).all()

        # A second render (steady state — buffers reused) still succeeds.
        again = dv.render_embedded(_StubRenderer())
        assert again is not None and len(again) == W * H * 4
    finally:
        if dv is not None:
            dv.destroy()
        ctx.destroy()
