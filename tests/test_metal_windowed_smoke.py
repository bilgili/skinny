"""Windowed Metal megakernel **present smoke** (task 6.5, display-gated).

Opens a window, builds a megakernel Renderer on a windowed `MetalContext`, loads
a head, and drives several frames through the real windowed path
(`render` → `_render_windowed_metal`: dispatch the megakernel into the offscreen
image, blit it onto the acquired slang-rhi surface image, present). Every frame
drains the device, so completing the loop without a hang means each present fence
signalled.

Double-gated: the compile gate (`RUN_METAL_MEGAKERNEL_COMPILE=1`, run under
`scripts/guarded_metal.sh`) plus a display — `slangpy.Window` raises on a headless
host, where the test skips cleanly.

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_MEGAKERNEL_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=300 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_windowed_smoke.py -q
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from skinny.backend_select import metal_available

pytest.importorskip("slangpy")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _PROJECT_ROOT / "src" / "skinny" / "shaders"
_HDR_DIR = _PROJECT_ROOT / "hdrs"
_HEAD_OBJ = _PROJECT_ROOT / "heads" / "head.obj"
_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"
_W = _H = 256


def _pump_until_ready(renderer, *, budget_s: float = 90.0) -> bool:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        renderer.update(0.016)
        if (
            renderer._backend_render_ready
            and renderer._num_instances >= 1
            and renderer._baked_source_idx >= 0
        ):
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"renders the megakernel on Metal (MTLCompilerService RAM spike); set "
        f"{_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_windowed_megakernel_smoke():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    if not _HEAD_OBJ.exists():
        pytest.skip(f"head asset not found: {_HEAD_OBJ}")
    import slangpy as spy

    try:
        window = spy.Window(width=_W, height=_H, title="skinny-metal-megakernel-smoke")
    except Exception as exc:  # noqa: BLE001 — no window server / headless host
        pytest.skip(f"no display for the windowed Metal smoke: {exc}")

    try:
        from skinny.metal_context import MetalContext
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path
        pytest.skip(f"renderer import needs the Vulkan SDK on the dylib path: {exc}")

    ctx = MetalContext(window=window, width=_W, height=_H)
    renderer = None
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="megakernel",
            hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
        )
        renderer.load_model_from_path(_HEAD_OBJ)
        assert _pump_until_ready(renderer), "head load/bake/pipeline build timed out"
        assert renderer.is_metal and renderer._backend_render_ready

        # Drive several windowed frames through render() → _render_windowed_metal.
        # No exception + no hang (each frame drains the device) == every present
        # fence signalled.
        for _ in range(8):
            window.process_events()
            renderer.update(0.016)
            renderer.render()
    finally:
        if renderer is not None:
            renderer.cleanup()
        ctx.destroy()
