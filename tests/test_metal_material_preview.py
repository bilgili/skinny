"""Metal **material-preview** render test (change metal-tool-dock-render, task 1.4).

Renders a material thumbnail through ``Renderer.render_material_preview`` on the
native ``MetalContext`` and asserts the returned buffer is the RGBA32F
``(pixels, size)`` shape the Material Graph dock reshapes — a lit primitive, not
all-zero / NaN. Before change metal-tool-dock-render P1 this raised
``AttributeError: 'ComputePipeline' object has no attribute
'descriptor_set_layout'`` on Metal (the Vulkan-only ``PreviewPipeline`` path); the
Metal branch dispatches ``preview_pass.slang`` by binding resources by name.

Scene: ``assets/three_materials_demo.usda`` (marble / wood / brass MaterialX
graph materials) — same asset the Vulkan headless smoke render uses, so at least
one ``material_id >= 1`` with a live graph is present. The megakernel pipeline
must build first (it owns the reflected MSL layout ``_pack_uniforms_msl`` /
``_build_metal_binds`` feed the preview dispatch) — the readiness pump asserts
that as a precondition.

DANGER — building the megakernel cold-compiles ``main_pass`` AND this test then
cold-compiles ``preview_pass`` (MTLCompilerService RAM spike). Run ONLY through
``scripts/guarded_metal.sh`` with the compile gate set:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_PREVIEW_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=420 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_material_preview.py -q
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from skinny.backend_select import metal_available

pytest.importorskip("slangpy")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _PROJECT_ROOT / "src" / "skinny" / "shaders"
_HDR_DIR = _PROJECT_ROOT / "hdrs"
_TATTOO_DIR = _PROJECT_ROOT / "tattoos"
_DEMO_SCENE = _PROJECT_ROOT / "assets" / "three_materials_demo.usda"
_COMPILE_GATE = "RUN_METAL_PREVIEW_COMPILE"
_PREVIEW_SIZE = 128


def _pump_until_ready(renderer, *, budget_s: float = 60.0) -> bool:
    """Drive update() until the demo scene has streamed its material instances
    and the megakernel pipeline (which owns the reflected MSL layout the preview
    dispatch reuses) has built. Returns readiness."""
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        renderer.update(0.025)
        scene = renderer._usd_scene
        if (
            renderer._backend_render_ready
            and scene is not None
            and len(scene.instances) >= 3
            and len(scene.materials) >= 2
        ):
            return True
        time.sleep(0.02)
    return False


def _first_previewable_material(renderer) -> int:
    """Pick a material_id the preview accepts (>= 1, in range). Prefer one with a
    live MaterialX graph (graph_id >= 2) so the graph dispatch path is exercised."""
    n = len(renderer._usd_scene.materials)
    graph_ids = renderer._material_graph_ids
    for mid in range(1, n):
        if int(graph_ids.get(mid, 0)) >= 2:
            return mid
    return 1 if n > 1 else 0


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"cold-compiles main_pass + preview_pass (MTLCompilerService RAM spike); "
        f"set {_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
@pytest.mark.skipif(not _DEMO_SCENE.exists(), reason="three_materials_demo.usda missing")
def test_metal_material_preview_renders():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    try:
        from skinny.metal_context import MetalContext
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path (renderer imports it)
        pytest.skip(f"renderer import needs the Vulkan SDK on the dylib path: {exc}")

    ctx = MetalContext(window=None, width=96, height=96)
    renderer = None
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="megakernel",
            hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
            tattoo_dir=_TATTOO_DIR if _TATTOO_DIR.is_dir() else None,
            usd_scene_path=_DEMO_SCENE,
        )
        assert _pump_until_ready(renderer), (
            "demo scene stream / megakernel pipeline build did not become ready"
        )
        assert renderer.is_metal, "expected the native Metal backend"

        mid = _first_previewable_material(renderer)
        assert mid >= 1, "no previewable material_id (>= 1) in the demo scene"

        # Dispatch the sphere preview (prim 0). Before P1 this raised on Metal.
        result = renderer.render_material_preview(mid, 0, size=_PREVIEW_SIZE)
        assert result is not None, "render_material_preview returned None on Metal"
        pixels, size = result
        assert size == _PREVIEW_SIZE

        # RGBA32F contract: the dock reshapes float32 (size, size, 4).
        assert len(pixels) == _PREVIEW_SIZE * _PREVIEW_SIZE * 16, len(pixels)
        arr = np.frombuffer(pixels, dtype=np.float32).reshape(
            _PREVIEW_SIZE, _PREVIEW_SIZE, 4)
        assert np.isfinite(arr).all(), "non-finite pixels in the Metal preview"
        # A lit primitive over an environment: some pixel must carry colour.
        assert float(arr[..., :3].max()) > 0.02, (
            "Metal preview frame is entirely black — the material did not shade"
        )

        # Re-dispatch at a different size to exercise the lazy resource teardown /
        # rebuild (image + pipeline are size-keyed) without wedging the GPU.
        result2 = renderer.render_material_preview(mid, 1, size=64)
        assert result2 is not None
        assert result2[1] == 64
        assert len(result2[0]) == 64 * 64 * 16
    finally:
        if renderer is not None:
            renderer.cleanup()
        ctx.destroy()
