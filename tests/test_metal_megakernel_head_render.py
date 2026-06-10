"""Metal megakernel **head-render** test (task 4.4).

Produces a real head-render frame on ``MetalContext``: load ``heads/head.obj``,
pump ``update()`` until the async OBJ load bakes the mesh + seeds a valid
identity TLAS instance + builds the megakernel pipeline, then dispatch through
the renderer's frame path and read the result back.

Why a real mesh (not the bare Renderer): the megakernel always traces
``marchHeadMesh(r, fc.numInstances)`` (``useMesh`` is hard-on). A bare headless
Renderer leaves ``instance_buffer`` zero-seeded — instance 0's transform is the
zero matrix, so the world→local ray becomes degenerate and BVH traversal walks
NaN bounds forever, hanging the GPU (the 420s hang the guard caught while
verifying 4.2/4.3). Loading a model runs the bake path
(``_rebake_if_needed`` → ``_bake_and_upload`` → ``_upload_instances([eye(4)])``),
which seeds a non-degenerate instance over a valid BVH. The test asserts that
seed as a *precondition* and only then dispatches, so a missing seed fails fast
instead of hanging.

DANGER — building the megakernel cold-compiles ``main_pass`` (MTLCompilerService
RAM spike) and the head bake adds CPU time. Run ONLY through
``scripts/guarded_metal.sh`` with the compile gate set:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_MEGAKERNEL_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=420 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_megakernel_head_render.py -q
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
_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"
_HEAD_OBJ = _PROJECT_ROOT / "heads" / "head.obj"
_INSTANCE_STRIDE = 144  # renderer.INSTANCE_STRIDE — one packed TLAS record


def _pump_until_ready(renderer, *, budget_s: float = 60.0) -> bool:
    """Drive update() until the async OBJ load has baked the mesh, seeded a TLAS
    instance, and built the megakernel pipeline. Returns readiness."""
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
        f"cold-compiles the main_pass megakernel (MTLCompilerService RAM spike); "
        f"set {_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_megakernel_head_render():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    if not _HEAD_OBJ.exists():
        pytest.skip(f"head asset not found: {_HEAD_OBJ}")
    try:
        from skinny.metal_context import MetalContext
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path (renderer imports it)
        pytest.skip(f"renderer import needs the Vulkan SDK on the dylib path: {exc}")

    ctx = MetalContext(window=None, width=96, height=96)
    renderer = None
    try:
        # hdr_dir lets a real HDR environment light the scene, so the head
        # renders nonzero (the Vulkan headless reference does the same).
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="megakernel",
            hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
            tattoo_dir=_TATTOO_DIR if _TATTOO_DIR.is_dir() else None,
        )
        renderer.load_model_from_path(_HEAD_OBJ)
        assert _pump_until_ready(renderer), (
            "head load/bake/pipeline build did not become ready in time"
        )

        # Precondition (guards the dispatch against the degenerate-instance hang):
        # a non-degenerate TLAS instance must be seeded over a valid BVH.
        assert renderer._num_instances >= 1
        first = renderer.instance_buffer.download_sync(_INSTANCE_STRIDE)
        assert any(first), "instance 0 transform is all-zero (would hang traversal)"
        assert renderer._backend_render_ready

        # Dispatch one head-render frame. The camera is framed to the loaded
        # head, so the tonemapped offscreen frame must not be all black — the
        # same signal the Vulkan headless reference asserts
        # (test_headless.TestMaterialXGraphDemoRender.test_render_headless_nonzero).
        raw = renderer.render_headless()
        assert isinstance(raw, (bytes, bytearray))
        assert len(raw) == renderer.width * renderer.height * 4
        assert any(raw), "head-render frame is entirely black"

        # The linear-HDR accumulation readback (used by the parity tests) is
        # well-formed and finite for the dispatched frame.
        arr, _samples = renderer.read_accumulation_hdr()
        assert arr.shape == (renderer.height, renderer.width, 4)
        assert np.isfinite(arr).all(), "non-finite values in the head-render frame"

        # Fence signalled each frame, no hang: render several more frames.
        for _ in range(3):
            renderer.update(0.016)
            again = renderer.render_headless()
            assert len(again) == renderer.width * renderer.height * 4
    finally:
        if renderer is not None:
            renderer.cleanup()
        ctx.destroy()
