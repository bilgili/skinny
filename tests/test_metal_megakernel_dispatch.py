"""Metal megakernel **dispatch-binds** test (tasks 4.2 + 4.3).

Drives the renderer's real Metal megakernel dispatch path
(``render_headless`` → ``_render_headless_metal`` → ``_render_megakernel_metal``
→ ``ComputePipeline.dispatch``) on a live Metal device. A clean dispatch is the
runtime proof of both tasks at once — they cannot be separated at the bind point:

* **4.2** every megakernel *buffer* the renderer holds (flat-material params,
  MaterialX skin/std params, the light buffers, gizmo, lens, env importance CDFs,
  neural + record buffers, …) binds by name with NO per-field scalar cursor write
  (design D4 — ``dispatch`` only does whole-resource ``cur[name] = native`` and
  ``cur["fc"].set_data(blob)``).
* **4.3** the bindless ``flatMaterialTextures`` pool (120 slots, every slot bound —
  unfilled ones get the default 1×1 texture), the shared ``commonSampler``, and
  the five discrete ``Texture2D`` + per-map ``SamplerState`` maps (env / tattoo /
  normal / roughness / displacement) bind within Apple's compute argument limits
  (design D8 / O4).

slang-rhi's Metal backend *aborts* on an unbound or mis-typed resource (the
``Unsupported binding type`` assert that forced the discrete-map split), so a
dispatch that returns finite pixels proves every bind above is correct. We also
assert the accumulation HDR is finite — a mis-pointed buffer feeding the
integrator surfaces as NaN/Inf, not a clean frame.

NO-GEOMETRY frame: the test forces ``_num_instances = 0`` so the megakernel takes
the zero-iteration mesh branch (``marchHeadMesh(r, fc.numInstances)`` with
count 0) — a bare headless Renderer has no mesh loaded, and traversing the
degenerate empty BVH hangs the GPU. Every buffer/texture/sampler is still bound
at dispatch (the bind-attachment proof) and the miss path samples ``envMap`` via
its discrete ``Texture2D`` + ``SamplerState``. Sampling the head normal/
roughness/displacement/tattoo maps requires a surface hit (real geometry); that
on-hit path is exercised by the 4.4 head-render frame, not here.

DANGER — building the megakernel cold-compiles ``main_pass`` and spikes RAM
inside Apple's ``MTLCompilerService`` daemon. Run ONLY through
``scripts/guarded_metal.sh`` with the compile gate set:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_MEGAKERNEL_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=420 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_megakernel_dispatch.py -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from skinny.backend_select import metal_available

pytest.importorskip("slangpy")

_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"


def _metal_megakernel_renderer(shader_dir, width=64, height=64):
    """Headless megakernel Renderer on Metal with its pipeline compiled, or a
    clean skip. Returns (ctx, renderer)."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    try:
        from skinny.metal_context import MetalContext
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path (renderer imports it)
        pytest.skip(f"renderer import needs the Vulkan SDK on the dylib path: {exc}")

    ctx = MetalContext(window=None, width=width, height=height)
    renderer = Renderer(vk_ctx=ctx, shader_dir=shader_dir, execution_mode="megakernel")
    renderer._scene_graph_fragments = []
    renderer._build_pipeline_for_current_graphs()
    return ctx, renderer


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"cold-compiles the main_pass megakernel (MTLCompilerService RAM spike); "
        f"set {_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_megakernel_dispatch_binds(shader_dir):
    ctx, renderer = _metal_megakernel_renderer(shader_dir)
    try:
        # The pipeline is built, so the headless path takes the real dispatch
        # branch (not the zeroed not-ready fallback).
        assert renderer._backend_render_ready, "pipeline not ready to dispatch"

        # No mesh is loaded on a bare Renderer; traversing the degenerate empty
        # BVH hangs the GPU. Zero instances → zero-iteration mesh branch → a clean
        # env/miss frame that still binds every resource.
        renderer._num_instances = 0

        # Binds every buffer (4.2) + bindless pool + commonSampler + discrete-map
        # samplers (4.3) and dispatches. Reaching this line without slang-rhi
        # aborting is the bind-correctness proof.
        out = renderer.render_headless()
        assert isinstance(out, (bytes, bytearray))
        assert len(out) == renderer.width * renderer.height * 4

        # A mis-pointed buffer feeding the integrator shows up as NaN/Inf in the
        # linear-HDR accumulation, not a clean frame.
        arr, _samples = renderer.read_accumulation_hdr()
        assert arr.shape == (renderer.height, renderer.width, 4)
        assert np.isfinite(arr).all(), "non-finite values in the dispatched HDR frame"
    finally:
        renderer.cleanup()
        ctx.destroy()
