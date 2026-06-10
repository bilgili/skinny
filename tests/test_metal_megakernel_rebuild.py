"""Metal megakernel **pipeline-rebuild** end-to-end test.

This is the integration proof behind the backend-neutral ``wait_idle`` seam: it
drives ``Renderer._build_pipeline_for_current_graphs`` through its rebuild branch
(``is_rebuild=True``) on a real Metal device. That branch calls
``self.ctx.wait_idle()`` — which on Metal must route to slang-rhi's
``Device.wait_for_idle`` rather than ``vk.vkDeviceWaitIdle`` (the latter crashes
with ``TypeError: an integer is required`` on a slangpy device). ``cleanup()``
hits the same seam on teardown.

DANGER — this is the ONE expensive operation in the suite. Each
``_build_pipeline_for_current_graphs`` call in megakernel mode cold-compiles the
full ``main_pass`` megakernel, which spikes RAM inside Apple's separate
``MTLCompilerService`` daemon. Run it ONLY through ``scripts/guarded_metal.sh`` so
the memory watchdog + graceful-only kill apply:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    PYTHONPATH=$PWD/src TIMEOUT_S=420 scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_megakernel_rebuild.py -q

The test self-skips unless ``RUN_METAL_MEGAKERNEL_COMPILE=1`` is exported, so an
ordinary ``pytest`` run (CI, a plain ``pytest tests/``) never triggers the spike
by accident.
"""

from __future__ import annotations

import os

import pytest

from skinny.backend_select import metal_available

# `renderer.py` imports `vulkan` unconditionally at module top, so the Vulkan SDK
# must be on the dylib path even though this test never touches a Vulkan device.
# (The guard wrapper's invocation sets VULKAN_SDK/DYLD_LIBRARY_PATH; if the import
# fails we skip rather than error so a misconfigured host stays green.)
pytest.importorskip("slangpy")

_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"cold-compiles the main_pass megakernel (MTLCompilerService RAM spike); "
        f"set {_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_megakernel_pipeline_rebuild(shader_dir):
    """First build → rebuild on Metal, then clean teardown — all through the
    backend-neutral ``wait_idle`` seam, with no ``vk.*`` call reaching the
    slangpy device."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")

    from skinny.metal_context import MetalContext
    from skinny.params import EXECUTION_MEGAKERNEL
    from skinny.renderer import Renderer

    ctx = MetalContext(window=None, width=64, height=64)
    renderer = None
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=shader_dir, execution_mode="megakernel"
        )
        # Megakernel is forced on Metal; the pipeline is built lazily, so nothing
        # is compiled yet.
        assert renderer.execution_mode_index == EXECUTION_MEGAKERNEL
        assert renderer.pipeline is None
        assert renderer._scene_bindings is None

        # First build (is_rebuild=False): cold-compiles main_pass, skips wait_idle.
        renderer._scene_graph_fragments = []
        renderer._build_pipeline_for_current_graphs()
        assert renderer.pipeline is not None
        assert renderer._scene_bindings is not None
        first = renderer._scene_bindings

        # Second build (is_rebuild=True): drains the device via ctx.wait_idle(),
        # destroys the old scene bindings, recompiles. This is the line that
        # crashed on Metal before the seam (vk.vkDeviceWaitIdle on a slangpy
        # device). A fresh pipeline object proves the rebuild actually ran.
        renderer._build_pipeline_for_current_graphs()
        assert renderer.pipeline is not None
        assert renderer._scene_bindings is not None
        assert renderer._scene_bindings is not first, (
            "rebuild did not replace the scene bindings"
        )
    finally:
        # cleanup() drains through the same seam; teardown must not raise.
        if renderer is not None:
            renderer.cleanup()
        ctx.destroy()
