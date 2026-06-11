"""Metal megakernel **binding-map coverage** test (task 4.1).

The Vulkan renderer binds megakernel resources by numbered descriptor slot; the
Metal pipeline binds them by **slang global name** (design D2 — bind-by-name).
``ComputePipeline.dispatch`` filters the renderer's bind dict to the names the
compiled ``mainImage`` module actually reflects (so a dead-stripped global is
harmlessly skipped). That same filter would *also* silently skip a **typo'd**
required name, leaving a real resource unbound and the frame reading garbage.

This test closes that gap: it builds the megakernel pipeline on a live Metal
device and asserts the renderer's binding map (``_build_metal_binds`` + the
bindless pool) drives the *same logical slots* the shader reflects —
1. a bedrock CORE set of always-referenced globals is present, and
2. every name the renderer binds is either reflected or in a documented
   dead-strip allowlist (so a name typo / binding-map drift fails loudly).

DANGER — building the megakernel pipeline cold-compiles ``main_pass`` and spikes
RAM inside Apple's ``MTLCompilerService`` daemon. Run ONLY through
``scripts/guarded_metal.sh`` and only with the compile gate set:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_MEGAKERNEL_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=420 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_megakernel_binding_map.py -q
"""

from __future__ import annotations

import os

import pytest

from skinny.backend_select import metal_available

pytest.importorskip("slangpy")

_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"

# Bedrock globals: ``mainImage`` references these unconditionally (geometry,
# accumulation/output images, flat + std material lookup, the bindless flat-tex
# pool + its shared sampler). They survive dead-stripping for any scene, so their
# absence means the binding map / reflection is broken, not merely scene-empty.
_CORE_GLOBALS = frozenset({
    "outputBuffer", "accumBuffer",
    "meshVertices", "meshIndices", "bvhNodes", "instances",
    "flatMaterials", "materialTypes",
    "flatMaterialTextures", "commonSampler",
})

# Names the renderer may bind that the compiled empty-scene ``mainImage`` is
# allowed to dead-strip (so binding them is a no-op, not a typo). Verified EMPTY
# on this Mac: every name ``_build_metal_binds`` returns — plus the bindless
# ``flatMaterialTextures`` pool — resolves to a reflected global, so the binding
# map drives the reflected slots with zero drift. A name appearing here later
# means that global became conditionally compiled; document why before adding it.
_DEAD_STRIP_ALLOWED = frozenset()


def _metal_megakernel_renderer(shader_dir):
    """Build a headless megakernel Renderer on Metal with its pipeline compiled,
    or skip cleanly. Returns (ctx, renderer)."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    try:
        from skinny.metal_context import MetalContext
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path (renderer imports it)
        pytest.skip(f"renderer import needs the Vulkan SDK on the dylib path: {exc}")

    ctx = MetalContext(window=None, width=64, height=64)
    renderer = Renderer(vk_ctx=ctx, shader_dir=shader_dir, execution_mode="megakernel")
    # Pipeline is lazy; compile the empty-scene megakernel (the spike).
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
def test_metal_megakernel_binding_map(shader_dir):
    ctx, renderer = _metal_megakernel_renderer(shader_dir)
    try:
        assert renderer.pipeline is not None, "megakernel pipeline not built"
        reflected = set(renderer.pipeline.global_names)

        # 1. Bedrock globals must be reflected — proves the binding map addresses
        #    the same logical slots the shader declares.
        missing_core = _CORE_GLOBALS - reflected
        assert not missing_core, (
            f"core megakernel globals absent from reflection: {sorted(missing_core)}"
        )

        # 2. Every name the renderer binds resolves to a reflected global (or a
        #    documented dead-strip). An unexplained name = typo / binding drift.
        bound = set(renderer._build_metal_binds())
        bound.add("flatMaterialTextures")  # bindless pool, bound separately
        unexplained = bound - reflected - _DEAD_STRIP_ALLOWED
        assert not unexplained, (
            "renderer binds names the megakernel does not reflect (typo or binding "
            f"drift; if a legit dead-strip, add to _DEAD_STRIP_ALLOWED): "
            f"{sorted(unexplained)}"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()
