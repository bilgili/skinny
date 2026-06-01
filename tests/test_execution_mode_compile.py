"""Mutual-exclusive compilation per execution mode (tasks 5.1–5.3).

The renderer compiles ONLY the selected backend. In `wavefront` mode it builds
the scene plumbing standalone (`scene_bindings_only`) and never compiles the
megakernel `main_pass` pipeline (`renderer.pipeline is None`); in `megakernel`
mode it builds the megakernel pipeline and never the wavefront stage pipelines.
A graph-set rebuild in wavefront mode stays megakernel-free.

GPU: needs the headless Vulkan/MoltenVK runtime + the build venv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
DEMO_SCENE = PROJECT_ROOT / "assets" / "three_materials_demo.usda"

pytestmark = pytest.mark.gpu

WIDTH = HEIGHT = 96


def _load(execution_mode):
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
        execution_mode=execution_mode,
    )
    deadline = 200
    while deadline > 0 and (
        renderer._usd_scene is None or len(renderer._usd_scene.instances) < 3
    ):
        renderer.update(0.025)
        deadline -= 1
    assert renderer._scene_bindings is not None, "scene bindings not built"
    return ctx, renderer


def _render(renderer, frames=6):
    for _ in range(frames):
        renderer.update(0.04)
        renderer.render_headless()
    return np.frombuffer(renderer.render_headless(), dtype=np.uint8)


def test_wavefront_mode_builds_no_megakernel_and_renders():
    """5.1: wavefront mode compiles no megakernel pipeline yet renders."""
    ctx, renderer = _load("wavefront")
    try:
        assert renderer.execution_mode_index == 1
        assert renderer.pipeline is None, "megakernel pipeline must not be built"
        # The scene bindings are the standalone (no-compile) plumbing.
        assert renderer._scene_bindings.pipeline is None
        assert renderer._scene_set0_layout is not None, "set-0 layout missing"

        frame = _render(renderer)
        assert int(frame.max()) > 0, "wavefront render produced an all-black frame"
        assert renderer._wavefront_path_pass is not None, "staged path pass not built"
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_megakernel_mode_builds_no_wavefront_pass_and_renders():
    """5.2: megakernel mode compiles no wavefront stage pipeline yet renders."""
    ctx, renderer = _load("megakernel")
    try:
        assert renderer.execution_mode_index == 0
        assert renderer.pipeline is not None, "megakernel pipeline not built"
        # In megakernel mode the scene bindings ARE the compiled pipeline.
        assert renderer._scene_bindings is renderer.pipeline
        assert renderer.pipeline.pipeline is not None

        frame = _render(renderer)
        assert int(frame.max()) > 0, "megakernel render produced an all-black frame"
        # No wavefront stage pipelines were built at any point.
        assert renderer._wavefront_path_pass is None
        assert renderer._wavefront_bdpt_pass is None
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_wavefront_graph_rebuild_never_builds_megakernel(monkeypatch):
    """5.3: a graph-set rebuild in wavefront mode (as when a model with a new
    material is added) never builds a compiled megakernel pipeline — every
    scene-bindings build stays compile-free."""
    import skinny.renderer as rmod

    compiled_megakernels = []
    real_init = rmod.ComputePipeline.__init__

    def spy_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        # A compiled megakernel has a non-None driver pipeline; the no-compile
        # `scene_bindings_only` build leaves it None.
        if self.pipeline is not None:
            compiled_megakernels.append(self.entry_module)

    monkeypatch.setattr(rmod.ComputePipeline, "__init__", spy_init)

    ctx, renderer = _load("wavefront")
    try:
        assert renderer.pipeline is None
        first_bindings = renderer._scene_bindings
        _render(renderer, frames=2)

        # Force a scene-bindings rebuild (the path a runtime material-graph add
        # takes: re-emit + rebuild the set-0 layout, gated on the fixed mode).
        renderer._build_pipeline_for_current_graphs()
        assert renderer._scene_bindings is not first_bindings, "rebuild did not occur"
        assert renderer.pipeline is None, "rebuild must not build a megakernel"
        assert renderer._scene_bindings.pipeline is None

        _render(renderer, frames=2)
        assert compiled_megakernels == [], (
            f"wavefront mode compiled a megakernel pipeline: {compiled_megakernels}"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()
