"""Headless A/B parity: megakernel vs staged wavefront path tracer (task 9.1).

Renders the demo scene with the `path` integrator in both execution modes and
asserts the wavefront accumulation matches the megakernel. The path tracer is
Monte Carlo and MoltenVK is not bit-reproducible, so two independent renders of
the *same* mode differ by a measurable noise floor; the wavefront (a different
sample sequence of the same estimator) must match the megakernel no worse than
that floor — i.e. it is the same integral, not a different image.
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
WARMUP = 48


def _match(a, b):
    d = np.abs(a - b).max(axis=2)
    tol = 8e-3 + 0.04 * np.abs(b).max(axis=2)
    return float((d <= tol).mean())


def test_wavefront_path_matches_megakernel():
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
    )
    try:
        deadline = 200
        while deadline > 0 and (
            renderer._usd_scene is None or len(renderer._usd_scene.instances) < 3
        ):
            renderer.update(0.025)
            deadline -= 1
        assert renderer.pipeline is not None
        renderer.integrator_index = 0  # path

        def render_mode(mode, frames=WARMUP):
            renderer.set_execution_mode(mode)
            renderer._material_version += 1  # force a clean accumulation
            for _ in range(frames):
                renderer.update(0.04)
                renderer.render_headless()
            return renderer.read_accumulation()[:, :, :3].copy()

        mega1 = render_mode(0)
        mega2 = render_mode(0)          # independent megakernel render → noise floor
        wave = render_mode(1)           # staged wavefront path tracer

        assert renderer.effective_execution_mode_index == 1, "wavefront not active"
        assert np.all(np.isfinite(wave))
        assert float(wave.max()) > 1e-2, "wavefront path render is black"

        noise_floor = _match(mega1, mega2)
        match_wave = _match(wave, mega1)
        assert noise_floor > 0.5, f"renderer too unstable to test ({noise_floor:.2f})"
        # The wavefront is the same estimator relocated across dispatches: it
        # must agree with the megakernel as well as two megakernel renders agree
        # with each other (both differ only by independent MC noise).
        assert match_wave >= noise_floor - 0.06, (
            f"wavefront path tracer diverges from the megakernel "
            f"(match {match_wave:.3f} vs megakernel-vs-megakernel floor "
            f"{noise_floor:.3f})"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_wavefront_path_tiled_streaming():
    """Tiled streaming (task 6.2): with a small stream cap the frame is
    processed in many fixed-size tiles (incl. a partial tail), the path-state
    buffer is sized to the cap — not the pixel count — and the image still
    matches the megakernel."""
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext
    from skinny.wavefront_layout import PATH_STATE_STRIDE

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
    )
    try:
        deadline = 200
        while deadline > 0 and (
            renderer._usd_scene is None or len(renderer._usd_scene.instances) < 3
        ):
            renderer.update(0.025)
            deadline -= 1
        assert renderer.pipeline is not None
        renderer.integrator_index = 0
        cap = 1000  # WIDTH*HEIGHT = 9216 ⇒ 10 tiles, last partial
        renderer._wf_stream_cap = cap

        def render_mode(mode, frames=WARMUP):
            renderer.set_execution_mode(mode)
            renderer._material_version += 1
            for _ in range(frames):
                renderer.update(0.04)
                renderer.render_headless()
            return renderer.read_accumulation()[:, :, :3].copy()

        mega1 = render_mode(0)
        mega2 = render_mode(0)
        wave = render_mode(1)

        # Path-state buffer is bounded by the stream cap, not the pixel count.
        assert renderer._wf_path_state_buf.size == cap * PATH_STATE_STRIDE, (
            f"path-state buffer {renderer._wf_path_state_buf.size} B != "
            f"cap {cap * PATH_STATE_STRIDE} B (should not scale with pixels)"
        )
        assert renderer._wavefront_path_pass.stream_size == cap
        assert renderer._wavefront_path_pass.num_pixels == WIDTH * HEIGHT

        noise_floor = _match(mega1, mega2)
        match_wave = _match(wave, mega1)
        assert noise_floor > 0.5, f"renderer too unstable ({noise_floor:.2f})"
        assert match_wave >= noise_floor - 0.06, (
            f"tiled wavefront diverges from the megakernel "
            f"(match {match_wave:.3f} vs floor {noise_floor:.3f})"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_wavefront_bdpt_matches_megakernel():
    """A/B parity for the bidirectional integrator (task 9.2): megakernel bdpt
    vs the staged wavefront bdpt (subpath walks + connection stage). Same
    noise-floor-relative criterion as the path A/B."""
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
    )
    try:
        deadline = 200
        while deadline > 0 and (
            renderer._usd_scene is None or len(renderer._usd_scene.instances) < 3
        ):
            renderer.update(0.025)
            deadline -= 1
        assert renderer.pipeline is not None
        renderer.integrator_index = 1  # bdpt
        assert renderer.WAVEFRONT_BDPT_SUPPORTED, "wavefront bdpt gated off"
        # Small stream cap → the bdpt subpath/aux buffers are tiled (10 tiles
        # over the 96² frame), exercising tiled streaming for bdpt too.
        cap = 1000
        renderer._wf_stream_cap = cap

        def render_mode(mode, frames=WARMUP):
            renderer.set_execution_mode(mode)
            renderer._material_version += 1
            for _ in range(frames):
                renderer.update(0.04)
                renderer.render_headless()
            return renderer.read_accumulation()[:, :, :3].copy()

        mega1 = render_mode(0)
        mega2 = render_mode(0)
        wave = render_mode(1)

        assert renderer.effective_execution_mode_index == 1, (
            "wavefront not active for bdpt (capability gate fell back)"
        )
        assert np.all(np.isfinite(wave))
        assert float(wave.max()) > 1e-2, "wavefront bdpt render is black"
        # bdpt subpath buffers are bounded by the stream cap, not the pixel count.
        assert renderer._wavefront_bdpt_pass.stream_size == cap
        assert renderer._wavefront_bdpt_pass.num_pixels == WIDTH * HEIGHT

        noise_floor = _match(mega1, mega2)
        match_wave = _match(wave, mega1)
        assert noise_floor > 0.5, f"renderer too unstable to test ({noise_floor:.2f})"
        assert match_wave >= noise_floor - 0.06, (
            f"wavefront bdpt diverges from the megakernel "
            f"(match {match_wave:.3f} vs floor {noise_floor:.3f})"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()
