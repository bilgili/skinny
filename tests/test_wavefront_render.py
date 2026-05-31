"""End-to-end A/B: env-only wavefront vs megakernel (P1 integration milestone).

Renders the demo scene in both execution modes and compares the linear-HDR
accumulation image at the background corners. The wavefront env pass shades
every pixel with the environment (no geometry), so it can only agree with the
megakernel where the megakernel's rays also miss geometry — the corners. A
match there proves the integration: the wavefront pipeline reads fc + envMap and
writes the accumulation image, gated by execution mode, producing the same
environment radiance as the megakernel.
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

WIDTH = HEIGHT = 128
WARMUP = 24


def _render(renderer, frames):
    for _ in range(frames):
        renderer.update(0.04)
        renderer.render_headless()
    return renderer.read_accumulation()[:, :, :3]


def test_wavefront_env_matches_megakernel_at_background():
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
    )
    try:
        # Pump async USD load until the scene + pipeline exist.
        deadline = 200
        while deadline > 0 and (
            renderer._usd_scene is None or len(renderer._usd_scene.instances) < 3
        ):
            renderer.update(0.025)
            deadline -= 1
        assert renderer.pipeline is not None, "megakernel pipeline not built"

        mega = _render(renderer, WARMUP)

        # Switch to wavefront (resets accumulation via the state hash) + re-render.
        renderer.set_execution_mode(1)  # EXECUTION_WAVEFRONT
        assert renderer.effective_execution_mode_index == 1
        wave = _render(renderer, WARMUP)

        # The env pass shaded every pixel with the environment and wrote the
        # accumulation image — it ran and read fc + env.
        assert np.all(np.isfinite(wave))
        assert float(wave.max()) > 1e-3, "wavefront env all black"

        # The env pass ignores geometry, so it agrees with the megakernel only
        # where the megakernel's rays also miss geometry (the background). Over
        # a scene of three small spheres that is most of the frame: require a
        # substantial fraction of pixels to match in linear HDR. The mismatched
        # pixels are the shaded sphere region.
        diff = np.abs(wave - mega)
        tol = 0.05 * np.abs(mega) + 3e-3
        matching = float((diff <= tol).all(axis=2).mean())
        assert matching >= 0.30, (
            f"only {matching:.0%} of pixels match the megakernel env "
            f"(expected the background to match)"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()
