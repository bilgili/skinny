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


def _patch_mean(mask: np.ndarray, cx: int, cy: int, half: int = 4) -> float:
    h, w = mask.shape
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    return float(mask[y0:y1, x0:x1].mean())


def test_wavefront_visibility_matches_known_geometry():
    """Primary-visibility (intersect) wavefront kernel: the hit mask must light
    up the demo scene's three known sphere positions and leave the corners
    (background) dark — proving the wavefront traverses the shared BVH."""
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
        # A megakernel frame settles the camera + uploads fc; geometry buffers
        # are now allocated, so the visibility pass can bind them.
        for _ in range(4):
            renderer.update(0.04)
            renderer.render_headless()

        renderer._wavefront_debug_pass = renderer.build_wavefront_trace_pass(
            "wavefront/wavefront_visibility", "wavefrontVisibility")
        renderer.set_execution_mode(1)  # EXECUTION_WAVEFRONT
        renderer.render_headless()       # dispatches the visibility pass
        mask = renderer.read_accumulation()[:, :, 0]  # hit flag in R

        assert np.all((mask >= -1e-3) & (mask <= 1.0 + 1e-3)), "mask not in [0,1]"

        hit = mask > 0.5
        frac = float(hit.mean())
        # Geometry was found (spheres detected), but is a minority of the frame.
        assert 0.01 < frac < 0.6, f"hit fraction {frac:.3f} out of expected range"

        # The four corners are background → miss.
        for cx, cy, label in [(0, 0, "tl"), (WIDTH - 1, 0, "tr"),
                              (0, HEIGHT - 1, "bl"), (WIDTH - 1, HEIGHT - 1, "br")]:
            assert _patch_mean(mask, cx, cy) < 0.25, f"{label} corner unexpectedly hit"

        # The three spheres are horizontally centered, so the hit region's
        # centroid sits near the image centre — robust to exact pixel positions.
        ys, xs = np.nonzero(hit)
        cx_hits, cy_hits = xs.mean(), ys.mean()
        assert abs(cx_hits - WIDTH / 2) < WIDTH * 0.18, f"hit centroid x={cx_hits:.0f} off-centre"
        assert abs(cy_hits - HEIGHT / 2) < HEIGHT * 0.30, f"hit centroid y={cy_hits:.0f} off-centre"
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_wavefront_hit_normals_are_sensible():
    """Hit-normal visualization: traceScene must return correct per-hit normals
    in the wavefront. The mapped normal (n*0.5+0.5) at sphere-hit pixels decodes
    to a unit vector, and at the silhouette centre faces roughly toward the
    camera — proving the shade stage gets a valid surface frame."""
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
        for _ in range(4):
            renderer.update(0.04)
            renderer.render_headless()

        renderer._wavefront_debug_pass = renderer.build_wavefront_trace_pass(
            "wavefront/wavefront_normal", "wavefrontNormal")
        renderer.set_execution_mode(1)
        renderer.render_headless()
        img = renderer.read_accumulation()[:, :, :3]

        # Decode mapped normals (c = n*0.5+0.5) back to vectors at hit pixels
        # (non-black). Each should be ~unit length.
        n = img * 2.0 - 1.0
        lum = img.sum(axis=2)
        hit_pixels = lum > 0.05  # mapped miss is (0,0,0); hits are >= ~(0.x)
        assert hit_pixels.sum() > 50, "no hit pixels in the normal buffer"
        lengths = np.linalg.norm(n[hit_pixels], axis=1)
        # Most hit-pixel normals decode to unit length (edges/AA add a few outliers).
        unit_frac = float((np.abs(lengths - 1.0) < 0.15).mean())
        assert unit_frac > 0.7, f"only {unit_frac:.0%} of hit normals are unit length"
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_wavefront_diffuse_shades_geometry():
    """Direct-lit diffuse wavefront shade: the geometry must come out shaded
    (a normal-dependent gradient over the spheres) and distinct from the
    environment background — proving the wavefront lights a surface with the hit
    normal + the scene's directional light. Not an A/B vs the megakernel (fixed
    albedo, single bounce) — a 'physically sensible lit surface' check."""
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
        for _ in range(4):
            renderer.update(0.04)
            renderer.render_headless()

        # Hit mask (locates the geometry).
        vis = renderer.build_wavefront_trace_pass(
            "wavefront/wavefront_visibility", "wavefrontVisibility")
        renderer._wavefront_debug_pass = vis
        renderer.set_execution_mode(1)
        renderer.render_headless()
        mask = renderer.read_accumulation()[:, :, 0] > 0.5
        vis.destroy()

        # Diffuse shade.
        diff = renderer.build_wavefront_trace_pass(
            "wavefront/wavefront_diffuse", "wavefrontDiffuse",
            include_env=True, include_lights=True)
        renderer._wavefront_debug_pass = diff
        renderer.render_headless()
        img = renderer.read_accumulation()[:, :, :3]

        assert np.all(np.isfinite(img))
        assert int(mask.sum()) > 50, "no geometry to shade"
        lum = img.sum(axis=2)
        # The shaded geometry varies across the surface (N·L + normal-sampled
        # ambient) rather than being a flat fill.
        assert float(lum[mask].std()) > 0.01, "shaded geometry is flat (no N-dependence)"
        # Geometry shading is distinct from the environment background.
        assert abs(float(lum[mask].mean()) - float(lum[~mask].mean())) > 1e-3, (
            "shaded geometry indistinguishable from background"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()
