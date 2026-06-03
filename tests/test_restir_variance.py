"""ReSTIR DI variance reduction (spec: Requirement "Variance reduction").

On a high-contrast many-light scene (a flat floor lit by a dim 128-triangle
emissive grid plus 3 small very bright emissive spots), stock NEE samples one
emissive triangle uniformly per pixel per frame — so it mostly picks a dim
triangle and rarely the bright spots, giving high variance at low spp. ReSTIR DI
reservoir-resamples M candidates weighted by the unshadowed target and keeps the
important one, so its image has LOWER error than stock NEE at equal low spp.

The reservoir accumulation image is a running average, so the raw image (NOT
divided by the sample count) is the mean radiance. The reference is the average of
a converged NEE and a converged ReSTIR render (both unbiased ⇒ same expectation,
lower combined noise, method-neutral).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SCENE = PROJECT_ROOT / "assets" / "restir_variance_demo.usda"

pytestmark = pytest.mark.gpu

WIDTH = 128
HEIGHT = 128
REUSE_NONE = 0
REUSE_RESTIR = 1
SPATIAL_CFG = dict(mLight=8, spatialK=5, spatialRadius=16.0, normalThresh=0.9,
                   depthThresh=0.1, mCap=20, mBsdf=1, flags=0x1)  # spatial-only (default)


def _have_vulkan() -> bool:
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


def _lum(a: np.ndarray) -> np.ndarray:
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _render(renderer, reuse_index, n_samples, base_seed, cfg=None) -> np.ndarray:
    """Raw running-average accumulation image (mean radiance — do NOT divide)."""
    renderer._restir_config = cfg
    renderer.reuse_index = int(reuse_index)
    for i in range(n_samples):
        renderer.frame_index = base_seed + i
        renderer.accum_frame = i
        renderer.render_headless()
    arr, _ = renderer.read_accumulation_hdr()
    return np.ascontiguousarray(arr, dtype=np.float32)[..., :3]


@needs_vulkan
@pytest.mark.skipif(not SCENE.exists(), reason="restir_variance_demo.usda missing")
def test_restir_reduces_variance_vs_nee():
    """ReSTIR DI (spatial, the shipped default) has lower error than stock NEE at
    equal low spp on the many-light scene."""
    from skinny.vk_context import VulkanContext
    from skinny.renderer import Renderer

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR, usd_scene_path=SCENE, execution_mode="wavefront",
        )
        try:
            deadline = 400
            while deadline > 0 and (
                renderer._usd_scene is None or len(renderer._usd_scene.instances) < 5
            ):
                renderer.update(0.025)
                deadline -= 1
            assert renderer._usd_scene is not None
            for _ in range(16):
                renderer.update(0.04)
            assert renderer._num_emissive_tris > 100, "scene should have many emissive triangles"

            # Method-neutral converged reference (both unbiased ⇒ same mean).
            ref = 0.5 * (_render(renderer, REUSE_NONE, 224, 9000)
                         + _render(renderer, REUSE_RESTIR, 224, 17000, SPATIAL_CFG))
            mask = _lum(ref) > 0.02
            assert int(mask.sum()) > WIDTH * HEIGHT // 4

            def rmse(img):
                return float(np.sqrt(np.mean((_lum(img)[mask] - _lum(ref)[mask]) ** 2)))

            spp = 24
            err_none = rmse(_render(renderer, REUSE_NONE, spp, 2000))
            err_restir = rmse(_render(renderer, REUSE_RESTIR, spp, 4000, SPATIAL_CFG))
            assert err_none > 0 and np.isfinite(err_restir)
            assert err_restir < 0.85 * err_none, (
                f"ReSTIR did not reduce variance: rmse {err_restir:.4f} vs NEE "
                f"{err_none:.4f} (ratio {err_restir / err_none:.3f} ≥ 0.85)"
            )
        finally:
            renderer.cleanup()
    finally:
        ctx.destroy()
