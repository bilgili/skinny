"""ReSTIR DI unified-light-domain convergence (spec: unified light-domain RIS).

The reuse seam gates the *entire* primary-hit ``allLightsNEE`` (directional +
sphere + emissive-triangle + environment light-sampled half) to zero at depth 0;
ReSTIR's resolve pass must re-add every light type or the MIS-weighted estimate
loses energy (biased darkening). These tests render scenes whose *only*
illumination is a sphere light / emissive-triangle panel and assert ReSTIR DI
still converges to stock NEE — the unbiased gate for the area-light candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SPHERE_SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"
EMISSIVE_SCENE = PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"

pytestmark = pytest.mark.gpu

WIDTH = 128
HEIGHT = 128
REUSE_NONE = 0
REUSE_RESTIR = 1


def _have_vulkan() -> bool:
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


def _lum(a: np.ndarray) -> np.ndarray:
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _accumulate(renderer, reuse_index: int, n_samples: int, base_seed: int) -> np.ndarray:
    renderer.reuse_index = int(reuse_index)
    for i in range(n_samples):
        renderer.frame_index = base_seed + i
        renderer.accum_frame = i
        renderer.render_headless()
    arr, samples = renderer.read_accumulation_hdr()
    return np.ascontiguousarray(arr, dtype=np.float32) / max(samples, 1)


def _settle(renderer, min_instances: int) -> None:
    deadline = 400
    while deadline > 0 and (
        renderer._usd_scene is None
        or len(renderer._usd_scene.instances) < min_instances
    ):
        renderer.update(0.025)
        deadline -= 1
    assert renderer._usd_scene is not None
    for _ in range(16):
        renderer.update(0.04)


def _converges(scene: Path, min_instances: int, spp: int = 80, tol: float = 0.03) -> None:
    from skinny.vk_context import VulkanContext
    from skinny.renderer import Renderer

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR, usd_scene_path=scene,
            execution_mode="wavefront",
        )
        try:
            _settle(renderer, min_instances)
            none_img = _accumulate(renderer, REUSE_NONE, spp, base_seed=1000)
            restir_img = _accumulate(renderer, REUSE_RESTIR, spp, base_seed=2000)
            assert renderer._restir_pass is not None, "ReSTIR pass not built"
            for name, img in (("none", none_img), ("restir", restir_img)):
                assert np.isfinite(img).all(), f"{name} has non-finite values"
            mean_none = float(_lum(none_img).mean())
            mean_restir = float(_lum(restir_img).mean())
            assert mean_none > 1e-3, f"reference ~black (mean={mean_none})"
            rel = abs(mean_restir - mean_none) / mean_none
            assert rel < tol, (
                f"ReSTIR did not converge to NEE on {scene.name}: "
                f"mean {mean_restir:.6g} vs {mean_none:.6g} (rel {rel:.4f} ≥ {tol})"
            )
        finally:
            renderer.cleanup()
    finally:
        ctx.destroy()


@needs_vulkan
@pytest.mark.skipif(not SPHERE_SCENE.exists(), reason="cornell_box_sphere.usda missing")
def test_restir_converges_sphere_light():
    """Sole illumination = a sphere light. ReSTIR must re-add the sphere's
    light-sampled NEE half via an area-light RIS candidate."""
    _converges(SPHERE_SCENE, min_instances=6)


@needs_vulkan
@pytest.mark.skipif(not EMISSIVE_SCENE.exists(), reason="cornell_box_emissive.usda missing")
def test_restir_converges_emissive_triangles():
    """Sole illumination = an emissive-triangle ceiling panel. ReSTIR must re-add
    the emissive-triangle light-sampled NEE half via an area-light RIS candidate."""
    _converges(EMISSIVE_SCENE, min_instances=6)
