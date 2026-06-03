"""End-to-end ReSTIR DI render tests (wavefront).

Milestone 1 (plumbing): the ReSTIR primary-direct pass relocates the NEE
light-sampling half to a separate wavefront pass at bounce 0 (shade's depth-0
reuseDirect gated to zero). Combined with shade's still-active BSDF-sampled
direct, the total is the same estimator as stock NEE — so the image must
CONVERGE to reuse=none. This validates the buffers / bounce-0 hook / descriptor
sets / radiance injection / capability gate before any reservoir math.
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
    """Deterministically accumulate n_samples wavefront frames with the given
    reuse mode; return the mean linear-HDR image. Switching reuse rebuilds the
    wavefront pass set (keyed on reuse_mode)."""
    renderer.reuse_index = int(reuse_index)
    for i in range(n_samples):
        renderer.frame_index = base_seed + i
        renderer.accum_frame = i
        renderer.render_headless()
    arr, samples = renderer.read_accumulation_hdr()
    return np.ascontiguousarray(arr, dtype=np.float32) / max(samples, 1)


@needs_vulkan
@pytest.mark.skipif(not DEMO_SCENE.exists(), reason="three_materials_demo.usda missing")
def test_restir_converges_to_nee():
    """ReSTIR DI primary-direct converges to stock NEE (wavefront), confirming
    the relocation is unbiased + the plumbing correct."""
    from skinny.vk_context import VulkanContext
    from skinny.renderer import Renderer

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
            execution_mode="wavefront",
        )
        try:
            deadline = 400
            while deadline > 0 and (
                renderer._usd_scene is None
                or len(renderer._usd_scene.instances) < 3
            ):
                renderer.update(0.025)
                deadline -= 1
            assert renderer._usd_scene is not None
            for _ in range(16):
                renderer.update(0.04)

            none_img = _accumulate(renderer, REUSE_NONE, 96, base_seed=1000)
            restir_img = _accumulate(renderer, REUSE_RESTIR, 96, base_seed=2000)
            assert renderer._restir_pass is not None, "ReSTIR pass was not built in wavefront"

            for name, img in (("none", none_img), ("restir", restir_img)):
                assert np.isfinite(img).all(), f"{name} has non-finite values"
            mean_none = float(_lum(none_img).mean())
            mean_restir = float(_lum(restir_img).mean())
            assert mean_none > 1e-3, f"reference ~black (mean={mean_none})"

            rel = abs(mean_restir - mean_none) / mean_none
            assert rel < 0.02, (
                f"ReSTIR did not converge to NEE: mean {mean_restir:.6g} vs "
                f"{mean_none:.6g} (rel {rel:.4f} ≥ 0.02)"
            )
        finally:
            renderer.cleanup()
    finally:
        ctx.destroy()


@needs_vulkan
@pytest.mark.skipif(not DEMO_SCENE.exists(), reason="three_materials_demo.usda missing")
def test_restir_megakernel_falls_back_to_identity():
    """ReSTIR selected on megakernel folds to identity (no reservoir pass; the
    reuseMode uniform packs 0 so the depth-0 gate stays inert)."""
    from skinny.vk_context import VulkanContext
    from skinny.renderer import Renderer

    ctx = VulkanContext(window=None, width=64, height=64)
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
            execution_mode="megakernel",
        )
        try:
            deadline = 400
            while deadline > 0 and (
                renderer._usd_scene is None
                or len(renderer._usd_scene.instances) < 3
            ):
                renderer.update(0.025)
                deadline -= 1
            renderer.reuse_index = REUSE_RESTIR
            for _ in range(4):
                renderer.update(0.04)
            renderer.frame_index = 4242
            renderer.accum_frame = 0
            renderer.render_headless()
            # No wavefront pass / ReSTIR pass on megakernel.
            assert getattr(renderer, "_restir_pass", None) is None
            arr, _ = renderer.read_accumulation_hdr()
            assert np.isfinite(arr).all() and float(arr[..., :3].max()) > 0.01
        finally:
            renderer.cleanup()
    finally:
        ctx.destroy()
