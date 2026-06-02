"""Deterministic baseline-parity gate for the pluggable scene-sampling seam.

The seam refactor must not change the rendered image when the default
proposal set (``{bsdf}``) and reuse mode (``none``) are active. This test
renders the MaterialX demo scene with a *pinned* RNG seed and a single
accumulation sample, then hashes the linear-HDR accumulation image (no
tonemap variance). The digest is captured to a local golden file on first
run and asserted byte-identical thereafter.

Determinism:
  - ``frame_index`` seeds the per-pixel PCG RNG (common.slang::createRNG),
    so it is pinned to a constant before the measured frame.
  - ``accum_frame = 0`` makes the running mean replace (weight 1/(n+1)=1),
    so the accumulation image is exactly one frame's radiance regardless of
    any stale contents.
  - The async USD load's pump count perturbs frame_index during streaming,
    but the measured frame overwrites it, so the digest is load-timing
    independent.

The golden is GPU/driver specific (MoltenVK float ops). It is a *local*
gate, not a CI fixture — the golden file is gitignored.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
DEMO_SCENE = PROJECT_ROOT / "assets" / "three_materials_demo.usda"
GOLDEN_FILE = Path(__file__).parent / "_sampling_parity_golden.txt"

pytestmark = pytest.mark.gpu

SEED_FRAME_INDEX = 4242
WIDTH = 128
HEIGHT = 128


def _have_vulkan() -> bool:
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


def _render_demo_digest() -> tuple[str, np.ndarray]:
    """Render the demo scene deterministically; return (sha256_hex, hdr_array)."""
    from skinny.vk_context import VulkanContext
    from skinny.renderer import Renderer

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    try:
        renderer = Renderer(
            vk_ctx=ctx,
            shader_dir=SHADER_DIR,
            hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR,
            usd_scene_path=DEMO_SCENE,
        )
        try:
            # Pump the async loader until the three spheres exist.
            deadline = 400
            while deadline > 0 and (
                renderer._usd_scene is None
                or len(renderer._usd_scene.instances) < 3
            ):
                renderer.update(0.025)
                deadline -= 1
            assert renderer._usd_scene is not None
            assert len(renderer._usd_scene.instances) >= 3, "demo spheres did not stream in"
            # Settle lazy state (mesh bake debounce, env upload, material types)
            # so the measured frame is content-stable.
            for _ in range(16):
                renderer.update(0.04)

            # Pin the measured frame: fixed RNG seed, single accumulation sample.
            renderer.frame_index = SEED_FRAME_INDEX
            renderer.accum_frame = 0
            renderer.render_headless()

            arr, samples = renderer.read_accumulation_hdr()
            assert samples == 1, f"expected single-sample accum, got {samples}"
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            return digest, arr
        finally:
            renderer.cleanup()
    finally:
        ctx.destroy()


@needs_vulkan
@pytest.mark.skipif(not DEMO_SCENE.exists(), reason="three_materials_demo.usda missing")
def test_baseline_parity():
    """Default {bsdf}/none output is byte-identical to the captured golden."""
    digest, arr = _render_demo_digest()
    finite = np.isfinite(arr).all()
    assert finite, "non-finite values in accumulation image"
    nonblack = float(arr[..., :3].max())
    assert nonblack > 0.01, f"image is ~black (max={nonblack}); render likely broke"

    if not GOLDEN_FILE.exists():
        GOLDEN_FILE.write_text(digest + "\n")
        pytest.skip(f"golden captured → {GOLDEN_FILE.name}: {digest[:16]}… (re-run to assert)")

    golden = GOLDEN_FILE.read_text().strip()
    assert digest == golden, (
        f"baseline parity BROKEN: {digest[:16]}… != golden {golden[:16]}…\n"
        f"max|val|={nonblack:.6g}. The seam refactor changed the default-path image."
    )


# ── Environment proposal: unbiasedness + variance reduction ────────

def _luminance(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def _accumulate(renderer, proposals, n_samples: int, base_seed: int) -> np.ndarray:
    """Deterministically accumulate ``n_samples`` frames with the given active
    proposal set; return the mean linear-HDR image (H, W, 4).

    Drives frame_index (the RNG seed) and accum_frame (the running-mean index)
    by hand so the result is reproducible and independent of any prior render.
    """
    renderer.active_proposals = list(proposals)
    for i in range(n_samples):
        renderer.frame_index = base_seed + i
        renderer.accum_frame = i
        renderer.render_headless()
    arr, samples = renderer.read_accumulation_hdr()
    return np.ascontiguousarray(arr, dtype=np.float32) / max(samples, 1)


@needs_vulkan
@pytest.mark.skipif(not DEMO_SCENE.exists(), reason="three_materials_demo.usda missing")
def test_env_proposal_unbiased_and_reduces_variance():
    """{bsdf,env} converges to the same image as {bsdf} (unbiased MIS) and has
    lower error at equal low sample count on an IBL-lit scene.

    Bias here would mean the mixture pdf or the NEE-coupling is wrong. IBL is
    isolated (direct lights off) so the environment proposal carries the signal.
    """
    from skinny.vk_context import VulkanContext
    from skinny.renderer import Renderer
    from skinny.sampling import BsdfProposal, EnvImportanceProposal

    bsdf = [BsdfProposal()]
    bsdf_env = [BsdfProposal(), EnvImportanceProposal()]

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    try:
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
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
            # Isolate IBL so the env proposal is the thing under test.
            renderer.direct_light_index = 1

            conv_bsdf = _accumulate(renderer, bsdf, 96, base_seed=1000)
            conv_env = _accumulate(renderer, bsdf_env, 96, base_seed=2000)
            lo_bsdf = _accumulate(renderer, bsdf, 16, base_seed=3000)
            lo_env = _accumulate(renderer, bsdf_env, 16, base_seed=3000)

            for name, img in (("conv_bsdf", conv_bsdf), ("conv_env", conv_env)):
                assert np.isfinite(img).all(), f"{name} has non-finite values"
            lum_bsdf = _luminance(conv_bsdf)
            lum_env = _luminance(conv_env)
            mean_bsdf = float(lum_bsdf.mean())
            mean_env = float(lum_env.mean())
            assert mean_bsdf > 1e-3, f"IBL render ~black (mean={mean_bsdf}); no env signal"

            # Unbiasedness: integrated radiance (a very low-variance statistic)
            # must agree — a wrong mixture pdf / NEE coupling would shift energy.
            rel = abs(mean_env - mean_bsdf) / mean_bsdf
            assert rel < 0.03, (
                f"env proposal biased: image-mean luminance {mean_env:.6g} vs "
                f"bsdf-only {mean_bsdf:.6g} (rel {rel:.4f} ≥ 0.03)"
            )

            # Variance reduction: at 16 spp the env-mixed image is closer to the
            # converged bsdf reference than bsdf-only is.
            ref = _luminance(conv_bsdf)
            err_bsdf = float(np.sqrt(np.mean((_luminance(lo_bsdf) - ref) ** 2)))
            err_env = float(np.sqrt(np.mean((_luminance(lo_env) - ref) ** 2)))
            assert err_env <= err_bsdf, (
                f"env proposal did not reduce error: rmse env {err_env:.5g} > "
                f"bsdf {err_bsdf:.5g}"
            )
        finally:
            renderer.cleanup()
    finally:
        ctx.destroy()
