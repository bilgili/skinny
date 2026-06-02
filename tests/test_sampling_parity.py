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
