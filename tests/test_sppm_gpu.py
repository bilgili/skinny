"""GPU correctness tests for the SPPM integrator (change photon-mapping-sppm).

Renders the cornell-box-sphere scene under the SPPM integrator and the path
tracer (both wavefront) and asserts the SPPM estimate is energy-consistent with
the path tracer — the end-to-end guard that the eye/grid/photon/update pipeline
deposits the indirect/caustic term once and disjoint from the NEE direct term
(no double-counting, no missing indirect). Implicitly exercises the counting-sort
grid + atomic deposit: a broken grid would mis-deposit and skew the energy.

Marked ``gpu`` (needs a Vulkan runtime + VULKAN_SDK on the dylib path) and skips
if the scene asset is absent (e.g. a bare worktree without the asset tree).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"


def _have_vulkan() -> bool:
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")
needs_scene = pytest.mark.skipif(not SCENE.exists(), reason="cornell_box_sphere asset absent")

_W = _H = 128


def _render(integrator_index: int, frames: int = 16):
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=_W, height=_H)
    try:
        r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
                     tattoo_dir=TATTOO_DIR, usd_scene_path=SCENE)
        for _ in range(200):
            if r._usd_scene is not None and len(r._usd_scene.instances) >= 1:
                break
            r.update(0.025)
        r.execution_mode_index = 1  # wavefront
        r.integrator_index = integrator_index
        r.accum_frame = 0
        raw = None
        for _ in range(frames):
            r.update(0.04)
            raw = r.render_headless()
        sppm_built = r._wavefront_sppm_pass is not None
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(_H, _W, 4)[:, :, :3].astype(np.float32)
        return arr, sppm_built
    finally:
        try:
            r.cleanup()
        except Exception:
            pass
        ctx.destroy()


@needs_vulkan
@needs_scene
def test_sppm_builds_and_renders_finite():
    arr, built = _render(2)  # INTEGRATOR_SPPM
    assert built, "SPPM pass was not constructed (fell back?)"
    assert np.isfinite(arr).all(), "SPPM produced non-finite pixels"
    assert (arr.sum(2) > 0).mean() > 0.5, "SPPM frame is mostly black"


@needs_vulkan
@needs_scene
def test_sppm_energy_matches_path_tracer():
    # SPPM (NEE direct + photon indirect) must be energy-consistent with the path
    # tracer's full GI: a double-counted direct term would push the ratio toward
    # ~2x; a missing indirect term would push it well below 1.
    path, _ = _render(0)
    sppm, built = _render(2)
    assert built
    ratio = float(sppm.mean()) / max(float(path.mean()), 1e-6)
    assert 0.85 <= ratio <= 1.15, f"SPPM/path energy ratio {ratio:.3f} out of band"
