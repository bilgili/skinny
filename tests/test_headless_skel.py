"""End-to-end headless check for UsdSkel skeletal animation.

Loads the ElephantWithMonochord asset (skinned via UsdSkel) and asserts the
rendered image changes between two time codes — i.e. the skinning deforms the
mesh through the playback path. GPU + Vulkan + USD + the asset required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
ELEPHANT = (
    PROJECT_ROOT / "assets" / "assets-main" / "full_assets"
    / "ElephantWithMonochord" / "SoC-ElephantWithMonochord.usdc"
)

pytestmark = pytest.mark.gpu


def _have_vulkan():
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")
needs_elephant = pytest.mark.skipif(
    not ELEPHANT.exists(), reason="ElephantWithMonochord asset not present"
)

W = H = 96


@needs_vulkan
@needs_elephant
class TestHeadlessSkeletal:
    @pytest.fixture(scope="class")
    def renderer(self):
        from skinny.renderer import Renderer
        from skinny.vk_context import VulkanContext

        ctx = VulkanContext(window=None, width=W, height=H)
        r = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR, usd_scene_path=ELEPHANT,
        )
        deadline = 500
        while deadline > 0 and (
            r._usd_scene is None or len(r._usd_scene.instances) < 2
        ):
            r.update(0.02)
            deadline -= 1
        try:
            yield r
        finally:
            r.cleanup()
            ctx.destroy()

    def _render_at(self, r, tc: float) -> np.ndarray:
        r.clock.playing = False
        r.clock.current_time_code = float(tc)
        last = None
        for _ in range(4):
            r.update(0.0)
            last = r.render_headless()
        return np.frombuffer(last, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].astype(np.float32)

    def test_skinned_stage_detected(self, renderer):
        assert renderer.clock.has_animation
        assert len(renderer._anim_index.skinned_mesh_paths) == 2
        assert renderer._skeletal is not None and renderer._skeletal.has_skinning

    def test_mesh_deforms_between_time_codes(self, renderer):
        a = self._render_at(renderer, 1.0)
        b = self._render_at(renderer, 1500.0)
        assert np.abs(a - b).mean() > 0.2
