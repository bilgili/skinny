"""End-to-end headless check for USD animation playback.

Loads a synthetic stage with a single time-sampled sphere translating across
the frame, renders at two time codes, and asserts the rendered image changes
(the instance moved). Proves the transform-animation re-eval path drives the
GPU without a mesh rebake. GPU + Vulkan + USD required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"

pytestmark = pytest.mark.gpu


def _have_vulkan():
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


def _have_usd():
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")
needs_usd = pytest.mark.skipif(not _have_usd(), reason="pxr/USD not installed")

ANIM_USDA = """#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 24
    timeCodesPerSecond = 24
    upAxis = "Y"
    metersPerUnit = 1
)

def Sphere "Ball"
{
    double3 xformOp:translate.timeSamples = {
        0: (-4, 0, 0),
        24: (4, 0, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
    double radius = 1.5
}

def Camera "Cam"
{
    float focalLength = 35
    float verticalAperture = 24
    double3 xformOp:translate = (0, 0, 14)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def DistantLight "Sun"
{
    float inputs:intensity = 3
    double3 xformOp:rotateXYZ = (-45, 20, 0)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
}
"""

W = H = 128


@needs_vulkan
@needs_usd
class TestHeadlessAnimationPlayback:
    @pytest.fixture(scope="class")
    def renderer(self, tmp_path_factory):
        from skinny.renderer import Renderer
        from skinny.vk_context import VulkanContext

        scene_path = tmp_path_factory.mktemp("anim") / "ball.usda"
        scene_path.write_text(ANIM_USDA)

        ctx = VulkanContext(window=None, width=W, height=H)
        r = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR, usd_scene_path=scene_path,
        )
        deadline = 200
        while deadline > 0 and (
            r._usd_scene is None or len(r._usd_scene.instances) < 1
        ):
            r.update(0.025)
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
        for _ in range(6):
            r.update(0.0)
            last = r.render_headless()
        return np.frombuffer(last, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].astype(np.float32)

    @staticmethod
    def _bright_centroid_x(img: np.ndarray) -> float:
        lum = img.mean(axis=2)
        mask = lum > (lum.mean() + lum.std())
        xs = np.where(mask.any(axis=0))[0]
        return float(xs.mean()) if len(xs) else -1.0

    def test_animation_detected(self, renderer):
        assert renderer.clock.has_animation
        assert renderer._anim_index is not None
        assert "/Ball" in renderer._anim_index.xform_paths

    def test_instance_moves_between_time_codes(self, renderer):
        a = self._render_at(renderer, 0.0)
        b = self._render_at(renderer, 24.0)
        assert np.abs(a - b).mean() > 1.0
        # Sphere translates x:-4→+4, so its bright centroid shifts rightward.
        assert self._bright_centroid_x(b) > self._bright_centroid_x(a) + 10.0
