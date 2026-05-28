"""End-to-end headless check for USD-declared controls (skinny:ui:*)."""

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


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")

SCENE_USDA = """#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1
)

def Sphere "Ball"
{
    double radius = 1.5
}

def SphereLight "Light"
{
    float inputs:intensity = 80
    float inputs:radius = 1
    double3 xformOp:translate = (4, 4, 4)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def Scope "SkinnyControls"
{
    def Scope "IBL"
    {
        custom token skinny:ui:type = "slider"
        custom string skinny:ui:target = "renderer:env_intensity"
        custom float skinny:ui:min = 0
        custom float skinny:ui:max = 4
        custom int skinny:ui:order = 0
    }
    def Scope "LightI"
    {
        custom token skinny:ui:type = "slider"
        custom string skinny:ui:target = "usd:/Light.inputs:intensity"
        custom float skinny:ui:min = 0
        custom float skinny:ui:max = 500
        custom int skinny:ui:order = 1
    }
}
"""

W = H = 96


@needs_vulkan
class TestHeadlessControls:
    @pytest.fixture(scope="class")
    def renderer(self, tmp_path_factory):
        from skinny.renderer import Renderer
        from skinny.vk_context import VulkanContext

        scene = tmp_path_factory.mktemp("ctl") / "scene.usda"
        scene.write_text(SCENE_USDA)
        ctx = VulkanContext(window=None, width=W, height=H)
        r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
                     tattoo_dir=TATTOO_DIR, usd_scene_path=scene)
        deadline = 300
        while deadline > 0 and (r._usd_scene is None or not r._usd_controls):
            r.update(0.02)
            deadline -= 1
        try:
            yield r
        finally:
            r.cleanup()
            ctx.destroy()

    def _render(self, r) -> np.ndarray:
        last = None
        for _ in range(5):
            r.update(0.0)
            last = r.render_headless()
        return np.frombuffer(last, np.uint8).reshape(H, W, 4)[:, :, :3].astype(np.float32)

    def _binding(self, r, target):
        from skinny.usd_loader import resolve_control_binding
        spec = next(c for c in r._usd_controls if c.target == target)
        return resolve_control_binding(r, spec)

    def test_controls_discovered(self, renderer):
        targets = {c.target for c in renderer._usd_controls}
        assert "renderer:env_intensity" in targets
        assert "usd:/Light.inputs:intensity" in targets

    def test_renderer_control_changes_render(self, renderer):
        _g, set_ = self._binding(renderer, "renderer:env_intensity")
        set_(0.1)
        a = self._render(renderer)
        set_(3.5)
        b = self._render(renderer)
        assert np.abs(a - b).mean() > 0.5

    def test_usd_control_changes_render(self, renderer):
        _g, set_ = self._binding(renderer, "usd:/Light.inputs:intensity")
        set_(0.0)
        a = self._render(renderer)
        set_(500.0)
        b = self._render(renderer)
        assert np.abs(a - b).mean() > 0.5
