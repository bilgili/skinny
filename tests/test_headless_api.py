"""Tests for the headless render API (skinny.headless) + loader stage support."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(
    not _have_usd() or not SCENE.exists(),
    reason="OpenUSD (pxr) not installed or cornell_box_sphere.usda missing",
)


SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"


def _have_vulkan() -> bool:
    try:
        import vulkan  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


@needs_usd
class TestLoadSceneFromStage:
    def test_stage_matches_path(self):
        from pxr import Usd
        from skinny.usd_loader import load_scene_from_stage, load_scene_from_usd

        by_path = load_scene_from_usd(SCENE)
        stage = Usd.Stage.Open(str(SCENE))
        by_stage = load_scene_from_stage(stage)

        assert len(by_stage.instances) == len(by_path.instances)
        assert len(by_stage.materials) == len(by_path.materials)
        assert len(by_stage.lights_dir) == len(by_path.lights_dir)


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestSetUsdScene:
    def test_set_usd_scene_renders_nonblack(self):
        from skinny.renderer import Renderer
        from skinny.usd_loader import load_scene_from_usd
        from skinny.vk_context import VulkanContext

        ctx = VulkanContext(window=None, width=128, height=128)
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR,
        )
        try:
            scene = load_scene_from_usd(SCENE)
            renderer.set_usd_scene(scene)
            assert renderer._is_usd_active()
            raw = b""
            for _ in range(8):
                renderer.update(0.016)
                raw = renderer.render_headless()
            assert any(b != 0 for b in raw), "USD scene should not render all-black"
        finally:
            renderer.cleanup()
            ctx.destroy()
