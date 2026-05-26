"""Tests for the headless render API (skinny.headless) + loader stage support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


class TestRenderOptions:
    def test_resolve_defaults(self):
        from skinny.headless import RenderOptions
        opts = RenderOptions()
        assert opts.samples == 64
        assert opts.integrator_index == 0   # path
        assert opts.tonemap_index == 0       # aces

    def test_integrator_bdpt(self):
        from skinny.headless import RenderOptions
        assert RenderOptions(integrator="bdpt").integrator_index == 1

    def test_tonemap_hable(self):
        from skinny.headless import RenderOptions
        assert RenderOptions(tonemap="hable").tonemap_index == 2

    def test_bad_integrator_raises(self):
        from skinny.headless import RenderOptions
        with pytest.raises(ValueError, match="integrator"):
            RenderOptions(integrator="nope")

    def test_bad_tonemap_raises(self):
        from skinny.headless import RenderOptions
        with pytest.raises(ValueError, match="tonemap"):
            RenderOptions(tonemap="nope")


def test_fmt_for_output():
    from skinny.headless import _fmt_for_output
    assert _fmt_for_output(Path("a.png"), None) == "png"
    assert _fmt_for_output(Path("a.JPG"), None) == "jpeg"
    assert _fmt_for_output(Path("a.png"), "exr") == "exr"
    assert _fmt_for_output(Path("a.png"), "jpg") == "jpeg"
    with pytest.raises(ValueError, match="format"):
        _fmt_for_output(Path("a.gif"), None)


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestHeadlessRender:
    def test_render_to_array_shape_nonblack(self):
        from skinny.headless import HeadlessRenderer
        with HeadlessRenderer(128, 128) as r:
            arr = r.render_to_array(SCENE, samples=8)
        assert arr.shape == (128, 128, 4)
        assert arr.dtype.name == "uint8"
        assert int(arr[..., :3].max()) > 0

    def test_render_scene_writes_png(self, tmp_path):
        from skinny.headless import HeadlessRenderer
        out = tmp_path / "frame.png"
        with HeadlessRenderer(128, 128) as r:
            r.render_scene(SCENE, out, samples=8)
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_render_scene_writes_exr(self, tmp_path):
        from skinny.headless import HeadlessRenderer
        out = tmp_path / "frame.exr"
        with HeadlessRenderer(96, 96) as r:
            r.render_scene(SCENE, out, samples=4)
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x76\x2f\x31\x01"  # OpenEXR magic

    def test_stage_mutation_changes_output(self):
        from pxr import Gf, Usd, UsdGeom
        from skinny.headless import HeadlessRenderer
        stage = Usd.Stage.Open(str(SCENE))
        # Pick the first Xformable mesh prim in the scene to translate.
        target = None
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Xformable(prim):
                target = UsdGeom.Xformable(prim)
                break
        assert target is not None, "no Xformable mesh prim found"
        # Use XformCommonAPI for robustness — works even if xformOps already exist.
        xform_api = UsdGeom.XformCommonAPI(target.GetPrim())
        with HeadlessRenderer(96, 96) as r:
            xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
            a = r.render_to_array(stage, samples=8)
            xform_api.SetTranslate(Gf.Vec3d(1.0, 1.0, 0.0))
            b = r.render_to_array(stage, samples=8)
        assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0


class TestFrameRange:
    def test_parse_start_end(self):
        from skinny.headless import _parse_frames
        assert _parse_frames("1:10") == (1.0, 10.0, 1.0)

    def test_parse_start_end_step(self):
        from skinny.headless import _parse_frames
        assert _parse_frames("0:48:2") == (0.0, 48.0, 2.0)

    def test_parse_bad_raises(self):
        from skinny.headless import _parse_frames
        with pytest.raises(ValueError, match="frames"):
            _parse_frames("5")

    def test_frame_times_inclusive(self):
        from skinny.headless import _frame_times
        assert _frame_times((1.0, 4.0, 1.0)) == [1.0, 2.0, 3.0, 4.0]

    def test_frame_times_step(self):
        from skinny.headless import _frame_times
        assert _frame_times((0.0, 10.0, 5.0)) == [0.0, 5.0, 10.0]


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestAnimation:
    def test_animation_writes_frames(self, tmp_path):
        from skinny.headless import HeadlessRenderer
        with HeadlessRenderer(64, 64) as r:
            paths = r.render_animation(
                SCENE, tmp_path, samples=4, frames=(0, 2, 1),
            )
        assert len(paths) == 3
        assert all(p.exists() for p in paths)
        assert all(p.read_bytes()[:4] == b"\x89PNG" for p in paths)


class TestCli:
    def test_parser_single(self):
        from skinny.headless import _build_parser
        ns = _build_parser().parse_args(
            ["scene.usda", "-o", "out.png", "--samples", "32"]
        )
        assert ns.source == "scene.usda"
        assert ns.output == "out.png"
        assert ns.samples == 32
        assert not ns.animate

    def test_parser_animate(self):
        from skinny.headless import _build_parser
        ns = _build_parser().parse_args(
            ["shot.usda", "--outdir", "frames", "--animate",
             "--frames", "1:48:2", "--fps", "24"]
        )
        assert ns.animate
        assert ns.outdir == "frames"
        assert ns.frames == "1:48:2"
        assert ns.fps == 24.0

    def test_parser_render_opts(self):
        from skinny.headless import _build_parser
        ns = _build_parser().parse_args(
            ["s.usda", "-o", "o.exr", "--integrator", "bdpt",
             "--tonemap", "hable", "--exposure", "0.5", "--no-direct",
             "--width", "800", "--height", "600"]
        )
        assert ns.integrator == "bdpt"
        assert ns.tonemap == "hable"
        assert ns.exposure == 0.5
        assert ns.no_direct is True
        assert ns.width == 800 and ns.height == 600
