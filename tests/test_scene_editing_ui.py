"""Headless renderer tests for the Spec 2 editing-resync behavior.

Covers the renderer-side guarantees the UI controls depend on: after add/remove
the derived scene graph is rebuilt + the version bumped, lights/camera are
re-read (so deleting a light drops it), and a runtime enable-toggle survives an
unrelated edit. GUI widgets themselves are smoke-tested manually (they cannot be
driven headless). Run via the repo-root 3.13 venv with VULKAN_SDK set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"

LIGHT_PATH = "/Cornell/CeilingLight"


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


def _have_vulkan() -> bool:
    try:
        import vulkan  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(
    not _have_usd() or not SCENE.exists(),
    reason="OpenUSD (pxr) not installed or cornell_box_sphere.usda missing",
)
needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


@pytest.fixture()
def editor():
    from pxr import Usd
    from skinny.renderer import Renderer
    from skinny.usd_loader import load_scene_from_stage
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=96, height=96)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR, tattoo_dir=TATTOO_DIR,
    )
    stage = Usd.Stage.Open(str(SCENE))
    scene = load_scene_from_stage(stage)
    renderer.set_usd_scene(scene, stage=stage)
    try:
        yield renderer, stage
    finally:
        renderer.cleanup()
        ctx.destroy()


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestResyncRefreshesSceneGraph:
    def test_add_rebuilds_graph_and_bumps_version(self, editor):
        renderer, _ = editor
        ver_before = renderer._scene_graph_version
        path = renderer.add_model(str(SCENE), parent_prim_path="/World", name="Added")
        assert renderer._scene_graph is not None
        assert renderer._scene_graph_version > ver_before
        paths = {n["path"] for n in renderer.list_nodes()}
        assert path in paths

    def test_remove_bumps_version(self, editor):
        renderer, _ = editor
        ver_before = renderer._scene_graph_version
        renderer.remove_node("/Cornell/TallBlock/Block")
        assert renderer._scene_graph_version > ver_before


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestResyncLights:
    def test_remove_light_drops_it(self, editor):
        renderer, _ = editor
        assert any(l.prim_path == LIGHT_PATH for l in renderer._usd_scene.lights_sphere)
        renderer.remove_node(LIGHT_PATH)
        assert all(l.prim_path != LIGHT_PATH for l in renderer._usd_scene.lights_sphere)

    def test_light_enable_preserved_across_unrelated_add(self, editor):
        renderer, _ = editor
        # Disable the ceiling light directly (simulating a UI toggle), then make
        # an unrelated edit; the toggle must survive the lights re-read.
        light = next(l for l in renderer._usd_scene.lights_sphere if l.prim_path == LIGHT_PATH)
        light.enabled = False
        renderer.add_model(str(SCENE), parent_prim_path="/World", name="Extra")
        survivor = next(
            l for l in renderer._usd_scene.lights_sphere if l.prim_path == LIGHT_PATH
        )
        assert survivor.enabled is False

    def test_default_lights_survive_resync(self, editor):
        renderer, _ = editor
        if renderer._default_light_stage is None:
            pytest.skip("no default-light stage in this build")
        renderer.remove_node("/Cornell/TallBlock/Block")
        paths = {n["path"] for n in renderer.list_nodes()} if False else None
        # Default lights live in the scene graph, not list_nodes(stage); check it.
        graph_paths = _collect_graph_paths(renderer._scene_graph)
        assert "/Skinny/DefaultDome" in graph_paths or "/Skinny/DefaultLight" in graph_paths


def _collect_graph_paths(node) -> set:
    out = set()
    if node is None:
        return out
    out.add(node.path)
    for c in node.children:
        out |= _collect_graph_paths(c)
    return out
