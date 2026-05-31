"""Headless integration check for the transform gizmo on a real USD scene.

Drives an actual drag through the renderer's gizmo API (real camera matrices,
real instance transform writeback) and asserts the manipulation touches the
right TRS channel:

- A world-rotate drag changes the instance's rotation, leaving translate/scale.
- A world-translate drag moves the instance along a single world axis, leaving
  its rotation.

This is the end-to-end A/B for the unified axis-angle rotate path (it exercises
the matrix→euler round-trip that replaced the old per-euler-component add) and
guards the renderer wiring (``set_gizmo_target`` basis, idle basis refresh).
Needs Vulkan + the demo scene; skipped otherwise.
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


def _have_vulkan():
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


@needs_vulkan
@pytest.mark.skipif(not DEMO_SCENE.exists(), reason="three_materials_demo.usda missing")
class TestGizmoDragOnScene:
    WIDTH = HEIGHT = 128

    @pytest.fixture(scope="class")
    def renderer_and_ctx(self):
        from skinny.vk_context import VulkanContext
        from skinny.renderer import Renderer
        ctx = VulkanContext(window=None, width=self.WIDTH, height=self.HEIGHT)
        renderer = Renderer(
            vk_ctx=ctx,
            shader_dir=SHADER_DIR,
            hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR,
            usd_scene_path=DEMO_SCENE,
        )
        deadline = 200
        while deadline > 0 and (
            renderer._usd_scene is None or len(renderer._usd_scene.instances) < 1
        ):
            renderer.update(0.025)
            deadline -= 1
        yield renderer
        renderer.cleanup()
        ctx.destroy()

    def _pivot_pixel(self, renderer):
        from skinny.gizmo import _project_to_pixel
        view = renderer.camera.view_matrix()
        proj = renderer.camera.projection_matrix(self.WIDTH / self.HEIGHT)
        pivot = renderer.gizmo.pivot_world
        return _project_to_pixel(pivot, view, proj, self.WIDTH, self.HEIGHT)

    def test_world_rotate_changes_only_rotation(self, renderer_and_ctx):
        from skinny.gizmo import GizmoMode
        from skinny.scene_graph import decompose_trs_matrix
        renderer = renderer_and_ctx
        assert renderer._usd_scene is not None and renderer._usd_scene.instances

        renderer.set_gizmo_target(0)
        renderer.gizmo.mode = GizmoMode.ROTATE_WORLD
        before = renderer._usd_scene.instances[0].transform.copy()
        t0, r0, s0 = decompose_trs_matrix(before)

        center = self._pivot_pixel(renderer)
        assert center is not None, "pivot must project on-screen"
        cx, cy = center[0], center[1]
        # Sweep ~90° around the pivot in screen space about the Y ring.
        assert renderer.gizmo_begin_drag("y", cx + 35.0, cy)
        renderer.gizmo_update_drag(cx, cy + 35.0)
        renderer.gizmo_end_drag()

        after = renderer._usd_scene.instances[0].transform
        t1, r1, s1 = decompose_trs_matrix(after)
        assert not np.allclose(r0, r1, atol=1e-3), "rotation should change"
        assert np.allclose(t0, t1, atol=1e-4), "translation must not move"
        assert np.allclose(s0, s1, atol=1e-4), "scale must not change"

    def test_world_translate_moves_single_axis(self, renderer_and_ctx):
        from skinny.gizmo import GizmoMode
        from skinny.scene_graph import decompose_trs_matrix
        renderer = renderer_and_ctx
        renderer.set_gizmo_target(0)
        renderer.gizmo.mode = GizmoMode.TRANSLATE_WORLD
        before = renderer._usd_scene.instances[0].transform.copy()
        t0, r0, _ = decompose_trs_matrix(before)

        center = self._pivot_pixel(renderer)
        assert center is not None
        cx, cy = center[0], center[1]
        assert renderer.gizmo_begin_drag("x", cx, cy)
        renderer.gizmo_update_drag(cx + 40.0, cy)
        renderer.gizmo_end_drag()

        after = renderer._usd_scene.instances[0].transform
        t1, r1, _ = decompose_trs_matrix(after)
        moved = np.abs(np.array(t1) - np.array(t0))
        assert moved.max() > 1e-3, "translation should change"
        # X is the largest-moving channel (single-axis drag).
        assert np.argmax(moved) == 0, f"expected X-dominant move, got {moved}"
        assert np.allclose(r0, r1, atol=1e-4), "rotation must not change"
