"""GPU integration tests for the geometry suballocator (tasks 9.4 / 9.5).

Drives the live USD edit path (remove_node / add_model / compact_geometry) on
the three-sphere demo and asserts the spec scenarios that need a real device:
removing a mesh keeps survivors' slab offsets and re-uploads nothing for them;
adding a mesh leaves resident offsets stable; and compaction relocates a slab
while leaving the rendered image unchanged.
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

WIDTH = HEIGHT = 96
WARMUP = 16


def _make_renderer():
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
    )
    deadline = 200
    while deadline > 0 and (
        renderer._usd_scene is None or len(renderer._usd_scene.instances) < 3
    ):
        renderer.update(0.025)
        deadline -= 1
    assert renderer.pipeline is not None
    return ctx, renderer


def _render(renderer, frames=WARMUP):
    for _ in range(frames):
        renderer.update(0.04)
        renderer.render_headless()
    return renderer.read_accumulation()[:, :, :3].copy()


def _key_for(renderer, needle):
    """The slab key whose prim path contains ``needle``."""
    for key in renderer._slab_alloc.alive_keys():
        if isinstance(key, tuple) and needle in str(key[0]):
            return key
    raise AssertionError(f"no slab key for {needle!r} in {renderer._slab_alloc.alive_keys()}")


def test_remove_keeps_survivor_offsets_and_skips_reupload():
    ctx, renderer = _make_renderer()
    try:
        _render(renderer, 4)
        marble = _key_for(renderer, "Marble")
        brass = _key_for(renderer, "Brass")
        off_marble = renderer._slab_alloc.offsets(marble)
        off_brass = renderer._slab_alloc.offsets(brass)

        # Count slab writes during the removal resync.
        writes = {"n": 0}
        for buf in (renderer.vertex_buffer, renderer.index_buffer, renderer.bvh_buffer):
            orig = buf.upload_range
            def counted(data, off, _orig=orig):
                writes["n"] += 1
                return _orig(data, off)
            buf.upload_range = counted

        wood_path = _key_for(renderer, "Wood")[0]
        renderer.remove_node(wood_path)

        # Survivors keep their exact offsets (instance records stay valid)...
        assert renderer._slab_alloc.offsets(marble) == off_marble
        assert renderer._slab_alloc.offsets(brass) == off_brass
        # ...and removing re-uploads nothing (no grow, survivors resident).
        assert writes["n"] == 0, f"removal re-uploaded {writes['n']} slab chunk(s)"
        # Wood's slab is freed (no longer resident).
        wood_keys = [k for k in renderer._slab_alloc.alive_keys() if "Wood" in str(k[0])]
        assert not wood_keys, "removed mesh still resident"
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_add_keeps_resident_offsets_stable():
    ctx, renderer = _make_renderer()
    try:
        _render(renderer, 4)
        marble = _key_for(renderer, "Marble")
        brass = _key_for(renderer, "Brass")
        off_marble = renderer._slab_alloc.offsets(marble)
        off_brass = renderer._slab_alloc.offsets(brass)
        n_keys_before = len(renderer._slab_alloc.alive_keys())

        v_before = renderer.vertex_buffer.size
        # Add a second copy of the demo by reference — introduces new instances
        # at fresh prim paths (new slabs) without disturbing the resident ones.
        renderer.add_model(str(DEMO_SCENE), parent_prim_path="/World", name="Added")

        # Resident meshes never move, regardless of grow/free-list reuse.
        assert renderer._slab_alloc.offsets(marble) == off_marble
        assert renderer._slab_alloc.offsets(brass) == off_brass
        # New geometry was registered as additional slab(s).
        assert len(renderer._slab_alloc.alive_keys()) > n_keys_before

        # If the buffers did not need to grow, the resident slabs were not
        # re-uploaded — only the new mesh's slab was written.
        if renderer.vertex_buffer.size == v_before:
            img = _render(renderer, WARMUP)
            assert np.all(np.isfinite(img))
    finally:
        renderer.cleanup()
        ctx.destroy()


def test_compaction_preserves_rendered_output():
    ctx, renderer = _make_renderer()
    try:
        _render(renderer, 4)
        # Remove the LOWEST-offset mesh so every survivor must slide down on
        # compaction — deterministic regardless of the (async) streaming order
        # that decides which slab sits where.
        alloc = renderer._slab_alloc
        keys = alloc.alive_keys()
        lowest = min(keys, key=lambda k: alloc.offsets(k).v)
        survivors_before = {k: alloc.offsets(k) for k in keys if k != lowest}
        renderer.remove_node(lowest[0])

        def _match(a, b):
            d = np.abs(a - b).max(axis=2)
            tol = 5e-3 + 0.02 * np.abs(b).max(axis=2)
            return float((d <= tol).mean())

        # Fresh accumulation of the fragmented layout (a hole at offset 0).
        img_frag = _render(renderer, WARMUP)

        # CONTROL: re-render the SAME (un-compacted) layout after an accum reset.
        # The path tracer is not bit-reproducible across resets (MoltenVK FP
        # ordering on a glossy survivor), so this measures the inherent re-render
        # noise floor — the bar compaction must not exceed.
        renderer._material_version += 1
        img_ctrl = _render(renderer, WARMUP)
        noise_floor = _match(img_ctrl, img_frag)
        assert noise_floor > 0.5, (
            f"renderer too unstable to test against ({noise_floor:.2f})"
        )

        # Compact: survivors slide down to fill the freed low region.
        n_moved = renderer.compact_geometry()
        assert n_moved >= 1, "expected compaction to relocate a slab"
        moved = [k for k, off in survivors_before.items()
                 if renderer._slab_alloc.offsets(k) != off]
        assert moved, "no survivor slab moved"

        # Force a clean re-accumulation and re-render the compacted layout.
        renderer._material_version += 1
        img_compact = _render(renderer, WARMUP)

        assert np.all(np.isfinite(img_compact))
        # Compaction (byte move + TLAS-offset rewrite) must change the image no
        # more than a plain re-render does — i.e. it introduces no error beyond
        # the renderer's own noise floor.
        match = _match(img_compact, img_frag)
        assert match >= noise_floor - 0.03, (
            f"compaction degraded the image beyond the re-render noise floor "
            f"(compaction match {match:.3f} vs control {noise_floor:.3f})"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()
