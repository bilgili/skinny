"""Wavefront per-material shade compile-win (per-material-pipeline, task 6.4 / 9.5).

Each material's shade module is its own compilation unit with a content-hash
SPIR-V cache, so:
  - rebuilding the same material set compiles nothing (all cache hits), and
  - introducing a new/changed material compiles exactly one kernel, leaving
    every resident material's pipeline a cache hit.

The second is simulated deterministically by evicting one material's cached
SPIR-V (equivalent to a previously-unseen graph) and asserting only it recompiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
DEMO_SCENE = PROJECT_ROOT / "assets" / "three_materials_demo.usda"

pytestmark = pytest.mark.gpu

WIDTH = HEIGHT = 64


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
    for _ in range(4):  # settle camera + allocate geometry buffers
        renderer.update(0.04)
        renderer.render_headless()
    return ctx, renderer


def test_shade_pipelines_cache_per_material():
    ctx, renderer = _make_renderer()
    try:
        # Warm the per-material SPIR-V cache (first build may hit or miss
        # depending on prior runs; this guarantees every module is cached).
        g0 = renderer.build_wavefront_shade_passes()
        names = [n for n, _ in g0.shade_compiles]
        keys = dict(g0.shade_keys)
        cache_dir = g0.cache_dir
        g0.destroy()

        assert len(names) >= 2, f"expected ≥2 material graphs, got {names}"
        # Each material has a distinct cache key — independent compilation units.
        assert len(set(keys.values())) == len(names), "material cache keys collide"

        # Rebuild the SAME material set → nothing recompiles (all cache hits).
        g1 = renderer.build_wavefront_shade_passes()
        assert all(cached for _, cached in g1.shade_compiles), (
            f"reusing existing materials recompiled: {g1.shade_compiles}"
        )
        g1.destroy()

        # Simulate a previously-unseen material: evict ONE module's cached
        # SPIR-V. The rebuild must recompile exactly that material and reuse the
        # rest from cache.
        victim = names[len(names) // 2]
        (cache_dir / f"{keys[victim]}.spv").unlink()
        g2 = renderer.build_wavefront_shade_passes()
        compiled = sorted(n for n, cached in g2.shade_compiles if not cached)
        g2.destroy()
        assert compiled == [victim], (
            f"expected only {victim!r} to recompile, got {compiled} "
            f"(full: {[(n, c) for n, c in g2.shade_compiles]})"
        )
    finally:
        renderer.cleanup()
        ctx.destroy()
