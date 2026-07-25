"""A rebuild over an unchanged shader tree must reuse the cached `.spv` and
never invoke `slangc` (change shader-variant-key-module, spec requirement
"spv cache hits survive the migration").

`gpu`-marked because it imports `vk_compute`, which imports `vulkan` at module
load — the default `.venv` sweep has no Vulkan SDK on the library path, so this
is DESELECTED there rather than silently skipped. It runs for real under the
guarded interpreter:

    export VULKAN_SDK=$HOME/VulkanSDK/<ver>/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    PYTHONPATH=src ./bin/python3.13 -m pytest tests/test_spv_cache_hit.py -m gpu

No GPU device is touched: only the compile-cache path is exercised.
"""

from __future__ import annotations

import pytest

from skinny.shader_variants import (
    Family,
    ShaderVariantKey,
    Target,
    slangc_flags,
)

pytestmark = pytest.mark.gpu


@pytest.fixture
def tree(tmp_path):
    shaders = tmp_path / "src" / "skinny" / "shaders"
    shaders.mkdir(parents=True)
    (tmp_path / "src" / "skinny" / "mtlx" / "genslang").mkdir(parents=True)
    (shaders / "preview_pass.slang").write_text("// preview\n", encoding="utf-8")
    return tmp_path, shaders


def _preview_pipeline(shaders, build_dir):
    """A PreviewPipeline positioned over the fixture tree, with no GPU device —
    `_compile_slang` touches only shader_dir / entry names / the cache dir."""
    from skinny.vk_compute import PreviewPipeline

    p = PreviewPipeline.__new__(PreviewPipeline)
    p.shader_dir = shaders
    p.entry_module = "preview_pass"
    p.entry_point = "previewMain"
    p._build_dir = lambda: build_dir
    return p


def test_rebuild_over_an_unchanged_tree_reuses_the_cache_without_slangc(
        tree, monkeypatch):
    from skinny import vk_compute

    tmp_path, shaders = tree
    build_dir = tmp_path / "build"
    pipe = _preview_pipeline(shaders, build_dir)

    # Seed the cache exactly as a previous build would have, under the key the
    # migrated flag tuple derives.
    flags = slangc_flags(
        ShaderVariantKey(Target.VULKAN, Family.PREVIEW),
        entry="previewMain",
        include_paths=(shaders, shaders.parent / "mtlx" / "genslang"))
    key = pipe._cache_key(shaders / "preview_pass.slang", flags)
    cache_dir = build_dir / vk_compute.ComputePipeline._CACHE_DIRNAME
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{key}.spv").write_bytes(b"CACHED-SPIRV")

    # Any slangc invocation now fails the test outright.
    def _no_slangc(*a, **kw):
        raise AssertionError("slangc was invoked despite a warm cache")

    monkeypatch.setattr(vk_compute.subprocess, "run", _no_slangc)
    monkeypatch.setattr(vk_compute.shutil, "which", lambda _: "/usr/bin/slangc")

    out = pipe._compile_slang()
    assert out.read_bytes() == b"CACHED-SPIRV"


def test_a_changed_shader_tree_misses_the_cache(tree, monkeypatch):
    """Negative control — without it, a cache lookup that always "hit" would
    pass the test above."""
    from skinny import vk_compute

    tmp_path, shaders = tree
    build_dir = tmp_path / "build"
    pipe = _preview_pipeline(shaders, build_dir)

    flags = slangc_flags(
        ShaderVariantKey(Target.VULKAN, Family.PREVIEW),
        entry="previewMain",
        include_paths=(shaders, shaders.parent / "mtlx" / "genslang"))
    src = shaders / "preview_pass.slang"
    key = pipe._cache_key(src, flags)
    cache_dir = build_dir / vk_compute.ComputePipeline._CACHE_DIRNAME
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{key}.spv").write_bytes(b"CACHED-SPIRV")

    src.write_text("// preview, edited\n", encoding="utf-8")
    assert pipe._cache_key(src, flags) != key

    calls = []

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        (shaders / "preview_pass.spv").write_bytes(b"FRESH-SPIRV")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(vk_compute.subprocess, "run", _fake_run)
    monkeypatch.setattr(vk_compute.shutil, "which", lambda _: "/usr/bin/slangc")

    out = pipe._compile_slang()
    assert calls, "a changed tree must fall through to slangc"
    assert out.read_bytes() == b"FRESH-SPIRV"
