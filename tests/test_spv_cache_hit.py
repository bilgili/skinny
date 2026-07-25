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


@pytest.mark.parametrize("spectral", [False, True])
def test_megakernel_warm_cache_returns_before_slangc(tree, monkeypatch, spectral):
    """The spectral megakernel is the case the spec calls out by name: its
    `SKINNY_SPECTRAL` define follows `-fvk-use-scalar-layout` in the hashed
    flag tuple, so it is the variant most sensitive to a splice-position
    change. Drive `ComputePipeline._compile_slang` itself for both variants.
    """
    from skinny import vk_compute

    tmp_path, shaders = tree
    (shaders / "main_pass.slang").write_text("// mega\n", encoding="utf-8")
    build_dir = tmp_path / "build"

    pipe = vk_compute.ComputePipeline.__new__(vk_compute.ComputePipeline)
    pipe.shader_dir = shaders
    pipe.entry_module = "main_pass"
    pipe.entry_point = "mainImage"
    pipe.spectral = spectral
    pipe._build_dir = lambda: build_dir

    flags = slangc_flags(
        ShaderVariantKey(Target.VULKAN, Family.MEGAKERNEL, spectral=spectral),
        entry="mainImage",
        include_paths=(shaders, shaders.parent / "mtlx" / "genslang"))
    assert (("-D", "SKINNY_SPECTRAL=1") == flags[-2:]) is spectral
    key = pipe._cache_key(shaders / "main_pass.slang", flags)
    cache_dir = build_dir / vk_compute.ComputePipeline._CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = b"CACHED-SPECTRAL" if spectral else b"CACHED-RGB"
    (cache_dir / f"{key}.spv").write_bytes(payload)

    def _no_slangc(*a, **kw):
        raise AssertionError("slangc was invoked despite a warm cache")

    monkeypatch.setattr(vk_compute.subprocess, "run", _no_slangc)
    monkeypatch.setattr(vk_compute.shutil, "which", lambda _: "/usr/bin/slangc")

    assert pipe._compile_slang().read_bytes() == payload


def test_rgb_and_spectral_megakernels_never_share_a_cache_entry(tree):
    """...and the two must not alias: a shared key would serve RGB SPIR-V to a
    spectral session (or the reverse) straight out of the warm cache."""
    tmp_path, shaders = tree
    (shaders / "main_pass.slang").write_text("// mega\n", encoding="utf-8")
    pipe = _preview_pipeline(shaders, tmp_path / "build")
    pipe.entry_module, pipe.entry_point = "main_pass", "mainImage"
    src = shaders / "main_pass.slang"
    inc = (shaders, shaders.parent / "mtlx" / "genslang")
    keys = {
        pipe._cache_key(src, slangc_flags(
            ShaderVariantKey(Target.VULKAN, Family.MEGAKERNEL, spectral=s),
            entry="mainImage", include_paths=inc))
        for s in (False, True)
    }
    assert len(keys) == 2


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
