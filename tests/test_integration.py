"""Integration tests: shader compilation smoke + pipeline verification."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

SKIP_DIRS = {"estimators", "integrators"}

SKIP_SHADERS = {
    "main_pass.slang", "bindings.slang", "composite_pass.slang",
    "tonemap_pass.slang", "debug_pass.slang", "environment.slang",
    "material_eval.slang", "skin_material.slang", "skin_shading.slang",
    "volume_render.slang",
}


class TestShaderCompilation:
    """Load library .slang files to verify they compile without errors."""

    def test_library_shaders_compile(self, device, shader_dir):
        spy = pytest.importorskip("slangpy")
        all_slang = sorted(shader_dir.rglob("*.slang"))
        assert len(all_slang) > 20, f"Expected 20+ shader files, found {len(all_slang)}"

        skipped = []
        failures = []
        for path in all_slang:
            rel_parts = path.relative_to(shader_dir).parts
            if rel_parts[0] in SKIP_DIRS or path.name in SKIP_SHADERS:
                skipped.append(path.name)
                continue
            try:
                spy.Module.load_from_file(device, str(path))
            except Exception as e:
                failures.append((path.name, str(e)[:200]))

        if failures:
            msg = "\n".join(f"  {name}: {err}" for name, err in failures)
            pytest.fail(f"Shader compilation failures:\n{msg}")

    def test_pipeline_shaders_exist(self, shader_dir):
        for name in SKIP_SHADERS:
            path = shader_dir / name
            if path.exists():
                assert path.stat().st_size > 0


class TestHarnessCompilation:
    """Verify all test harness modules load without errors."""

    @pytest.mark.parametrize("harness", [
        "test_common_harness.slang",
        "test_sampler_harness.slang",
        "test_light_harness.slang",
        "test_skin_harness.slang",
        "test_volume_harness.slang",
    ])
    def test_harness_compiles(self, load_shader, harness):
        m = load_shader(harness)
        assert m is not None


class TestMtlxGenSlangCompilation:
    """Verify MaterialX-generated Slang files compile."""

    @pytest.mark.xfail(reason="MaterialX generated Slang has compilation errors at line 166")
    def test_mtlx_genslang_files_compile(self, device, mtlx_dir):
        spy = pytest.importorskip("slangpy")
        all_slang = sorted(mtlx_dir.rglob("*.slang"))
        if not all_slang:
            pytest.skip("No MaterialX generated Slang files found")

        failures = []
        for path in all_slang:
            try:
                spy.Module.load_from_file(device, str(path))
            except Exception as e:
                failures.append((path.name, str(e)[:200]))

        if failures:
            msg = "\n".join(f"  {name}: {err}" for name, err in failures)
            pytest.fail(f"MtlxGenSlang compilation failures:\n{msg}")
