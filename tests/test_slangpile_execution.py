"""GPU execution tests for slangpile-compiled kernels.

These tests compile @sp.shader Python functions to Slang,
load via slangpy, and verify results on GPU.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

from tests.helpers import assert_near

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.xfail(reason="slangpy Windows UTF-8 encoding issue in dispatch kernel generation"),
]

_TEMP_DIR = tempfile.mkdtemp(prefix="slangpile_exec_test_")
_EXEC_OUT = Path(_TEMP_DIR) / "exec_output"
_EXEC_OUT.mkdir(exist_ok=True)


@pytest.fixture
def exec_out():
    d = _EXEC_OUT / str(id(object()))
    d.mkdir(exist_ok=True)
    return d


def _make_module(name: str, code: str) -> types.ModuleType:
    path = Path(_TEMP_DIR) / f"{name}.py"
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def slangpile_device(shader_dir):
    spy = pytest.importorskip("slangpy")
    try:
        return spy.create_device(include_paths=[str(shader_dir)])
    except Exception:
        pytest.skip("No Vulkan device available")


class TestBasicExecution:
    def test_add_function(self, slangpile_device, exec_out):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def gpu_add(a: sp.float32, b: sp.float32) -> sp.float32:
    return a + b
"""
        mod = _make_module("test_gpu_add", code)
        compiled = sp.compile_module(mod)
        runtime = compiled.load_slangpy(exec_out, device=slangpile_device)
        result = float(runtime.gpu_add(3.0, 4.0))
        assert_near(result, 7.0)

    def test_multiply(self, slangpile_device, exec_out):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def gpu_mul(a: sp.float32, b: sp.float32) -> sp.float32:
    return a * b
"""
        mod = _make_module("test_gpu_mul", code)
        compiled = sp.compile_module(mod)
        runtime = compiled.load_slangpy(exec_out, device=slangpile_device)
        result = float(runtime.gpu_mul(3.0, 5.0))
        assert_near(result, 15.0)

    def test_conditional(self, slangpile_device, exec_out):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def gpu_max_val(a: sp.float32, b: sp.float32) -> sp.float32:
    if a > b:
        return a
    else:
        return b
"""
        mod = _make_module("test_gpu_cond", code)
        compiled = sp.compile_module(mod)
        runtime = compiled.load_slangpy(exec_out, device=slangpile_device)
        assert_near(float(runtime.gpu_max_val(3.0, 5.0)), 5.0)
        assert_near(float(runtime.gpu_max_val(7.0, 2.0)), 7.0)


class TestCrossValidation:
    """Compare slangpile reference kernels against .slang implementations."""

    def test_power_heuristic_match(self, slangpile_device, common_harness, exec_out):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def ref_power_heuristic(pdf_a: sp.float32, pdf_b: sp.float32) -> sp.float32:
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    return a2 / max(a2 + b2, 1e-12)
"""
        mod = _make_module("test_ph_xval", code)
        compiled = sp.compile_module(mod)
        runtime = compiled.load_slangpy(exec_out, device=slangpile_device)

        for a, b in [(1.0, 1.0), (2.0, 1.0), (1.0, 5.0), (10.0, 0.1)]:
            ref = float(runtime.ref_power_heuristic(a, b))
            gpu = float(common_harness.test_powerHeuristic(a, b))
            assert_near(ref, gpu, rel=1e-4)

    def test_uniform_sphere_pdf_match(self, slangpile_device, common_harness, exec_out):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def ref_uniform_pdf() -> sp.float32:
    return 1.0 / (4.0 * 3.14159265358979)
"""
        mod = _make_module("test_updf_xval", code)
        compiled = sp.compile_module(mod)
        runtime = compiled.load_slangpy(exec_out, device=slangpile_device)

        ref = float(runtime.ref_uniform_pdf())
        gpu = float(common_harness.test_uniformSpherePdf())
        assert_near(ref, gpu, rel=1e-4)
