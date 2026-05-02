"""Unit tests for the slangpile Python→Slang transpiler."""

from __future__ import annotations

import sys
import types

import pytest

from skinny.slangpile.registry import is_shader, get_shader, is_extern
from skinny.slangpile.diagnostics import SlangPileError


import importlib.util
import tempfile
from pathlib import Path

_TEMP_DIR = tempfile.mkdtemp(prefix="slangpile_test_")


def _make_module(name: str, code: str) -> types.ModuleType:
    """Create a temporary Python module from source on disk so inspect.getsource works."""
    path = Path(_TEMP_DIR) / f"{name}.py"
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestShaderDecorator:
    def test_marks_function(self):
        from skinny import slangpile as sp

        @sp.shader
        def my_fn(a: sp.float32) -> sp.float32:
            return a

        assert is_shader(my_fn)

    def test_preserves_name(self):
        from skinny import slangpile as sp

        @sp.shader
        def calculate_stuff(a: sp.float32) -> sp.float32:
            return a

        shader = get_shader(calculate_stuff)
        assert shader.name == "calculate_stuff"

    def test_preserves_module_name(self):
        from skinny import slangpile as sp

        @sp.shader
        def my_fn(a: sp.float32) -> sp.float32:
            return a

        shader = get_shader(my_fn)
        assert shader.module_name is not None


class TestExternDeclaration:
    def test_creates_extern(self):
        from skinny import slangpile as sp

        ext = sp.extern(
            name="sin",
            module="slang:core",
            args=[sp.float32],
            returns=sp.float32,
        )
        assert is_extern(ext)

    def test_extern_name(self):
        from skinny import slangpile as sp

        ext = sp.extern(name="cos", args=[sp.float32], returns=sp.float32)
        assert ext.slang_name == "cos"

    def test_extern_cannot_call(self):
        from skinny import slangpile as sp

        ext = sp.extern(name="sin", args=[sp.float32], returns=sp.float32)
        with pytest.raises(RuntimeError, match="declarations"):
            ext(1.0)


class TestCompiler:
    def test_simple_add(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def add_floats(a: sp.float32, b: sp.float32) -> sp.float32:
    return a + b
"""
        mod = _make_module("test_simple_add", code)
        compiled = sp.compile_module(mod)
        assert "add_floats" in compiled.source
        assert "float" in compiled.source
        assert "return (a + b);" in compiled.source

    def test_arithmetic_ops(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def math_ops(a: sp.float32, b: sp.float32) -> sp.float32:
    return a * b - a / b + a
"""
        mod = _make_module("test_arith", code)
        compiled = sp.compile_module(mod)
        assert "*" in compiled.source
        assert "-" in compiled.source
        assert "/" in compiled.source
        assert "+" in compiled.source

    def test_if_else(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def clamp_positive(x: sp.float32) -> sp.float32:
    if x > 0.0:
        return x
    else:
        return 0.0
"""
        mod = _make_module("test_if", code)
        compiled = sp.compile_module(mod)
        assert "if" in compiled.source
        assert "else" in compiled.source

    def test_annotated_local(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def with_local(a: sp.float32) -> sp.float32:
    x: sp.float32 = a * 2.0
    return x
"""
        mod = _make_module("test_local", code)
        compiled = sp.compile_module(mod)
        assert "float x" in compiled.source

    def test_vector_type_annotation(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def vec_func(v: sp.float32x3) -> sp.float32x3:
    return v
"""
        mod = _make_module("test_vec", code)
        compiled = sp.compile_module(mod)
        assert "float3" in compiled.source

    def test_comparison_operators(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def compare(a: sp.float32, b: sp.float32) -> sp.float32:
    if a < b:
        return a
    return b
"""
        mod = _make_module("test_cmp", code)
        compiled = sp.compile_module(mod)
        assert "<" in compiled.source

    def test_extern_call_generates_import(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

ext_sin = sp.extern(name="sin", module="slang:core", args=[sp.float32], returns=sp.float32)

@sp.shader
def use_sin(x: sp.float32) -> sp.float32:
    return ext_sin(x)
"""
        mod = _make_module("test_ext", code)
        compiled = sp.compile_module(mod)
        assert "sin(x)" in compiled.source
        assert "slang:core" in compiled.imports

    def test_builtin_math_calls(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def use_builtins(a: sp.float32, b: sp.float32) -> sp.float32:
    return max(a, min(b, abs(a)))
"""
        mod = _make_module("test_builtins", code)
        compiled = sp.compile_module(mod)
        assert "max(" in compiled.source
        assert "min(" in compiled.source
        assert "abs(" in compiled.source

    def test_missing_annotation_raises(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def bad(a):
    return a
"""
        mod = _make_module("test_bad_annot", code)
        with pytest.raises(SlangPileError):
            sp.compile_module(mod)

    def test_docstring_skipped(self):
        from skinny import slangpile as sp

        code = '''
from skinny import slangpile as sp

@sp.shader
def documented(a: sp.float32) -> sp.float32:
    """This is a docstring."""
    return a
'''
        mod = _make_module("test_doc", code)
        compiled = sp.compile_module(mod)
        assert "docstring" not in compiled.source
        assert "return a;" in compiled.source


class TestCompiledModuleIO:
    @pytest.fixture(autouse=True)
    def _io_dir(self):
        out = Path(_TEMP_DIR) / "io_output"
        out.mkdir(exist_ok=True)
        self._out = out

    def test_write_creates_file(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def disk_test(a: sp.float32) -> sp.float32:
    return a
"""
        mod = _make_module("test_disk", code)
        compiled = sp.compile_module(mod)
        slang_path = compiled.write(self._out)
        assert slang_path.exists()
        content = slang_path.read_text()
        assert "disk_test" in content

    def test_write_creates_source_map(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def mapped(a: sp.float32) -> sp.float32:
    return a
"""
        mod = _make_module("test_map", code)
        compiled = sp.compile_module(mod)
        compiled.write(self._out)
        map_files = list(self._out.rglob("*.map.json"))
        assert len(map_files) >= 1

    def test_write_creates_manifest(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def manifest_test(a: sp.float32) -> sp.float32:
    return a
"""
        mod = _make_module("test_manifest", code)
        compiled = sp.compile_module(mod)
        compiled.write(self._out)
        manifest = self._out / "slangpile_manifest.json"
        assert manifest.exists()


class TestSlangTypes:
    def test_scalar_types(self):
        from skinny.slangpile import types as t

        assert t.float32.slang_name == "float"
        assert t.float64.slang_name == "double"
        assert t.int32.slang_name == "int"
        assert t.uint32.slang_name == "uint"
        assert t.float16.slang_name == "half"

    def test_vector_types(self):
        from skinny.slangpile import types as t

        assert t.float32x3.slang_name == "float3"
        assert t.float32x4.slang_name == "float4"
        assert t.int32x2.slang_name == "int2"

    def test_matrix_types(self):
        from skinny.slangpile import types as t

        assert t.float32x3x3.slang_name == "float3x3"
        assert t.float32x4x4.slang_name == "float4x4"

    def test_is_slang_type(self):
        from skinny.slangpile.types import is_slang_type, float32, float32x3

        assert is_slang_type(float32)
        assert is_slang_type(float32x3)
        assert not is_slang_type(42)
        assert not is_slang_type("float")
