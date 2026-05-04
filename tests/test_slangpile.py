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


class TestForLoop:
    def test_range_one_arg(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def sum_range(n: sp.int32) -> sp.int32:
    total: sp.int32 = 0
    for i in range(n):
        total += i
    return total
"""
        mod = _make_module("test_for1", code)
        compiled = sp.compile_module(mod)
        assert "for (int i = 0; i < n; i++)" in compiled.source
        assert "total += i;" in compiled.source

    def test_range_two_args(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def sum_range2(a: sp.int32, b: sp.int32) -> sp.int32:
    total: sp.int32 = 0
    for i in range(a, b):
        total += i
    return total
"""
        mod = _make_module("test_for2", code)
        compiled = sp.compile_module(mod)
        assert "for (int i = a; i < b; i++)" in compiled.source

    def test_range_three_args(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def stepped(n: sp.int32) -> sp.int32:
    total: sp.int32 = 0
    for i in range(0, n, 2):
        total += i
    return total
"""
        mod = _make_module("test_for3", code)
        compiled = sp.compile_module(mod)
        assert "for (int i = 0; i < n; i += 2)" in compiled.source

    def test_nested_for(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def nested(n: sp.int32) -> sp.int32:
    total: sp.int32 = 0
    for i in range(n):
        for j in range(n):
            total += 1
    return total
"""
        mod = _make_module("test_nested_for", code)
        compiled = sp.compile_module(mod)
        assert "for (int i = 0; i < n; i++)" in compiled.source
        assert "for (int j = 0; j < n; j++)" in compiled.source


class TestWhileLoop:
    def test_while(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def countdown(n: sp.int32) -> sp.int32:
    x: sp.int32 = n
    while x > 0:
        x -= 1
    return x
"""
        mod = _make_module("test_while", code)
        compiled = sp.compile_module(mod)
        assert "while ((x > 0))" in compiled.source
        assert "x -= 1;" in compiled.source


class TestBreakContinue:
    def test_break(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def find_first(n: sp.int32) -> sp.int32:
    result: sp.int32 = 0
    for i in range(n):
        if i > 5:
            break
        result = i
    return result
"""
        mod = _make_module("test_break", code)
        compiled = sp.compile_module(mod)
        assert "break;" in compiled.source

    def test_continue(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def skip_evens(n: sp.int32) -> sp.int32:
    total: sp.int32 = 0
    for i in range(n):
        if i % 2 == 0:
            continue
        total += i
    return total
"""
        mod = _make_module("test_continue", code)
        compiled = sp.compile_module(mod)
        assert "continue;" in compiled.source


class TestAugmentedAssignment:
    def test_all_ops(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def aug_ops(a: sp.float32) -> sp.float32:
    x: sp.float32 = a
    x += 1.0
    x -= 2.0
    x *= 3.0
    x /= 4.0
    return x
"""
        mod = _make_module("test_aug", code)
        compiled = sp.compile_module(mod)
        assert "x += 1.0;" in compiled.source
        assert "x -= 2.0;" in compiled.source
        assert "x *= 3.0;" in compiled.source
        assert "x /= 4.0;" in compiled.source


class TestBitwiseOperators:
    def test_bitwise_and_or_xor(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def bitops(a: sp.uint32, b: sp.uint32) -> sp.uint32:
    x = a & b
    y = a | b
    z = a ^ b
    return x
"""
        mod = _make_module("test_bitwise", code)
        compiled = sp.compile_module(mod)
        assert "& b)" in compiled.source
        assert "| b)" in compiled.source
        assert "^ b)" in compiled.source

    def test_shift(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def shifts(a: sp.uint32) -> sp.uint32:
    x = a << 2
    y = a >> 1
    return x
"""
        mod = _make_module("test_shift", code)
        compiled = sp.compile_module(mod)
        assert "<< 2)" in compiled.source
        assert ">> 1)" in compiled.source

    def test_bitwise_not(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def bitnot(a: sp.uint32) -> sp.uint32:
    return ~a
"""
        mod = _make_module("test_bitnot", code)
        compiled = sp.compile_module(mod)
        assert "(~a)" in compiled.source


class TestArraySubscript:
    def test_subscript_read(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def read_elem(v: sp.float32x3) -> sp.float32:
    return v[0]
"""
        mod = _make_module("test_subscript", code)
        compiled = sp.compile_module(mod)
        assert "v[0]" in compiled.source


class TestVectorConstructor:
    def test_float3_constructor(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def make_vec(x: sp.float32, y: sp.float32, z: sp.float32) -> sp.float32x3:
    return sp.float32x3(x, y, z)
"""
        mod = _make_module("test_vec_ctor", code)
        compiled = sp.compile_module(mod)
        assert "float3(x, y, z)" in compiled.source

    def test_float3_broadcast(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def zero_vec() -> sp.float32x3:
    return sp.float32x3(0.0)
"""
        mod = _make_module("test_vec_broadcast", code)
        compiled = sp.compile_module(mod)
        assert "float3(0.0)" in compiled.source


class TestBoolOps:
    def test_and_or(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def logic(a: sp.float32, b: sp.float32) -> sp.float32:
    if a > 0.0 and b > 0.0:
        return a
    if a < 0.0 or b < 0.0:
        return b
    return 0.0
"""
        mod = _make_module("test_boolop", code)
        compiled = sp.compile_module(mod)
        assert "&&" in compiled.source
        assert "||" in compiled.source


class TestReassignment:
    def test_reassign_no_redeclare(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def reassign(a: sp.float32) -> sp.float32:
    x: sp.float32 = a
    x = a * 2.0
    return x
"""
        mod = _make_module("test_reassign", code)
        compiled = sp.compile_module(mod)
        assert "float x = a;" in compiled.source
        assert "x = (a * 2.0);" in compiled.source
        assert compiled.source.count("float x") == 1


class TestElseIf:
    def test_elif_chain(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def classify(x: sp.float32) -> sp.int32:
    if x > 1.0:
        return 2
    elif x > 0.0:
        return 1
    else:
        return 0
"""
        mod = _make_module("test_elif", code)
        compiled = sp.compile_module(mod)
        assert "else if" in compiled.source


class TestMethodCall:
    def test_method_on_object(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def call_method(v: sp.float32x3) -> sp.float32:
    return v.length()
"""
        mod = _make_module("test_method", code)
        compiled = sp.compile_module(mod)
        assert "v.length()" in compiled.source


class TestExpandedBuiltins:
    def test_reflect(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def do_reflect(v: sp.float32x3, n: sp.float32x3) -> sp.float32x3:
    return reflect(v, n)
"""
        mod = _make_module("test_reflect", code)
        compiled = sp.compile_module(mod)
        assert "reflect(v, n)" in compiled.source

    def test_clamp_pow_cross(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def math_ext(a: sp.float32, b: sp.float32x3, c: sp.float32x3) -> sp.float32x3:
    x = clamp(a, 0.0, 1.0)
    y = cross(b, c)
    z = pow(a, 2.0)
    return y
"""
        mod = _make_module("test_ext_builtins", code)
        compiled = sp.compile_module(mod)
        assert "clamp(a, 0.0, 1.0)" in compiled.source
        assert "cross(b, c)" in compiled.source
        assert "pow(a, 2.0)" in compiled.source


class TestUninitializedLocal:
    def test_declared_no_value(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def uninit(a: sp.float32) -> sp.float32:
    x: sp.float32
    x = a * 2.0
    return x
"""
        mod = _make_module("test_uninit", code)
        compiled = sp.compile_module(mod)
        assert "float x;" in compiled.source
        assert "x = (a * 2.0);" in compiled.source


class TestExpressionStatement:
    def test_bare_call(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

ext_doSomething = sp.extern(name="doSomething", args=[sp.float32], returns=sp.float32)

@sp.shader
def caller(a: sp.float32) -> sp.float32:
    ext_doSomething(a)
    return a
"""
        mod = _make_module("test_bare_call", code)
        compiled = sp.compile_module(mod)
        assert "doSomething(a);" in compiled.source


class TestStruct:
    def test_simple_struct(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.struct
class MyVec:
    x: sp.float32
    y: sp.float32
    z: sp.float32
"""
        mod = _make_module("test_struct_simple", code)
        compiled = sp.compile_module(mod)
        assert "struct MyVec" in compiled.source
        assert "float x;" in compiled.source
        assert "float y;" in compiled.source
        assert "float z;" in compiled.source

    def test_struct_with_conformance(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.struct(conforms_to="ISampler")
class UniformSampler:
    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        return sp.float32x3(0.0, 1.0, 0.0)

    def pdf(self, L: sp.float32x3) -> sp.float32:
        return 1.0
"""
        mod = _make_module("test_struct_conform", code)
        compiled = sp.compile_module(mod)
        assert "struct UniformSampler : ISampler" in compiled.source
        assert "float3 sampleDirection(float2 u)" in compiled.source
        assert "float pdf(float3 L)" in compiled.source

    def test_struct_with_fields_and_methods(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.struct(conforms_to="ISampler")
class LambertSampler:
    N: sp.float32x3

    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        return self.N

    def pdf(self, L: sp.float32x3) -> sp.float32:
        return max(dot(L, self.N), 0.0)
"""
        mod = _make_module("test_struct_fields_methods", code)
        compiled = sp.compile_module(mod)
        assert "struct LambertSampler : ISampler" in compiled.source
        assert "float3 N;" in compiled.source
        assert "float3 sampleDirection(float2 u)" in compiled.source

    def test_mutating_method(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.struct
class Counter:
    value: sp.int32

    @sp.mutating
    def increment(self) -> None:
        self.value += 1
"""
        mod = _make_module("test_mutating", code)
        compiled = sp.compile_module(mod)
        assert "[mutating] void increment()" in compiled.source
        assert "value += 1;" in compiled.source


class TestSlangImport:
    def test_slang_import(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

@sp.shader
def identity(x: sp.float32) -> sp.float32:
    return x
"""
        mod = _make_module("test_slang_import", code)
        compiled = sp.compile_module(mod)
        assert "import common;" in compiled.source
        assert "import interfaces;" in compiled.source
        assert "common" in compiled.imports
        assert "interfaces" in compiled.imports


class TestVerbatim:
    def test_verbatim_block(self):
        from skinny import slangpile as sp

        code = '''
from skinny import slangpile as sp

_binding = sp.verbatim("static const float PI = 3.14159265;")

@sp.shader
def use_pi(x: sp.float32) -> sp.float32:
    return x
'''
        mod = _make_module("test_verbatim", code)
        compiled = sp.compile_module(mod)
        assert "static const float PI = 3.14159265;" in compiled.source


class TestInoutOut:
    def test_inout_parameter(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def modify(x: sp.inout(sp.float32)) -> None:
    x += 1.0
"""
        mod = _make_module("test_inout", code)
        compiled = sp.compile_module(mod)
        assert "inout float x" in compiled.source
        assert "void modify" in compiled.source

    def test_out_parameter(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

@sp.shader
def get_basis(N: sp.float32x3, T: sp.out(sp.float32x3), B: sp.out(sp.float32x3)) -> None:
    T = sp.float32x3(1.0, 0.0, 0.0)
    B = sp.float32x3(0.0, 1.0, 0.0)
"""
        mod = _make_module("test_out", code)
        compiled = sp.compile_module(mod)
        assert "out float3 T" in compiled.source
        assert "out float3 B" in compiled.source


class TestUniformSphereSamplerTranspile:
    """Integration test: transpile uniform_sphere sampler from Python, verify output matches hand-written Slang."""

    def test_transpile_uniform_sphere(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

sampleUniformSphere = sp.extern(
    name="sampleUniformSphere", module="common",
    args=[sp.float32x2], returns=sp.float32x3,
)
uniformSpherePdf = sp.extern(
    name="uniformSpherePdf", module="common",
    args=[], returns=sp.float32,
)

@sp.struct(conforms_to="ISampler")
class UniformSphereSampler:
    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        return sampleUniformSphere(u)

    def pdf(self, L: sp.float32x3) -> sp.float32:
        return uniformSpherePdf()
"""
        mod = _make_module("test_uniform_sphere_transpile", code)
        compiled = sp.compile_module(mod)
        src = compiled.source

        assert "import common;" in src
        assert "import interfaces;" in src
        assert "struct UniformSphereSampler : ISampler" in src
        assert "float3 sampleDirection(float2 u)" in src
        assert "sampleUniformSphere(u)" in src
        assert "float pdf(float3 L)" in src
        assert "uniformSpherePdf()" in src


class TestLambertSamplerTranspile:
    """Integration test: transpile lambert sampler from Python."""

    def test_transpile_lambert(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

sampleCosineHemisphere = sp.extern(
    name="sampleCosineHemisphere", module="common",
    args=[sp.float32x2, sp.float32x3], returns=sp.float32x3,
)
cosineHemispherePdf = sp.extern(
    name="cosineHemispherePdf", module="common",
    args=[sp.float32x3, sp.float32x3], returns=sp.float32,
)

@sp.struct(conforms_to="ISampler")
class LambertSampler:
    N: sp.float32x3

    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        return sampleCosineHemisphere(u, self.N)

    def pdf(self, L: sp.float32x3) -> sp.float32:
        return cosineHemispherePdf(L, self.N)
"""
        mod = _make_module("test_lambert_transpile", code)
        compiled = sp.compile_module(mod)
        src = compiled.source

        assert "struct LambertSampler : ISampler" in src
        assert "float3 N;" in src
        assert "sampleCosineHemisphere(u, N)" in src
        assert "cosineHemispherePdf(L, N)" in src


class TestGGXSamplerTranspile:
    """Integration: transpile GGX sampler — exercises locals, out params, reflect, PI."""

    def test_transpile_ggx(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

buildBasis = sp.extern(
    name="buildBasis", module="common",
    args=[sp.float32x3, sp.out(sp.float32x3), sp.out(sp.float32x3)],
    returns=sp.float32,
)

PI = sp.extern(name="PI", module="common", args=[], returns=sp.float32)

@sp.struct(conforms_to="ISampler")
class GGXSampler:
    N: sp.float32x3
    V: sp.float32x3
    roughness: sp.float32

    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        a = max(self.roughness * self.roughness, 1e-4)
        a2 = a * a
        cosTheta = sqrt((1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y))
        sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta))
        phi = 2.0 * PI * u.x
        T: sp.float32x3
        B: sp.float32x3
        buildBasis(self.N, T, B)
        H = normalize(
            T * cos(phi) * sinTheta +
            B * sin(phi) * sinTheta +
            self.N * cosTheta
        )
        return reflect(-self.V, H)

    def pdf(self, L: sp.float32x3) -> sp.float32:
        H = normalize(self.V + L)
        NdotH = max(dot(self.N, H), 0.0)
        VdotH = max(dot(self.V, H), 1e-6)
        if dot(self.N, L) <= 0.0 or NdotH <= 0.0:
            return 0.0
        a = max(self.roughness * self.roughness, 1e-4)
        a2 = a * a
        d = NdotH * NdotH * (a2 - 1.0) + 1.0
        D = a2 / (PI * d * d)
        return D * NdotH / (4.0 * VdotH)
"""
        mod = _make_module("test_ggx_transpile", code)
        compiled = sp.compile_module(mod)
        src = compiled.source

        assert "struct GGXSampler : ISampler" in src
        assert "float3 N;" in src
        assert "float3 V;" in src
        assert "float roughness;" in src
        assert "float3 sampleDirection(float2 u)" in src
        assert "float pdf(float3 L)" in src
        assert "reflect(" in src
        assert "buildBasis(N, T, B)" in src
        assert "normalize(" in src
        assert "cos(phi)" in src
        assert "sin(phi)" in src


class TestGenerics:
    def test_generic_function(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

TA = sp.generic("TA", "ISampler")
TB = sp.generic("TB", "ISampler")

powerHeuristic = sp.extern(
    name="powerHeuristic", module="common",
    args=[sp.float32, sp.float32], returns=sp.float32,
)

@sp.shader(generics={"TA": "ISampler", "TB": "ISampler"})
def misPrimaryWeight(primary: TA, companion: TB, L: sp.float32x3) -> sp.float32:
    return powerHeuristic(primary.pdf(L), companion.pdf(L))
"""
        mod = _make_module("test_generics", code)
        compiled = sp.compile_module(mod)
        src = compiled.source
        assert "<TA : ISampler, TB : ISampler>" in src
        assert "float misPrimaryWeight" in src
        assert "TA primary" in src
        assert "TB companion" in src
        assert "primary.pdf(L)" in src
        assert "companion.pdf(L)" in src


class TestHenyeyGreensteinTranspile:
    def test_transpile_hg(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

buildBasis = sp.extern(name="buildBasis", module="common", args=[sp.float32x3, sp.out(sp.float32x3), sp.out(sp.float32x3)], returns=sp.float32)
PI = sp.extern(name="PI", module="common", args=[], returns=sp.float32)

@sp.struct(conforms_to="ISampler")
class HenyeyGreensteinSampler:
    forward: sp.float32x3
    g: sp.float32

    def sampleDirection(self, u: sp.float32x2) -> sp.float32x3:
        cosTheta: sp.float32
        if abs(self.g) > 1e-3:
            t = (1.0 - self.g * self.g) / (1.0 - self.g + 2.0 * self.g * u.x)
            cosTheta = (1.0 + self.g * self.g - t * t) / (2.0 * self.g)
        else:
            cosTheta = 1.0 - 2.0 * u.x
        cosTheta = clamp(cosTheta, -1.0, 1.0)
        sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta))
        phi = 2.0 * PI * u.y
        T: sp.float32x3
        B: sp.float32x3
        buildBasis(self.forward, T, B)
        return normalize(
            T * cos(phi) * sinTheta +
            B * sin(phi) * sinTheta +
            self.forward * cosTheta
        )

    def pdf(self, L: sp.float32x3) -> sp.float32:
        cosTheta = dot(normalize(L), self.forward)
        g2 = self.g * self.g
        num = 1.0 - g2
        den = pow(1.0 + g2 - 2.0 * self.g * cosTheta, 1.5)
        return num / (4.0 * PI * den)
"""
        mod = _make_module("test_hg_transpile", code)
        compiled = sp.compile_module(mod)
        src = compiled.source
        assert "struct HenyeyGreensteinSampler : ISampler" in src
        assert "float3 forward;" in src
        assert "float g;" in src
        assert "clamp(cosTheta, -1.0, 1.0)" in src
        assert "buildBasis(forward, T, B)" in src
        assert "pow(" in src


class TestMISCombineTranspile:
    def test_transpile_mis_combine(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

TA = sp.generic("TA", "ISampler")
TB = sp.generic("TB", "ISampler")

powerHeuristic = sp.extern(
    name="powerHeuristic", module="common",
    args=[sp.float32, sp.float32], returns=sp.float32,
)

@sp.shader(generics={"TA": "ISampler", "TB": "ISampler"})
def misPrimaryWeight(primary: TA, companion: TB, L: sp.float32x3) -> sp.float32:
    return powerHeuristic(primary.pdf(L), companion.pdf(L))

@sp.shader(generics={"TA": "ISampler", "TB": "ISampler"})
def misCompanionWeight(primary: TA, companion: TB, L: sp.float32x3) -> sp.float32:
    return powerHeuristic(companion.pdf(L), primary.pdf(L))
"""
        mod = _make_module("test_mis_transpile", code)
        compiled = sp.compile_module(mod)
        src = compiled.source
        assert "float misCompanionWeight<TA : ISampler, TB : ISampler>" in src
        assert "float misPrimaryWeight<TA : ISampler, TB : ISampler>" in src
        assert "powerHeuristic(primary.pdf(L), companion.pdf(L))" in src
        assert "powerHeuristic(companion.pdf(L), primary.pdf(L))" in src


class TestConst:
    def test_static_const_emitted(self):
        from skinny import slangpile as sp

        code = """
from skinny import slangpile as sp

MY_CONST = sp.const("MY_CONST", sp.float32, 3.14)

@sp.shader
def use_const(x: sp.float32) -> sp.float32:
    return x * MY_CONST
"""
        mod = _make_module("test_const_emit", code)
        compiled = sp.compile_module(mod)
        src = compiled.source
        assert "static const float MY_CONST = 3.14;" in src
        assert "MY_CONST" in src


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
