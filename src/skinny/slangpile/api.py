from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Callable

from .compiler import ModuleCompiler
from .registry import (
    ExternFunction,
    GenericTypeParam,
    SlangConst,
    SlangImport,
    StructDefinition,
    StructField,
    StructMethod,
    StructType,
    Verbatim,
    register_extern,
    register_shader,
)
from .runtime import configure_runtime, load_compiled_module
from .types import SlangType, is_slang_type


def shader(fn: Callable[..., object] | None = None, *, generics: dict[str, str] | None = None):
    def wrap(f: Callable[..., object]) -> Callable[..., object]:
        return register_shader(f, generics=generics)

    if fn is not None:
        return register_shader(fn)
    return wrap


def generic(name: str, constraint: str) -> GenericTypeParam:
    return GenericTypeParam(name)


def extern_type(name: str) -> SlangType:
    return SlangType(name, name)


def extern(
    name: str,
    *,
    module: str | None = None,
    args: list[SlangType] | tuple[SlangType, ...],
    returns: SlangType,
) -> ExternFunction:
    return register_extern(ExternFunction(name, module, tuple(args), returns))


def struct(cls: type | None = None, *, conforms_to: str | None = None):
    def wrap(cls_inner: type) -> StructType:
        module_name = getattr(cls_inner, "__module__", None) or "<unknown>"

        fields: list[StructField] = []
        hints = {}
        if hasattr(cls_inner, "__annotations__"):
            hints = cls_inner.__annotations__
        for fname, ftype in hints.items():
            if is_slang_type(ftype):
                fields.append(StructField(fname, ftype))

        methods: list[StructMethod] = []
        for mname, mval in vars(cls_inner).items():
            if mname.startswith("_"):
                continue
            if callable(mval):
                mutating = getattr(mval, "_slangpile_mutating", False)
                methods.append(StructMethod(fn=mval, name=mname, mutating=mutating))

        defn = StructDefinition(
            name=cls_inner.__name__,
            module_name=module_name,
            conforms_to=conforms_to,
            fields=fields,
            methods=methods,
        )
        return StructType(defn)

    if cls is not None:
        return wrap(cls)
    return wrap


def mutating(fn: Callable[..., object]) -> Callable[..., object]:
    fn._slangpile_mutating = True
    return fn


def slang_import(module_name: str) -> SlangImport:
    return SlangImport(module_name)


def verbatim(code: str) -> Verbatim:
    return Verbatim(code)


def const(name: str, type: SlangType, value: str | int | float) -> SlangConst:
    return SlangConst(name, type, value)


class _InoutWrapper(SlangType):
    def __init__(self, inner: SlangType):
        super().__init__(f"inout_{inner.name}", f"inout {inner.slang_name}")
        self.inner = inner


class _OutWrapper(SlangType):
    def __init__(self, inner: SlangType):
        super().__init__(f"out_{inner.name}", f"out {inner.slang_name}")
        self.inner = inner


def inout(t: SlangType) -> SlangType:
    return _InoutWrapper(t)


def out(t: SlangType) -> SlangType:
    return _OutWrapper(t)


def compile_module(module: ModuleType | str):
    if isinstance(module, str):
        module = importlib.import_module(module)
    return ModuleCompiler(module).compile()


def build_module(module: ModuleType | str, out_dir: str | Path, *, verify: bool = False, slangc: str | None = None):
    compiled = compile_module(module)
    compiled.write(out_dir)
    if verify:
        return compiled.verify(out_dir, slangc=slangc)
    return compiled


def load_module(
    module: ModuleType | str,
    out_dir: str | Path,
    *,
    device: object | None = None,
    include_paths: list[str | Path] | None = None,
    slangpy_module: object | None = None,
):
    compiled = compile_module(module)
    return load_compiled_module(
        compiled,
        out_dir,
        device=device,
        include_paths=include_paths,
        slangpy_module=slangpy_module,
    )
