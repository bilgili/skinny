from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Callable

from .compiler import ModuleCompiler
from .registry import ExternFunction, register_extern, register_shader
from .runtime import configure_runtime, load_compiled_module
from .types import SlangType


def shader(fn: Callable[..., object]) -> Callable[..., object]:
    return register_shader(fn)


def extern(
    name: str,
    *,
    module: str | None = None,
    args: list[SlangType] | tuple[SlangType, ...],
    returns: SlangType,
) -> ExternFunction:
    return register_extern(ExternFunction(name, module, tuple(args), returns))


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
