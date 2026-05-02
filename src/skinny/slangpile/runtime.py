from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import get_shader


@dataclass(frozen=True)
class SlangPyModule:
    """Runtime wrapper around a SlangPy-loaded module."""

    compiled_module: Any
    slang_path: Path
    device: object
    module: object

    def __getattr__(self, name: str) -> object:
        return getattr(self.module, name)

    def call(self, name: str, *args: object, **kwargs: object) -> object:
        return getattr(self.module, name)(*args, **kwargs)


@dataclass
class RuntimeConfig:
    out_dir: Path = Path(".slangpile/generated")
    device: object | None = None
    include_paths: list[Path] | None = None
    slangpy_module: object | None = None


_runtime_config = RuntimeConfig()
_runtime_cache: dict[tuple[str, Path, int | None, int | None], SlangPyModule] = {}


def configure_runtime(
    *,
    out_dir: str | Path | None = None,
    device: object | None = None,
    include_paths: list[str | Path] | None = None,
    slangpy_module: object | None = None,
    clear_cache: bool = True,
) -> None:
    """Configure implicit calls made by decorated shader functions.

    Normal users do not have to call this; by default SlangPile writes generated
    Slang to `.slangpile/generated` and imports `slangpy` on first shader call.
    Tests and applications with an existing SlangPy device can override that.
    """

    global _runtime_config
    _runtime_config = RuntimeConfig(
        out_dir=Path(out_dir) if out_dir is not None else _runtime_config.out_dir,
        device=device,
        include_paths=[Path(p) for p in include_paths] if include_paths is not None else None,
        slangpy_module=slangpy_module,
    )
    if clear_cache:
        _runtime_cache.clear()


def call_shader(shader_callable: object, *args: object, **kwargs: object) -> object:
    shader = get_shader(shader_callable)
    if shader is None:
        raise RuntimeError("object is not a SlangPile shader")
    runtime = _load_runtime_module(shader.module_name)
    return runtime.call(shader.name, *args, **kwargs)


def _load_runtime_module(module_name: str) -> SlangPyModule:
    config = _runtime_config
    out_dir = config.out_dir.resolve()
    key = (module_name, out_dir, id(config.device) if config.device is not None else None, id(config.slangpy_module) if config.slangpy_module is not None else None)
    cached = _runtime_cache.get(key)
    if cached is not None:
        return cached

    from .api import compile_module

    compiled = compile_module(importlib.import_module(module_name))
    runtime = load_compiled_module(
        compiled,
        out_dir,
        device=config.device,
        include_paths=config.include_paths,
        slangpy_module=config.slangpy_module,
    )
    _runtime_cache[key] = runtime
    return runtime


def load_compiled_module(
    compiled_module: Any,
    out_dir: str | Path,
    *,
    device: object | None = None,
    include_paths: list[str | Path] | None = None,
    slangpy_module: object | None = None,
) -> SlangPyModule:
    """Write a compiled module and load it with SlangPy.

    `slangpy_module` is injectable so tests can validate integration without a
    GPU runtime. In normal use this imports the installed `slangpy` package.
    """

    slang_path = compiled_module.write(out_dir)
    spy = slangpy_module if slangpy_module is not None else _import_slangpy()
    include_path_values = _include_paths(out_dir, include_paths)
    runtime_device = device if device is not None else spy.create_device(include_paths=include_path_values)
    loaded = spy.Module.load_from_file(runtime_device, str(slang_path))
    return SlangPyModule(compiled_module, slang_path, runtime_device, loaded)


def _include_paths(out_dir: str | Path, include_paths: list[str | Path] | None) -> list[Path]:
    paths = [Path(out_dir).resolve()]
    for path in include_paths or []:
        resolved = Path(path).resolve()
        if resolved not in paths:
            paths.append(resolved)
    return paths


def _import_slangpy() -> object:
    try:
        return importlib.import_module("slangpy")
    except ImportError as exc:
        raise RuntimeError(
            "SlangPy runtime support requires the optional dependency: "
            "install with `pip install slangpile[slangpy]` or `pip install slangpy`."
        ) from exc
