from __future__ import annotations

from dataclasses import dataclass
from functools import update_wrapper
from typing import Callable

from .types import SlangType


@dataclass(frozen=True)
class ShaderFunction:
    fn: Callable[..., object]
    module_name: str
    name: str


class ShaderCallable:
    def __init__(self, fn: Callable[..., object], shader: ShaderFunction):
        self.__slangpile_shader__ = shader
        self.__wrapped__ = fn
        update_wrapper(self, fn)

    def __call__(self, *args: object, **kwargs: object) -> object:
        from .runtime import call_shader

        return call_shader(self, *args, **kwargs)


@dataclass(frozen=True)
class ExternFunction:
    slang_name: str
    module: str | None
    args: tuple[SlangType, ...]
    returns: SlangType
    python_name: str | None = None

    def __call__(self, *args: object) -> object:
        raise RuntimeError("slangpile extern functions are declarations and cannot run in Python")


_shaders: dict[object, ShaderFunction] = {}
_externs: dict[ExternFunction, ExternFunction] = {}


def register_shader(fn: Callable[..., object]) -> ShaderCallable:
    shader = ShaderFunction(fn=fn, module_name=fn.__module__, name=fn.__name__)
    wrapped = ShaderCallable(fn, shader)
    _shaders[wrapped] = shader
    return wrapped


def register_extern(extern: ExternFunction) -> ExternFunction:
    _externs[extern] = extern
    return extern


def is_shader(value: object) -> bool:
    return callable(value) and hasattr(value, "__slangpile_shader__")


def get_shader(value: object) -> ShaderFunction | None:
    return getattr(value, "__slangpile_shader__", None)


def is_extern(value: object) -> bool:
    return isinstance(value, ExternFunction)
