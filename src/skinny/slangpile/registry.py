from __future__ import annotations

from dataclasses import dataclass, field
from functools import update_wrapper
from typing import Callable

from .types import SlangType


@dataclass(frozen=True)
class ShaderFunction:
    fn: Callable[..., object]
    module_name: str
    name: str
    generics: dict[str, str] = field(default_factory=dict)


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


@dataclass
class StructField:
    name: str
    type: SlangType


@dataclass
class StructMethod:
    fn: Callable[..., object]
    name: str
    mutating: bool = False


@dataclass
class StructDefinition:
    name: str
    module_name: str
    conforms_to: str | None
    fields: list[StructField] = field(default_factory=list)
    methods: list[StructMethod] = field(default_factory=list)


class StructType(SlangType):
    def __init__(self, definition: StructDefinition):
        super().__init__(definition.name, definition.name)
        self.definition = definition


class SlangImport:
    def __init__(self, module_name: str):
        self.module_name = module_name


class Verbatim:
    def __init__(self, code: str):
        self.code = code


@dataclass(frozen=True)
class SlangConst:
    name: str
    type: SlangType
    value: str | int | float


_shaders: dict[object, ShaderFunction] = {}
_externs: dict[ExternFunction, ExternFunction] = {}


class GenericTypeParam(SlangType):
    def __init__(self, name: str):
        super().__init__(name, name)


def register_shader(fn: Callable[..., object], generics: dict[str, str] | None = None) -> ShaderCallable:
    shader = ShaderFunction(fn=fn, module_name=fn.__module__, name=fn.__name__, generics=generics or {})
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


def is_struct(value: object) -> bool:
    return isinstance(value, StructType)


def get_struct(value: object) -> StructDefinition | None:
    if isinstance(value, StructType):
        return value.definition
    return None
