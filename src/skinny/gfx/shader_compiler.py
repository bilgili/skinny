"""Backend-agnostic Slang compilation skeleton.

Step 1 deposits the surface API and target dispatch only — the cache and
slangc invocation logic currently in ``vk_compute.py::_compile_slang``
moves here in Step 3 (parameterised on `target`). The cache directory
becomes ``build/shader_cache/<target>/<hash>.<ext>`` so SPIR-V and MSL
blobs do not collide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ShaderTarget = Literal["spirv", "metal"]


@dataclass(frozen=True)
class CompileRequest:
    source: Path
    entry_point: str
    target: ShaderTarget
    stage: Literal["compute", "vertex", "fragment"] = "compute"
    defines: tuple[tuple[str, str], ...] = ()
    includes: tuple[Path, ...] = ()
    extra_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CompileResult:
    blob_path: Path
    cache_hit: bool


def compile_slang(req: CompileRequest, cache_root: Path) -> CompileResult:
    """Invoke slangc for the requested target. Implemented in Step 3.

    Step 3 will move the body of ``vk_compute.py::_compile_slang`` (lines
    286–352) here, generalising the ``-target spirv`` flag to switch between
    SPIR-V (Vulkan) and Metal IR.
    """
    raise NotImplementedError("compile_slang lands in Step 3")


def target_extension(target: ShaderTarget) -> str:
    return {"spirv": "spv", "metal": "metallib"}[target]
