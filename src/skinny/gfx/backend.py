"""Top-level Backend ABC.

The renderer takes a `Backend` instance, asks it for a `Device` and an
optional `Presenter`, and never touches the underlying vendor API.

`Backend.shader_target()` tells the shader compiler which slangc target to
emit (``"spirv"`` for Vulkan, ``"metal"`` for native Metal — MoltenVK still
runs the Vulkan backend, so it returns ``"spirv"`` too).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skinny.gfx.device import Device
    from skinny.gfx.presenter import Presenter


@dataclass(frozen=True)
class BackendCaps:
    """Feature flags the renderer queries before opting into optional paths."""

    bindless_textures: bool
    scalar_block_layout: bool
    push_descriptors: bool
    max_storage_buffer_bindings: int
    max_compute_workgroup_size: tuple[int, int, int]


class Backend(ABC):
    name: str

    @classmethod
    @abstractmethod
    def create(
        cls,
        *,
        window=None,
        width: int = 1280,
        height: int = 720,
        enable_validation: bool = True,
        gpu_preference: str | None = None,
    ) -> "Backend":
        """Construct a ready-to-use backend. ``window=None`` selects headless
        mode (no presenter)."""

    @property
    @abstractmethod
    def caps(self) -> BackendCaps: ...

    @property
    @abstractmethod
    def device(self) -> "Device": ...

    @property
    @abstractmethod
    def presenter(self) -> "Presenter | None": ...

    @abstractmethod
    def shader_target(self) -> Literal["spirv", "metal"]: ...

    @abstractmethod
    def destroy(self) -> None: ...
