"""Pipeline + descriptor abstractions.

A `DescriptorLayout` declares which slots a pipeline expects; a
`DescriptorSet` is a concrete binding of resources to those slots. The
renderer builds layouts once at pipeline creation and rewrites sets per
frame (e.g. binding 1 swaps between the swapchain image and the offscreen
output).

Metal will translate `DescriptorLayout` to an argument-buffer layout and
`DescriptorSet` to a populated argument-buffer instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skinny.gfx.types import BindingKind, Format, ShaderStage

if TYPE_CHECKING:
    from skinny.gfx.resources import Buffer, Image, Sampler, ShaderModule


@dataclass(frozen=True)
class BindingDecl:
    slot: int
    kind: BindingKind
    count: int = 1
    stages: ShaderStage = ShaderStage.COMPUTE


@dataclass(frozen=True)
class PushConstantRange:
    offset: int
    size: int
    stages: ShaderStage = ShaderStage.COMPUTE


class DescriptorLayout(ABC):
    """Immutable description of the binding slots a pipeline reads from."""

    bindings: tuple[BindingDecl, ...]
    push_constants: tuple[PushConstantRange, ...]

    @abstractmethod
    def destroy(self) -> None: ...


class DescriptorSet(ABC):
    """Concrete binding of resources to the slots of a DescriptorLayout."""

    layout: DescriptorLayout

    @abstractmethod
    def write_buffer(self, slot: int, buffer: "Buffer",
                     array_index: int = 0) -> None: ...

    @abstractmethod
    def write_image(self, slot: int, image: "Image",
                    sampler: "Sampler | None" = None,
                    array_index: int = 0) -> None: ...

    @abstractmethod
    def commit(self) -> None:
        """Apply pending writes. Vulkan: vkUpdateDescriptorSets. Metal: encode
        into the argument buffer. Allowed to be a no-op if writes were already
        eager."""


class ComputePipeline(ABC):
    layout: DescriptorLayout
    module: "ShaderModule"

    @abstractmethod
    def destroy(self) -> None: ...


@dataclass(frozen=True)
class GraphicsPipelineDesc:
    """Minimal graphics pipeline description; expanded as debug_viewport
    is ported (Step 5)."""

    vertex_module: "ShaderModule"
    fragment_module: "ShaderModule"
    color_format: Format
    depth_format: Format = Format.UNDEFINED


class GraphicsPipeline(ABC):
    layout: DescriptorLayout

    @abstractmethod
    def destroy(self) -> None: ...
