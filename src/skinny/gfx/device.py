"""Device + Queue ABCs — resource and pipeline factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from skinny.gfx.types import (
    BufferUsage,
    Extent2D,
    Format,
    ImageUsage,
)

if TYPE_CHECKING:
    from skinny.gfx.command import CommandList, Fence, Semaphore
    from skinny.gfx.pipeline import (
        BindingDecl,
        ComputePipeline,
        DescriptorLayout,
        DescriptorSet,
        GraphicsPipeline,
        GraphicsPipelineDesc,
        PushConstantRange,
    )
    from skinny.gfx.resources import (
        Buffer,
        Image,
        Sampler,
        SamplerDesc,
        ShaderModule,
    )


class Queue(ABC):
    @abstractmethod
    def submit(
        self,
        command_lists: list["CommandList"],
        wait_semaphores: list["Semaphore"] | None = None,
        signal_semaphores: list["Semaphore"] | None = None,
        fence: "Fence | None" = None,
    ) -> None: ...

    @abstractmethod
    def wait_idle(self) -> None: ...


class Device(ABC):
    """Owns the logical device handle. All resources are created through it."""

    # ── Resources ───────────────────────────────────────────
    @abstractmethod
    def create_buffer(
        self,
        size: int,
        usage: BufferUsage,
        host_visible: bool = False,
    ) -> "Buffer": ...

    @abstractmethod
    def create_image(
        self,
        extent: Extent2D,
        format: Format,
        usage: ImageUsage,
    ) -> "Image": ...

    @abstractmethod
    def create_sampler(self, desc: "SamplerDesc") -> "Sampler": ...

    # ── Shader + pipeline ───────────────────────────────────
    @abstractmethod
    def create_shader_module(
        self,
        blob: bytes,
        entry_point: str,
    ) -> "ShaderModule": ...

    @abstractmethod
    def create_descriptor_layout(
        self,
        bindings: "list[BindingDecl]",
        push_constants: "list[PushConstantRange] | None" = None,
    ) -> "DescriptorLayout": ...

    @abstractmethod
    def allocate_descriptor_set(
        self,
        layout: "DescriptorLayout",
    ) -> "DescriptorSet": ...

    @abstractmethod
    def create_compute_pipeline(
        self,
        module: "ShaderModule",
        layout: "DescriptorLayout",
    ) -> "ComputePipeline": ...

    @abstractmethod
    def create_graphics_pipeline(
        self,
        desc: "GraphicsPipelineDesc",
        layout: "DescriptorLayout",
    ) -> "GraphicsPipeline": ...

    # ── Command + sync ──────────────────────────────────────
    @abstractmethod
    def create_command_list(self) -> "CommandList": ...

    @abstractmethod
    def create_fence(self, signaled: bool = False) -> "Fence": ...

    @abstractmethod
    def create_semaphore(self) -> "Semaphore": ...

    # ── Queues ──────────────────────────────────────────────
    @property
    @abstractmethod
    def compute_queue(self) -> Queue: ...

    @property
    @abstractmethod
    def graphics_queue(self) -> Queue:
        """May be the same handle as compute_queue when the backend exposes
        a single universal queue (Metal, single-family Vulkan)."""

    @abstractmethod
    def wait_idle(self) -> None: ...
