"""Command list + sync primitive ABCs.

Models a Vulkan-style explicit recording API. Metal collapses much of the
barrier work into automatic hazard tracking, so most barrier calls become
no-ops on the Metal backend, but the renderer keeps the explicit calls so
the same code drives both backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from skinny.gfx.types import Extent2D, ImageState, PipelineStage

if TYPE_CHECKING:
    from skinny.gfx.pipeline import (
        ComputePipeline,
        DescriptorSet,
        GraphicsPipeline,
    )
    from skinny.gfx.resources import Buffer, Image


class Fence(ABC):
    @abstractmethod
    def wait(self, timeout_ns: int = 2**63 - 1) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def is_signaled(self) -> bool: ...

    @abstractmethod
    def destroy(self) -> None: ...


class Semaphore(ABC):
    @abstractmethod
    def destroy(self) -> None: ...


class CommandList(ABC):
    @abstractmethod
    def begin(self) -> None: ...

    @abstractmethod
    def end(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    # ── Compute ─────────────────────────────────────────────
    @abstractmethod
    def bind_compute_pipeline(self, pipeline: "ComputePipeline") -> None: ...

    @abstractmethod
    def bind_descriptor_set(
        self,
        pipeline: "ComputePipeline | GraphicsPipeline",
        descriptor_set: "DescriptorSet",
        set_index: int = 0,
    ) -> None: ...

    @abstractmethod
    def push_constants(
        self,
        pipeline: "ComputePipeline | GraphicsPipeline",
        data: bytes | memoryview,
        offset: int = 0,
    ) -> None: ...

    @abstractmethod
    def dispatch(self, group_x: int, group_y: int, group_z: int = 1) -> None: ...

    # ── Barriers ────────────────────────────────────────────
    @abstractmethod
    def memory_barrier(
        self,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
    ) -> None:
        """Global memory barrier. Used between compute dispatches that read
        each other's output (e.g. accumulation image cross-frame fence)."""

    @abstractmethod
    def image_barrier(
        self,
        image: "Image",
        old_state: ImageState,
        new_state: ImageState,
        src_stage: PipelineStage = PipelineStage.COMPUTE_SHADER,
        dst_stage: PipelineStage = PipelineStage.COMPUTE_SHADER,
    ) -> None: ...

    # ── Transfer ────────────────────────────────────────────
    @abstractmethod
    def copy_buffer_to_image(
        self,
        buffer: "Buffer",
        image: "Image",
        offset: int = 0,
    ) -> None: ...

    @abstractmethod
    def copy_image_to_buffer(
        self,
        image: "Image",
        buffer: "Buffer",
        offset: int = 0,
    ) -> None: ...

    @abstractmethod
    def blit_image(
        self,
        src: "Image",
        dst: "Image",
        dst_extent: Extent2D | None = None,
    ) -> None:
        """Copy + scale src into dst (linear filter). Used for the offscreen-
        to-swapchain blit when render resolution differs from window size."""
