"""Skinny rendering backend abstraction.

Public surface used by the renderer:

    from skinny.gfx import select_backend, BufferUsage, ImageUsage, Format

Concrete backends live in ``skinny.gfx.vulkan`` and ``skinny.gfx.metal``.
The vulkan backend is wired in Step 2; the metal backend is stubbed in
Step 7. ``select_backend`` chooses between them based on
``$SKINNY_BACKEND`` (``vulkan`` | ``metal``) or the explicit ``prefer``
argument; default is Vulkan everywhere (MoltenVK is the macOS fallback
under that same backend, requiring no code change).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from skinny.gfx.backend import Backend, BackendCaps
from skinny.gfx.command import CommandList, Fence, Semaphore
from skinny.gfx.device import Device, Queue
from skinny.gfx.pipeline import (
    BindingDecl,
    ComputePipeline,
    DescriptorLayout,
    DescriptorSet,
    GraphicsPipeline,
    GraphicsPipelineDesc,
    PushConstantRange,
)
from skinny.gfx.presenter import AcquiredImage, Presenter
from skinny.gfx.resources import (
    Buffer,
    Image,
    Sampler,
    SamplerDesc,
    ShaderModule,
)
from skinny.gfx.types import (
    AddressMode,
    BindingKind,
    BufferUsage,
    Extent2D,
    Extent3D,
    FilterMode,
    Format,
    ImageState,
    ImageUsage,
    PipelineStage,
    ShaderStage,
)

if TYPE_CHECKING:
    pass


def select_backend(prefer: str | None = None) -> type[Backend]:
    """Pick the backend class based on env / explicit preference.

    Defaults to Vulkan; on macOS, MoltenVK transparently runs the Vulkan
    backend on top of Metal, so no code change is needed for the fallback
    path. Pass ``prefer="metal"`` (or set ``SKINNY_BACKEND=metal``) to
    select the native Metal backend (currently stubs).
    """
    choice = (os.environ.get("SKINNY_BACKEND") or prefer or "vulkan").lower()
    if choice == "metal":
        from skinny.gfx.metal import MetalBackend
        return MetalBackend
    if choice in ("vulkan", "moltenvk"):
        from skinny.gfx.vulkan import VulkanBackend
        return VulkanBackend
    raise ValueError(f"Unknown backend: {choice!r}")


__all__ = [
    "AcquiredImage",
    "AddressMode",
    "Backend",
    "BackendCaps",
    "BindingDecl",
    "BindingKind",
    "Buffer",
    "BufferUsage",
    "CommandList",
    "ComputePipeline",
    "DescriptorLayout",
    "DescriptorSet",
    "Device",
    "Extent2D",
    "Extent3D",
    "Fence",
    "FilterMode",
    "Format",
    "GraphicsPipeline",
    "GraphicsPipelineDesc",
    "Image",
    "ImageState",
    "ImageUsage",
    "PipelineStage",
    "Presenter",
    "PushConstantRange",
    "Queue",
    "Sampler",
    "SamplerDesc",
    "Semaphore",
    "ShaderModule",
    "ShaderStage",
    "select_backend",
]
