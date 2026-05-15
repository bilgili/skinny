"""Backend-agnostic enums and small value types for the skinny rendering layer.

These mirror the Vulkan vocabulary closely because that is the renderer's
native idiom; concrete backends translate them (Metal collapses several
ImageState values into no-ops since Metal tracks hazards automatically).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, IntEnum, auto


class Format(IntEnum):
    UNDEFINED = 0
    R8G8B8A8_UNORM = 1
    R8G8B8A8_SRGB = 2
    B8G8R8A8_UNORM = 3
    B8G8R8A8_SRGB = 4
    R16G16B16A16_SFLOAT = 5
    R32G32B32A32_SFLOAT = 6
    R32_UINT = 7
    R32_SFLOAT = 8
    D32_SFLOAT = 9


class BufferUsage(Flag):
    NONE = 0
    UNIFORM = auto()
    STORAGE = auto()
    INDEX = auto()
    VERTEX = auto()
    INDIRECT = auto()
    TRANSFER_SRC = auto()
    TRANSFER_DST = auto()


class ImageUsage(Flag):
    NONE = 0
    SAMPLED = auto()
    STORAGE = auto()
    COLOR_ATTACHMENT = auto()
    DEPTH_ATTACHMENT = auto()
    TRANSFER_SRC = auto()
    TRANSFER_DST = auto()


class ImageState(IntEnum):
    """Logical image layout / access state.

    Vulkan maps these to ``VK_IMAGE_LAYOUT_*`` + access masks. Metal mostly
    treats them as no-ops but keeps the explicit transitions in the call
    graph so the renderer logic is identical across backends.
    """

    UNDEFINED = 0
    GENERAL = 1
    SHADER_READ = 2
    SHADER_WRITE = 3
    TRANSFER_SRC = 4
    TRANSFER_DST = 5
    COLOR_ATTACHMENT = 6
    DEPTH_ATTACHMENT = 7
    PRESENT = 8


class BindingKind(IntEnum):
    UNIFORM_BUFFER = 0
    STORAGE_BUFFER = 1
    STORAGE_IMAGE = 2
    SAMPLED_IMAGE = 3
    COMBINED_IMAGE_SAMPLER = 4
    SAMPLER = 5


class ShaderStage(Flag):
    NONE = 0
    VERTEX = auto()
    FRAGMENT = auto()
    COMPUTE = auto()
    ALL_GRAPHICS = VERTEX | FRAGMENT


class PipelineStage(Flag):
    NONE = 0
    COMPUTE_SHADER = auto()
    VERTEX_SHADER = auto()
    FRAGMENT_SHADER = auto()
    TRANSFER = auto()
    COLOR_ATTACHMENT_OUTPUT = auto()
    TOP = auto()
    BOTTOM = auto()


class FilterMode(IntEnum):
    NEAREST = 0
    LINEAR = 1


class AddressMode(IntEnum):
    REPEAT = 0
    CLAMP_TO_EDGE = 1
    CLAMP_TO_BORDER = 2
    MIRRORED_REPEAT = 3


@dataclass(frozen=True)
class Extent2D:
    width: int
    height: int


@dataclass(frozen=True)
class Extent3D:
    width: int
    height: int
    depth: int = 1
