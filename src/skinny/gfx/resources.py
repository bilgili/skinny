"""Resource ABCs: Buffer, Image, Sampler, ShaderModule.

Concrete backends own real GPU handles; these classes expose only the
operations the renderer needs (uploads, mapped pointers, view access for
descriptor writes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from skinny.gfx.types import (
    AddressMode,
    BufferUsage,
    Extent2D,
    FilterMode,
    Format,
    ImageUsage,
)


class Buffer(ABC):
    size: int
    usage: BufferUsage
    host_visible: bool

    @abstractmethod
    def upload(self, data: bytes | memoryview, offset: int = 0) -> None:
        """Copy host data into the buffer. host_visible buffers map; device-
        local buffers go through a staging buffer + transfer queue submit."""

    @abstractmethod
    def upload_sync(self, data: bytes | memoryview, offset: int = 0) -> None:
        """Same as upload but blocks until the GPU has the data visible."""

    @abstractmethod
    def map(self) -> memoryview:
        """host_visible only. Raises if device-local."""

    @abstractmethod
    def unmap(self) -> None: ...

    @abstractmethod
    def destroy(self) -> None: ...


class Image(ABC):
    extent: Extent2D
    format: Format
    usage: ImageUsage

    @abstractmethod
    def upload(self, data: bytes | memoryview) -> None:
        """Tightly packed pixels in `format`'s native byte order."""

    @abstractmethod
    def destroy(self) -> None: ...


class Sampler(ABC):
    @abstractmethod
    def destroy(self) -> None: ...


class SamplerDesc:
    """Plain value-object so call sites don't depend on backend types."""

    __slots__ = ("min_filter", "mag_filter", "mip_filter",
                 "address_u", "address_v", "address_w",
                 "max_anisotropy")

    def __init__(
        self,
        *,
        min_filter: FilterMode = FilterMode.LINEAR,
        mag_filter: FilterMode = FilterMode.LINEAR,
        mip_filter: FilterMode = FilterMode.LINEAR,
        address_u: AddressMode = AddressMode.REPEAT,
        address_v: AddressMode = AddressMode.REPEAT,
        address_w: AddressMode = AddressMode.REPEAT,
        max_anisotropy: float = 1.0,
    ) -> None:
        self.min_filter = min_filter
        self.mag_filter = mag_filter
        self.mip_filter = mip_filter
        self.address_u = address_u
        self.address_v = address_v
        self.address_w = address_w
        self.max_anisotropy = max_anisotropy


class ShaderModule(ABC):
    """Compiled shader blob loaded into the GPU driver.

    Vulkan: VkShaderModule wrapping SPIR-V.
    Metal: MTLLibrary + MTLFunction reference for an MSL entry point.
    """

    entry_point: str

    @abstractmethod
    def destroy(self) -> None: ...
