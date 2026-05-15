"""Concrete Vulkan resource implementations.

These wrap raw Vulkan handles (VkBuffer/VkImage/VkSampler/VkShaderModule)
plus the device memory backing them. Logic that needs a queue submit
(device-local upload, image layout transition) routes through the
device's one-shot helper rather than recreating command buffers ad-hoc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cffi
import vulkan as vk

from skinny.gfx.resources import (
    Buffer,
    Image,
    Sampler,
    SamplerDesc,
    ShaderModule,
)
from skinny.gfx.types import (
    BufferUsage,
    Extent2D,
    Format,
    ImageState,
    ImageUsage,
)
from skinny.gfx.vulkan._helpers import (
    find_memory_type,
    format_bytes,
    vk_address_mode,
    vk_buffer_usage,
    vk_filter,
    vk_format,
    vk_image_usage,
    vk_mipmap_mode,
)

if TYPE_CHECKING:
    from skinny.gfx.vulkan.device import VulkanDevice

_FFI = cffi.FFI()


class VulkanBuffer(Buffer):
    def __init__(
        self,
        device: "VulkanDevice",
        size: int,
        usage: BufferUsage,
        host_visible: bool,
    ) -> None:
        self._device = device
        self.size = max(int(size), 16)  # GPU drivers dislike zero-size
        self.usage = usage
        self.host_visible = host_visible

        vk_usage = vk_buffer_usage(usage)
        # Device-local buffers always need TRANSFER_DST so upload() can
        # stage into them; host-visible buffers map directly.
        if not host_visible:
            vk_usage |= vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT

        info = vk.VkBufferCreateInfo(
            size=self.size,
            usage=vk_usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.handle = vk.vkCreateBuffer(device.handle, info, None)

        reqs = vk.vkGetBufferMemoryRequirements(device.handle, self.handle)
        props = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            if host_visible
            else vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        mem_type = find_memory_type(device.physical_device, reqs.memoryTypeBits, props)
        self._alloc_size = reqs.size
        self.memory = vk.vkAllocateMemory(
            device.handle,
            vk.VkMemoryAllocateInfo(
                allocationSize=reqs.size,
                memoryTypeIndex=mem_type,
            ),
            None,
        )
        vk.vkBindBufferMemory(device.handle, self.handle, self.memory, 0)

        self._mapped: memoryview | None = None
        if host_visible:
            self._mapped_ptr = vk.vkMapMemory(
                device.handle, self.memory, 0, self._alloc_size, 0
            )
        else:
            self._mapped_ptr = None

    def upload(self, data: bytes | memoryview, offset: int = 0) -> None:
        payload = bytes(data)
        if offset + len(payload) > self.size:
            raise ValueError(
                f"VulkanBuffer.upload: {offset + len(payload)}B > buffer {self.size}B"
            )
        if self.host_visible:
            self._mapped_ptr[offset:offset + len(payload)] = payload
            return
        # Device-local: stage + queue copy.
        self._device.upload_to_buffer(self, payload, offset)

    def upload_sync(self, data: bytes | memoryview, offset: int = 0) -> None:
        self.upload(data, offset)
        if not self.host_visible:
            # upload_to_buffer is already synchronous (one-shot + waitIdle).
            return

    def map(self) -> memoryview:
        if not self.host_visible:
            raise RuntimeError("map() requires host_visible=True")
        # vkMapMemory returns a cffi buffer that already exposes the
        # buffer protocol; memoryview wraps it directly.
        return memoryview(self._mapped_ptr)

    def unmap(self) -> None:
        # Persistent map; nothing to do until destroy().
        return None

    def destroy(self) -> None:
        if self._mapped_ptr is not None:
            vk.vkUnmapMemory(self._device.handle, self.memory)
            self._mapped_ptr = None
        vk.vkDestroyBuffer(self._device.handle, self.handle, None)
        vk.vkFreeMemory(self._device.handle, self.memory, None)


class VulkanImage(Image):
    def __init__(
        self,
        device: "VulkanDevice",
        extent: Extent2D,
        format: Format,
        usage: ImageUsage,
    ) -> None:
        self._device = device
        self.extent = extent
        self.format = format
        self.usage = usage
        self._byte_count = extent.width * extent.height * format_bytes(format)

        vk_usage = vk_image_usage(usage)
        # SAMPLED implies an upload path; force TRANSFER_DST so upload()
        # works without a separate flag.
        if usage & ImageUsage.SAMPLED and not (usage & ImageUsage.TRANSFER_DST):
            vk_usage |= vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT

        info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk_format(format),
            extent=vk.VkExtent3D(width=extent.width, height=extent.height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk_usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self.handle = vk.vkCreateImage(device.handle, info, None)

        reqs = vk.vkGetImageMemoryRequirements(device.handle, self.handle)
        mem_type = find_memory_type(
            device.physical_device,
            reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        self.memory = vk.vkAllocateMemory(
            device.handle,
            vk.VkMemoryAllocateInfo(allocationSize=reqs.size, memoryTypeIndex=mem_type),
            None,
        )
        vk.vkBindImageMemory(device.handle, self.handle, self.memory, 0)

        view_info = vk.VkImageViewCreateInfo(
            image=self.handle,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=vk_format(format),
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        self.view = vk.vkCreateImageView(device.handle, view_info, None)

        # Initial transition: UNDEFINED → useful state. Storage images go to
        # GENERAL; sampled images go to SHADER_READ so empty samples don't
        # error before the first upload.
        if usage & ImageUsage.STORAGE:
            initial_state = ImageState.GENERAL
        elif usage & ImageUsage.SAMPLED:
            initial_state = ImageState.SHADER_READ
        else:
            initial_state = ImageState.GENERAL
        self.current_state = ImageState.UNDEFINED
        device.transition_image(self, ImageState.UNDEFINED, initial_state)
        self.current_state = initial_state

    def upload(self, data: bytes | memoryview) -> None:
        payload = bytes(data)
        if len(payload) != self._byte_count:
            raise ValueError(
                f"VulkanImage.upload: got {len(payload)}B, expected {self._byte_count}B"
            )
        self._device.upload_to_image(self, payload)

    def destroy(self) -> None:
        vk.vkDestroyImageView(self._device.handle, self.view, None)
        vk.vkDestroyImage(self._device.handle, self.handle, None)
        vk.vkFreeMemory(self._device.handle, self.memory, None)


class VulkanSampler(Sampler):
    def __init__(self, device: "VulkanDevice", desc: SamplerDesc) -> None:
        self._device = device
        info = vk.VkSamplerCreateInfo(
            magFilter=vk_filter(desc.mag_filter),
            minFilter=vk_filter(desc.min_filter),
            mipmapMode=vk_mipmap_mode(desc.mip_filter),
            addressModeU=vk_address_mode(desc.address_u),
            addressModeV=vk_address_mode(desc.address_v),
            addressModeW=vk_address_mode(desc.address_w),
            anisotropyEnable=vk.VK_TRUE if desc.max_anisotropy > 1.0 else vk.VK_FALSE,
            maxAnisotropy=desc.max_anisotropy,
            compareEnable=vk.VK_FALSE,
            compareOp=vk.VK_COMPARE_OP_ALWAYS,
            minLod=0.0,
            maxLod=0.0,
            borderColor=vk.VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            unnormalizedCoordinates=vk.VK_FALSE,
        )
        self.handle = vk.vkCreateSampler(device.handle, info, None)

    def destroy(self) -> None:
        vk.vkDestroySampler(self._device.handle, self.handle, None)


class VulkanShaderModule(ShaderModule):
    def __init__(self, device: "VulkanDevice", blob: bytes, entry_point: str) -> None:
        self._device = device
        self.entry_point = entry_point
        info = vk.VkShaderModuleCreateInfo(codeSize=len(blob), pCode=blob)
        self.handle = vk.vkCreateShaderModule(device.handle, info, None)

    def destroy(self) -> None:
        vk.vkDestroyShaderModule(self._device.handle, self.handle, None)
