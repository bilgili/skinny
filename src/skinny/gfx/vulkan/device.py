"""VulkanDevice — wraps the logical device, queues, and command pool.

Owns the resource factory methods. Pipeline + descriptor creation lands in
Step 3; for Step 2 those raise so the wiring is testable but doesn't claim
to be complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import vulkan as vk

from skinny.gfx.device import Device, Queue
from skinny.gfx.types import (
    BufferUsage,
    Extent2D,
    Format,
    ImageState,
    ImageUsage,
)
from skinny.gfx.vulkan._helpers import vk_image_layout
from skinny.gfx.vulkan.command import VulkanCommandList, VulkanQueue
from skinny.gfx.vulkan.resources import (
    VulkanBuffer,
    VulkanImage,
    VulkanSampler,
    VulkanShaderModule,
)
from skinny.gfx.vulkan.sync import VulkanFence, VulkanSemaphore

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


class VulkanDevice(Device):
    def __init__(
        self,
        backend,
        physical_device,
        compute_family: int,
        present_family: int | None,
    ) -> None:
        self._backend = backend
        self.physical_device = physical_device
        self._compute_family = compute_family
        self._present_family = present_family

        unique = {compute_family}
        if present_family is not None:
            unique.add(present_family)
        queue_create_infos = [
            vk.VkDeviceQueueCreateInfo(
                queueFamilyIndex=idx,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            for idx in unique
        ]

        indexing_features = vk.VkPhysicalDeviceVulkan12Features(
            descriptorBindingPartiallyBound=vk.VK_TRUE,
            shaderSampledImageArrayNonUniformIndexing=vk.VK_TRUE,
            scalarBlockLayout=vk.VK_TRUE,
        )
        device_extensions = (
            [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME] if present_family is not None else []
        )
        device_info = vk.VkDeviceCreateInfo(
            pNext=indexing_features,
            queueCreateInfoCount=len(queue_create_infos),
            pQueueCreateInfos=queue_create_infos,
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=device_extensions,
            pEnabledFeatures=vk.VkPhysicalDeviceFeatures(),
        )
        self.handle = vk.vkCreateDevice(physical_device, device_info, None)

        compute_handle = vk.vkGetDeviceQueue(self.handle, compute_family, 0)
        self._compute_queue = VulkanQueue(self.handle, compute_handle, compute_family)
        if present_family is not None and present_family != compute_family:
            present_handle = vk.vkGetDeviceQueue(self.handle, present_family, 0)
            self._graphics_queue = VulkanQueue(self.handle, present_handle, present_family)
        else:
            self._graphics_queue = self._compute_queue

        pool_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=compute_family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.handle, pool_info, None)

    # ── Resources ──────────────────────────────────────────────────

    def create_buffer(
        self,
        size: int,
        usage: BufferUsage,
        host_visible: bool = False,
    ) -> "Buffer":
        return VulkanBuffer(self, size, usage, host_visible)

    def create_image(
        self,
        extent: Extent2D,
        format: Format,
        usage: ImageUsage,
    ) -> "Image":
        return VulkanImage(self, extent, format, usage)

    def create_sampler(self, desc: "SamplerDesc") -> "Sampler":
        return VulkanSampler(self, desc)

    def create_shader_module(
        self,
        blob: bytes,
        entry_point: str,
    ) -> "ShaderModule":
        return VulkanShaderModule(self, blob, entry_point)

    # ── Pipeline / descriptor (Step 3) ─────────────────────────────

    def create_descriptor_layout(
        self,
        bindings: "list[BindingDecl]",
        push_constants: "list[PushConstantRange] | None" = None,
    ) -> "DescriptorLayout":
        raise NotImplementedError("DescriptorLayout lands in Step 3")

    def allocate_descriptor_set(
        self,
        layout: "DescriptorLayout",
    ) -> "DescriptorSet":
        raise NotImplementedError("DescriptorSet lands in Step 3")

    def create_compute_pipeline(
        self,
        module: "ShaderModule",
        layout: "DescriptorLayout",
    ) -> "ComputePipeline":
        raise NotImplementedError("ComputePipeline lands in Step 3")

    def create_graphics_pipeline(
        self,
        desc: "GraphicsPipelineDesc",
        layout: "DescriptorLayout",
    ) -> "GraphicsPipeline":
        raise NotImplementedError("GraphicsPipeline lands in Step 5")

    # ── Command + sync ─────────────────────────────────────────────

    def create_command_list(self) -> "CommandList":
        return VulkanCommandList(self)

    def create_fence(self, signaled: bool = False) -> "Fence":
        return VulkanFence(self.handle, signaled)

    def create_semaphore(self) -> "Semaphore":
        return VulkanSemaphore(self.handle)

    # ── Queues ─────────────────────────────────────────────────────

    @property
    def compute_queue(self) -> Queue:
        return self._compute_queue

    @property
    def graphics_queue(self) -> Queue:
        return self._graphics_queue

    def wait_idle(self) -> None:
        vk.vkDeviceWaitIdle(self.handle)

    # ── One-shot helpers (used by VulkanBuffer/VulkanImage) ────────

    def _alloc_one_shot_cmd(self):
        info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vk.vkAllocateCommandBuffers(self.handle, info)[0]

    def _submit_one_shot(self, record_fn) -> None:
        cmd = self._alloc_one_shot_cmd()
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        record_fn(cmd)
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self._compute_queue.handle, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self._compute_queue.handle)
        vk.vkFreeCommandBuffers(self.handle, self.command_pool, 1, [cmd])

    def transition_image(
        self,
        image: VulkanImage,
        old_state: ImageState,
        new_state: ImageState,
    ) -> None:
        from skinny.gfx.vulkan._helpers import vk_access_for_state

        def record(cmd) -> None:
            barrier = vk.VkImageMemoryBarrier(
                srcAccessMask=vk_access_for_state(old_state),
                dstAccessMask=vk_access_for_state(new_state),
                oldLayout=vk_image_layout(old_state),
                newLayout=vk_image_layout(new_state),
                image=image.handle,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1,
                ),
            )
            vk.vkCmdPipelineBarrier(
                cmd,
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, None, 0, None, 1, [barrier],
            )

        self._submit_one_shot(record)

    def upload_to_buffer(
        self,
        buffer: VulkanBuffer,
        data: bytes,
        offset: int,
    ) -> None:
        # Stage-via-host-visible buffer for device-local destinations.
        staging = VulkanBuffer(
            self, len(data),
            BufferUsage.TRANSFER_SRC,
            host_visible=True,
        )
        try:
            staging._mapped_ptr[0:len(data)] = data  # type: ignore[index]

            def record(cmd) -> None:
                region = vk.VkBufferCopy(
                    srcOffset=0, dstOffset=offset, size=len(data),
                )
                vk.vkCmdCopyBuffer(cmd, staging.handle, buffer.handle, 1, [region])

            self._submit_one_shot(record)
        finally:
            staging.destroy()

    def upload_to_image(
        self,
        image: VulkanImage,
        data: bytes,
    ) -> None:
        staging = VulkanBuffer(
            self, len(data),
            BufferUsage.TRANSFER_SRC,
            host_visible=True,
        )
        try:
            staging._mapped_ptr[0:len(data)] = data  # type: ignore[index]

            current = image.current_state
            sub = vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            )

            def record(cmd) -> None:
                from skinny.gfx.vulkan._helpers import vk_access_for_state
                to_dst = vk.VkImageMemoryBarrier(
                    srcAccessMask=vk_access_for_state(current),
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk_image_layout(current),
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    image=image.handle,
                    subresourceRange=sub,
                )
                vk.vkCmdPipelineBarrier(
                    cmd,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, None, 0, None, 1, [to_dst],
                )
                region = vk.VkBufferImageCopy(
                    bufferOffset=0,
                    bufferRowLength=0,
                    bufferImageHeight=0,
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0, baseArrayLayer=0, layerCount=1,
                    ),
                    imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    imageExtent=vk.VkExtent3D(
                        width=image.extent.width,
                        height=image.extent.height,
                        depth=1,
                    ),
                )
                vk.vkCmdCopyBufferToImage(
                    cmd, staging.handle, image.handle,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, [region],
                )
                # Restore the prior layout.
                final_state = (
                    ImageState.SHADER_READ
                    if image.usage & ImageUsage.SAMPLED
                    else ImageState.GENERAL
                )
                to_final = vk.VkImageMemoryBarrier(
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk_access_for_state(final_state),
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk_image_layout(final_state),
                    image=image.handle,
                    subresourceRange=sub,
                )
                vk.vkCmdPipelineBarrier(
                    cmd,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 0, None, 0, None, 1, [to_final],
                )
                image.current_state = final_state

            self._submit_one_shot(record)
        finally:
            staging.destroy()

    # ── Cleanup ────────────────────────────────────────────────────

    def destroy(self) -> None:
        vk.vkDestroyCommandPool(self.handle, self.command_pool, None)
        vk.vkDestroyDevice(self.handle, None)
