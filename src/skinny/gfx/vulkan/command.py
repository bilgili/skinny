"""VulkanCommandList + VulkanQueue."""

from __future__ import annotations

from typing import TYPE_CHECKING

import vulkan as vk

from skinny.gfx.command import CommandList
from skinny.gfx.device import Queue
from skinny.gfx.types import Extent2D, ImageState, PipelineStage
from skinny.gfx.vulkan._helpers import (
    vk_access_for_state,
    vk_image_layout,
    vk_pipeline_stage,
)

if TYPE_CHECKING:
    from skinny.gfx.command import Fence, Semaphore
    from skinny.gfx.pipeline import (
        ComputePipeline,
        DescriptorSet,
        GraphicsPipeline,
    )
    from skinny.gfx.resources import Buffer, Image
    from skinny.gfx.vulkan.device import VulkanDevice
    from skinny.gfx.vulkan.resources import VulkanImage


class VulkanQueue(Queue):
    def __init__(self, device, handle, family_index: int) -> None:
        self._device = device
        self.handle = handle
        self.family_index = family_index

    def submit(
        self,
        command_lists: list[CommandList],
        wait_semaphores: list["Semaphore"] | None = None,
        signal_semaphores: list["Semaphore"] | None = None,
        fence: "Fence | None" = None,
    ) -> None:
        cmds = [cl.handle for cl in command_lists]  # type: ignore[attr-defined]
        wait_handles = [s.handle for s in (wait_semaphores or [])]
        signal_handles = [s.handle for s in (signal_semaphores or [])]
        wait_stages = [vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT] * len(wait_handles)
        info = vk.VkSubmitInfo(
            waitSemaphoreCount=len(wait_handles),
            pWaitSemaphores=wait_handles,
            pWaitDstStageMask=wait_stages,
            commandBufferCount=len(cmds),
            pCommandBuffers=cmds,
            signalSemaphoreCount=len(signal_handles),
            pSignalSemaphores=signal_handles,
        )
        fence_handle = fence.handle if fence is not None else vk.VK_NULL_HANDLE  # type: ignore[attr-defined]
        vk.vkQueueSubmit(self.handle, 1, [info], fence_handle)

    def wait_idle(self) -> None:
        vk.vkQueueWaitIdle(self.handle)


class VulkanCommandList(CommandList):
    def __init__(self, device: "VulkanDevice") -> None:
        self._device = device
        alloc = vk.VkCommandBufferAllocateInfo(
            commandPool=device.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.handle = vk.vkAllocateCommandBuffers(device.handle, alloc)[0]
        self._bound_pipeline_layout = None

    def begin(self) -> None:
        info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(self.handle, info)

    def end(self) -> None:
        vk.vkEndCommandBuffer(self.handle)

    def reset(self) -> None:
        vk.vkResetCommandBuffer(self.handle, 0)

    # ── Compute ────────────────────────────────────────────────────

    def bind_compute_pipeline(self, pipeline: "ComputePipeline") -> None:
        vk.vkCmdBindPipeline(
            self.handle,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.handle,  # type: ignore[attr-defined]
        )
        self._bound_pipeline_layout = pipeline.pipeline_layout  # type: ignore[attr-defined]

    def bind_descriptor_set(
        self,
        pipeline: "ComputePipeline | GraphicsPipeline",
        descriptor_set: "DescriptorSet",
        set_index: int = 0,
    ) -> None:
        vk.vkCmdBindDescriptorSets(
            self.handle,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout,  # type: ignore[attr-defined]
            set_index, 1, [descriptor_set.handle],  # type: ignore[attr-defined]
            0, None,
        )

    def push_constants(
        self,
        pipeline: "ComputePipeline | GraphicsPipeline",
        data: bytes | memoryview,
        offset: int = 0,
    ) -> None:
        payload = bytes(data)
        vk.vkCmdPushConstants(
            self.handle,
            pipeline.pipeline_layout,  # type: ignore[attr-defined]
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset, len(payload), payload,
        )

    def dispatch(self, group_x: int, group_y: int, group_z: int = 1) -> None:
        vk.vkCmdDispatch(self.handle, group_x, group_y, group_z)

    # ── Barriers ───────────────────────────────────────────────────

    def memory_barrier(
        self,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
    ) -> None:
        barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            self.handle,
            vk_pipeline_stage(src_stage),
            vk_pipeline_stage(dst_stage),
            0, 1, [barrier], 0, None, 0, None,
        )

    def image_barrier(
        self,
        image: "Image",
        old_state: ImageState,
        new_state: ImageState,
        src_stage: PipelineStage = PipelineStage.COMPUTE_SHADER,
        dst_stage: PipelineStage = PipelineStage.COMPUTE_SHADER,
    ) -> None:
        img: "VulkanImage" = image  # type: ignore[assignment]
        barrier = vk.VkImageMemoryBarrier(
            srcAccessMask=vk_access_for_state(old_state),
            dstAccessMask=vk_access_for_state(new_state),
            oldLayout=vk_image_layout(old_state),
            newLayout=vk_image_layout(new_state),
            image=img.handle,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            self.handle,
            vk_pipeline_stage(src_stage),
            vk_pipeline_stage(dst_stage),
            0, 0, None, 0, None, 1, [barrier],
        )
        img.current_state = new_state

    # ── Transfer ───────────────────────────────────────────────────

    def copy_buffer_to_image(
        self,
        buffer: "Buffer",
        image: "Image",
        offset: int = 0,
    ) -> None:
        img: "VulkanImage" = image  # type: ignore[assignment]
        region = vk.VkBufferImageCopy(
            bufferOffset=offset,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(
                width=img.extent.width, height=img.extent.height, depth=1,
            ),
        )
        vk.vkCmdCopyBufferToImage(
            self.handle,
            buffer.handle,  # type: ignore[attr-defined]
            img.handle,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, [region],
        )

    def copy_image_to_buffer(
        self,
        image: "Image",
        buffer: "Buffer",
        offset: int = 0,
    ) -> None:
        img: "VulkanImage" = image  # type: ignore[assignment]
        region = vk.VkBufferImageCopy(
            bufferOffset=offset,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(
                width=img.extent.width, height=img.extent.height, depth=1,
            ),
        )
        vk.vkCmdCopyImageToBuffer(
            self.handle,
            img.handle,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            buffer.handle,  # type: ignore[attr-defined]
            1, [region],
        )

    def blit_image(
        self,
        src: "Image",
        dst: "Image",
        dst_extent: Extent2D | None = None,
    ) -> None:
        s: "VulkanImage" = src  # type: ignore[assignment]
        d: "VulkanImage" = dst  # type: ignore[assignment]
        target = dst_extent or d.extent
        region = vk.VkImageBlit(
            srcSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            srcOffsets=[
                vk.VkOffset3D(x=0, y=0, z=0),
                vk.VkOffset3D(x=s.extent.width, y=s.extent.height, z=1),
            ],
            dstSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            dstOffsets=[
                vk.VkOffset3D(x=0, y=0, z=0),
                vk.VkOffset3D(x=target.width, y=target.height, z=1),
            ],
        )
        vk.vkCmdBlitImage(
            self.handle,
            s.handle, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            d.handle, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, [region],
            vk.VK_FILTER_LINEAR,
        )
