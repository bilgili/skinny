"""VulkanPresenter — wraps VK_KHR_swapchain.

Acquires images from the swapchain, hands them to the renderer wrapped as
``VulkanImage`` (constructed with an externally-owned VkImage handle so it
participates in the same descriptor-write code path as offscreen images),
and submits the present operation.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import vulkan as vk

from skinny.gfx.presenter import AcquiredImage, Presenter
from skinny.gfx.resources import Image
from skinny.gfx.types import Extent2D, Format, ImageState, ImageUsage
from skinny.gfx.vulkan._helpers import vk_format
from skinny.gfx.vulkan.sync import VulkanSemaphore

if TYPE_CHECKING:
    from skinny.gfx.vulkan.backend import VulkanBackend


class _SwapchainImage(Image):
    """Image view + handle pair for one swapchain slot.

    Memory is owned by the swapchain, not this object — destroy() only
    releases the view we created.
    """

    def __init__(
        self,
        device,
        handle,
        view,
        extent: Extent2D,
        format: Format,
    ) -> None:
        self._device = device
        self.handle = handle
        self.view = view
        self.extent = extent
        self.format = format
        self.usage = ImageUsage.COLOR_ATTACHMENT | ImageUsage.STORAGE | ImageUsage.TRANSFER_DST
        self.current_state = ImageState.UNDEFINED

    def upload(self, data) -> None:
        raise RuntimeError("Cannot upload host data to a swapchain image")

    def destroy(self) -> None:
        vk.vkDestroyImageView(self._device, self.view, None)


class VulkanPresenter(Presenter):
    def __init__(self, backend: "VulkanBackend", window) -> None:
        self._backend = backend
        self._device = backend.device.handle
        self._physical_device = backend.physical_device
        self._window = window
        self._instance = backend.instance
        self._queue_family = backend.device._compute_family

        self._load_instance_funcs()
        self.surface = self._create_surface(window)

        # Confirm the chosen queue family supports presentation on this surface.
        if not self._vkGetPhysicalDeviceSurfaceSupportKHR(
            self._physical_device, self._queue_family, self.surface
        ):
            raise RuntimeError(
                "Compute queue family does not support presentation on this surface"
            )

        self._load_device_funcs()
        self._images: list[_SwapchainImage] = []
        self.format = Format.B8G8R8A8_UNORM
        self._vk_format = vk_format(self.format)
        self._vk_color_space = vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        self.extent = Extent2D(backend._init_width, backend._init_height)
        self._build_swapchain(self.extent.width, self.extent.height)

        # Per-frame acquire semaphores. Two-frame ring matches MAX_FRAMES_IN_FLIGHT
        # in the renderer; rotating prevents reusing a still-pending semaphore.
        self._acquire_sems = [VulkanSemaphore(self._device) for _ in range(2)]
        self._frame = 0

    # ── ICD function loading ────────────────────────────────────────

    def _load_instance_funcs(self) -> None:
        get = vk.vkGetInstanceProcAddr
        self._vkGetPhysicalDeviceSurfaceSupportKHR = get(
            self._instance, "vkGetPhysicalDeviceSurfaceSupportKHR"
        )
        self._vkGetPhysicalDeviceSurfaceCapabilitiesKHR = get(
            self._instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
        )
        self._vkGetPhysicalDeviceSurfaceFormatsKHR = get(
            self._instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
        )
        self._vkGetPhysicalDeviceSurfacePresentModesKHR = get(
            self._instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"
        )
        self._vkDestroySurfaceKHR = get(self._instance, "vkDestroySurfaceKHR")

    def _load_device_funcs(self) -> None:
        get = vk.vkGetDeviceProcAddr
        self._vkCreateSwapchainKHR = get(self._device, "vkCreateSwapchainKHR")
        self._vkGetSwapchainImagesKHR = get(self._device, "vkGetSwapchainImagesKHR")
        self._vkDestroySwapchainKHR = get(self._device, "vkDestroySwapchainKHR")
        self._vkAcquireNextImageKHR = get(self._device, "vkAcquireNextImageKHR")
        self._vkQueuePresentKHR = get(self._device, "vkQueuePresentKHR")

    # ── Surface ────────────────────────────────────────────────────

    def _create_surface(self, window):
        import glfw
        instance_handle = int(vk.ffi.cast("uintptr_t", self._instance))
        surface = ctypes.c_void_p(0)
        result = glfw.create_window_surface(
            instance_handle, window, None, ctypes.byref(surface)
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create Vulkan surface: {result}")
        return vk.ffi.cast("VkSurfaceKHR", surface.value)

    # ── Swapchain ──────────────────────────────────────────────────

    def _build_swapchain(self, width: int, height: int) -> None:
        caps = self._vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            self._physical_device, self.surface
        )
        formats = self._vkGetPhysicalDeviceSurfaceFormatsKHR(
            self._physical_device, self.surface
        )
        modes = self._vkGetPhysicalDeviceSurfacePresentModesKHR(
            self._physical_device, self.surface
        )

        chosen_format = formats[0]
        for fmt in formats:
            if (
                fmt.format == self._vk_format
                and fmt.colorSpace == self._vk_color_space
            ):
                chosen_format = fmt
                break
        self._vk_format = chosen_format.format
        self._vk_color_space = chosen_format.colorSpace

        chosen_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if vk.VK_PRESENT_MODE_MAILBOX_KHR in modes:
            chosen_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR

        extent = vk.VkExtent2D(width=width, height=height)
        image_count = caps.minImageCount + 1
        if caps.maxImageCount > 0:
            image_count = min(image_count, caps.maxImageCount)

        info = vk.VkSwapchainCreateInfoKHR(
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=self._vk_format,
            imageColorSpace=self._vk_color_space,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=(
                vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                | vk.VK_IMAGE_USAGE_STORAGE_BIT
                | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
            ),
            imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=1,
            pQueueFamilyIndices=[self._queue_family],
            preTransform=caps.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=chosen_mode,
            clipped=vk.VK_TRUE,
        )
        self.swapchain = self._vkCreateSwapchainKHR(self._device, info, None)
        self.extent = Extent2D(width, height)

        raw_images = self._vkGetSwapchainImagesKHR(self._device, self.swapchain)
        self._images = []
        for img in raw_images:
            view_info = vk.VkImageViewCreateInfo(
                image=img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self._vk_format,
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
            view = vk.vkCreateImageView(self._device, view_info, None)
            self._images.append(_SwapchainImage(
                self._device, img, view,
                Extent2D(width, height), self.format,
            ))

    # ── Presenter API ──────────────────────────────────────────────

    def image_count(self) -> int:
        return len(self._images)

    def acquire(self) -> AcquiredImage:
        sem = self._acquire_sems[self._frame]
        index = self._vkAcquireNextImageKHR(
            self._device,
            self.swapchain,
            2**64 - 1,
            sem.handle,
            vk.VK_NULL_HANDLE,
        )
        return AcquiredImage(
            image=self._images[index],
            index=index,
            acquire_semaphore=sem,
        )

    def present(self, acquired: AcquiredImage, wait_semaphore=None) -> None:
        wait_handles = [wait_semaphore.handle] if wait_semaphore is not None else []
        info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=len(wait_handles),
            pWaitSemaphores=wait_handles,
            swapchainCount=1,
            pSwapchains=[self.swapchain],
            pImageIndices=[acquired.index],
        )
        self._vkQueuePresentKHR(self._backend.device.graphics_queue.handle, info)
        self._frame = (self._frame + 1) % len(self._acquire_sems)

    def recreate(self, width: int, height: int) -> None:
        vk.vkDeviceWaitIdle(self._device)
        for img in self._images:
            img.destroy()
        self._vkDestroySwapchainKHR(self._device, self.swapchain, None)
        self._build_swapchain(width, height)

    def destroy(self) -> None:
        for sem in self._acquire_sems:
            sem.destroy()
        for img in self._images:
            img.destroy()
        self._vkDestroySwapchainKHR(self._device, self.swapchain, None)
        self._vkDestroySurfaceKHR(self._instance, self.surface, None)
