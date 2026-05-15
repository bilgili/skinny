"""VulkanFence + VulkanSemaphore."""

from __future__ import annotations

import vulkan as vk

from skinny.gfx.command import Fence, Semaphore


class VulkanFence(Fence):
    def __init__(self, device, signaled: bool = False) -> None:
        self._device = device
        flags = vk.VK_FENCE_CREATE_SIGNALED_BIT if signaled else 0
        info = vk.VkFenceCreateInfo(flags=flags)
        self.handle = vk.vkCreateFence(device, info, None)

    def wait(self, timeout_ns: int = 2**63 - 1) -> None:
        vk.vkWaitForFences(self._device, 1, [self.handle], vk.VK_TRUE, timeout_ns)

    def reset(self) -> None:
        vk.vkResetFences(self._device, 1, [self.handle])

    def is_signaled(self) -> bool:
        # python-vulkan returns None on VK_SUCCESS and raises on error/non-
        # success codes (VK_NOT_READY, VK_TIMEOUT). Treat the raise as the
        # "not signaled yet" path.
        try:
            vk.vkWaitForFences(self._device, 1, [self.handle], vk.VK_TRUE, 0)
            return True
        except Exception:
            return False

    def destroy(self) -> None:
        vk.vkDestroyFence(self._device, self.handle, None)


class VulkanSemaphore(Semaphore):
    def __init__(self, device) -> None:
        self._device = device
        self.handle = vk.vkCreateSemaphore(device, vk.VkSemaphoreCreateInfo(), None)

    def destroy(self) -> None:
        vk.vkDestroySemaphore(self._device, self.handle, None)
