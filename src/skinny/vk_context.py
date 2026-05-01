"""Vulkan context — instance, device, queues, and swapchain management."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import glfw
import vulkan as vk


@dataclass
class SwapchainInfo:
    swapchain: object
    images: list
    image_views: list
    format: int
    extent: object


class VulkanContext:
    """Manages the core Vulkan objects needed for compute-based rendering."""

    VALIDATION_LAYERS = ["VK_LAYER_KHRONOS_validation"]
    DEVICE_EXTENSIONS = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]

    def __init__(self, window, width: int, height: int, enable_validation: bool = True) -> None:
        self.width = width
        self.height = height
        self._enable_validation = enable_validation

        self.instance = self._create_instance()
        self._load_instance_functions()
        self.surface = self._create_surface(window)
        self.physical_device = self._pick_physical_device()
        self.queue_family_indices = self._find_queue_families()
        self.device = self._create_device()
        self._load_device_functions()
        self.compute_queue = vk.vkGetDeviceQueue(
            self.device, self.queue_family_indices["compute"], 0
        )
        self.present_queue = vk.vkGetDeviceQueue(
            self.device, self.queue_family_indices["present"], 0
        )
        self.swapchain_info = self._create_swapchain()
        self.command_pool = self._create_command_pool()

    # ── Instance ─────────────────────────────────────────────────

    def _create_instance(self):
        app_info = vk.VkApplicationInfo(
            pApplicationName="Skinny",
            applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            pEngineName="Skinny Engine",
            engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0),
        )

        extensions = glfw.get_required_instance_extensions()
        layers = self.VALIDATION_LAYERS if self._enable_validation else []

        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
        )
        return vk.vkCreateInstance(create_info, None)

    def _load_instance_functions(self):
        get = vk.vkGetInstanceProcAddr
        self._vkGetPhysicalDeviceSurfaceSupportKHR = get(
            self.instance, "vkGetPhysicalDeviceSurfaceSupportKHR"
        )
        self._vkGetPhysicalDeviceSurfaceCapabilitiesKHR = get(
            self.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
        )
        self._vkGetPhysicalDeviceSurfaceFormatsKHR = get(
            self.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
        )
        self._vkGetPhysicalDeviceSurfacePresentModesKHR = get(
            self.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"
        )
        self._vkDestroySurfaceKHR = get(self.instance, "vkDestroySurfaceKHR")

    def _load_device_functions(self):
        get = vk.vkGetDeviceProcAddr
        self._vkCreateSwapchainKHR = get(self.device, "vkCreateSwapchainKHR")
        self._vkGetSwapchainImagesKHR = get(self.device, "vkGetSwapchainImagesKHR")
        self._vkDestroySwapchainKHR = get(self.device, "vkDestroySwapchainKHR")
        self.vkAcquireNextImageKHR = get(self.device, "vkAcquireNextImageKHR")
        self.vkQueuePresentKHR = get(self.device, "vkQueuePresentKHR")

    # ── Surface ──────────────────────────────────────────────────

    def _create_surface(self, window):
        # Convert cffi VkInstance to raw integer for ctypes-based glfw
        instance_handle = int(vk.ffi.cast("uintptr_t", self.instance))
        surface = ctypes.c_void_p(0)
        result = glfw.create_window_surface(
            instance_handle, window, None, ctypes.byref(surface)
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create Vulkan surface: {result}")
        return vk.ffi.cast("VkSurfaceKHR", surface.value)

    # ── Physical device ──────────────────────────────────────────

    def _pick_physical_device(self):
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("No Vulkan-capable GPU found")

        # Prefer discrete GPU
        for dev in devices:
            props = vk.vkGetPhysicalDeviceProperties(dev)
            if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return dev
        return devices[0]

    # ── Queue families ───────────────────────────────────────────

    def _find_queue_families(self) -> dict[str, int]:
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        indices: dict[str, int | None] = {"compute": None, "present": None}

        for i, family in enumerate(families):
            if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                indices["compute"] = i
            if self._vkGetPhysicalDeviceSurfaceSupportKHR(self.physical_device, i, self.surface):
                indices["present"] = i
            if all(v is not None for v in indices.values()):
                break

        if any(v is None for v in indices.values()):
            raise RuntimeError(f"Required queue families not found: {indices}")
        return indices

    # ── Logical device ───────────────────────────────────────────

    def _create_device(self):
        unique_families = set(self.queue_family_indices.values())
        queue_create_infos = [
            vk.VkDeviceQueueCreateInfo(
                queueFamilyIndex=idx,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            for idx in unique_families
        ]

        # Phase C-4: enable Vulkan 1.2 descriptor-indexing features so the
        # bindless flat-material texture array (binding 14) can hold partially
        # populated slots and tolerate non-uniform index accesses across
        # threads in the same dispatch (different pixels hit different
        # materials, so the texture index is not invocation-uniform).
        indexing_features = vk.VkPhysicalDeviceVulkan12Features(
            descriptorBindingPartiallyBound=vk.VK_TRUE,
            shaderSampledImageArrayNonUniformIndexing=vk.VK_TRUE,
            scalarBlockLayout=vk.VK_TRUE,
        )

        device_create_info = vk.VkDeviceCreateInfo(
            pNext=indexing_features,
            queueCreateInfoCount=len(queue_create_infos),
            pQueueCreateInfos=queue_create_infos,
            enabledExtensionCount=len(self.DEVICE_EXTENSIONS),
            ppEnabledExtensionNames=self.DEVICE_EXTENSIONS,
            pEnabledFeatures=vk.VkPhysicalDeviceFeatures(),
        )
        return vk.vkCreateDevice(self.physical_device, device_create_info, None)

    # ── Swapchain ────────────────────────────────────────────────

    def _create_swapchain(self) -> SwapchainInfo:
        capabilities = self._vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            self.physical_device, self.surface
        )
        formats = self._vkGetPhysicalDeviceSurfaceFormatsKHR(self.physical_device, self.surface)
        present_modes = self._vkGetPhysicalDeviceSurfacePresentModesKHR(
            self.physical_device, self.surface
        )

        # Pick B8G8R8A8_UNORM (storage-compatible) if available, else first
        chosen_format = formats[0]
        for fmt in formats:
            if (
                fmt.format == vk.VK_FORMAT_B8G8R8A8_UNORM
                and fmt.colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                chosen_format = fmt
                break

        # Prefer mailbox, fall back to FIFO
        chosen_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if vk.VK_PRESENT_MODE_MAILBOX_KHR in present_modes:
            chosen_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR

        extent = vk.VkExtent2D(width=self.width, height=self.height)
        image_count = capabilities.minImageCount + 1
        if capabilities.maxImageCount > 0:
            image_count = min(image_count, capabilities.maxImageCount)

        queue_indices = list(set(self.queue_family_indices.values()))
        sharing_mode = (
            vk.VK_SHARING_MODE_CONCURRENT
            if len(queue_indices) > 1
            else vk.VK_SHARING_MODE_EXCLUSIVE
        )

        create_info = vk.VkSwapchainCreateInfoKHR(
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=chosen_format.format,
            imageColorSpace=chosen_format.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=(
                vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT
            ),
            imageSharingMode=sharing_mode,
            queueFamilyIndexCount=len(queue_indices),
            pQueueFamilyIndices=queue_indices,
            preTransform=capabilities.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=chosen_mode,
            clipped=vk.VK_TRUE,
        )
        swapchain = self._vkCreateSwapchainKHR(self.device, create_info, None)

        images = self._vkGetSwapchainImagesKHR(self.device, swapchain)

        image_views = []
        for img in images:
            view_info = vk.VkImageViewCreateInfo(
                image=img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=chosen_format.format,
                components=vk.VkComponentMapping(
                    r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            image_views.append(vk.vkCreateImageView(self.device, view_info, None))

        return SwapchainInfo(
            swapchain=swapchain,
            images=images,
            image_views=image_views,
            format=chosen_format.format,
            extent=extent,
        )

    # ── Command pool ─────────────────────────────────────────────

    def _create_command_pool(self):
        pool_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.queue_family_indices["compute"],
        )
        return vk.vkCreateCommandPool(self.device, pool_info, None)

    def allocate_command_buffers(self, count: int = 1) -> list:
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=count,
        )
        return vk.vkAllocateCommandBuffers(self.device, alloc_info)

    # ── Cleanup ──────────────────────────────────────────────────

    def destroy(self) -> None:
        vk.vkDeviceWaitIdle(self.device)

        for view in self.swapchain_info.image_views:
            vk.vkDestroyImageView(self.device, view, None)
        self._vkDestroySwapchainKHR(self.device, self.swapchain_info.swapchain, None)
        vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        vk.vkDestroyDevice(self.device, None)
        self._vkDestroySurfaceKHR(self.instance, self.surface, None)
        vk.vkDestroyInstance(self.instance, None)
