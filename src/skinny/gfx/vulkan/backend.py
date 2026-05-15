"""VulkanBackend — instance + physical device + Device + Presenter.

Replaces the Step 1 stub. Constructs everything in ``create()``; callers
get a ready-to-use Backend handle.

Existing ``skinny.vk_context.VulkanContext`` is left untouched — Step 4
will switch ``renderer.py`` over to this backend; until then both paths
coexist.
"""

from __future__ import annotations

from typing import Literal

import vulkan as vk

from skinny.gfx.backend import Backend, BackendCaps
from skinny.gfx.vulkan.device import VulkanDevice
from skinny.gfx.vulkan.presenter import VulkanPresenter
from skinny.hardware import GpuInfo, select_gpu


_VALIDATION_LAYERS = ["VK_LAYER_KHRONOS_validation"]


class VulkanBackend(Backend):
    name = "vulkan"

    def __init__(
        self,
        instance,
        gpu: GpuInfo,
        device: VulkanDevice,
        presenter: VulkanPresenter | None,
        init_width: int,
        init_height: int,
    ) -> None:
        self.instance = instance
        self.gpu = gpu
        self.physical_device = gpu.vk_physical_device
        self._device = device
        self._presenter = presenter
        self._init_width = init_width
        self._init_height = init_height
        self._caps = BackendCaps(
            bindless_textures=True,
            scalar_block_layout=True,
            push_descriptors=False,
            max_storage_buffer_bindings=64,
            max_compute_workgroup_size=(1024, 1024, 64),
        )

    # ── Construction ───────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        window=None,
        width: int = 1280,
        height: int = 720,
        enable_validation: bool = True,
        gpu_preference: str | None = None,
    ) -> "VulkanBackend":
        instance = cls._create_instance(window, enable_validation)
        gpu = select_gpu(instance, gpu_preference)
        physical_device = gpu.vk_physical_device

        compute_family, present_family = cls._find_queue_families(
            instance, physical_device, window,
        )

        # Device creation has to happen before the presenter because the
        # presenter needs a logical device to load the swapchain entry
        # points off of. We pass `present_family=None` when there's no
        # window, which short-circuits the swapchain extension request.
        device = VulkanDevice(
            backend=None,  # set below
            physical_device=physical_device,
            compute_family=compute_family,
            present_family=present_family if window is not None else None,
        )
        backend = cls(
            instance=instance,
            gpu=gpu,
            device=device,
            presenter=None,
            init_width=width,
            init_height=height,
        )
        device._backend = backend

        if window is not None:
            backend._presenter = VulkanPresenter(backend, window)
        return backend

    # ── Instance + queue families (lifted from vk_context.py) ──────

    @staticmethod
    def _create_instance(window, enable_validation: bool):
        app_info = vk.VkApplicationInfo(
            pApplicationName="Skinny",
            applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            pEngineName="Skinny Engine",
            engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0),
        )
        if window is None:
            extensions = []
        else:
            import glfw
            extensions = glfw.get_required_instance_extensions()

        layers = _VALIDATION_LAYERS if enable_validation else []
        info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
        )
        return vk.vkCreateInstance(info, None)

    @staticmethod
    def _find_queue_families(instance, physical_device, window):
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
        compute = None
        present = None

        if window is None:
            for i, family in enumerate(families):
                if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    return i, None
            raise RuntimeError("No compute queue family found")

        # Windowed: need surface support too. Build a temporary surface
        # only to query support, then destroy it; the real surface lives
        # on the presenter.
        import ctypes
        get = vk.vkGetInstanceProcAddr
        query = get(instance, "vkGetPhysicalDeviceSurfaceSupportKHR")
        destroy = get(instance, "vkDestroySurfaceKHR")
        import glfw
        surface = ctypes.c_void_p(0)
        if glfw.create_window_surface(
            int(vk.ffi.cast("uintptr_t", instance)),
            window, None, ctypes.byref(surface),
        ) != vk.VK_SUCCESS:
            raise RuntimeError("Failed to create probe surface")
        probe_surface = vk.ffi.cast("VkSurfaceKHR", surface.value)
        try:
            for i, family in enumerate(families):
                if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    compute = i
                if query(physical_device, i, probe_surface):
                    present = i
                if compute is not None and present is not None:
                    break
        finally:
            destroy(instance, probe_surface, None)

        if compute is None or present is None:
            raise RuntimeError(
                f"Required queue families not found: compute={compute}, present={present}"
            )
        return compute, present

    # ── Backend interface ──────────────────────────────────────────

    @property
    def caps(self) -> BackendCaps:
        return self._caps

    @property
    def device(self) -> VulkanDevice:
        return self._device

    @property
    def presenter(self) -> VulkanPresenter | None:
        return self._presenter

    def shader_target(self) -> Literal["spirv", "metal"]:
        return "spirv"

    def destroy(self) -> None:
        self._device.wait_idle()
        if self._presenter is not None:
            self._presenter.destroy()
            self._presenter = None
        self._device.destroy()
        vk.vkDestroyInstance(self.instance, None)
