"""Native Metal context built on SlangPy's ``DeviceType.metal``.

``MetalContext`` is the Metal sibling of :class:`skinny.vk_context.VulkanContext`.
It constructs a native Metal device through SlangPy / slang-rhi (no MoltenVK, no
raw PyObjC) and exposes the **same duck-typed surface the renderer reads off the
Vulkan context** — ``width`` / ``height``, ``compute_queue`` / ``present_queue``,
``swapchain_info``, ``gpu_info``, ``allocate_command_buffers``,
``recreate_swapchain``, ``destroy``, the ``backend_name`` / ``is_metal``
predicate, and the capability flags — so later phases can drive it duck-typed
without Metal-specific knowledge
at the context layer.

**Foundation phase (P1).** This covers device bring-up, a trivial compute
dispatch (via :mod:`skinny.metal_compute`), and a windowed clear + present only.
The full renderer (megakernel head render, materials, ReSTIR, neural, wavefront)
is not yet ported to Metal and is staged in later changes. The module is import-
and platform-guarded so non-Apple-Silicon hosts never construct it.

Present path (resolves design O2): slang-rhi exposes a ``Surface`` we drive
directly — ``configure`` / ``acquire_next_image`` / ``present`` — so there is no
manual ``CAMetalLayer``. A GLFW window is bridged via its Cocoa ``NSWindow``
pointer (``glfw.get_cocoa_window`` → ``WindowHandle(nswindow=…)``); a
``slangpy.Window`` is passed straight through.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass
class MetalSwapchainInfo:
    """Duck-typed analogue of :class:`skinny.vk_context.SwapchainInfo`.

    Wraps the slang-rhi ``Surface`` plus its current configuration. The renderer
    only reads the shape (extent + format) in the paths that exist today; the
    ``Surface`` itself owns the image ring.
    """

    surface: object
    format: object
    width: int
    height: int


@dataclass
class MetalGpuInfo:
    """Duck-typed analogue of :class:`skinny.hardware.GpuInfo` for the Metal path.

    The Qt / web front-ends and the video encoder read three fields off
    ``ctx.gpu_info`` — ``name``, ``is_discrete``, and ``preferred_h264_encoder``.
    Apple-Silicon GPUs are unified-memory (integrated, never discrete) and encode
    H.264 through VideoToolbox, so the latter two are fixed; ``name`` comes from
    slang-rhi's ``Device.info.adapter_name`` (e.g. ``"Apple M5 Pro"``). Kept local
    to this module so the Metal backend never imports :mod:`skinny.hardware`,
    which pulls in the ``vulkan`` extension.
    """

    name: str
    is_discrete: bool = False
    preferred_h264_encoder: str = "h264_videotoolbox"


class MetalContext:
    """Manages the native Metal device + surface for compute-based rendering."""

    backend_name = "metal"
    is_metal = True

    # Conservative foundation-phase capability flags (design D5): the renderer
    # stays on its fp32 and file-handoff paths on Metal until the shared-storage
    # MTLBuffer / MTLSharedEvent / fp16 equivalents arrive in later phases.
    supports_external_memory = False
    supports_external_semaphore = False
    supports_fp16_storage = False
    supports_fp16_compute = False

    def __init__(
        self,
        window=None,
        width: int = 1280,
        height: int = 720,
        *,
        enable_validation: bool = False,
        gpu_preference: str | None = None,
    ) -> None:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "MetalContext requires Apple-Silicon macOS "
                f"(got platform={sys.platform!r} machine={platform.machine()!r})"
            )
        import slangpy as spy

        self._spy = spy
        self.width = int(width)
        self.height = int(height)
        self._headless = window is None
        # gpu_preference / enable_validation are accepted for surface parity with
        # VulkanContext; slang-rhi picks the system default Metal device and owns
        # its own validation, so they are no-ops here in P1.

        self.device = spy.create_device(type=spy.DeviceType.metal)

        # Duck-typed parity with VulkanContext.gpu_info — front-ends read
        # gpu_info.name and the encoder reads preferred_h264_encoder. slang-rhi
        # owns device selection (gpu_preference is a P1 no-op), so the name comes
        # from its reported adapter.
        self.gpu_info = MetalGpuInfo(name=self.device.info.adapter_name)
        print(f"[GPU] Selected: {self.gpu_info.name} (Metal)")

        # SlangPy records and submits work on the device's implicit queue
        # (ComputeKernel.dispatch / submit_command_buffer); there are no separate
        # VkQueue handles to expose. Mirror the VulkanContext attribute names as
        # duck-typed placeholders so the surface shape matches.
        self.compute_queue = None
        self.present_queue = None

        self.surface = None
        self.swapchain_info = None
        if not self._headless:
            self.surface = self._create_surface(window)
            self.swapchain_info = self._configure_surface(self.width, self.height)

    # ── Surface ──────────────────────────────────────────────────

    def _native_window_handle(self, window):
        import glfw

        nswindow = glfw.get_cocoa_window(window)
        if not nswindow:
            raise RuntimeError("glfw.get_cocoa_window returned a null NSWindow")
        return self._spy.WindowHandle(nswindow=int(nswindow))

    def _create_surface(self, window):
        spy = self._spy
        # A slangpy.Window is passed straight through; anything else is treated
        # as a GLFW window and bridged via its Cocoa NSWindow pointer.
        if isinstance(window, spy.Window):
            return self.device.create_surface(window)
        return self.device.create_surface(self._native_window_handle(window))

    def _surface_format(self):
        spy = self._spy
        info = self.surface.info
        fmt = info.preferred_format
        if fmt == spy.Format.undefined and info.formats:
            fmt = info.formats[0]
        return fmt

    def _configure_surface(self, width: int, height: int) -> MetalSwapchainInfo:
        spy = self._spy
        fmt = self._surface_format()
        self.surface.configure(
            width=int(width),
            height=int(height),
            format=fmt,
            usage=spy.TextureUsage.render_target | spy.TextureUsage.copy_destination,
            vsync=True,
        )
        return MetalSwapchainInfo(
            surface=self.surface, format=fmt, width=int(width), height=int(height)
        )

    # ── Command buffers ──────────────────────────────────────────

    def allocate_command_buffers(self, count: int = 1) -> list:
        """Surface parity with :meth:`VulkanContext.allocate_command_buffers`.

        slang-rhi records through fresh ``CommandEncoder`` objects (finished into
        a ``CommandBuffer`` and submitted), so return that many encoders.
        """
        return [self.device.create_command_encoder() for _ in range(count)]

    # ── Present (foundation proof) ───────────────────────────────

    def present_clear(self, color=(0.0, 0.0, 0.0, 1.0)) -> bool:
        """Acquire the next swapchain image, clear it to ``color``, and present.

        Returns ``True`` when a frame was presented, ``False`` when the swapchain
        had no image ready this tick. Raises on a headless context. This is the
        windowed present foundation proof — the GPU fence must signal every frame
        (no per-field cursor writes are made anywhere on the Metal path; see
        :mod:`skinny.metal_compute` for the ``set_data`` byte-blob discipline).
        """
        if self.surface is None:
            raise RuntimeError("present_clear requires a windowed MetalContext")
        spy = self._spy
        image = self.surface.acquire_next_image()
        if image is None:
            return False
        encoder = self.device.create_command_encoder()
        encoder.clear_texture_float(image, clear_value=spy.math.float4(*color))
        self.device.submit_command_buffer(encoder.finish())
        self.surface.present()
        self.device.wait_for_idle()
        return True

    # ── Sync ─────────────────────────────────────────────────────

    def wait_idle(self) -> None:
        """Block until the device has finished all submitted work.

        Backend-neutral seam (mirrors :meth:`VulkanContext.wait_idle`) so the
        renderer can call ``self.ctx.wait_idle()`` without branching on
        ``is_metal``. Delegates to slang-rhi's ``Device.wait_for_idle``.
        """
        self.device.wait_for_idle()

    # ── Swapchain lifecycle ──────────────────────────────────────

    def recreate_swapchain(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        if self.surface is not None:
            self.swapchain_info = self._configure_surface(width, height)

    # ── Cleanup ──────────────────────────────────────────────────

    def destroy(self) -> None:
        try:
            self.device.wait_for_idle()
        except Exception:  # noqa: BLE001 — best-effort drain before teardown
            pass
        if self.surface is not None:
            try:
                self.surface.unconfigure()
            except Exception:  # noqa: BLE001
                pass
            self.surface = None
        self.swapchain_info = None
        try:
            self.device.close()
        except Exception:  # noqa: BLE001 — device may already be torn down
            pass
