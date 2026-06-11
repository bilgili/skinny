"""Native Metal context built on SlangPy's ``DeviceType.metal``.

``MetalContext`` is the Metal sibling of :class:`skinny.vk_context.VulkanContext`.
It constructs a native Metal device through SlangPy / slang-rhi (no MoltenVK, no
raw PyObjC) and exposes the **same duck-typed surface the renderer reads off the
Vulkan context** — ``width`` / ``height``, ``compute_queue`` / ``present_queue``,
``swapchain_info``, ``allocate_command_buffers``, ``recreate_swapchain``,
``destroy``, the ``backend_name`` / ``is_metal`` predicate, and the capability
flags — so later phases can drive it duck-typed without Metal-specific knowledge
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


class MetalContext:
    """Manages the native Metal device + surface for compute-based rendering."""

    backend_name = "metal"
    is_metal = True

    # External-memory / -semaphore interop stays off on Metal: frozen neural
    # weights load by buffer upload, not a GPU↔GPU handoff (design D6), so the
    # renderer keeps its file-handoff path here.
    supports_external_memory = False
    supports_external_semaphore = False
    # fp16 flags are class-level defaults; ``__init__`` overrides them per device
    # from the slang-rhi feature probe (``_probe_fp16``). They stay ``False`` until
    # a constructed device reports ``Feature.half``.
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

        # Probe fp16 support on the real device (design D6 / task 1.1). slang-rhi
        # exposes a single ``Feature.half`` covering half-precision storage +
        # compute, so both flags follow it. NOTE: slang-rhi 0.42's Metal backend
        # under-reports this — ``has_feature(half)`` is ``False`` even on Apple
        # Silicon, which natively supports MSL ``half`` — so the renderer
        # conservatively stays on fp32 (D6's "device without fp16 → fp32" branch).
        # Enabling fp16 neural storage despite the flag (an empirical compile
        # probe) is deferred to the neural phase.
        fp16 = self._probe_fp16(self.device)
        self.supports_fp16_storage = fp16
        self.supports_fp16_compute = fp16

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

    # ── Capability probes ────────────────────────────────────────

    @staticmethod
    def _probe_fp16(device) -> bool:
        """Whether the slang-rhi device reports half-precision support.

        Returns ``device.has_feature(Feature.half)``, guarded so a slangpy build
        without that enum value (or a device that raises) degrades to ``False``
        rather than crashing context construction.
        """
        import slangpy as spy

        half = getattr(spy.Feature, "half", None)
        if half is None:
            return False
        try:
            return bool(device.has_feature(half))
        except Exception:  # noqa: BLE001 — unknown feature ⇒ treat as unsupported
            return False

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
