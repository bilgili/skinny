"""Backend-selection seam shared by every front-end.

Two entry points:

- :func:`select_backend` resolves the active GPU backend name (``"vulkan"`` or
  ``"metal"``) from the ``--backend`` flag, the ``SKINNY_BACKEND`` environment
  variable, the persisted setting, and the ``auto`` default — in that precedence
  order.
- :func:`make_context` constructs the matching context object (a
  :class:`skinny.vk_context.VulkanContext` or
  :class:`skinny.metal_context.MetalContext`), both of which expose the same
  duck-typed surface the renderer consumes.

The renderer runs the full megakernel on both backends. ``auto`` resolves to
**Metal on Apple-Silicon macOS** (when a ``DeviceType.metal`` device constructs)
and to **Vulkan** everywhere else; an explicit ``--backend metal`` on a host
where the device cannot construct raises a clear error rather than degrading.
"""

from __future__ import annotations

import os
import platform
import sys

BACKEND_CHOICES = ("auto", "metal", "vulkan")


def metal_available() -> tuple[bool, str]:
    """Return ``(ok, reason)`` for whether a native Metal device can be built.

    ``ok`` is ``True`` only on Apple-Silicon macOS where SlangPy constructs a
    ``DeviceType.metal`` device. ``reason`` names the missing requirement when
    ``ok`` is ``False`` (used to build the clear error for an explicit but
    unavailable ``--backend metal``). The probe device is closed immediately.
    """
    if sys.platform != "darwin":
        return False, "native Metal requires macOS"
    if platform.machine() != "arm64":
        return False, "native Metal requires Apple Silicon (arm64)"
    try:
        import slangpy as spy
    except Exception as exc:  # noqa: BLE001 — any import failure ⇒ unavailable
        return False, f"slangpy is unavailable ({exc})"
    try:
        device = spy.create_device(type=spy.DeviceType.metal)
    except Exception as exc:  # noqa: BLE001 — device build failure ⇒ unavailable
        return False, f"the Metal device did not construct ({exc})"
    try:
        device.close()
    except Exception:  # noqa: BLE001 — best-effort cleanup of the probe device
        pass
    return True, ""


def select_backend(prefer: str | None = None, *, persisted: str | None = None) -> str:
    """Resolve the active backend name to ``"vulkan"`` or ``"metal"``.

    Precedence: explicit ``prefer`` (the ``--backend`` flag) > ``SKINNY_BACKEND``
    env > ``persisted`` setting > ``auto``. ``auto`` resolves to ``metal`` on
    Apple-Silicon macOS when a Metal device constructs, else ``vulkan``. An
    explicit ``metal`` request that cannot construct a Metal device raises a
    ``RuntimeError`` naming the missing requirement rather than degrading.
    """
    env = os.environ.get("SKINNY_BACKEND") or None
    choice = (prefer or env or persisted or "auto").strip().lower()
    if choice not in BACKEND_CHOICES:
        raise ValueError(
            f"unknown backend {choice!r} (expected one of {BACKEND_CHOICES})"
        )

    if choice == "vulkan":
        return "vulkan"

    if choice == "metal":
        ok, reason = metal_available()
        if not ok:
            raise RuntimeError(
                f"--backend metal requested but native Metal is unavailable: "
                f"{reason}. Use --backend vulkan (or 'auto')."
            )
        return "metal"

    # auto → metal on Apple-Silicon macOS when the device constructs, else vulkan.
    ok, _reason = metal_available()
    return "metal" if ok else "vulkan"


def make_context(
    backend: str,
    window=None,
    width: int = 1280,
    height: int = 720,
    **kw,
):
    """Construct the context for ``backend`` (``"vulkan"`` or ``"metal"``).

    Extra keyword arguments (``gpu_preference``, ``enable_validation``) are
    forwarded to the chosen context, which both accept the same names. Imports
    are deferred so a host without one backend's runtime never imports it.
    """
    backend = backend.strip().lower()
    if backend == "vulkan":
        from skinny.vk_context import VulkanContext

        return VulkanContext(window, width, height, **kw)
    if backend == "metal":
        from skinny.metal_context import MetalContext

        return MetalContext(window, width, height, **kw)
    raise ValueError(f"unknown backend {backend!r} (expected 'vulkan' or 'metal')")


def resource_module(ctx):
    """Return the GPU-resource module matching ``ctx``'s backend.

    The renderer builds its GPU resources (``StorageBuffer``, ``StorageImage``,
    ``SampledImage``, ``UniformBuffer``, ``ComputePipeline``, …) through one of
    two sibling modules that expose the **same public API**:
    :mod:`skinny.vk_compute` for a :class:`~skinny.vk_context.VulkanContext` and
    :mod:`skinny.metal_compute` for a :class:`~skinny.metal_context.MetalContext`.
    Resolving the module once from ``ctx.is_metal`` keeps the construction sites
    backend-agnostic (no per-site ``if ctx.is_metal`` branch). Imports are
    deferred so a host without one backend's runtime never imports it (notably
    ``vk_compute`` pulls in the ``vulkan`` extension, which the Metal path must
    not require).
    """
    if getattr(ctx, "is_metal", False):
        from skinny import metal_compute

        return metal_compute
    from skinny import vk_compute

    return vk_compute
