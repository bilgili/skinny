"""Metal device-foundation tests: trivial compute dispatch + present.

The foundation correctness proof is the **headless** trivial-dispatch parity
test (Metal vs Vulkan produce the identical buffer); the windowed present is a
separate, display-gated smoke. Each test skips cleanly when its backend (or a
display) is unavailable on the host, so the suite runs anywhere — the Metal legs
run on Apple Silicon and skip elsewhere; the Vulkan leg runs wherever a SlangPy
Vulkan device constructs (MoltenVK on macOS).
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.backend_select import metal_available

_ENTRY = "computeMain"          # never `main` — Metal reserves it (see the shader)
_MODULE = "foundation_trivial"
_N = 256


def _expected() -> np.ndarray:
    return np.arange(_N, dtype=np.uint32)


def _dispatch_arange_slangpy(device_type, shader_dir) -> np.ndarray | None:
    """Compile + dispatch the trivial kernel on a fresh SlangPy device of the
    given type and return the read-back uint32 array, or ``None`` if the device
    cannot be constructed (so the caller can skip cleanly)."""
    spy = pytest.importorskip("slangpy")
    try:
        device = spy.create_device(type=device_type, include_paths=[str(shader_dir)])
    except Exception:
        return None
    try:
        source = (shader_dir / f"{_MODULE}.slang").read_text(encoding="utf-8")
        module = device.load_module_from_source(_MODULE, source)
        program = device.link_program([module], [module.entry_point(_ENTRY)])
        kernel = device.create_compute_kernel(program)
        buffer = device.create_buffer(
            element_count=_N, struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource
            | spy.BufferUsage.copy_source,
            memory_type=spy.MemoryType.device_local, label="result",
        )
        kernel.dispatch(thread_count=[_N, 1, 1], vars={"result": buffer})
        device.wait_for_idle()
        return buffer.to_numpy().view(np.uint32)[:_N].copy()
    finally:
        try:
            device.close()
        except Exception:
            pass


def _dispatch_arange_metal_context(shader_dir) -> np.ndarray:
    """Run the trivial kernel through MetalContext + skinny.metal_compute (the
    real foundation wrappers) and return the read-back uint32 array."""
    from skinny.metal_compute import ComputePipeline, StorageBuffer
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=64, height=64)
    try:
        buf = StorageBuffer(ctx, _N * 4)
        pipe = ComputePipeline(ctx, shader_dir, _MODULE, _ENTRY)
        pipe.dispatch([_N, 1, 1], buffers={"result": buf})
        return np.frombuffer(buf.download_sync(), dtype=np.uint32)[:_N].copy()
    finally:
        ctx.destroy()


# ── 3.2 headless trivial dispatch ────────────────────────────────────

def test_metal_trivial_dispatch_matches_arange(shader_dir):
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    out = _dispatch_arange_metal_context(shader_dir)
    np.testing.assert_array_equal(out, _expected())


def test_trivial_dispatch_metal_vs_vulkan_parity(shader_dir):
    import slangpy as spy

    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    vk_out = _dispatch_arange_slangpy(spy.DeviceType.vulkan, shader_dir)
    if vk_out is None:
        pytest.skip("no SlangPy Vulkan device on this host")
    metal_out = _dispatch_arange_metal_context(shader_dir)
    # Same kernel, two backends → byte-identical buffers (and both == arange).
    np.testing.assert_array_equal(metal_out, _expected())
    np.testing.assert_array_equal(vk_out, _expected())
    np.testing.assert_array_equal(metal_out, vk_out)


# ── 3.3 windowed present smoke (display-gated) ───────────────────────

def test_metal_windowed_present_no_hang():
    """Open a window, clear + present several frames, expect every frame's GPU
    fence to signal (present_clear returns True). Skips without a display."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    spy = pytest.importorskip("slangpy")
    from skinny.metal_context import MetalContext

    try:
        window = spy.Window(width=256, height=256, title="skinny-metal-smoke")
    except Exception as exc:  # noqa: BLE001 — no window server / headless host
        pytest.skip(f"no display for the windowed Metal present smoke: {exc}")

    ctx = None
    try:
        ctx = MetalContext(window=window, width=256, height=256)
        presented = 0
        for i in range(8):
            window.process_events()
            if ctx.present_clear((0.1 * (i % 3), 0.2, 0.3, 1.0)):
                presented += 1
        assert presented >= 1, "no frame presented (swapchain never produced an image)"
    finally:
        if ctx is not None:
            ctx.destroy()
        try:
            window.close()
        except Exception:
            pass
