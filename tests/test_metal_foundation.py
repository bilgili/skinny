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
    buf = pipe = None
    try:
        buf = StorageBuffer(ctx, _N * 4)
        pipe = ComputePipeline(ctx, shader_dir, _MODULE, _ENTRY)
        pipe.dispatch_kernel([_N, 1, 1], buffers={"result": buf})
        return np.frombuffer(buf.download_sync(_N * 4), dtype=np.uint32)[:_N].copy()
    finally:
        # Explicit per-resource teardown before the device closes — repeated
        # device/pipeline churn without this is the Metal-test leak vector.
        if pipe is not None:
            pipe.destroy()
        if buf is not None:
            buf.destroy()
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


# ── device drain seam (wait_idle) ────────────────────────────────────

def test_metal_wait_idle_runs():
    """`MetalContext.wait_idle()` is the backend-neutral device drain the
    renderer's pipeline-rebuild path calls instead of `vk.vkDeviceWaitIdle`
    (which crashes on a Metal device). Prove the seam exists and drains a
    fresh device without error. No ComputePipeline → no megakernel compile →
    no MTLCompilerService spike, so this is safe to run unguarded."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=64, height=64)
    try:
        assert ctx.is_metal is True
        ctx.wait_idle()  # must not raise; routes to Device.wait_for_idle
    finally:
        ctx.destroy()


# ── 1.6 indirect-dispatch primitive + capability flags ───────────────

def _dispatch_arange_indirect(shader_dir, groups: int):
    """Run the trivial kernel through ``ComputePipeline.dispatch_indirect`` with a
    GPU-side ``(groups, 1, 1)`` args triple. ``numthreads`` is 64, so ``groups``
    workgroups write ``arange(64 * groups)``. Returns ``(out, supported)`` where
    ``supported`` is the device's ``supports_indirect_dispatch`` probe result, so
    the caller can assert the native and CPU-readback-fallback paths agree."""
    from skinny.metal_compute import ComputePipeline, StorageBuffer
    from skinny.metal_context import MetalContext

    n = 64 * groups
    ctx = MetalContext(window=None, width=64, height=64)
    res = args = pipe = None
    try:
        res = StorageBuffer(ctx, n * 4)
        args = StorageBuffer(ctx, 12, indirect=True)
        args.upload_sync(np.array([groups, 1, 1], dtype=np.uint32).tobytes())
        pipe = ComputePipeline(ctx, shader_dir, _MODULE, _ENTRY)
        pipe.dispatch_indirect(args, 0, bindings={"result": res})
        out = np.frombuffer(res.download_sync(n * 4), dtype=np.uint32)[:n].copy()
        return out, bool(ctx.supports_indirect_dispatch)
    finally:
        if pipe is not None:
            pipe.destroy()
        if args is not None:
            args.destroy()
        if res is not None:
            res.destroy()
        ctx.destroy()


def test_metal_dispatch_indirect_matches_direct(shader_dir):
    """The indirect dispatch (native or CPU-readback fallback, selected by the
    one-time ``supports_indirect_dispatch`` probe) issues the same group count as a
    direct dispatch — both produce ``arange``."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    indirect, _supported = _dispatch_arange_indirect(shader_dir, groups=4)  # 4*64=256
    direct = _dispatch_arange_metal_context(shader_dir)                     # _N == 256
    expected = np.arange(_N, dtype=np.uint32)
    np.testing.assert_array_equal(indirect, expected)
    np.testing.assert_array_equal(indirect, direct)


def test_metal_capability_flags_reflect_device():
    """fp16 + indirect-dispatch capability flags are real booleans probed from the
    device (design D2/D6), not left as ``None``/unset."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=64, height=64)
    try:
        for flag in ("supports_fp16_storage", "supports_fp16_compute",
                     "supports_indirect_dispatch"):
            assert isinstance(getattr(ctx, flag), bool), flag
        # External interop stays off on Metal (design D6 non-goal).
        assert ctx.supports_external_memory is False
        assert ctx.supports_external_semaphore is False
    finally:
        ctx.destroy()


# ── gpu_info duck-typed parity ───────────────────────────────────────

def test_metal_context_exposes_gpu_info():
    """`MetalContext.gpu_info` mirrors `VulkanContext.gpu_info` — the Qt/web
    front-ends read `gpu_info.name` and the video encoder reads
    `gpu_info.preferred_h264_encoder`. A missing attribute crashes window
    construction (`AttributeError: 'MetalContext' object has no attribute
    'gpu_info'`). No ComputePipeline → no megakernel compile, safe unguarded."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=64, height=64)
    try:
        gi = ctx.gpu_info
        assert isinstance(gi.name, str) and gi.name  # e.g. "Apple M5 Pro"
        assert gi.is_discrete is False               # Apple Silicon = unified mem
        assert gi.preferred_h264_encoder == "h264_videotoolbox"
    finally:
        ctx.destroy()


# ── 1.4 single-frame multi-pass encoder + barrier ────────────────────

def test_metal_frame_encoder_multipass_barrier(shader_dir):
    """Two compute stages encoded into ONE command encoder with a global barrier
    between them and a SINGLE submit: stage 1 writes ``tid``, stage 2 adds 1000.
    The result is ``arange(N) + 1000`` iff stage 2 observed stage 1's writes —
    proving the multi-pass encoder + barrier replace per-stage ``wait_for_idle``."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    from skinny.metal_compute import ComputePipeline, MetalFrameEncoder, StorageBuffer
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=64, height=64)
    buf = write = bias = None
    try:
        buf = StorageBuffer(ctx, _N * 4)
        write = ComputePipeline(ctx, shader_dir, _MODULE, _ENTRY)        # computeMain
        bias = ComputePipeline(ctx, shader_dir, _MODULE, "addBias")
        groups = (_N // 64, 1, 1)  # 64 threads/group

        enc = MetalFrameEncoder(ctx)
        enc.dispatch(write, groups, bindings={"result": buf})
        enc.barrier()
        enc.dispatch(bias, groups, bindings={"result": buf})
        enc.submit()

        out = np.frombuffer(buf.download_sync(_N * 4), dtype=np.uint32)[:_N]
        np.testing.assert_array_equal(out, np.arange(_N, dtype=np.uint32) + 1000)
    finally:
        for obj in (write, bias, buf):
            if obj is not None:
                obj.destroy()
        ctx.destroy()


# ── image-format token resolution ────────────────────────────────────

def test_metal_format_tokens_resolve_to_real_slangpy_formats():
    """Every value in `_FORMAT_TOKENS` / `_VKFORMAT_INTS` must name a real
    `slangpy.Format` member, else `_resolve_format`'s `getattr(spy.Format, …)`
    raises `AttributeError: type object 'Format' has no attribute '…'` at texture-
    load / render time (slang-rhi spells 8-bit sRGB `rgba8_unorm_srgb`, not the
    `rgba8_srgb` neutral token). Pure enum-name check — no device, no compile."""
    spy = pytest.importorskip("slangpy")
    from skinny.metal_compute import _FORMAT_TOKENS, _VKFORMAT_INTS, _resolve_format

    for name in (*_FORMAT_TOKENS.values(), *_VKFORMAT_INTS.values()):
        assert hasattr(spy.Format, name), f"slangpy.Format has no member {name!r}"

    # The sRGB neutral token and VkFormat R8G8B8A8_SRGB (43) both land on the
    # slang-rhi sRGB format — this is the exact regression that crashed the
    # three-materials USD scene render on Metal.
    assert _resolve_format(spy, "rgba8_srgb") == spy.Format.rgba8_unorm_srgb
    assert _resolve_format(spy, 43) == spy.Format.rgba8_unorm_srgb


# ── Metal no-op for Vulkan-only descriptor writes ────────────────────

def test_update_texture_pool_descriptors_noop_on_metal():
    """`_update_texture_pool_descriptors` writes Vulkan descriptor set 14; on
    Metal `descriptor_sets is None` and the pool is bound by name at dispatch, so
    it must short-circuit. `_upload_flat_materials` calls it on every material
    upload — without the guard the Metal USD-scene render crashes with
    `TypeError: 'NoneType' object is not iterable`. Stub via `__new__` (no device,
    no megakernel compile); `descriptor_sets` left None to prove it is untouched."""
    try:
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path (renderer imports it)
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    r = Renderer.__new__(Renderer)  # bypass __init__ — no GPU/context needed
    r.is_metal = True
    r.descriptor_sets = None
    r._update_texture_pool_descriptors()  # must return without iterating None


def test_rebind_descriptors_noop_on_metal():
    """`_rebind_scene_descriptors` / `_rebind_aux_material_descriptors` rewrite
    Vulkan descriptor sets after a buffer realloc; on Metal `_build_metal_binds`
    re-reads each live buffer at dispatch, so both must short-circuit. They run on
    the instance/material grow path (e.g. a 20+-instance USD scene) — without the
    guard the Vulkan `VkDescriptorBufferInfo(buffer=<slangpy buffer>)` call raises
    `TypeError: an integer is required`. Stub via `__new__`, no buffers set, to
    prove the methods return before touching any Vulkan/buffer attribute."""
    try:
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path (renderer imports it)
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    r = Renderer.__new__(Renderer)
    r.is_metal = True
    r._rebind_scene_descriptors()        # must return before reading instance_buffer
    r._rebind_aux_material_descriptors()  # must return before reading std_surface_buffer


def test_debug_viewport_refuses_metal_cleanly():
    """The debug viewport is a Vulkan **graphics** rasteriser; the compute-only
    Metal backend can't build it. `open()` must raise a clear RuntimeError on a
    Metal context (not a cryptic `AttributeError: 'MetalContext' object has no
    attribute 'queue_family_indices'`) and leave the viewport closed, so the Qt
    dock's showEvent degrades gracefully. Stub via `__new__` — no GPU."""
    import types

    try:
        from skinny.debug_viewport import DebugViewport
    except OSError as exc:  # libvulkan not on the dylib path (module imports vulkan)
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    dv = DebugViewport.__new__(DebugViewport)
    dv._vk_ctx = types.SimpleNamespace(is_metal=True)
    dv._open = False
    dv._embedded = True
    with pytest.raises(RuntimeError, match="Vulkan backend"):
        dv.open()
    assert dv._open is False


def test_bxdf_eval_degrades_on_metal():
    """`request_bxdf_eval` (BXDF/BSSRDF visualiser) does a Vulkan-only one-shot
    readback (`command_pool` / `vkDeviceWaitIdle` / `descriptor_sets`). On Metal
    `self.pipeline` is non-None, so the old `pipeline is None` guard wouldn't catch
    it and it crashed on `self.ctx.command_pool`. It must instead degrade to a
    zeroed grid via the callback. Stub via `__new__`, pipeline non-None, is_metal
    True — proves it never reaches the Vulkan path (no ctx needed)."""
    try:
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path (renderer imports it)
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    r = Renderer.__new__(Renderer)
    r.is_metal = True
    r.pipeline = object()  # non-None: only the is_metal arm may divert control
    got = []
    r.request_bxdf_eval({"n_theta": 4, "n_phi": 3}, got.append)
    assert len(got) == 1
    assert got[0].shape == (4, 3, 3)
    assert not got[0].any()  # all zeros — graceful empty grid


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
