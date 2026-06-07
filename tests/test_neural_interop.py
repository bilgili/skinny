"""Interop CUDA weight-handoff (change neural-online-training, task 5.2).

The CUDA *import* half of ``--neural-handoff interop``: CUDA imports the
Vulkan-exported weight buffers (bindings 33/34) and the exported timeline
semaphore, writes new weights straight into GPU memory, and the renderer's
frame-end swap waits the semaphore — no CPU round-trip.

These tests are hardware-bound: they need a CUDA device, ``cuda-python``, and a
Vulkan device that advertises ``VK_KHR_external_memory`` (+ external semaphore).
Every test skips cleanly where any of those is absent, so the suite stays green
on Mac/MoltenVK and torch-/CUDA-free boxes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skinny.sampling import neural_handoff_interop as interop
from skinny.sampling.neural_handoff_interop import (
    InteropWeightPublisher,
    interop_available,
)
from skinny.sampling.neural_weights import NeuralBuildConfig, make_dummy_weights
from skinny.sampling.path_records import RECORD_DTYPE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SCENE = PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"


def _records(n: int, tag: float) -> np.ndarray:
    r = np.zeros(n, dtype=RECORD_DTYPE)
    r["pos"] = tag
    r["wi_local"] = [0.0, 0.0, 1.0]
    r["contrib"] = 1.0
    return r


def _require_cuda():
    ok, reason = interop_available()
    if not ok:
        pytest.skip(reason)


def _vk_external():
    """A headless VulkanContext with external memory, or skip."""
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=16, height=16)
    if not getattr(ctx, "supports_external_memory", False):
        ctx.destroy()
        pytest.skip("device lacks VK_KHR_external_memory — interop unavailable")
    return ctx


def _vk_external_sema():
    """A headless VulkanContext with external timeline-semaphore support, or skip."""
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=16, height=16)
    if not getattr(ctx, "supports_external_semaphore", False):
        ctx.destroy()
        pytest.skip("device lacks external timeline-semaphore support")
    return ctx


def test_timeline_semaphore_cuda_signal_host_wait():
    """A Vulkan-exported timeline semaphore, imported into CUDA, can be signalled
    from a CUDA stream to a value the Vulkan host wait then observes — the
    tear-free ordering primitive for the CUDA-write → Vulkan-read swap."""
    _require_cuda()
    from skinny.vk_compute import ExternalTimelineSemaphore

    ctx = _vk_external_sema()
    try:
        sema = ExternalTimelineSemaphore(ctx)
        try:
            if sema.export_handle() is None:
                pytest.skip("semaphore export handle unavailable on this device")
            assert sema.value() == 0
            cuts = interop.CudaExternalTimeline.from_semaphore(sema)
            try:
                cuts.signal(7)
                sema.wait(7, timeout_ns=2_000_000_000)
                assert sema.value() >= 7
            finally:
                cuts.close()
        finally:
            sema.destroy()
    finally:
        ctx.destroy()


def test_renderer_interop_online_loop():
    """End-to-end on the renderer: under ``--neural-handoff interop`` the online
    loop trains (placeholder is fine — we exercise the handoff, not the optimiser),
    publishes weights straight into the CUDA-shared buffers, and the frame-end swap
    host-waits the timeline and bumps ``networkVersion`` — no CPU re-upload."""
    _require_cuda()
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=64, height=64)
    try:
        if not (getattr(ctx, "supports_external_memory", False)
                and getattr(ctx, "supports_external_semaphore", False)):
            pytest.skip("device lacks external memory + timeline semaphore")
        r = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR, tattoo_dir=TATTOO_DIR,
            usd_scene_path=SCENE, execution_mode="wavefront", neural_handoff="interop",
        )
        deadline = 200
        while deadline > 0 and (
            r._usd_scene is None or len(r._usd_scene.instances) < 1
            or r._scene_bindings is None
        ):
            r.update(0.025)
            deadline -= 1
        assert r._scene_bindings is not None, "scene bindings never built"

        assert r.neural_weights_buffer.external
        assert r.neural_timeline_semaphore is not None

        r.enable_online_training(handoff="interop")
        assert isinstance(r._neural_publisher, InteropWeightPublisher)
        assert r._neural_network_version == 0

        rng = np.random.default_rng(0)
        r._neural_replay.add(_records(2_000, tag=1.0))
        version = r.online_train_and_publish(rng)
        assert version == 1
        assert r._online_frame_end_swap() is True
        assert r._neural_network_version == 1

        # A second cycle keeps advancing the version through the CUDA handoff.
        r._neural_replay.add(_records(2_000, tag=2.0))
        assert r.online_train_and_publish(rng) == 2
        assert r._online_frame_end_swap() is True
        assert r._neural_network_version == 2

        r._neural_publisher.close()
    finally:
        ctx.destroy()


def test_interop_publish_swap_roundtrip():
    """The full CUDA half: publish() writes weights+biases straight into the
    Vulkan-exported buffers and signals the timeline; swap() host-waits it and
    advances the version; acquire_for_render() returns no host weights (they live
    in GPU memory). The bytes in the exported buffers match the published net."""
    _require_cuda()
    from skinny.vk_compute import ExternalTimelineSemaphore, StorageBuffer

    cfg = NeuralBuildConfig()
    nw = make_dummy_weights(cfg)
    nw.weights[:] = (np.arange(nw.weights.size, dtype=np.float32) * 0.001).astype(np.float32)
    nw.biases[:] = (np.arange(nw.biases.size, dtype=np.float32) * 0.01).astype(np.float32)

    ctx = _vk_external_sema()
    try:
        if not getattr(ctx, "supports_external_memory", False):
            pytest.skip("device lacks external memory")
        wbuf = StorageBuffer(ctx, max(len(nw.weight_bytes), 4), external=True)
        bbuf = StorageBuffer(ctx, max(len(nw.bias_bytes), 4), external=True)
        sema = ExternalTimelineSemaphore(ctx)
        try:
            if wbuf.export_handle() is None or sema.export_handle() is None:
                pytest.skip("export handles unavailable on this device")
            pub = InteropWeightPublisher(
                weights_buffer=wbuf, biases_buffer=bbuf, timeline_semaphore=sema,
                expect_arch=cfg.arch, precision=cfg.precision)
            try:
                assert pub.current_version() == 0
                assert pub.publish(nw) == 1
                assert pub.swap() is True
                assert pub.current_version() == 1
                host_w, ver = pub.acquire_for_render()
                assert host_w is None and ver == 1

                want_w = nw.weight_bytes_for(cfg.precision)
                want_b = nw.bias_bytes_for(cfg.precision)
                assert pub._weights_cuda.read(len(want_w)) == want_w
                assert pub._biases_cuda.read(len(want_b)) == want_b
            finally:
                pub.close()
        finally:
            sema.destroy()
            wbuf.destroy()
            bbuf.destroy()
    finally:
        ctx.destroy()


def test_external_buffer_cuda_roundtrip():
    """A Vulkan-exported buffer, imported into CUDA, round-trips bytes written
    by ``cudaMemcpy`` and read back — proving the shared mapping is the same
    device memory the Vulkan buffer backs."""
    _require_cuda()
    from skinny.vk_compute import StorageBuffer

    ctx = _vk_external()
    try:
        sb = StorageBuffer(ctx, 4096, external=True)
        try:
            if sb.export_handle() is None:
                pytest.skip("export handle unavailable on this device")
            cbuf = interop.CudaExternalBuffer.from_storage_buffer(sb)
            try:
                payload = bytes((i * 7) & 0xFF for i in range(1024))
                cbuf.write(payload)
                got = cbuf.read(len(payload))
                assert got == payload
            finally:
                cbuf.close()
        finally:
            sb.destroy()
    finally:
        ctx.destroy()
