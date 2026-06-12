"""Metal shared-storage buffer mode + `supports_shared_memory` capability flag
(change metal-neural-interop, tasks 1.2–1.4).

Proves the UMA interop seam the neural weight handoff builds on:

* a `StorageBuffer(shared=True)` allocates host-visible storage whose
  `write_in_place` bytes a subsequent compute dispatch observes, with no
  staging upload call;
* a buffer allocated without shared mode behaves exactly as before
  (staged `upload_sync` / `download_sync` round-trip, `shared` False);
* `MetalContext` reports `supports_shared_memory` True on an Apple-Silicon
  unified-memory device while `supports_external_memory` /
  `supports_external_semaphore` stay False.

Compiles one trivial single-thread kernel in-process (MTLCompilerService blip
comparable to the foundation tests, far below a megakernel build) — still
gpu-marked so `-m 'not gpu'` sweeps skip it per the thermal rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.backend_select import metal_available

pytestmark = pytest.mark.gpu

_SRC = (
    "StructuredBuffer<uint> probe_in;\n"
    "RWStructuredBuffer<uint> probe_out;\n"
    "[shader(\"compute\")] [numthreads(1, 1, 1)]\n"
    "void probeMain(uint3 tid : SV_DispatchThreadID)"
    " { probe_out[0] = probe_in[0] + 1u; }\n"
)


def _require_metal():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")


def _make_ctx():
    from skinny.metal_context import MetalContext

    return MetalContext(window=None, width=64, height=64)


def _dispatch_increment(ctx, in_buf, out_buf) -> int:
    """Dispatch `probe_out[0] = probe_in[0] + 1` once and return the result."""
    import slangpy as spy

    device = ctx.device
    module = device.load_module_from_source("shared_storage_probe", _SRC)
    program = device.link_program([module], [module.entry_point("probeMain")])
    pipeline = device.create_compute_pipeline(program=program)
    ro = device.create_root_shader_object(program)
    cursor = spy.ShaderCursor(ro)
    cursor["probe_in"] = in_buf.buffer
    cursor["probe_out"] = out_buf.buffer
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.bind_pipeline(pipeline, ro)
    cpass.dispatch_compute(spy.math.uint3(1, 1, 1))
    cpass.end()
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    return int(np.frombuffer(out_buf.download_sync(4), dtype=np.uint32)[0])


def test_shared_memory_capability_flags():
    """Spec: shared-memory flag reflects unified memory; external flags stay off."""
    _require_metal()
    ctx = _make_ctx()
    try:
        assert ctx.supports_shared_memory is True
        assert ctx.supports_external_memory is False
        assert ctx.supports_external_semaphore is False
    finally:
        ctx.destroy()


def test_shared_buffer_in_place_write_visible_to_dispatch():
    """Spec: shared-storage buffer accepts in-place host writes a dispatch sees."""
    from skinny.metal_compute import StorageBuffer

    _require_metal()
    ctx = _make_ctx()
    shared = out = None
    try:
        shared = StorageBuffer(ctx, 16, shared=True)
        assert shared.shared is True
        out = StorageBuffer(ctx, 16)
        out.fill_zero_sync()

        shared.write_in_place(np.array([41], dtype=np.uint32).tobytes())
        assert _dispatch_increment(ctx, shared, out) == 42

        # In-place rewrite between submissions — the next dispatch reads the
        # new bytes with no staging upload call.
        shared.write_in_place(np.array([99], dtype=np.uint32).tobytes())
        assert _dispatch_increment(ctx, shared, out) == 100

        # Offset write composes with the shadow (word 0 keeps its bytes).
        shared.write_in_place(np.array([7], dtype=np.uint32).tobytes(), offset=4)
        assert _dispatch_increment(ctx, shared, out) == 100

        with pytest.raises(ValueError):
            shared.write_in_place(b"\x00" * 32)
    finally:
        for buf in (shared, out):
            if buf is not None:
                buf.destroy()
        ctx.destroy()


def test_non_shared_buffer_unchanged():
    """A default StorageBuffer keeps the staged-upload behavior and reports
    `shared` False; `external` stays a no-op and `export_handle()` None."""
    from skinny.metal_compute import StorageBuffer

    _require_metal()
    ctx = _make_ctx()
    buf = None
    try:
        buf = StorageBuffer(ctx, 16, external=True)
        assert buf.shared is False
        assert buf.external is False
        assert buf.export_handle() is None
        payload = np.arange(4, dtype=np.uint32).tobytes()
        buf.upload_sync(payload)
        assert buf.download_sync(16) == payload
    finally:
        if buf is not None:
            buf.destroy()
        ctx.destroy()
