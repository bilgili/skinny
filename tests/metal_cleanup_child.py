"""Subprocess bodies for the Metal dispatch-hygiene kill harness.

Driven by ``tests/test_metal_cleanup.py`` (gpu-marked tests) — this module is
never collected by pytest (no ``test_`` prefix) and each invocation owns the
one guarded Metal device of its process. Modes (``sys.argv[1]``):

- ``probe``    — construct a MetalContext, run one trivial foundation dispatch
                 (``foundation_trivial.slang`` → ``computeMain`` arange),
                 verify the read-back, destroy, exit 0. This is the "GPU is
                 usable" probe the parent runs after clean exits and SIGKILLs.
- ``render``   — bounded accumulation loop: one short command buffer per
                 iteration (dispatch + synchronous read-back), a sentinel file
                 (``sys.argv[2]``) written after frame 1 so the parent knows
                 GPU work is in flight before it SIGKILLs us. Never calls
                 ``destroy()`` on the kill path — proving queued bounded work
                 dies with the process is the point.
- ``atexit``   — construct + dispatch, then exit WITHOUT calling ``destroy()``;
                 the atexit hook MetalContext registered must drain + close and
                 (with ``SKINNY_METAL_TEARDOWN_MARKER`` set by the parent)
                 print ``SKINNY_METAL_TEARDOWN_RAN``.
- ``beacon``   — mmap kernel-beacon wiring (change metal-kernel-beacon):
                 ``argv[2]`` = beacon-cell path, ``argv[3]`` = frame-1 sentinel,
                 ``argv[4]`` = kernel entry name (default ``mainImage``). Opens
                 the cell, initializes it to ``KERNEL_ID_NONE``, then per
                 iteration stamps that entry's kernel id through
                 ``BeaconWriter.stamp`` BEFORE a bounded dispatch. The parent
                 SIGTERMs mid-loop and reads the cell: the last stamp names the
                 kernel the child was on. Never hangs the GPU (bounded buffers);
                 the "wedge" is simulated by the SIGTERM interrupt.
- ``beacon_wavefront`` — drives the ACTUAL trace-gated wavefront per-kernel-submit
                 path (change beacon-wavefront-attribution): ``argv[2]`` = frame-1
                 sentinel. The parent sets ``SKINNY_METAL_TRACE=<cell>`` in the
                 env, so ``MetalContext`` turns ``ctx.trace`` on and creates the
                 beacon writer to that cell, and ``MetalFrameEncoder.dispatch``
                 submits + drains one command buffer per kernel. The child encodes
                 a three-stage frame through ONE ``MetalFrameEncoder``: stage A
                 (``wfPathGenerate``) drains, a barrier, then the LATER-encoded
                 stage B (``wfPathShade``) that runs a bounded-but-long loop. Under
                 the per-kernel-submit rule the child blocks in B's drain and never
                 reaches the stage C (``wfPathResolve``) dispatch, so the cell names
                 B — the in-flight kernel — not the later-encoded C. The parent
                 SIGTERMs while B is in flight (sentinel is written right before B).
                 The loop has a hard trip count so it self-terminates and never
                 truly wedges the GPU (metal-dispatch-hygiene).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _REPO_ROOT / "src" / "skinny" / "shaders"
_MODULE = "foundation_trivial"
_ENTRY = "computeMain"  # never `main` — Metal reserves it
_N = 256


def _make_ctx():
    from skinny.metal_context import MetalContext

    return MetalContext(window=None, width=64, height=64)


def _dispatch_arange(ctx) -> bool:
    """One bounded command buffer: trivial dispatch + synchronous read-back."""
    import numpy as np

    from skinny.metal_compute import ComputePipeline, StorageBuffer

    buf = pipe = None
    try:
        buf = StorageBuffer(ctx, _N * 4)
        pipe = ComputePipeline(ctx, _SHADER_DIR, _MODULE, _ENTRY)
        pipe.dispatch_kernel([_N, 1, 1], buffers={"result": buf})
        out = np.frombuffer(buf.download_sync(_N * 4), dtype=np.uint32)[:_N]
        return bool((out == np.arange(_N, dtype=np.uint32)).all())
    finally:
        if pipe is not None:
            pipe.destroy()
        if buf is not None:
            buf.destroy()


def _mode_probe() -> int:
    ctx = _make_ctx()
    try:
        ok = _dispatch_arange(ctx)
    finally:
        ctx.destroy()
    print("PROBE_OK" if ok else "PROBE_BAD", flush=True)
    return 0 if ok else 1


def _mode_render(sentinel: str, budget_s: float = 120.0) -> int:
    """Bounded accumulation loop; the parent SIGKILLs us mid-loop."""
    import numpy as np  # noqa: F401 — keep import cost out of the timed loop

    from skinny.metal_compute import ComputePipeline, StorageBuffer

    ctx = _make_ctx()
    buf = StorageBuffer(ctx, _N * 4)
    pipe = ComputePipeline(ctx, _SHADER_DIR, _MODULE, _ENTRY)
    deadline = time.monotonic() + budget_s
    frame = 0
    while time.monotonic() < deadline:
        # One short command buffer per iteration (watchdog-bounded by design);
        # the synchronous read-back closes the buffer before the next begins.
        pipe.dispatch_kernel([_N, 1, 1], buffers={"result": buf})
        buf.download_sync(_N * 4)
        frame += 1
        if frame == 1:
            Path(sentinel).write_text("frame1", encoding="utf-8")
    # Not killed within the budget (parent bug / manual run): exit cleanly.
    pipe.destroy()
    buf.destroy()
    ctx.destroy()
    return 0


def _mode_beacon(beacon_path: str, sentinel: str, entry: str = "mainImage",
                 budget_s: float = 120.0) -> int:
    """Stamp a known kernel id before each bounded dispatch; the parent SIGTERMs
    mid-loop and reads the cell to report the kernel by name."""
    import numpy as np  # noqa: F401 — keep import cost out of the timed loop

    from skinny.metal_beacon import KERNEL_ID_NONE, BeaconWriter, kernel_id_for
    from skinny.metal_compute import ComputePipeline, StorageBuffer

    kid = kernel_id_for(entry)  # KeyError here = a bad test entry, fail loud
    ctx = _make_ctx()
    buf = StorageBuffer(ctx, _N * 4)
    pipe = ComputePipeline(ctx, _SHADER_DIR, _MODULE, _ENTRY)
    writer = BeaconWriter(beacon_path)
    writer.stamp(KERNEL_ID_NONE)  # initialize the cell before the first dispatch
    deadline = time.monotonic() + budget_s
    frame = 0
    while time.monotonic() < deadline:
        writer.stamp(kid)  # stamp the kernel id BEFORE the dispatch that may hang
        pipe.dispatch_kernel([_N, 1, 1], buffers={"result": buf})
        buf.download_sync(_N * 4)
        frame += 1
        if frame == 1:
            Path(sentinel).write_text("frame1", encoding="utf-8")
    # Not killed within the budget (parent bug / manual run): exit cleanly.
    writer.close()
    pipe.destroy()
    buf.destroy()
    ctx.destroy()
    return 0


# Self-contained wavefront stub: three compute entry points whose names resolve
# to real kernel ids in the frozen beacon table (wfPathGenerate=2, wfPathShade=8,
# wfPathResolve=7). No includes, no gated code — the beacon stamp is driven purely
# by the host from each pipeline's entry-point name. Stage B (wfPathShade) runs a
# HARD-BOUNDED long loop (never infinite → never wedges the GPU); the LCG carry
# keeps the compiler from hoisting it, and the result store keeps it live.
# ponytail: the trip count is a physical GPU calibration knob — big enough that
# the parent's SIGTERM lands while B is in flight (> ~1 s on any Apple GPU), small
# enough that the OS watchdog reclaims it if the SIGTERM path somehow stalls.
_WF_STUB_TRIP = 400_000_000
_WF_STUB_SRC = f"""
RWStructuredBuffer<uint> result;

[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathGenerate(uint3 tid : SV_DispatchThreadID)
{{
    result[tid.x] = tid.x;
}}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathShade(uint3 tid : SV_DispatchThreadID)
{{
    uint acc = tid.x + 1u;
    [loop]
    for (uint i = 0u; i < {_WF_STUB_TRIP}u; ++i) {{
        acc = acc * 1664525u + 1013904223u;
    }}
    result[tid.x] = acc;
}}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathResolve(uint3 tid : SV_DispatchThreadID)
{{
    result[tid.x] = tid.x + 7u;
}}
"""


def _mode_beacon_wavefront(sentinel: str) -> int:
    """Drive the trace-gated ``MetalFrameEncoder`` per-kernel-submit path.

    ``ctx.trace`` is on (parent set ``SKINNY_METAL_TRACE=<cell>``), so each
    ``enc.dispatch`` submits + drains its own command buffer and stamps the mmap
    cell from the pipeline's entry name. Encodes A → barrier → B(hang) → barrier →
    C; under the fix the child blocks in B's drain and never encodes C.
    """
    from skinny.metal_compute import MetalFrameEncoder, StorageBuffer
    from skinny.metal_wavefront import _EntryPipeline

    ctx = _make_ctx()
    # The whole point of this mode: the encoder must be per-kernel-submitting.
    assert getattr(ctx, "trace", False), (
        "SKINNY_METAL_TRACE must be set so MetalFrameEncoder per-kernel-submits"
    )
    spy = ctx._spy
    opts = spy.SlangCompilerOptions()
    opts.matrix_layout = spy.SlangMatrixLayout.column_major
    session = ctx.device.create_slang_session(compiler_options=opts)
    module = session.load_module_from_source(
        "beacon_wf_stub", _WF_STUB_SRC, "beacon_wf_stub.slang"
    )
    pipes = {e: _EntryPipeline(ctx, session, module, e)
             for e in ("wfPathGenerate", "wfPathShade", "wfPathResolve")}
    buf = StorageBuffer(ctx, _N * 4)
    groups = [(_N + 63) // 64, 1, 1]

    enc = MetalFrameEncoder(ctx)
    # Stage A — completes under trace (submit + drain).
    enc.dispatch(pipes["wfPathGenerate"], groups, bindings={"result": buf})
    enc.barrier()  # runs on the fresh empty encoder A already submitted
    # Tell the parent stage A drained and B is the next (in-flight) kernel, so it
    # SIGTERMs while B — the LATER-encoded stage — is the one in flight.
    Path(sentinel).write_text("stageA-done", encoding="utf-8")
    # Stage B — the LATER-encoded, bounded-long hang. Under the per-kernel-submit
    # rule this submits + drains, so the child BLOCKS here and never encodes C.
    enc.dispatch(pipes["wfPathShade"], groups, bindings={"result": buf})
    enc.barrier()
    # Stage C — reached ONLY if the fix is absent (batched encode would stamp this
    # last-encoded kernel over B). Under the fix this line is never reached.
    enc.dispatch(pipes["wfPathResolve"], groups, bindings={"result": buf})
    enc.submit()  # frame-end submit on a possibly-empty encoder — safe by design
    buf.destroy()
    ctx.destroy()
    return 0


def _mode_atexit() -> int:
    # Module-level ref: the requirement's scenario is a context still ALIVE at
    # interpreter exit (the lifecycle registry holds only weakrefs by design, so
    # a GC'd context is correctly absent — that is not the case under test).
    global _ATEXIT_KEEPALIVE
    _ATEXIT_KEEPALIVE = _make_ctx()
    _dispatch_arange(_ATEXIT_KEEPALIVE)
    # Intentionally NO destroy(): the atexit hook registered at context
    # construction must run the teardown on normal interpreter exit.
    print("CHILD_EXITING", flush=True)
    return 0


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "probe"
    if mode == "probe":
        return _mode_probe()
    if mode == "render":
        return _mode_render(argv[2])
    if mode == "atexit":
        return _mode_atexit()
    if mode == "beacon":
        entry = argv[4] if len(argv) > 4 else "mainImage"
        return _mode_beacon(argv[2], argv[3], entry)
    if mode == "beacon_wavefront":
        return _mode_beacon_wavefront(argv[2])
    print(f"unknown mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
