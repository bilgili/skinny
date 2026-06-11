"""Metal tool-buffer read/write probe (6.1 diagnostic + regression guard).

Isolates the mechanism behind ``read_structural_aov`` WITHOUT compiling
``main_pass`` — it dispatches the tiny ``tool_probe`` kernel (foundation class,
no MTLCompilerService wedge risk) and checks two things the structural-parity
test depends on:

  1. CPU→GPU: a host write to ``toolBuffer[0].x`` (the armed tool mode) is
     visible to a Metal compute dispatch.
  2. GPU→CPU: a value the GPU writes into a ``HostStorageBuffer`` is read back
     to the host by ``read()``.

The kernel poisons-then-conditionally-overwrites so the three failure modes are
distinguishable (see ``tool_probe.slang``). This is the bug that made the first
6.1 runs report ``metal_hits=0``: ``HostStorageBuffer.write`` re-uploads the
full host shadow (which never sees GPU writes), so a ``write`` *after* the GPU
dispatch clobbers the GPU output — ``read_structural_aov`` therefore reads
before it disarms. This probe pins the read-back behaviour that fix relies on.

Safe to run unguarded: tiny kernel, same compile class as
``test_metal_foundation``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from skinny.backend_select import metal_available

pytest.importorskip("slangpy")

_SHADER_DIR = Path(__file__).resolve().parent.parent / "src" / "skinny" / "shaders"
_N = 4096                 # threads / structural slots the kernel fills
_BASE = 16                # float4 slot where the kernel starts writing
_CAP = 8448               # tool-buffer float4 capacity (128*64 + headroom)


def test_metal_tool_buffer_gpu_write_readback():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    from skinny.metal_compute import ComputePipeline, HostStorageBuffer
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=64, height=64)
    pipe = buf = None
    try:
        buf = HostStorageBuffer(ctx, _CAP * 16)
        pipe = ComputePipeline(ctx, _SHADER_DIR, "tool_probe", "computeMain")

        # Arm tool mode 4 at slot 0; poison the output region so a slot the GPU
        # never wrote is distinct from a real write.
        buf.write(struct.pack("<Ifff", 4, 0.0, 0.0, 0.0), offset=0)
        buf.write(b"\xee" * (_N * 16), offset=_BASE * 16)

        pipe.dispatch_kernel([_N, 1, 1], buffers={"toolBuffer": buf})

        out = np.frombuffer(
            buf.read(_N * 16, offset=_BASE * 16), dtype=np.float32
        ).reshape(_N, 4)

        poison = np.frombuffer(b"\xee" * 4, dtype=np.float32)[0]
        still_poison = int(np.count_nonzero(out[:, 0] == poison))
        mode_unseen = int(np.count_nonzero(out[:, 0] == -1.0))
        print(
            f"\n[probe] still_poison={still_poison} mode_unseen={mode_unseen} "
            f"row0={out[0].tolist()} row4095={out[_N - 1].tolist()}"
        )

        # (2) GPU→CPU: no slot may remain poison — the GPU wrote every one and
        # read() must reflect those writes.
        assert still_poison == 0, (
            f"{still_poison} slots never read back the GPU write "
            "(HostStorageBuffer.read does not reflect GPU writes on Metal)"
        )
        # (1) CPU→GPU: the kernel saw toolMode==4 (no 'mode unseen' marker).
        assert mode_unseen == 0, (
            f"{mode_unseen} threads did not see the armed tool mode "
            "(host toolBuffer[0] write not visible to the Metal dispatch)"
        )
        # Exact sentinel: x=index, yzw=(2,3,4).
        expected = np.stack(
            [np.arange(_N, dtype=np.float32),
             np.full(_N, 2.0, np.float32),
             np.full(_N, 3.0, np.float32),
             np.full(_N, 4.0, np.float32)], axis=1)
        np.testing.assert_array_equal(out, expected)
    finally:
        if pipe is not None:
            pipe.destroy()
        if buf is not None:
            buf.destroy()
        ctx.destroy()
