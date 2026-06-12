"""Probe: compile the Metal wavefront path pass with the neural proposal
un-stubbed (``SKINNY_METAL_NEURAL=1``) plus the ``MetalNeuralProposalPass``
in-process, and dump the reflected surface — per-entry global counts (the
31-buffer-slot-cap pressure check, phase 6) and the neural weight-buffer
presence in the kernels that need them. Run under scripts/guarded_metal.sh —
pipeline creation spikes MTLCompilerService.

    PYTHONPATH=$PWD/src TIMEOUT_S=600 scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 scripts/probe_metal_neural_compile.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"

NEURAL_BUFS = {"neuralWeights", "neuralBiases", "neuralLayers"}


def main() -> int:
    from skinny.megakernel_sources import emit_megakernel_sources
    from skinny.metal_context import MetalContext
    from skinny.metal_wavefront import (
        MetalNeuralProposalPass,
        MetalWavefrontPathPass,
    )

    emit_megakernel_sources(SHADER_DIR, [])
    ctx = MetalContext(window=None, width=64, height=64)
    print(f"[probe] device ok; supports_indirect_dispatch="
          f"{ctx.supports_indirect_dispatch} "
          f"fp16_storage={ctx.supports_fp16_storage}", flush=True)

    t0 = time.monotonic()
    p = MetalWavefrontPathPass(ctx, SHADER_DIR, 4096, 4096,
                               build_catchall=True, neural_active=True)
    dt = time.monotonic() - t0
    print(f"[probe] path pass (neural_active=True) compiled "
          f"{sorted(p._entries)} in {dt:.1f}s", flush=True)

    t0 = time.monotonic()
    np_ = MetalNeuralProposalPass(ctx, SHADER_DIR, p, 4096)
    dt = time.monotonic() - t0
    print(f"[probe] neural pre-pass compiled in {dt:.1f}s; "
          f"npc_blob={len(np_.npc_blob())}B", flush=True)

    ok = True
    # Shade kernels evaluate the inline inverse pdf → must see the weight bufs.
    need_weights = {"wfPathShadeFlat", "wfPathShade", "wfNeuralProposal"}
    for e, ep in sorted(list(p._entries.items()) + list(np_._entries.items())):
        names = sorted(ep.global_names)
        weights = NEURAL_BUFS & ep.global_names
        print(f"[probe] {e}: {len(names)} globals; neural-bufs={sorted(weights)}",
              flush=True)
        print(f"[probe]   {names}", flush=True)
        if e in need_weights and weights != NEURAL_BUFS:
            print(f"[probe]   MISSING weight buffers in {e}", flush=True)
            ok = False
    has_np_binds = {"npState", "npHits", "npOut"} <= set(np_.bind_map)
    if not has_np_binds:
        print("[probe] MISSING npState/npHits/npOut bind map", flush=True)
        ok = False
    np_.destroy()
    p.destroy()
    print(f"[probe] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
