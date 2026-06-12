"""Probe: compile the Metal ReSTIR DI pass set in-process and dump the
reflected surface (Reservoir/GBuf MSL strides, rpc/fc presence, per-entry
global counts — the 31-buffer-slot-cap pressure check). Run under
scripts/guarded_metal.sh — pipeline creation spikes MTLCompilerService.

    PYTHONPATH=$PWD/src TIMEOUT_S=600 scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 scripts/probe_metal_restir_compile.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"


def main() -> int:
    from skinny.megakernel_sources import emit_megakernel_sources
    from skinny.metal_context import MetalContext
    from skinny.metal_wavefront import MetalRestirDiPass

    emit_megakernel_sources(SHADER_DIR, [])
    ctx = MetalContext(window=None, width=64, height=64)
    print(f"[probe] device ok; supports_indirect_dispatch="
          f"{ctx.supports_indirect_dispatch}", flush=True)
    t0 = time.monotonic()
    p = MetalRestirDiPass(ctx, SHADER_DIR, 4096)
    dt = time.monotonic() - t0
    print(f"[probe] compiled {sorted(p._entries)} in {dt:.1f}s", flush=True)
    print(f"[probe] strides reservoir={p.reservoir_stride} "
          f"gbuf={p.gbuf_stride}", flush=True)
    blob = p.rpc_blob()
    print(f"[probe] rpc_blob={len(blob)}B", flush=True)
    ok = True
    for e, ep in sorted(p._entries.items()):
        names = sorted(ep.global_names)
        has_rpc = "rpc" in ep.global_names
        has_fc = "fc" in ep.global_names
        bound_restir = {"wfState", "wfHits", "wfReservoirA", "wfReservoirB",
                        "wfGBuffer"} & ep.global_names
        print(f"[probe] {e}: {len(names)} globals; rpc={has_rpc} fc={has_fc} "
              f"restir-binds={sorted(bound_restir)}", flush=True)
        if not has_rpc or not has_fc:
            ok = False
        print(f"[probe]   {names}", flush=True)
    p.destroy()
    print(f"[probe] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
