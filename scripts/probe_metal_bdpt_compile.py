"""Probe: compile the Metal wavefront BDPT pipelines in-process for every walk
mode and dump the reflected surface (strides, wfTile/fc presence, layout
sanity). Run under scripts/guarded_metal.sh — pipeline creation spikes
MTLCompilerService.

    PYTHONPATH=$PWD/src TIMEOUT_S=600 scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 scripts/probe_metal_bdpt_compile.py
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
    from skinny.metal_wavefront import MetalWavefrontBdptPass

    emit_megakernel_sources(SHADER_DIR, [])
    ctx = MetalContext(window=None, width=64, height=64)
    print(f"[probe] device ok; supports_indirect_dispatch="
          f"{ctx.supports_indirect_dispatch}", flush=True)
    for walk_mode in MetalWavefrontBdptPass.WALK_MODES:
        t0 = time.monotonic()
        p = MetalWavefrontBdptPass(ctx, SHADER_DIR, 4096, 4096,
                                   walk_mode=walk_mode)
        dt = time.monotonic() - t0
        print(f"[probe] {walk_mode}: compiled {sorted(p._entries)} "
              f"in {dt:.1f}s", flush=True)
        print(f"[probe] {walk_mode}: strides vertex={p.vertex_stride} "
              f"aux={p.aux_stride}; fc uniform_size={p.uniform_size} "
              f"fields={len(p.uniform_layout)} "
              f"std_surface_stride={p.std_surface_stride}", flush=True)
        ok = True
        for e, ep in p._entries.items():
            has_tile = "wfTile" in ep.global_names
            has_fc = "fc" in ep.global_names
            print(f"[probe]   {e}: wfTile={has_tile} fc={has_fc} "
                  f"globals={len(ep.global_names)}", flush=True)
            if not has_tile:
                print(f"[probe] FAIL: {e} did not reflect wfTile", flush=True)
                ok = False
        if not p.uniform_layout:
            print(f"[probe] FAIL: {walk_mode} reflected no fc layout", flush=True)
            ok = False
        p.destroy()
        if not ok:
            ctx.destroy()
            return 1
    ctx.destroy()
    print("[probe] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
