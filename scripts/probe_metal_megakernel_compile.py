#!/usr/bin/env python3
"""Minimal headless probe: cold-compile the full main_pass.slang megakernel on
the Metal target and reflect it. This is the single dangerous operation (the
MTLCompilerService RAM spike) isolated with the smallest possible surface — no
GLFW window, no surface, no full-frame dispatch. Run it ONLY through
``scripts/guarded_metal.sh`` so the memory watchdog + graceful-only kill apply.

Proves: 4.0/4.0a (megakernel links on Metal) + 3.1 reflection (MSL ``fc`` size).
Exits 0 on success; closes the device in a finally so a guard SIGINT unwinds
cleanly and does not orphan MTLCompilerService.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    import slangpy as spy
    import skinny  # locate the package to find shaders/

    shader_dir = Path(skinny.__file__).resolve().parent / "shaders"
    print(f"[probe] shader_dir = {shader_dir}", flush=True)

    # Minimal duck-typed ctx: ComputePipeline only touches ctx._spy + ctx.device.
    class _Ctx:
        pass

    ctx = _Ctx()
    ctx._spy = spy
    ctx.is_metal = True
    print("[probe] creating Metal device...", flush=True)
    ctx.device = spy.create_device(type=spy.DeviceType.metal)
    print("[probe] device OK; cold-compiling main_pass megakernel (the spike)...",
          flush=True)

    try:
        from skinny.metal_compute import ComputePipeline

        pipe = ComputePipeline(ctx, shader_dir)  # _build() = cold compile
        print(f"[probe] COMPILE OK: globals={len(pipe.global_names)} "
              f"uniform_size={pipe.uniform_size}B", flush=True)
        # Spot-check a few reflected MSL offsets (design D3 / task 3.1).
        for key in ("camera.position", "focusPlaneOrigin", "focusPlaneNormal",
                    "zoomMin", "pickPixel"):
            if key in pipe.uniform_layout:
                off, size = pipe.uniform_layout[key]
                print(f"[probe]   fc.{key:18s} offset={off:4d} size={size}", flush=True)
        pipe.destroy()
        print("[probe] PASS", flush=True)
        return 0
    finally:
        try:
            ctx.device.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
