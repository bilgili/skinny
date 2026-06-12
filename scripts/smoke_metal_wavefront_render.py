"""Smoke: headless Metal wavefront render of the three-materials demo scene.

Builds a Renderer on MetalContext with execution_mode="wavefront", pumps the
async USD load, renders a few accumulation frames, and prints image stats.
Run under scripts/guarded_metal.sh (in-process Metal pipeline compiles):

    export VULKAN_SDK=...; export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    PYTHONPATH=$PWD/src TIMEOUT_S=420 scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 scripts/smoke_metal_wavefront_render.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"


def main() -> int:
    from skinny.metal_context import MetalContext
    from skinny.renderer import Renderer

    ctx = MetalContext(window=None, width=96, height=96)
    r = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=ROOT / "hdrs",
        tattoo_dir=ROOT / "tattoos",
        usd_scene_path=ROOT / "assets" / "three_materials_demo.usda",
        execution_mode="wavefront",
    )
    deadline = 200
    while deadline > 0 and (
        r._usd_scene is None or len(r._usd_scene.instances) < 3
    ):
        r.update(0.025)
        deadline -= 1
    print(f"[smoke] scene bindings: {r._scene_bindings is not None}; "
          f"execution_modes={r.execution_modes}; "
          f"mode={r.execution_mode_index}/{r.effective_execution_mode_index}",
          flush=True)
    assert r._scene_bindings is not None
    assert r.pipeline is None, "wavefront mode must not compile the megakernel"
    r.integrator_index = 0
    r._material_version += 1
    for i in range(8):
        r.update(0.04)
        px = r.render_headless()
        if i == 0:
            print(f"[smoke] frame0 rendered ({len(px)} bytes)", flush=True)
    arr = r.read_accumulation()[:, :, :3]
    print(f"[smoke] accum: finite={np.isfinite(arr).all()} "
          f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} "
          f"nonzero={float((arr > 0).mean()):.3f}", flush=True)
    pass_ = r._wavefront_path_pass
    print(f"[smoke] pass: entries={sorted(pass_._entries)} "
          f"catchall={pass_.build_catchall} stream={pass_.stream_size} "
          f"graphs={len(pass_.graph_param_layouts)}", flush=True)
    ok = bool(np.isfinite(arr).all()) and arr.max() > 0
    print("[smoke] OK" if ok else "[smoke] FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
