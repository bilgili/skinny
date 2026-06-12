"""Diagnose Metal-vs-Vulkan wavefront divergence: render both, dump the worst
pixels + per-frame stats. Run under guarded_metal.sh with the SDK env."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"
RES = 64
SAMPLES = 96


def _pump(r, budget_s=120.0):
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._backend_render_ready and r._num_instances >= 3:
            return True
        time.sleep(0.02)
    return False


def converge(ctx, frames=SAMPLES):
    from skinny.renderer import Renderer
    r = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, execution_mode="wavefront",
        hdr_dir=ROOT / "hdrs", tattoo_dir=ROOT / "tattoos",
        usd_scene_path=ROOT / "assets" / "three_materials_demo.usda",
    )
    r.integrator_index = 0
    assert _pump(r)
    r.accum_frame = 0
    for _ in range(frames):
        r.update(0.016)
        r.render_headless()
    arr, n = r.read_accumulation_hdr()
    return (arr[..., :3] / max(1, n)).astype(np.float64)


def main() -> int:
    from skinny.metal_context import MetalContext
    from skinny.vk_context import VulkanContext

    m_ctx = MetalContext(window=None, width=RES, height=RES)
    m = converge(m_ctx)
    m_ctx.destroy()
    v_ctx = VulkanContext(window=None, width=RES, height=RES)
    v = converge(v_ctx)
    v_ctx.destroy()

    np.save("/tmp/wf_metal.npy", m)
    np.save("/tmp/wf_vulkan.npy", v)
    err = np.abs(m - v).max(axis=2)
    order = np.argsort(err.ravel())[::-1][:12]
    print(f"err: max={err.max():.3f} mean={err.mean():.5f} "
          f"p99={np.percentile(err, 99):.4f}")
    for idx in order:
        y, x = divmod(int(idx), RES)
        print(f"  ({x:2d},{y:2d}) err={err[y, x]:9.3f} "
              f"metal={np.round(m[y, x], 3)} vulkan={np.round(v[y, x], 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
