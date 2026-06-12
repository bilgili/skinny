"""Bisect Metal wavefront divergence with single deterministic frames:
metal-wavefront vs metal-megakernel vs vulkan-wavefront (same RNG seeds).
Run under guarded_metal.sh (compiles BOTH the wavefront pipelines and the
megakernel on Metal)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"
RES = 64


def _pump(r, budget_s=180.0):
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._backend_render_ready and r._num_instances >= 3:
            return True
        time.sleep(0.02)
    return False


def render_one(ctx, mode, frames=1):
    from skinny.renderer import Renderer
    r = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, execution_mode=mode,
        hdr_dir=ROOT / "hdrs", tattoo_dir=ROOT / "tattoos",
        usd_scene_path=ROOT / "assets" / "three_materials_demo.usda",
    )
    r.integrator_index = 0
    assert _pump(r), f"scene not ready ({mode})"
    r.accum_frame = 0
    for _ in range(frames):
        r.update(0.016)
        r.render_headless()
    arr, n = r.read_accumulation_hdr()
    return (arr[..., :3] / max(1, n)).astype(np.float64)


def stats(name, a, b):
    err = np.abs(a - b).max(axis=2)
    rel = float(np.mean((a - b) ** 2) / (np.mean(b ** 2) + 1e-8))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    print(f"[{name}] rel_mse={rel:.5f} corr={corr:.5f} "
          f"meanA={a.mean():.5f} meanB={b.mean():.5f} errmax={err.max():.4f} "
          f"frac(err>0.01)={float((err > 0.01).mean()):.4f}")


def main() -> int:
    from skinny.metal_context import MetalContext
    from skinny.vk_context import VulkanContext

    ctx = MetalContext(window=None, width=RES, height=RES)
    m_wf = render_one(ctx, "wavefront")
    ctx.destroy()

    ctx = MetalContext(window=None, width=RES, height=RES)
    m_mk = render_one(ctx, "megakernel")
    ctx.destroy()

    ctx = VulkanContext(window=None, width=RES, height=RES)
    v_wf = render_one(ctx, "wavefront")
    ctx.destroy()

    np.save("/tmp/one_m_wf.npy", m_wf)
    np.save("/tmp/one_m_mk.npy", m_mk)
    np.save("/tmp/one_v_wf.npy", v_wf)
    stats("metalWF vs metalMK ", m_wf, m_mk)
    stats("metalWF vs vulkanWF", m_wf, v_wf)
    stats("metalMK vs vulkanWF", m_mk, v_wf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
