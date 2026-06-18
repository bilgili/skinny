"""Render a clean BDPT reference (bsdf-only, no neural) to PNG + HDR .npy.

BDPT (integrator_index=1) resolves caustics far better than unidirectional PT, so
it is a much cleaner ground-truth reference for the glass-caustics variance than
the 512spp PT reference. Metal pins wavefront+BDPT to the megakernel (renderer.py
~1330), which is fine here (no neural pre-pass needed for the reference).

Run (Vulkan SDK sourced for the renderer import):
  PYTHONPATH=src:tests <py> scripts/nis_render_bdpt_ref.py --scene glass \
     --backend metal --res 512 --spp 1024 --no-direct
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
import guiding_variance_sweep as gv  # noqa: E402  (reuse _accumulate_to/_save_png/_auto_exposure)

ROOT = Path("/Users/ahmetbilgili/projects/skinny")
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"
SCENES = {"cornell": gv.SCENES["cornell"],
          "glass": gv.ASSETS / "glass_caustics_test.usda"}


def build(backend, scene_path, res, execution_mode, no_direct):
    from skinny.renderer import Renderer
    ctx = gv._make_context(backend, res)
    r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=ROOT / "hdrs",
                 tattoo_dir=ROOT / "tattoos", usd_scene_path=scene_path,
                 execution_mode=execution_mode)
    r.integrator_index = 1            # BDPT
    r.proposal_preset_index = r.proposal_preset_from_token("bsdf")
    for _ in range(400):
        r.update(0.025)
        if (r._usd_scene is not None and len(r._usd_scene.instances) >= 1
                and r._scene_bindings is not None):
            break
    if r._scene_bindings is None:
        raise RuntimeError("scene bindings never built")
    if no_direct:
        r.light_intensity = 0.0
        r.direct_light_index = 1
    return ctx, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="glass")
    ap.add_argument("--backend", default="metal", choices=("metal", "vulkan"))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--no-direct", action="store_true")
    ap.add_argument("--execution-mode", default="megakernel")
    ap.add_argument("--out", default=str(ROOT / "renders_bdpt"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sp = SCENES[args.scene]

    print(f"[bdpt] {args.scene} BDPT {args.spp}spp res{args.res} {args.backend} "
          f"{args.execution_mode} no_direct={args.no_direct}", flush=True)
    ctx, r = build(args.backend, sp, args.res, args.execution_mode, args.no_direct)
    try:
        assert r.integrator_index == 1, f"integrator not BDPT: {r.integrator_index}"
        print(f"[bdpt] integrator={r.integrator_modes[r.integrator_index]} "
              f"exec={getattr(r,'execution_mode','?')}", flush=True)
        img = gv._accumulate_to(r, args.spp, 0)   # fixed harness => true mean
    finally:
        r.cleanup(); ctx.destroy()

    nd = "_nodirect" if args.no_direct else ""
    base = out / f"{args.scene}_bdpt_ref_{args.spp}spp{nd}_{args.backend}"
    np.save(str(base.with_suffix(".npy")), img.astype(np.float32))
    exp = gv._auto_exposure(img)
    gv._save_png(str(base.with_suffix(".png")), img, exp)
    print(f"[bdpt] saved {base}.png  mean={float(img.mean()):.4f} max={float(img.max()):.3f} "
          f"exposure={exp:.3f}", flush=True)


if __name__ == "__main__":
    main()
