"""Render the NIS sweep scenes to PNG (bsdf vs bsdf,neural + a high-spp bsdf
reference) so the renderings can be eyeballed. Reuses the variance-sweep GPU
plumbing; tonemaps the linear-HDR accumulation (Reinhard + sRGB gamma).

Run (Vulkan SDK sourced for the renderer import):
  PYTHONPATH=src:tests <py> scripts/nis_render_to_png.py --scenes cornell glass \
     --backend metal --res 192 --spp 96
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
import guiding_variance_sweep as gv  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

NETS = {
    "cornell": "/Users/ahmetbilgili/projects/skinny/.skinny_neural/cornell_metal.nfw1",
    "glass": "/Users/ahmetbilgili/projects/skinny/.skinny_neural/glass_metal.nfw1",
}
SCENE_PATH = {
    "cornell": gv.SCENES["cornell"],
    "glass": gv.ASSETS / "glass_caustics_test.usda",
}


def _lum(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def auto_exposure(ref_img, pct=50.0, target=0.45):
    """Map the reference's median (pct=50) luminance to a mid value so the scene is
    visible and highlights/fireflies clip to white; shared across a scene's images
    for fair brightness (only noise differs)."""
    key = float(np.percentile(_lum(ref_img), pct))
    return target / max(key, 1e-8)


def tonemap(lin, exposure):
    lin = np.maximum(lin.astype(np.float64) * exposure, 0.0)
    return np.clip(lin ** (1.0 / 2.2), 0.0, 1.0)   # exposed clip + sRGB gamma


def cell(proposals, net=None):
    c = {"proposals": proposals, "reuse": "none", "chart": "V1",
         "encoding": "E0", "temporal": "off", "precision": "fp32"}
    if net:
        c["net"] = net
    return c


def render(backend, scene_path, c, res, spp, seed=12345, no_direct=False):
    ctx, r = gv._build_renderer(backend, scene_path, "path", c, res)
    try:
        if no_direct:
            r.light_intensity = 0.0       # zero the analytic distant ("direct") light
            r.direct_light_index = 1      # belt-and-suspenders: disable analytic direct
        img = gv._accumulate_to(r, spp, seed)   # mean linear-HDR (H,W,3)
    finally:
        r.cleanup(); ctx.destroy()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["cornell", "glass"])
    ap.add_argument("--backend", default="metal", choices=("metal", "vulkan"))
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--spp", type=int, default=96)
    ap.add_argument("--ref-spp", type=int, default=512)
    ap.add_argument("--exposure", type=float, default=0.0,
                    help="0 = auto (from reference 99th-percentile); >0 = fixed")
    ap.add_argument("--no-direct", action="store_true",
                    help="zero the analytic distant (direct) light; scene lit only by "
                         "the emissive panel (more indirect-dominated, guiding matters more)")
    ap.add_argument("--out", default="/Users/ahmetbilgili/projects/skinny/renders")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for name in args.scenes:
        sp = SCENE_PATH[name]; net = NETS[name]
        jobs = [("bsdf", cell("bsdf"), args.spp),
                ("neural", cell("bsdf,neural", net), args.spp),
                ("bsdf_ref", cell("bsdf"), args.ref_spp)]
        imgs = {}
        for tag, c, spp in jobs:
            print(f"[render] {name} {tag} {spp}spp res{args.res} {args.backend} …", flush=True)
            imgs[tag] = (render(args.backend, sp, c, args.res, spp, no_direct=args.no_direct), spp)
            hdr, _ = imgs[tag]
            print(f"[render]   {tag} mean={float(hdr.mean()):.5f} max={float(hdr.max()):.4f}",
                  flush=True)
        # shared exposure from the clean reference (fair brightness across cells)
        exp = args.exposure if args.exposure > 0 else auto_exposure(imgs["bsdf_ref"][0])
        print(f"[render] {name} exposure={exp:.2f}", flush=True)
        for tag, (hdr, spp) in imgs.items():
            f = out / f"{name}_{tag}_{spp}spp_{args.backend}.png"
            mpimg.imsave(str(f), tonemap(hdr, exp))
            np.save(str(f.with_suffix(".npy")), hdr.astype(np.float32))
            print(f"[render]   saved {f}", flush=True)


if __name__ == "__main__":
    main()
