"""Glass RQ-vs-PQ head-to-head render (Metal, direct off).

Renders bsdf / bsdf+neural(RQ) / bsdf+neural(PQ) at equal spp, computes variance
= MSE vs the clean BDPT 1024spp reference (downsampled to the render res), and
saves the three images. RQ = "my method", PQ = NIS piecewise-quadratic — both
offline-trained on the SAME glass records (V1 chart), so only the coupling differs.

Run (Vulkan SDK sourced): PYTHONPATH=src:tests <py> scripts/nis_glass_rqvspq.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "tests")
import guiding_variance_sweep as gv  # noqa: E402

NN = "/Users/ahmetbilgili/projects/skinny/.skinny_neural"
BDPT = "/Users/ahmetbilgili/projects/skinny/renders_bdpt/glass_bdpt_ref_1024spp_nodirect_metal.npy"
OUT = Path("/Users/ahmetbilgili/projects/skinny/renders_rqvspq"); OUT.mkdir(exist_ok=True)
RES, SPP, SEED = 512, 64, 4242   # 512^2 == BDPT ref res (no downsample)
SCENE = gv.ASSETS / "glass_caustics_test.usda"


def cell(proposals, coupling=None, net=None):
    c = {"proposals": proposals, "reuse": "none", "chart": "V1",
         "encoding": "E0", "temporal": "off", "precision": "fp32"}
    if coupling:
        c["coupling"] = coupling
    if net:
        c["net"] = net
    return c


def render(c):
    ctx, r = gv._build_renderer("metal", SCENE, "path", c, RES, no_direct=True)
    try:
        if "neural" in c["proposals"] and not r._neural_active():
            raise RuntimeError("neural not active")
        return gv._accumulate_to(r, SPP, SEED)
    finally:
        r.cleanup(); ctx.destroy()


def downsample(img, res):
    f = img.shape[0] // res
    return img.reshape(res, f, res, f, img.shape[2]).mean(axis=(1, 3)) if f > 1 else img


def lum(i):
    return 0.2126 * i[..., 0] + 0.7152 * i[..., 1] + 0.0722 * i[..., 2]


import os
NETTAG = os.environ.get("NIS_NETTAG", "glass_off")   # glass_off (unclamped) | glass_offc99
cells = [("bsdf", cell("bsdf")),
         ("rqs",  cell("bsdf,neural", "rqs",    f"{NN}/{NETTAG}_rqs.nfw1")),
         ("pq",   cell("bsdf,neural", "nis-pq", f"{NN}/{NETTAG}_pq.nfw1"))]

ref = downsample(np.load(BDPT).astype(np.float64), RES)
print(f"glass RQ-vs-PQ  res{RES} {SPP}spp direct-off  (var = MSE vs BDPT-1024 ref)\n")
imgs, var = {}, {}
for tag, c in cells:
    img = render(c).astype(np.float64)
    imgs[tag] = img
    var[tag] = float(np.mean((img - ref) ** 2))
    print(f"  {tag:5s} mean={img.mean():.4f}  var={var[tag]:.4e}")

print(f"\n  bsdf/RQ ratio = {var['bsdf']/var['rqs']:.3f}   "
      f"bsdf/PQ ratio = {var['bsdf']/var['pq']:.3f}   "
      f"RQ/PQ ratio = {var['rqs']/var['pq']:.3f}  (>1 = PQ better)")

# save tonemapped images (shared exposure from ref median)
exp = 0.45 / max(float(np.median(lum(ref))), 1e-8)
for tag in imgs:
    srgb = np.clip(np.maximum(imgs[tag] * exp, 0) ** (1 / 2.2), 0, 1)
    import matplotlib.image as mpimg
    mpimg.imsave(str(OUT / f"glass_{NETTAG}_{tag}_{SPP}spp.png"), srgb)
print(f"\nsaved images -> {OUT}")
