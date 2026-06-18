"""Glass encoding sweep: E0/E1/E3 x {RQ, PQ} at chart V1 (Metal, direct off).

Tests whether the conditioner encoding (AXIS 2, Jacobian-free) improves either
coupling in the renderer. Charts are V1-only here: the shader implements just the
Lambert V1 chart (neural_flow.slang), so V0/V2/V5 are NOT renderable without the
renderer-chart-selection change. Encodings E0 (raw) / E1 (gamma-band every
path-regime scalar) / E3 (E1 + raw tail) ARE renderable.

Nets are offline-trained per (encoding x coupling) on the SAME clamped glass
records (p99), V1 chart: glass_offc99_{rqs,pq} (E0), glass_offc99_{E1,E3}_{rqs,pq}.
Each cell renders at N paired seeds (common random numbers), var = MSE vs the
BDPT-1024 reference; reports per-cell mean var +/- 95% t-CI and paired ratios.

ALWAYS DUMPS every render as a tonemapped PNG into renders_rqvspq/enc_sweep/.

Run (Vulkan SDK sourced): PYTHONPATH=src:tests:scripts <py> \
    scripts/nis_glass_encoding_sweep.py [N_SEEDS]
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.image as mpimg

sys.path.insert(0, "src"); sys.path.insert(0, "tests"); sys.path.insert(0, "scripts")
import guiding_variance_sweep as gv  # noqa: E402

NN = "/Users/ahmetbilgili/projects/skinny/.skinny_neural"
BDPT = "/Users/ahmetbilgili/projects/skinny/renders_bdpt/glass_bdpt_ref_1024spp_nodirect_metal.npy"
OUT = Path("/Users/ahmetbilgili/projects/skinny/renders_rqvspq"); OUT.mkdir(exist_ok=True)
DUMP = OUT / "enc_sweep"; DUMP.mkdir(exist_ok=True)
RES, SPP = 512, 64                 # 512^2 == BDPT ref res (no downsample)
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED_BASE, SEED_STRIDE = 1000, 1000   # disjoint streams [base+k*stride, +SPP)
SCENE = gv.ASSETS / "glass_caustics_test.usda"
ENCODINGS = ["E0", "E1", "E3"]
COUPLINGS = [("rqs", "rqs"), ("pq", "nis-pq")]   # (suffix/label, cli coupling)

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}


def tcrit(n):
    return T975.get(n - 1, 1.96)


def ci(vals):
    a = np.asarray(vals, float)
    n = len(a)
    if n < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(tcrit(n) * a.std(ddof=1) / np.sqrt(n))


def net_path(enc, suff):
    # E0 clamped nets keep the legacy name (no encoding segment).
    return f"{NN}/glass_offc99_{suff}.nfw1" if enc == "E0" \
        else f"{NN}/glass_offc99_{enc}_{suff}.nfw1"


def cell(proposals, enc, coupling=None, net=None):
    c = {"proposals": proposals, "reuse": "none", "chart": "V1",
         "encoding": enc, "temporal": "off", "precision": "fp32"}
    if coupling:
        c["coupling"] = coupling
    if net:
        c["net"] = net
    return c


def render(c, seed):
    ctx, r = gv._build_renderer("metal", SCENE, "path", c, RES, no_direct=True)
    try:
        if "neural" in c["proposals"] and not r._neural_active():
            raise RuntimeError("neural not active")
        return gv._accumulate_to(r, SPP, seed)
    finally:
        r.cleanup(); ctx.destroy()


def downsample(img, res):
    f = img.shape[0] // res
    return img.reshape(res, f, res, f, img.shape[2]).mean(axis=(1, 3)) if f > 1 else img


def lum(i):
    return 0.2126 * i[..., 0] + 0.7152 * i[..., 1] + 0.0722 * i[..., 2]


# Build the cell list: bsdf (encoding-independent) + {enc x coupling}.
cells = [("bsdf", "-", cell("bsdf", "E0"))]
for enc in ENCODINGS:
    for suff, coup in COUPLINGS:
        cells.append((suff, enc, cell("bsdf,neural", enc, coup, net_path(enc, suff))))

ref = downsample(np.load(BDPT).astype(np.float64), RES)
exp = 0.45 / max(float(np.median(lum(ref))), 1e-8)   # shared tonemap exposure
mpimg.imsave(str(DUMP / "_ref_bdpt1024.png"),
             np.clip((np.maximum(ref * exp, 0)) ** (1 / 2.2), 0, 1))

seeds = [SEED_BASE + k * SEED_STRIDE for k in range(N_SEEDS)]
print(f"glass ENCODING SWEEP  E0/E1/E3 x RQ/PQ  chart V1  res{RES} {SPP}spp direct-off  "
      f"{N_SEEDS} seeds\n  (var = MSE vs BDPT-1024 ref; dumping every render -> {DUMP})\n")

# key = (suff, enc); var[key] = per-seed MSE list, paired by seed index.
var = {(s, e): [] for s, e, _ in cells}
mean_lin = {(s, e): [] for s, e, _ in cells}
for si, sd in enumerate(seeds):
    parts = []
    for suff, enc, c in cells:
        img = render(c, sd).astype(np.float64)
        var[(suff, enc)].append(float(np.mean((img - ref) ** 2)))
        mean_lin[(suff, enc)].append(float(img.mean()))
        # DUMP every render (tonemapped, shared exposure)
        srgb = np.clip(np.maximum(img * exp, 0) ** (1 / 2.2), 0, 1)
        name = f"glass_bsdf_seed{sd}" if suff == "bsdf" else f"glass_{suff}_{enc}_seed{sd}"
        mpimg.imsave(str(DUMP / f"{name}.png"), srgb)
        parts.append(f"{suff}/{enc}={var[(suff, enc)][-1]:.3e}")
    print(f"  seed {sd}: " + "  ".join(parts))

bsdf_v = var[("bsdf", "-")]
res = {"res": RES, "spp": SPP, "seeds": seeds, "chart": "V1",
       "note": "charts V1-only (shader); encodings E0/E1/E3 swept", "cells": {}}
print("\n  per-cell  mean var +/- 95% t-CI   bsdf/X (paired, >1 = X beats bsdf):")
m_bsdf, h_bsdf = ci(bsdf_v)
print(f"    bsdf        {m_bsdf:.4e} +/- {h_bsdf:.2e}")
res["cells"]["bsdf"] = {"var_mean": m_bsdf, "var_ci95": h_bsdf, "var_per_seed": bsdf_v}
for enc in ENCODINGS:
    for suff, _ in COUPLINGS:
        v = var[(suff, enc)]
        m, h = ci(v)
        ratio = [bsdf_v[i] / v[i] for i in range(N_SEEDS)]   # paired
        rm, rh = ci(ratio)
        sig = "  <-- beats bsdf" if (rm - rh) > 1 else ("  WORSE" if (rm + rh) < 1 else "")
        print(f"    {suff:4s} {enc}    {m:.4e} +/- {h:.2e}   bsdf/X {rm:.3f} +/- {rh:.3f}{sig}")
        res["cells"][f"{suff}_{enc}"] = {
            "var_mean": m, "var_ci95": h, "var_per_seed": v,
            "bsdf_over_X_mean": rm, "bsdf_over_X_ci95": rh,
            "mean_lin": float(np.mean(mean_lin[(suff, enc)]))}

print("\n  RQ-vs-PQ per encoding  (RQ/PQ paired, <1 = RQ beats PQ):")
res["rq_vs_pq"] = {}
for enc in ENCODINGS:
    rv, pv = var[("rqs", enc)], var[("pq", enc)]
    r = [rv[i] / pv[i] for i in range(N_SEEDS)]
    rm, rh = ci(r)
    flag = "  RQ wins" if (rm + rh) < 1 else ("  PQ wins" if (rm - rh) > 1 else "  tie")
    print(f"    {enc}:  RQ/PQ {rm:.3f} +/- {rh:.3f}{flag}")
    res["rq_vs_pq"][enc] = {"mean": rm, "ci95": rh}

print("\n  best encoding per coupling (lowest mean var):")
for suff, _ in COUPLINGS:
    best = min(ENCODINGS, key=lambda e: np.mean(var[(suff, e)]))
    vs = {e: float(np.mean(var[(suff, e)])) for e in ENCODINGS}
    print(f"    {suff}: {best}  ({'  '.join(f'{e}={vs[e]:.3e}' for e in ENCODINGS)})")
    res["cells"].setdefault("_best", {})[suff] = best

outp = OUT / f"encsweep_glass_offc99_{RES}_{SPP}spp_{N_SEEDS}seeds.json"
outp.write_text(json.dumps(res, indent=2))
print(f"\nsaved -> {outp}\n  dumped {len(cells) * N_SEEDS} renders + ref -> {DUMP}")
