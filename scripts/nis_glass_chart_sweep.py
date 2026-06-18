"""Glass chart sweep: V0/V1/V5 x {RQ, PQ} at encoding E0 (Metal, direct off).

Tests whether the hemisphere chart (renderer-chart-selection) improves either
coupling in the renderer. V0 cylindrical / V1 Lambert (default) / V5 equirectangular
(non-equal-area). Each cell renders at N paired seeds (common random numbers),
var = MSE vs the BDPT-1024 reference; reports per-cell var +/- 95% t-CI and paired
ratios. Also a sanity gate: every neural cell's mean MUST stay ~0.225 (unbiased) —
a wrong chart Jacobian would bias the estimator.

Nets are offline-trained per (chart x coupling) on the SAME clamped glass records
(p99): V1=glass_offc99_{rqs,pq}, V0=glass_V0c99_{rqs,pq}, V5=glass_V5c99_{rqs,pq}.

ALWAYS DUMPS every render as a tonemapped PNG into renders_rqvspq/chart_sweep/.

Run (Vulkan SDK sourced): PYTHONPATH=src:tests:scripts <py> \
    scripts/nis_glass_chart_sweep.py [N_SEEDS]
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
DUMP = OUT / "chart_sweep"; DUMP.mkdir(exist_ok=True)
RES, SPP = 512, 64
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED_BASE, SEED_STRIDE = 1000, 1000
SCENE = gv.ASSETS / "glass_caustics_test.usda"
CHARTS = ["V0", "V1", "V5"]
COUPLINGS = [("rqs", "rqs"), ("pq", "nis-pq")]

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}


def tcrit(n):
    return T975.get(n - 1, 1.96)


def ci(vals):
    a = np.asarray(vals, float); n = len(a)
    if n < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(tcrit(n) * a.std(ddof=1) / np.sqrt(n))


def net_path(chart, suff):
    # V1 clamped E0 nets use the legacy name; V0/V5 carry the chart tag.
    return f"{NN}/glass_offc99_{suff}.nfw1" if chart == "V1" \
        else f"{NN}/glass_{chart}c99_{suff}.nfw1"


def cell(proposals, chart, coupling=None, net=None):
    c = {"proposals": proposals, "reuse": "none", "chart": chart,
         "encoding": "E0", "temporal": "off", "precision": "fp32"}
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


cells = [("bsdf", "-", cell("bsdf", "V1"))]
for chart in CHARTS:
    for suff, coup in COUPLINGS:
        cells.append((suff, chart, cell("bsdf,neural", chart, coup, net_path(chart, suff))))

ref = downsample(np.load(BDPT).astype(np.float64), RES)
exp = 0.45 / max(float(np.median(lum(ref))), 1e-8)
mpimg.imsave(str(DUMP / "_ref_bdpt1024.png"),
             np.clip((np.maximum(ref * exp, 0)) ** (1 / 2.2), 0, 1))

seeds = [SEED_BASE + k * SEED_STRIDE for k in range(N_SEEDS)]
print(f"glass CHART SWEEP  V0/V1/V5 x RQ/PQ  encoding E0  res{RES} {SPP}spp direct-off  "
      f"{N_SEEDS} seeds\n  (var = MSE vs BDPT-1024 ref; mean must stay ~0.225 = unbiased; "
      f"dumping every render -> {DUMP})\n")

var = {(s, c): [] for s, c, _ in cells}
mean_lin = {(s, c): [] for s, c, _ in cells}
for sd in seeds:
    parts = []
    for suff, chart, c in cells:
        img = render(c, sd).astype(np.float64)
        var[(suff, chart)].append(float(np.mean((img - ref) ** 2)))
        mean_lin[(suff, chart)].append(float(img.mean()))
        srgb = np.clip(np.maximum(img * exp, 0) ** (1 / 2.2), 0, 1)
        name = f"glass_bsdf_seed{sd}" if suff == "bsdf" else f"glass_{suff}_{chart}_seed{sd}"
        mpimg.imsave(str(DUMP / f"{name}.png"), srgb)
        parts.append(f"{suff}/{chart}={var[(suff, chart)][-1]:.3e}")
    print(f"  seed {sd}: " + "  ".join(parts))

bsdf_v = var[("bsdf", "-")]
res = {"res": RES, "spp": SPP, "seeds": seeds, "encoding": "E0",
       "note": "charts V0/V1/V5 swept at E0", "cells": {}}
print("\n  per-cell  mean var +/- 95% t-CI   [mean lin]   bsdf/X (paired, >1 = X beats bsdf):")
m_bsdf, h_bsdf = ci(bsdf_v)
print(f"    bsdf        {m_bsdf:.4e} +/- {h_bsdf:.2e}   [{np.mean(mean_lin[('bsdf','-')]):.4f}]")
res["cells"]["bsdf"] = {"var_mean": m_bsdf, "var_ci95": h_bsdf, "var_per_seed": bsdf_v}
bsdf_mean_lin = float(np.mean(mean_lin[("bsdf", "-")]))
for chart in CHARTS:
    for suff, _ in COUPLINGS:
        v = var[(suff, chart)]
        m, h = ci(v)
        ml = float(np.mean(mean_lin[(suff, chart)]))
        ratio = [bsdf_v[i] / v[i] for i in range(N_SEEDS)]
        rm, rh = ci(ratio)
        sig = "  <-- beats bsdf" if (rm - rh) > 1 else ("  WORSE" if (rm + rh) < 1 else "")
        bias = "  !!BIASED" if abs(ml - bsdf_mean_lin) > 0.01 else ""
        print(f"    {suff:4s} {chart}    {m:.4e} +/- {h:.2e}   [{ml:.4f}]{bias}   "
              f"bsdf/X {rm:.3f} +/- {rh:.3f}{sig}")
        res["cells"][f"{suff}_{chart}"] = {
            "var_mean": m, "var_ci95": h, "var_per_seed": v,
            "bsdf_over_X_mean": rm, "bsdf_over_X_ci95": rh, "mean_lin": ml}

print("\n  RQ-vs-PQ per chart  (RQ/PQ paired, <1 = RQ beats PQ):")
res["rq_vs_pq"] = {}
for chart in CHARTS:
    rv, pv = var[("rqs", chart)], var[("pq", chart)]
    r = [rv[i] / pv[i] for i in range(N_SEEDS)]
    rm, rh = ci(r)
    flag = "  RQ wins" if (rm + rh) < 1 else ("  PQ wins" if (rm - rh) > 1 else "  tie")
    print(f"    {chart}:  RQ/PQ {rm:.3f} +/- {rh:.3f}{flag}")
    res["rq_vs_pq"][chart] = {"mean": rm, "ci95": rh}

print("\n  best chart per coupling (lowest mean var):")
for suff, _ in COUPLINGS:
    best = min(CHARTS, key=lambda ch: np.mean(var[(suff, ch)]))
    vs = {ch: float(np.mean(var[(suff, ch)])) for ch in CHARTS}
    print(f"    {suff}: {best}  ({'  '.join(f'{ch}={vs[ch]:.3e}' for ch in CHARTS)})")
    res["cells"].setdefault("_best", {})[suff] = best

outp = OUT / f"chartsweep_glass_{RES}_{SPP}spp_{N_SEEDS}seeds.json"
outp.write_text(json.dumps(res, indent=2))
print(f"\nsaved -> {outp}\n  dumped {len(cells) * N_SEEDS} renders + ref -> {DUMP}")
