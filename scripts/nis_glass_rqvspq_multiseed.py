"""Glass RQ-vs-PQ head-to-head, MULTI-SEED (Metal, direct off).

Firms up the single-seed headline (RQ+clamp ~-10% vs PQ, ~-9% vs bsdf on glass)
with confidence intervals. Renders bsdf / bsdf+neural(RQ) / bsdf+neural(PQ) at N
independent seeds, computes per-seed variance = MSE vs the clean BDPT 1024spp
reference, then reports:
  - per-cell mean var +/- 95% t-CI across seeds,
  - PAIRED ratios (same seed shared across cells = common random numbers): the
    bsdf/RQ, bsdf/PQ, RQ/PQ ratio computed per seed, then mean +/- 95% t-CI.
The paired ratio is the statistically right comparator: shared seeds correlate
the noise, shrinking the ratio's variance.

Nets are offline-trained on the SAME glass records (V1 chart) so only the
coupling differs. Default NETTAG=glass_offc99 (firefly-clamped p99 = the headline
"my method"); set NIS_NETTAG=glass_off for the unclamped control.

Run (no Vulkan needed, Metal): PYTHONPATH=src:tests:scripts <py> \
    scripts/nis_glass_rqvspq_multiseed.py [N_SEEDS]
"""
import sys
import os
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "tests"); sys.path.insert(0, "scripts")
import guiding_variance_sweep as gv  # noqa: E402

NN = "/Users/ahmetbilgili/projects/skinny/.skinny_neural"
BDPT = "/Users/ahmetbilgili/projects/skinny/renders_bdpt/glass_bdpt_ref_1024spp_nodirect_metal.npy"
OUT = Path("/Users/ahmetbilgili/projects/skinny/renders_rqvspq"); OUT.mkdir(exist_ok=True)
RES, SPP = 512, 64                 # 512^2 == BDPT ref res (no downsample)
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEED_BASE, SEED_STRIDE = 1000, 1000   # streams [base+k*stride, +SPP); stride>>SPP = disjoint
SCENE = gv.ASSETS / "glass_caustics_test.usda"
NETTAG = os.environ.get("NIS_NETTAG", "glass_offc99")  # clamped headline by default

# Student-t 0.975 quantiles (two-sided 95%) by degrees of freedom = N-1.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}


def tcrit(n):
    return T975.get(n - 1, 1.96)


def ci(vals):
    """mean, half-width of 95% t-CI."""
    a = np.asarray(vals, float)
    n = len(a)
    if n < 2:
        return float(a.mean()), float("nan")
    sem = a.std(ddof=1) / np.sqrt(n)
    return float(a.mean()), float(tcrit(n) * sem)


def cell(proposals, coupling=None, net=None):
    c = {"proposals": proposals, "reuse": "none", "chart": "V1",
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


cells = [("bsdf", cell("bsdf")),
         ("rqs",  cell("bsdf,neural", "rqs",    f"{NN}/{NETTAG}_rqs.nfw1")),
         ("pq",   cell("bsdf,neural", "nis-pq", f"{NN}/{NETTAG}_pq.nfw1"))]

ref = downsample(np.load(BDPT).astype(np.float64), RES)
seeds = [SEED_BASE + k * SEED_STRIDE for k in range(N_SEEDS)]
print(f"glass RQ-vs-PQ MULTI-SEED  net={NETTAG}  res{RES} {SPP}spp direct-off  "
      f"{N_SEEDS} seeds {seeds}\n  (var = MSE vs BDPT-1024 ref; paired = same seed across cells)\n")

# var[tag] = list of per-seed MSE; ratios paired per seed.
var = {t: [] for t, _ in cells}
mean_lin = {t: [] for t, _ in cells}
for s in seeds:
    row = {}
    for tag, c in cells:
        img = render(c, s).astype(np.float64)
        row[tag] = img
        var[tag].append(float(np.mean((img - ref) ** 2)))
        mean_lin[tag].append(float(img.mean()))
    print(f"  seed {s}: " + "  ".join(
        f"{t}={var[t][-1]:.4e}" for t, _ in cells))

print("\n  per-cell  mean var +/- 95% t-CI (mean linear in [], unbiasedness check):")
res = {"net": NETTAG, "res": RES, "spp": SPP, "seeds": seeds, "cells": {}}
for tag, _ in cells:
    m, h = ci(var[tag])
    ml, _ = ci(mean_lin[tag])
    print(f"    {tag:5s} {m:.4e} +/- {h:.2e}   [mean {ml:.4f}]")
    res["cells"][tag] = {"var_mean": m, "var_ci95": h, "var_per_seed": var[tag],
                         "mean_lin": ml}

# paired ratios (per seed), then mean +/- CI
pairs = [("bsdf/rqs", "bsdf", "rqs"), ("bsdf/pq", "bsdf", "pq"),
         ("rqs/pq", "rqs", "pq")]
print("\n  PAIRED ratios  mean +/- 95% t-CI  (bsdf/X >1 = X beats bsdf; rqs/pq <1 = RQ beats PQ):")
res["paired_ratios"] = {}
for name, a, b in pairs:
    per = [var[a][i] / var[b][i] for i in range(N_SEEDS)]
    m, h = ci(per)
    flag = ""
    if name == "rqs/pq":
        flag = "  <-- RQ wins" if (m + h) < 1 else ("  (CI crosses 1)" if (m - h) < 1 else "  PQ wins")
    print(f"    {name:9s} {m:.3f} +/- {h:.3f}{flag}")
    res["paired_ratios"][name] = {"mean": m, "ci95": h, "per_seed": per}

# also ratio-of-means for reference
print("\n  ratio-of-means (unpaired, for reference):")
vm = {t: float(np.mean(var[t])) for t, _ in cells}
print(f"    bsdf/rqs={vm['bsdf']/vm['rqs']:.3f}  bsdf/pq={vm['bsdf']/vm['pq']:.3f}  "
      f"rqs/pq={vm['rqs']/vm['pq']:.3f}")

outp = OUT / f"multiseed_{NETTAG}_{RES}_{SPP}spp_{N_SEEDS}seeds.json"
outp.write_text(json.dumps(res, indent=2))
print(f"\nsaved -> {outp}")
