"""Offline diagnostic: does a trained NFW1 net encode a PEAKED directional
distribution, or is it ~uniform (a no-op proposal)?

Loads NFW1 weights into the spline_flow ConditionalSplineFlow2D (exactly as the
renderer's TorchTrainingBackend.warm_start), evaluates log q on the unit square
over a grid for several conditions, and reports:
  - norm check  ∫ q dz  (should be ~1)
  - concentration  C = ∫ q log q dz  (nats; 0 = uniform, >0 = peaked)
  - peak/mean density ratio
  - conditioning sensitivity: how much q changes between two conditions.

Run:
  PYTHONPATH=src ./bin/python3.13 scripts/nis_eval_net_density.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ahmetbilgili/projects/skinny")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/ahmetbilgili/projects/spline_flow")

import torch  # noqa: E402
from train import ConditionalSplineFlow2D  # noqa: E402
from skinny.sampling.neural_weights import (  # noqa: E402
    load_neural_weights, NF_COND,
)

_LINEAR_IDX = (0, 2, 4)


def build_flow_from_nfw1(path):
    nw = load_neural_weights(path)
    model = ConditionalSplineFlow2D(cond_dim=NF_COND, num_layers=nw.layers,
                                    num_bins=nw.bins, hidden=nw.hidden)
    hi = 0
    with torch.no_grad():
        for coupling in model.layers:
            for li in _LINEAR_IDX:
                w_off, b_off, in_dim, out_dim = (int(x) for x in nw.headers[hi]); hi += 1
                w = nw.weights[w_off:w_off + out_dim * in_dim].reshape(out_dim, in_dim)
                b = nw.biases[b_off:b_off + out_dim]
                coupling.net[li].weight.copy_(torch.from_numpy(np.ascontiguousarray(w, np.float32)))
                coupling.net[li].bias.copy_(torch.from_numpy(np.ascontiguousarray(b, np.float32)))
    model.eval()
    return model


def grid_density(model, cond_vec, n=64):
    """q(z|cond) on an n×n grid of the unit square. cond_vec: [9]."""
    xs = (np.arange(n) + 0.5) / n
    gx, gy = np.meshgrid(xs, xs)
    z = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    zt = torch.from_numpy(z)
    ct = torch.from_numpy(np.asarray(cond_vec, np.float32)[None, :])
    with torch.no_grad():
        logq = model.log_pdf_square(zt, ct).squeeze(-1).numpy()
    q = np.exp(logq)
    return q  # length n*n, cell area = 1/(n*n)


def stats(q, n=64):
    area = 1.0 / (n * n)
    integral = float(q.sum() * area)            # ~1 if normalized
    qln = float((q * np.log(np.clip(q, 1e-12, None))).sum() * area)  # ∫ q log q
    return integral, qln, float(q.max() / max(q.mean(), 1e-12))


# A spread of plausible Cornell/glass conditions: pos in [-1,1]^3, unit N, unit wo.
def unit(v):
    v = np.asarray(v, np.float64); return v / np.linalg.norm(v)

CONDS = {
    "floor_up_steepwo":  [0.0, -0.8, 0.0, *unit([0, 1, 0]), *unit([0.1, 0.9, 0.3])],
    "floor_up_grazewo":  [0.0, -0.8, 0.0, *unit([0, 1, 0]), *unit([0.9, 0.2, 0.1])],
    "leftwall_right":    [-0.9, 0.0, 0.0, *unit([1, 0, 0]), *unit([0.8, 0.3, 0.2])],
    "backwall_fwd":      [0.0, 0.0, -0.9, *unit([0, 0, 1]), *unit([0.2, 0.3, 0.9])],
}

NETS = {
    "v17_cornell": ".skinny_neural/weights_v000017.nfw1",
    "v22_glass":   ".skinny_neural/weights_v000022.nfw1",
}

print(f"torch {torch.__version__}  (cpu)\n")
# random-init reference (no weight load) = "what an untrained flow looks like"
ref = ConditionalSplineFlow2D(cond_dim=NF_COND, num_layers=6, num_bins=24, hidden=96).eval()
print("=== random-init reference flow ===")
for cname, cv in CONDS.items():
    integ, conc, pk = stats(grid_density(ref, cv))
    print(f"  {cname:18s} ∫q={integ:.3f}  C=∫qlnq={conc:+.4f} nats  peak/mean={pk:.2f}")

for name, path in NETS.items():
    print(f"\n=== {name}  ({path}) ===")
    m = build_flow_from_nfw1(path)
    qs = {}
    for cname, cv in CONDS.items():
        q = grid_density(m, cv)
        qs[cname] = q
        integ, conc, pk = stats(q)
        print(f"  {cname:18s} ∫q={integ:.3f}  C=∫qlnq={conc:+.4f} nats  peak/mean={pk:.2f}")
    # conditioning sensitivity: L1 distance between density at different conds
    keys = list(qs)
    area = 1.0 / q.size
    d = float(np.abs(qs[keys[0]] - qs[keys[1]]).sum() * area)
    d2 = float(np.abs(qs[keys[0]] - qs[keys[3]]).sum() * area)
    print(f"  cond-sensitivity  L1(floor_steep, floor_graze)={d:.3f}   "
          f"L1(floor_steep, backwall)={d2:.3f}   (0=ignores condition)")
