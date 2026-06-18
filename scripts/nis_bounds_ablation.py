"""Controlled A/B: isolate the scene-bounds bug through the REAL build_dataset_np
code path.

Same synthetic Cornell-scale records (world pos in [-0.7, 4.0], a
position-DEPENDENT incident lobe so the net MUST use the position channel). Train
two identical flows that differ ONLY in how build_dataset_np normalises position:
  BROKEN  bounds=(0,1)        -> the online trainer's _bounds() fallback (the bug)
  FIXED   bounds=(bmin,bext)  -> the scene AABB, == shader neuralCondition + the fix
Then evaluate BOTH at INFERENCE normalisation (the AABB the shader always uses) on
held-out positions. If the bug is real, the BROKEN net (trained on a different
position scale than it is queried with) gives a much worse test NLL and fails to
track the position-dependent lobe; the FIXED net tracks it.

Run: PYTHONPATH=src ./bin/python3.13 scripts/nis_bounds_ablation.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "/Users/ahmetbilgili/projects/spline_flow")
import torch  # noqa: E402
from train import ConditionalSplineFlow2D  # noqa: E402
from skinny.sampling.path_records import RECORD_DTYPE  # noqa: E402
from skinny.sampling.training_backends import build_dataset_np  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
BMIN = np.array([-0.7, -0.7, -0.7], np.float32)   # Cornell-scale world AABB
BEXT = np.array([4.7, 4.7, 4.7], np.float32)
NCOND = 9


def make_records(n):
    """Records with a position-dependent target azimuth: phi_t = 2*pi*px_norm."""
    pos = (np.random.rand(n, 3).astype(np.float32) * BEXT + BMIN)
    px_norm = (pos[:, 0] - BMIN[0]) / BEXT[0]            # [0,1], the signal
    phi_t = 2 * np.pi * px_norm + np.random.randn(n) * 0.15
    cth = np.clip(0.7 + np.random.randn(n) * 0.06, 0.05, 0.99)
    sth = np.sqrt(np.maximum(0.0, 1 - cth ** 2))
    wi = np.stack([sth * np.cos(phi_t), cth, sth * np.sin(phi_t)], 1).astype(np.float32)
    recs = np.zeros(n, dtype=RECORD_DTYPE)
    recs["pos"] = pos
    recs["normal"] = np.array([0, 1, 0], np.float32)
    recs["wo"] = np.array([0, 1, 0], np.float32)
    recs["wi_local"] = wi
    recs["contrib"] = np.array([1, 1, 1], np.float32)    # uniform positive weight
    return recs, px_norm


def cond_inference(pos):
    """The shader's neuralCondition position normalisation (AABB → [-1,1])."""
    p = (pos - BMIN) / BEXT * 2 - 1
    N = np.tile([0, 1, 0], (len(pos), 1)); wo = np.tile([0, 1, 0], (len(pos), 1))
    return np.concatenate([p, N, wo], 1).astype(np.float32)


def train_flow(cond, z, w, steps=1200):
    torch.manual_seed(1)
    m = ConditionalSplineFlow2D(cond_dim=NCOND, num_layers=6, num_bins=24, hidden=96)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    ct = torch.from_numpy(cond); zt = torch.from_numpy(z); wt = torch.from_numpy(w)
    n = ct.shape[0]; m.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (4096,))
        opt.zero_grad(set_to_none=True)
        lq = m.log_pdf_square(zt[idx], ct[idx]).squeeze(-1)
        loss = -(wt[idx] * lq).sum() / wt[idx].sum().clamp_min(1e-12)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval(); return m


def test_nll(m, cond_inf, z):
    with torch.no_grad():
        lq = m.log_pdf_square(torch.from_numpy(z), torch.from_numpy(cond_inf)).squeeze(-1)
    return float(-lq.mean())   # lower = better; 0 = uniform


def peak_u(m, pos_row):
    """Argmax-u of q at a single position (does the lobe track position?)."""
    n = 128; xs = (np.arange(n) + 0.5) / n
    gx, gy = np.meshgrid(xs, xs)
    z = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    c = np.tile(cond_inference(pos_row[None, :]), (z.shape[0], 1))
    with torch.no_grad():
        q = torch.exp(m.log_pdf_square(torch.from_numpy(z), torch.from_numpy(c)).squeeze(-1)).numpy()
    return float(gx.ravel()[q.argmax()])   # u of the peak


recs, _ = make_records(60000)
cond_brk, z, w = build_dataset_np(recs, (np.zeros(3, np.float32), np.ones(3, np.float32)))
cond_fix, z2, w2 = build_dataset_np(recs, (BMIN, BEXT))
assert np.allclose(z, z2) and np.allclose(w, w2), "only cond should differ"
print(f"records kept: {z.shape[0]}   broken px-cond range [{cond_brk[:,0].min():.2f},"
      f"{cond_brk[:,0].max():.2f}]   fixed px-cond range [{cond_fix[:,0].min():.2f},{cond_fix[:,0].max():.2f}]")

m_brk = train_flow(cond_brk, z, w)
m_fix = train_flow(cond_fix, z, w)

# held-out test set, evaluated at INFERENCE normalisation (what the shader uses)
tst, _ = make_records(20000)
cinf = cond_inference(tst["pos"]);
# z target for test
from skinny.sampling.training_backends import _hemisphere_to_square_np  # noqa: E402
ztst = np.clip(_hemisphere_to_square_np(tst["wi_local"]), 1e-4, 1 - 1e-4).astype(np.float32)
print(f"\n{'net':24s} test-NLL@inference (lower=better, 0=uniform)")
print(f"{'BROKEN bounds=(0,1)':24s} {test_nll(m_brk, cinf, ztst):+.4f}")
print(f"{'FIXED  bounds=AABB':24s} {test_nll(m_fix, cinf, ztst):+.4f}")

print(f"\nLobe tracking (true peak-u = px_norm):")
print(f"  {'px_norm':>8s} {'BROKEN u':>9s} {'FIXED u':>8s}")
for pn in (0.15, 0.5, 0.85):
    pos = np.array([BMIN[0] + pn * BEXT[0], 1.0, 1.0], np.float32)
    print(f"  {pn:8.2f} {peak_u(m_brk, pos):9.2f} {peak_u(m_fix, pos):8.2f}")
