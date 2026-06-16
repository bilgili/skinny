"""Confirm the lever: train the SHIPPED flow arch on a synthetic radiance-weighted
dataset and watch concentration C=∫q ln q grow with training steps.

If the deployed online net (C~0.30 nats, scene-agnostic) is merely UNDERTRAINED,
the same arch+loss should reach much higher C here with more steps. This isolates
training BUDGET as the bottleneck (vs a fundamental arch/wiring limit).

Synthetic data mimics an online guiding record stream: directions drawn from the
BSDF (cosine) sampler, contribution-weighted toward a fixed incident lobe
(condition-dependent), exactly the (z, w) contract of build_dataset_np.

Run: PYTHONPATH=src ./bin/python3.13 scripts/nis_train_curve.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ahmetbilgili/projects/skinny")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/ahmetbilgili/projects/spline_flow")

import torch  # noqa: E402
from train import ConditionalSplineFlow2D  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
NCOND = 9


def cosine_hemisphere(n):
    u1 = np.random.rand(n); u2 = np.random.rand(n)
    r = np.sqrt(u1); phi = 2 * np.pi * u2
    x = r * np.cos(phi); z = r * np.sin(phi); y = np.sqrt(np.maximum(0.0, 1 - u1))
    return np.stack([x, y, z], axis=1)  # y-up flow-local


def hemi_to_square(w):
    w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-8)
    phi = np.arctan2(w[:, 2], w[:, 0]); phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.stack([phi / (2 * np.pi), np.clip(w[:, 1], 0, 1)], axis=1)


def make_batch(n, lobe_dir, kappa=8.0):
    """BSDF(cosine)-sampled dirs, contribution-weighted toward lobe_dir (vMF-like).
    Returns cond[n,9], z[n,2], w[n] (mean-normalised), as build_dataset_np does."""
    wi = cosine_hemisphere(n)
    d = lobe_dir / np.linalg.norm(lobe_dir)
    w = np.exp(kappa * (wi @ d - 1.0))                 # incident-radiance weight
    cond = np.tile(np.array([0, -0.8, 0, 0, 1, 0, 0.1, 0.9, 0.3], np.float32), (n, 1))
    z = np.clip(hemi_to_square(wi), 1e-4, 1 - 1e-4).astype(np.float32)
    keep = w > 0
    w = (w / max(w.mean(), 1e-12)).astype(np.float32)
    return cond[keep], z[keep], w[keep]


def concentration(model, cond_vec, n=64):
    xs = (np.arange(n) + 0.5) / n
    gx, gy = np.meshgrid(xs, xs)
    z = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))
    c = torch.from_numpy(np.asarray(cond_vec, np.float32)[None, :])
    with torch.no_grad():
        q = torch.exp(model.log_pdf_square(z, c).squeeze(-1)).numpy()
    area = 1.0 / (n * n)
    return float((q * np.log(np.clip(q, 1e-12, None))).sum() * area), float(q.max() / q.mean())


LOBE = np.array([0.3, 0.7, 0.6])  # a fixed incident lobe (the "true" radiance dir)
model = ConditionalSplineFlow2D(cond_dim=NCOND, num_layers=6, num_bins=24, hidden=96)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
cond_eval = [0, -0.8, 0, 0, 1, 0, 0.1, 0.9, 0.3]

print("steps | last NLL |   C=∫q ln q | peak/mean")
C0, pk0 = concentration(model, cond_eval)
print(f"   0  |    --    |   {C0:+.3f}    |  {pk0:.2f}   (random init)")
milestones = {50, 200, 1000, 3000}
model.train()
for step in range(1, 3001):
    cond, z, w = make_batch(4096, LOBE)
    ct = torch.from_numpy(cond); zt = torch.from_numpy(z); wt = torch.from_numpy(w)
    opt.zero_grad(set_to_none=True)
    logq = model.log_pdf_square(zt, ct).squeeze(-1)
    loss = -(wt * logq).sum() / wt.sum().clamp_min(1e-12)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    if step in milestones:
        model.eval(); C, pk = concentration(model, cond_eval); model.train()
        print(f" {step:4d}  |  {float(loss):+.3f}  |   {C:+.3f}    |  {pk:.2f}")
