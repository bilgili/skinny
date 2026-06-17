"""Parity: my shader nf_decode_pq / nf_pq_fwd / nf_pq_inv (numpy mirror, matching
the Slang line-for-line) vs spline_flow's reference _decode_pq / pq_forward /
pq_inverse. Validates the coupling MATH before any GPU render.

Run: PYTHONPATH=src ./bin/python3.13 scripts/nis_pq_parity.py
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/ahmetbilgili/projects/spline_flow-nis-baseline")
import torch  # noqa: E402
from train import pq_forward, pq_inverse  # noqa: E402  (reference)

K = 24
EPS = 1e-8


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)   # stable


# ---- numpy mirror of the SHADER (neural_flow.slang) ----
def sh_decode_pq(params):
    mw = params[:K].max()
    w = np.exp(params[:K] - mw); w = w / w.sum()
    w = 1e-4 + w; w = w / w.sum()
    v = softplus(params[K:]) + 1e-3                      # K+1
    area = max((w * 0.5 * (v[:-1] + v[1:])).sum(), 1e-8)
    return w, v / area


def sh_pq_fwd(x, w, v):
    x = np.clip(x, 0, 1)
    xk = np.concatenate([[0.0], np.cumsum(w)])
    yk = np.concatenate([[0.0], np.cumsum(w * 0.5 * (v[:-1] + v[1:]))])
    idx = min(max(int(np.sum(x >= xk) - 1), 0), K - 1)
    t = np.clip((x - xk[idx]) / max(w[idx], EPS), 0, 1)
    v0, v1 = v[idx], v[idx + 1]
    y = yk[idx] + max(w[idx], EPS) * (v0 * t + 0.5 * (v1 - v0) * t * t)
    return np.clip(y, 0, 1)


def sh_pq_inv(y, w, v):
    y = np.clip(y, 0, 1)
    xk = np.concatenate([[0.0], np.cumsum(w)])
    yk = np.concatenate([[0.0], np.cumsum(w * 0.5 * (v[:-1] + v[1:]))])
    idx = min(max(int(np.sum(y >= yk) - 1), 0), K - 1)
    wi = max(w[idx], EPS); v0, v1 = v[idx], v[idx + 1]
    a = wi * 0.5 * (v1 - v0); b = wi * v0; c = -(y - yk[idx])
    disc = max(b * b - 4 * a * c, 0.0)
    t_quad = (2 * c) / min(-b - np.sqrt(disc), -EPS)
    t_lin = -c / max(b, EPS)
    t = np.clip(t_lin if abs(a) < EPS else t_quad, 0, 1)
    return np.clip(xk[idx] + t * wi, 0, 1)


def ref_decode_pq(params):
    """spline_flow _decode_pq (torch), single sample."""
    raw = torch.tensor(params, dtype=torch.float64)
    w = torch.softmax(raw[:K], 0); w = 1e-4 + w; w = w / w.sum()
    v = torch.nn.functional.softplus(raw[K:]) + 1e-3
    area = (w * 0.5 * (v[:-1] + v[1:])).sum()
    return w.numpy(), (v / area.clamp_min(1e-8)).numpy()


rng = np.random.default_rng(0)
max_dw = max_dv = max_dfwd = max_dinv = 0.0
for trial in range(2000):
    params = rng.standard_normal(2 * K + 1).astype(np.float64) * 1.5
    w_s, v_s = sh_decode_pq(params)
    w_r, v_r = ref_decode_pq(params)
    max_dw = max(max_dw, np.abs(w_s - w_r).max())
    max_dv = max(max_dv, np.abs(v_s - v_r).max())
    # forward/inverse vs reference (reference expects [B,D,*] shapes)
    x = float(rng.random())
    W = torch.tensor(w_r)[None, None]; V = torch.tensor(v_r)[None, None]
    y_ref = float(pq_forward(torch.tensor([[x]], dtype=torch.float64), W, V)[0][0, 0])
    y_sh = sh_pq_fwd(x, w_r, v_r)
    max_dfwd = max(max_dfwd, abs(y_sh - y_ref))
    yq = float(rng.random())
    x_ref = float(pq_inverse(torch.tensor([[yq]], dtype=torch.float64), W, V)[0][0, 0])
    x_sh = sh_pq_inv(yq, w_r, v_r)
    max_dinv = max(max_dinv, abs(x_sh - x_ref))

print(f"trials=2000  max|Δ| vs spline_flow reference:")
print(f"  decode widths   : {max_dw:.2e}")
print(f"  decode vertices : {max_dv:.2e}")
print(f"  pq_forward      : {max_dfwd:.2e}")
print(f"  pq_inverse      : {max_dinv:.2e}")
ok = max(max_dw, max_dv, max_dfwd, max_dinv) < 1e-6
print("PARITY", "OK" if ok else "FAIL")
