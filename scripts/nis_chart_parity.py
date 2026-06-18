"""Chart parity: numpy mirror of the shader charts vs spline_flow CHART.

Validates the neural_flow.slang transliterations of the V0/V1/V5 square<->direction
maps + log-Jacobian against the spline_flow `CHART` reference (the trainer's
ground truth), so a net trained on a chart renders the matching distribution. The
numpy mirrors below are byte-for-byte transliterations of the slang functions
(nf_chart_square_to_dir / nf_chart_dir_to_square / nf_chart_logjac and the V1
nf_square_to_hemi / nf_hemi_to_square). Mirrors nis_pq_parity.py's approach
(numpy mirror vs python; the GPU shader-vs-python check is separate).

Run: PYTHONPATH=src SKINNY_SPLINE_FLOW=<spline_flow> ./bin/python3.13 \
       scripts/nis_chart_parity.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, "src")
SF = os.environ.get("SKINNY_SPLINE_FLOW",
                    "/Users/ahmetbilgili/projects/spline_flow-nis-baseline")
sys.path.insert(0, SF)
import torch  # noqa: E402
from train import CHART  # noqa: E402

PI = np.pi
ZEPS = 1e-5
EPS = 1e-8
LOG2PI = float(np.log(2.0 * PI))
LOGPI2 = float(np.log(PI * PI))


# ---- slang mirrors (transliterations of neural_flow.slang) ----
def _concentric_square_to_disk(z):
    u = np.clip(z[:, 0], 0.0, 1.0); v = np.clip(z[:, 1], 0.0, 1.0)
    a = 2.0 * u - 1.0; b = 2.0 * v - 1.0
    a_dom = np.abs(a) >= np.abs(b)
    r = np.where(a_dom, np.abs(a), np.abs(b))
    a_s = np.where(a == 0.0, 1.0, a); b_s = np.where(b == 0.0, 1.0, b)
    q = PI / 4.0
    right = a_dom & (a >= 0.0); left = a_dom & (a < 0.0); top = (~a_dom) & (b >= 0.0)
    phi = 1.5 * PI + q * (a / (-b_s))
    phi = np.where(right, q * (b / a_s), phi)
    phi = np.where(left, PI - q * (b / (-a_s)), phi)
    phi = np.where(top, 0.5 * PI - q * (a / b_s), phi)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def _disk_to_square(c):
    r = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)
    phi = np.arctan2(c[:, 1], c[:, 0])
    phi = np.where(phi < -PI / 4.0, phi + 2.0 * PI, phi)
    fourpi = 4.0 / PI
    w0 = phi < PI / 4.0
    w1 = (phi >= PI / 4.0) & (phi < 3.0 * PI / 4.0)
    w2 = (phi >= 3.0 * PI / 4.0) & (phi < 5.0 * PI / 4.0)
    a = (phi - 1.5 * PI) * r * fourpi; b = -r
    a = np.where(w0, r, a); b = np.where(w0, phi * r * fourpi, b)
    a = np.where(w1, (0.5 * PI - phi) * r * fourpi, a); b = np.where(w1, r, b)
    a = np.where(w2, -r, a); b = np.where(w2, (PI - phi) * r * fourpi, b)
    u = (a + 1.0) * 0.5; v = (b + 1.0) * 0.5
    return np.stack([np.clip(u, ZEPS, 1.0 - ZEPS), np.clip(v, ZEPS, 1.0 - ZEPS)], axis=1)


def v1_s2d(z):
    c = _concentric_square_to_disk(z)
    r2 = np.clip(c[:, 0] ** 2 + c[:, 1] ** 2, 0.0, 1.0)
    cos_t = 1.0 - r2
    sin_t = np.sqrt(np.maximum(1.0 - cos_t ** 2, 0.0))
    r = np.sqrt(r2)
    cphi = np.where(r > EPS, c[:, 0] / np.maximum(r, EPS), 1.0)
    sphi = np.where(r > EPS, c[:, 1] / np.maximum(r, EPS), 0.0)
    return np.stack([sin_t * cphi, cos_t, sin_t * sphi], axis=1)


def v1_d2s(w):
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    cos_t = np.clip(w[:, 1], -1.0, 1.0)
    r = np.sqrt(np.maximum(1.0 - cos_t, 0.0))
    sin_t = np.sqrt(np.maximum(1.0 - cos_t ** 2, 0.0))
    cphi = np.where(sin_t > EPS, w[:, 0] / np.maximum(sin_t, EPS), 1.0)
    sphi = np.where(sin_t > EPS, w[:, 2] / np.maximum(sin_t, EPS), 0.0)
    return _disk_to_square(np.stack([r * cphi, r * sphi], axis=1))


def v0_s2d(z):
    u = np.clip(z[:, 0], 0.0, 1.0); v = np.clip(z[:, 1], 0.0, 1.0)
    phi = 2.0 * PI * u; cos_t = v
    sin_t = np.sqrt(np.maximum(1.0 - cos_t ** 2, 0.0))
    return np.stack([sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)], axis=1)


def v0_d2s(w):
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    phi = np.arctan2(w[:, 2], w[:, 0])
    phi = np.where(phi < 0.0, phi + 2.0 * PI, phi)
    return np.stack([phi / (2.0 * PI), np.clip(w[:, 1], 0.0, 1.0)], axis=1)


def v5_s2d(z):
    u = np.clip(z[:, 0], 0.0, 1.0); v = np.clip(z[:, 1], 0.0, 1.0)
    phi = 2.0 * PI * u; theta = 0.5 * PI * v
    sin_t = np.sin(theta); cos_t = np.cos(theta)
    return np.stack([sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)], axis=1)


def v5_d2s(w):
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    theta = np.arccos(np.clip(w[:, 1], -1.0, 1.0))
    phi = np.arctan2(w[:, 2], w[:, 0])
    phi = np.where(phi < 0.0, phi + 2.0 * PI, phi)
    u = phi / (2.0 * PI); v = theta / (0.5 * PI)
    return np.stack([np.clip(u, ZEPS, 1.0 - ZEPS), np.clip(v, ZEPS, 1.0 - ZEPS)], axis=1)


def v5_logjac(z):
    v = np.clip(z[:, 1], 0.0, 1.0)
    sin_t = np.maximum(np.sin(0.5 * PI * v), EPS)
    return LOGPI2 + np.log(sin_t)


MIRROR = {
    "V0": (v0_s2d, v0_d2s, lambda z: np.full(z.shape[0], LOG2PI)),
    "V1": (v1_s2d, v1_d2s, lambda z: np.full(z.shape[0], LOG2PI)),
    "V5": (v5_s2d, v5_d2s, v5_logjac),
}

TOL = 1e-5
rng = np.random.default_rng(7)
# sample z off the exact boundary (the chart inverses clamp there)
z = rng.uniform(0.02, 0.98, size=(20000, 2)).astype(np.float64)

print(f"chart parity — numpy slang-mirror vs spline_flow CHART  (tol {TOL:g})\n")
ok = True
for name in ("V0", "V1", "V5"):
    s2d, d2s, ljac = MIRROR[name]
    ref = CHART[name]
    zt = torch.from_numpy(z)
    # forward square -> direction
    w_mir = s2d(z)
    w_ref = ref.square_to_direction(zt, None).numpy()
    e_fwd = float(np.abs(w_mir - w_ref).max())
    # inverse direction -> square (feed the reference's own directions)
    s_mir = d2s(w_ref)
    s_ref = ref.direction_to_square(torch.from_numpy(w_ref), None).numpy()
    e_inv = float(np.abs(s_mir - s_ref).max())
    # log-Jacobian
    lj_mir = ljac(z)
    lj_ref = ref.log_jac(zt)
    lj_ref = (np.full(z.shape[0], float(lj_ref)) if np.isscalar(lj_ref) or
              getattr(lj_ref, "ndim", 0) == 0 else np.asarray(lj_ref).reshape(-1))
    e_jac = float(np.abs(lj_mir - lj_ref).max())
    # round-trip identity (mirror only)
    e_rt = float(np.abs(d2s(s2d(z)) - z).max())
    bad = max(e_fwd, e_inv, e_jac) > TOL
    ok = ok and not bad
    flag = "FAIL" if bad else "ok"
    print(f"  {name}: fwd {e_fwd:.2e}  inv {e_inv:.2e}  logjac {e_jac:.2e}  "
          f"roundtrip {e_rt:.2e}  [{flag}]")

print("\n" + ("PARITY OK — all charts match spline_flow within tol"
              if ok else "PARITY FAILED"))
sys.exit(0 if ok else 1)
