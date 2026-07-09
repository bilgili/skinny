"""Runtime access to the vendored pbrt-v4 spectral tables.

Two datasets, extracted verbatim from pbrt-v4 by ``_extract_pbrt_spectra.py``
(run once; the ``.npz`` files are checked in):

* the sRGB→spectrum sigmoid-coefficient table (Jakob & Hanika 2019) — the exact
  ``sRGBToSpectrumTable`` pbrt uses, RES=64; :func:`rgb_to_sigmoid_coeffs`
  reproduces pbrt's ``RGBToSpectrumTable::operator()`` trilinear lookup and
  :func:`sigmoid_poly` its ``RGBSigmoidPolynomial`` evaluation.
* the CIE D65 illuminant SPD and the named-metal complex IOR curves
  (Ag/Al/Au/Cu ``eta``/``k``), resampled onto skinny's 360–830 nm / 5 nm grid.

Named-glass dispersion uses a documented Cauchy fit (no bulk table): the pbrt
named glasses are close to BK7-family crowns; ``named_glass_ior`` evaluates
``n(λ) = A + B/λ²`` with published BK7 coefficients — a documented
approximation, adequate for the hero-wavelength dispersion path.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import _normalize_metal_key

_DATA_DIR = Path(__file__).resolve().parent
_LAMBDA = np.arange(360.0, 830.0 + 1.0, 5.0)

# Cauchy coefficients for a BK7-family crown glass (n = A + B/λ², λ in µm).
# n(589 nm) ≈ 1.5168, normal dispersion (blue > red). Documented approximation.
_GLASS_CAUCHY = {"default": (1.5046, 0.00420), "bk7": (1.5046, 0.00420)}


@functools.lru_cache(maxsize=1)
def _load_upsample():
    d = np.load(_DATA_DIR / "rgb2spec_srgb.npz")
    return int(d["res"]), d["scale"].astype(np.float64), d["data"].astype(np.float64)


@functools.lru_cache(maxsize=1)
def _load_curves():
    return {k: v.astype(np.float64) for k, v in np.load(_DATA_DIR / "spectral_curves.npz").items()}


def load_srgb_upsample_table():
    """Return ``(res, scale[res], data[3,res,res,res,3])`` — the pbrt sRGB table."""
    return _load_upsample()


def _find_interval(nodes: np.ndarray, x: float) -> int:
    # pbrt FindInterval: largest i in [0, n-2] with nodes[i] < x.
    i = int(np.searchsorted(nodes, x, side="right")) - 1
    return int(np.clip(i, 0, nodes.size - 2))


def rgb_to_sigmoid_coeffs(rgb) -> np.ndarray:
    """Sigmoid-polynomial coefficients (c0, c1, c2) for an sRGB reflectance.

    Bit-faithful port of pbrt ``RGBToSpectrumTable::operator()`` (color.cpp).
    Input is clamped to [0, 1]; a uniform RGB takes pbrt's closed-form branch.
    """
    res, zn, data = _load_upsample()
    r, g, b = (float(np.clip(c, 0.0, 1.0)) for c in rgb)
    if r == g == b:
        denom = np.sqrt(r * (1.0 - r)) if 0.0 < r < 1.0 else 0.0
        c2 = 0.0 if denom == 0.0 else (r - 0.5) / denom
        return np.array([0.0, 0.0, c2])
    rgb3 = (r, g, b)
    maxc = 0 if (r > g and r > b) else (1 if g > b else 2)
    z = rgb3[maxc]
    x = rgb3[(maxc + 1) % 3] * (res - 1) / z
    y = rgb3[(maxc + 2) % 3] * (res - 1) / z
    xi = min(int(x), res - 2)
    yi = min(int(y), res - 2)
    zi = _find_interval(zn, z)
    dx, dy = x - xi, y - yi
    dz = (z - zn[zi]) / (zn[zi + 1] - zn[zi])

    def lerp(t, a, b):
        return a + t * (b - a)

    out = np.empty(3)
    for i in range(3):
        def co(ddx, ddy, ddz, _i=i):
            return data[maxc, zi + ddz, yi + ddy, xi + ddx, _i]

        out[i] = lerp(
            dz,
            lerp(dy, lerp(dx, co(0, 0, 0), co(1, 0, 0)), lerp(dx, co(0, 1, 0), co(1, 1, 0))),
            lerp(dy, lerp(dx, co(0, 0, 1), co(1, 0, 1)), lerp(dx, co(0, 1, 1), co(1, 1, 1))),
        )
    return out


def sigmoid_poly(coeffs, lam_nm) -> np.ndarray:
    """Evaluate the sigmoid polynomial S(λ) = s(c0λ²+c1λ+c2) (pbrt convention)."""
    c0, c1, c2 = coeffs
    lam = np.asarray(lam_nm, dtype=np.float64)
    poly = c2 + lam * (c1 + lam * c0)
    return 0.5 + poly / (2.0 * np.sqrt(1.0 + poly * poly))


def d65_spd() -> np.ndarray:
    """CIE D65 relative SPD on the internal 5 nm grid (pbrt values)."""
    return _load_curves()["d65"].copy()


def named_metal_spectrum(name: str):
    """Return ``(eta_on_lambda, k_on_lambda)`` for a named metal, or None."""
    key = _normalize_metal_key(name)
    aliases = {"gold": "au", "silver": "ag", "aluminium": "al", "aluminum": "al", "copper": "cu"}
    key = aliases.get(key, key)
    curves = _load_curves()
    if f"{key}_eta" not in curves:
        return None
    return curves[f"{key}_eta"].copy(), curves[f"{key}_k"].copy()


def named_glass_ior(name: str, lam_nm):
    """Cauchy dispersion n(λ) for a named glass, or None. Documented BK7 fit."""
    n = (name or "").strip().lower()
    for prefix in ("glass-", "glass_"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    coeff = _GLASS_CAUCHY.get(n, _GLASS_CAUCHY["default"])
    a, b = coeff
    lam_um = np.asarray(lam_nm, dtype=np.float64) * 1e-3
    return a + b / (lam_um * lam_um)
