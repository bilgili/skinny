"""Runtime access to the vendored pbrt-v4 spectral tables.

Two datasets, extracted verbatim from pbrt-v4 by ``_extract_pbrt_spectra.py``
(run once; the ``.npz`` files are checked in):

* the sRGB→spectrum sigmoid-coefficient table (Jakob & Hanika 2019) — the exact
  ``sRGBToSpectrumTable`` pbrt uses, RES=64; :func:`rgb_to_sigmoid_coeffs`
  reproduces pbrt's ``RGBToSpectrumTable::operator()`` trilinear lookup and
  :func:`sigmoid_poly` its ``RGBSigmoidPolynomial`` evaluation.
* the named illuminant SPDs, the named-metal complex IOR curves
  (Ag/Al/Au/Cu/CuZn/MgO/TiO2 ``eta``/``k``) and the named-glass ``eta`` curves,
  resampled onto skinny's 360–830 nm / 5 nm grid.

Named-glass dispersion evaluates ``n(λ) = A + B/λ_µm²`` from **per-glass**
Cauchy coefficients least-squares fit to pbrt's own tabulated eta (see
``_extract_pbrt_spectra.fit_cauchy``), so every pbrt named glass carries its own
dispersion and its own d-line index. Max fit residual over the visible is 3e-4
(BK7) to 7.5e-3 (F11); the raw curves are vendored so the residual is tested
hostlessly. A 2-term Cauchy is deliberate — a third term was measured and does
not improve the fit (the residual is interpolation error in pbrt's sparse table)
while costing a `FlatMaterialParams` layout change.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import _normalize_metal_key

_DATA_DIR = Path(__file__).resolve().parent
_LAMBDA = np.arange(360.0, 830.0 + 1.0, 5.0)

# Per-glass Cauchy coefficients (A, B) for n(λ) = A + B/λ_µm², least-squares fit
# to pbrt's tabulated eta over 360-830 nm. Regenerate with
# `python -m skinny.pbrt.data._extract_pbrt_spectra --print-tables`.
#
# "default" is the fallback for a named-but-unrecognised glass and is BK7's fit
# *by definition* — pbrt is the single source of truth for every entry here, so
# the fallback reuses a real glass rather than inventing an unsourced constant.
# What removes the silence is the import-time APPROX note naming the
# substitution (spectra.named_glass_key), not a distinct number.
_GLASS_CAUCHY = {
    "default": (1.50431, 0.004267),  # == bk7
    "bk7": (1.50431, 0.004267),  # max resid 3.1e-04
    "baf10": (1.64775, 0.007720),  # max resid 1.4e-03
    "fk51a": (1.47768, 0.003035),  # max resid 1.7e-04
    "lasf9": (1.80852, 0.014634),  # max resid 4.6e-03
    "f5": (1.63949, 0.011655),  # max resid 4.0e-03
    "f10": (1.68848, 0.013994),  # max resid 5.9e-03
    "f11": (1.73547, 0.017399),  # max resid 7.5e-03
    # Schott catalogue names for the same glasses. pbrt's public name for the dense
    # flints is `glass-F5`/`-F10`/`-F11`, but it reads them from the arrays
    # `GlassSF5_eta`/`GlassSF10_eta`/`GlassSF11_eta` — SF11 *is* F11. Accepting both
    # spellings keeps a hand-authored `glass_dispersion = "sf11"` from silently
    # falling back to BK7 (i.e. losing its dispersion entirely).
    "sf5": (1.63949, 0.011655),  # == f5
    "sf10": (1.68848, 0.013994),  # == f10
    "sf11": (1.73547, 0.017399),  # == f11
}

# Refractive index at the sodium d-line (589.3 nm) — the scalar IOR a glass is
# quoted by, and what the RGB (non-spectral) build renders with. Read from
# pbrt's tabulated eta, same provenance as _GLASS_CAUCHY.
_GLASS_IOR_D = {
    "default": 1.51673,  # == bk7
    "bk7": 1.51673,
    "baf10": 1.66988,
    "fk51a": 1.48651,
    "lasf9": 1.85004,
    "f5": 1.67254,
    "f10": 1.72806,
    "f11": 1.78448,
    "sf5": 1.67254,  # == f5   (Schott spelling, see _GLASS_CAUCHY)
    "sf10": 1.72806,  # == f10
    "sf11": 1.78448,  # == f11
}


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
        # Uniform RGB → constant spectrum s(c2). pbrt lets the r∈{0,1} endpoints
        # divide by zero and ride the IEEE ±inf into sigmoid → {0, 1}; a finite
        # ±1e9 reproduces that limit (sigmoid(±1e9) ≈ {1, 0} to ~1e-9). The prior
        # `c2 = 0` collapsed BOTH endpoints to 0.5, so pure white reflected 50 %
        # (white furnace never closed) and pure black reflected 50 %.
        if r >= 1.0:
            c2 = 1.0e9
        elif r <= 0.0:
            c2 = -1.0e9
        else:
            c2 = (r - 0.5) / np.sqrt(r * (1.0 - r))
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


def normalize_glass_key(name: str) -> str:
    """Strip a ``glass-``/``glass_`` prefix and lower-case; ``""`` for a falsy name."""
    n = (name or "").strip().lower()
    for prefix in ("glass-", "glass_"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    return n


def glass_is_known(name: str) -> bool:
    """True if *name* names a glass with its own vendored coefficients.

    ``"default"`` is a fallback, not a pbrt glass, so it reads as unknown — the
    caller reports the substitution rather than silently adopting BK7's
    dispersion for an unrecognised name.
    """
    key = normalize_glass_key(name)
    return bool(key) and key != "default" and key in _GLASS_CAUCHY


def named_glass_ior(name: str, lam_nm):
    """Cauchy dispersion n(λ) for a named glass. Unknown names fall back to BK7."""
    a, b = _GLASS_CAUCHY.get(normalize_glass_key(name), _GLASS_CAUCHY["default"])
    lam_um = np.asarray(lam_nm, dtype=np.float64) * 1e-3
    return a + b / (lam_um * lam_um)


def named_glass_cauchy(name: str):
    """Return the Cauchy coefficients ``(A, B)`` for a named glass, or ``None``.

    The dispersion law is ``n(λ_µm) = A + B / λ_µm²`` — the same fit
    :func:`named_glass_ior` evaluates. Each recognised pbrt glass resolves to its
    **own** coefficients; an unknown-but-present name falls back to ``"default"``
    (== BK7) so the GPU packer always has a usable dispersion. Returns ``None``
    only for a ``None``/empty name. Use :func:`glass_is_known` to tell a
    recognised glass from a fallback — this function cannot report that.
    """
    if not name or not name.strip():
        return None
    a, b = _GLASS_CAUCHY.get(normalize_glass_key(name), _GLASS_CAUCHY["default"])
    return float(a), float(b)


def named_glass_ior_d(name: str):
    """Scalar refractive index at the sodium d-line (589.3 nm), or ``None``.

    This is what the RGB (non-spectral) build renders a named glass with; the
    spectral build substitutes the Cauchy base index instead. Unknown-but-present
    names fall back to ``"default"`` (== BK7), mirroring :func:`named_glass_cauchy`.
    """
    if not name or not name.strip():
        return None
    return float(_GLASS_IOR_D.get(normalize_glass_key(name), _GLASS_IOR_D["default"]))


def named_illuminant_spectrum(name: str):
    """Vendored SPD for a pbrt named illuminant on the 5 nm grid, or ``None``.

    Covers ``stdillum-A``/``-D50``/``-D65``/``-F1``…``-F12`` and ``illum-acesD60``
    (pbrt's scene-addressable illuminants; the ``canon_*``/``ilford_*`` entries in
    pbrt's table are camera sensor responses, not scene spectra, and are out of
    scope). Matching is case-insensitive. ``stdillum-D65`` aliases the ``d65``
    array rather than storing a second copy of the same pbrt symbol.

    The SPD is the raw pbrt curve — pbrt's load-time ``normalize`` is not applied
    (see ``_extract_pbrt_spectra``); every consumer rescales, so absolute scale
    cancels.
    """
    if not name or not name.strip():
        return None
    key = name.strip().lower()
    if key in ("stdillum-d65", "d65"):
        return d65_spd()
    curves = _load_curves()
    lookup = {f"illum_{k}".lower(): k for k in
              (n[len("illum_"):] for n in curves if n.startswith("illum_"))}
    arr = lookup.get(f"illum_{key}")
    return curves[f"illum_{arr}"].copy() if arr is not None else None
