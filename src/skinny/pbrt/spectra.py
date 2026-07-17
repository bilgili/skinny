"""Spectrum -> linear-RGB reduction for the pbrt importer.

skinny is an RGB renderer, so every pbrt spectrum is reduced to linear RGB:

* ``rgb`` parameters pass through unchanged (the dominant case).
* ``blackbody [T]`` -> Planckian SPD integrated against the CIE XYZ colour
  matching functions, then XYZ -> linear sRGB, normalised to unit luminance
  (chromaticity only; magnitude comes from the light's ``L``/``scale``).
* sampled ``spectrum [l v l v ...]`` -> integrated to XYZ -> linear RGB
  (illuminant: direct; reflectance: under an equal-energy whitepoint —
  a documented simplification). A constant SPD short-circuits to the
  achromatic ``[v, v, v]``, matching pbrt's achromatic treatment.
* named conductor spectra (``metal-Au-eta`` / ``-k``) -> tabulated RGB IOR.

The CIE CMFs use the Wyman-Sloan-Shirley (JCGT 2013) analytic multi-Gaussian
fit, so no bulk spectral table is vendored. The residual RGB-vs-spectral
divergence is inherent and documented in the parity matrix.
"""

from __future__ import annotations

import numpy as np

from .data import NAMED_METAL_IOR, _normalize_metal_key
from .data import spectral_tables as _st

# Canonical named-conductor keys the spectral path binds vendored eta/k curves for
# (spectral_tables.named_metal_spectrum). The alias map mirrors that function so
# the importer, this module, and spectral_tables agree on normalization.
_CONDUCTOR_CANON = frozenset({"au", "ag", "al", "cu", "cuzn", "mgo", "tio2"})
_CONDUCTOR_ALIASES = {
    "gold": "au", "silver": "ag", "aluminium": "al", "aluminum": "al", "copper": "cu",
    "brass": "cuzn",
}

# XYZ (D65) -> linear sRGB / Rec.709
_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

_LAMBDA = np.arange(360.0, 830.0 + 1.0, 5.0)  # nm sample grid

# numpy 2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _gauss(x, mu, sigma1, sigma2):
    sigma = np.where(x < mu, sigma1, sigma2)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cie_xyz_bar(lam):
    """CIE 1931 2-deg colour-matching functions (Wyman et al. analytic fit)."""
    lam = np.asarray(lam, dtype=np.float64)
    x = (
        1.056 * _gauss(lam, 599.8, 37.9, 31.0)
        + 0.362 * _gauss(lam, 442.0, 16.0, 26.7)
        - 0.065 * _gauss(lam, 501.1, 20.4, 26.2)
    )
    y = 0.821 * _gauss(lam, 568.8, 46.9, 40.5) + 0.286 * _gauss(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _gauss(lam, 437.0, 11.8, 36.0) + 0.681 * _gauss(lam, 459.0, 26.0, 13.8)
    return x, y, z


_XBAR, _YBAR, _ZBAR = cie_xyz_bar(_LAMBDA)
_Y_INTEGRAL = _trapz(_YBAR, _LAMBDA)


def xyz_to_linear_srgb(xyz) -> np.ndarray:
    return _XYZ_TO_SRGB @ np.asarray(xyz, dtype=np.float64)


def spd_to_xyz(values) -> np.ndarray:
    """Integrate an SPD sampled on the internal grid to (X, Y, Z)."""
    values = np.asarray(values, dtype=np.float64)
    x = _trapz(values * _XBAR, _LAMBDA)
    y = _trapz(values * _YBAR, _LAMBDA)
    z = _trapz(values * _ZBAR, _LAMBDA)
    return np.array([x, y, z])


def planck(lam_nm, temperature_k: float) -> np.ndarray:
    """Planck blackbody spectral radiance (relative), wavelength in nm."""
    lam = np.asarray(lam_nm, dtype=np.float64) * 1e-9  # -> metres
    h = 6.62606957e-34
    c = 299792458.0
    kb = 1.3806488e-23
    return (2.0 * h * c * c) / (lam**5 * (np.exp(h * c / (lam * kb * temperature_k)) - 1.0))


def blackbody_rgb(temperature_k: float) -> np.ndarray:
    """Linear-RGB chromaticity of a blackbody at *temperature_k*, unit luminance."""
    spd = planck(_LAMBDA, temperature_k)
    xyz = spd_to_xyz(spd)
    if xyz[1] > 0:
        xyz = xyz / xyz[1]  # normalise luminance Y -> 1
    rgb = xyz_to_linear_srgb(xyz)
    return np.clip(rgb, 0.0, None)


def sampled_spectrum_to_rgb(pairs, *, illuminant: bool = False) -> np.ndarray:
    """Reduce a pbrt sampled spectrum ``[l v l v ...]`` to linear RGB.

    A constant SPD (all sample values exactly equal) is achromatic in pbrt, so
    it maps straight to ``[v, v, v]`` — projecting it through XYZ would tint it
    because the equal-energy whitepoint is not the sRGB (D65) white. Colored
    spectra are integrated to XYZ and divided by the CMF integral, matching
    pbrt's ``SpectrumToXYZ`` (which divides by ``CIE_Y_integral`` for *every*
    spectrum). Reflectance is thereby taken under an equal-energy whitepoint — a
    documented simplification.

    The division applies to both branches so the projection stays continuous with
    the achromatic shortcut above as a spectrum approaches constant. It used to be
    skipped for illuminants, which made a colored illuminant ~107× (``_Y_INTEGRAL``)
    a constant one of the same magnitude: ``[400 10 700 10]`` → ``[10, 10, 10]``
    but ``[400 10 700 10.000001]`` → ``[1283, 1015, 971]``.
    """
    pairs = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    lam, val = pairs[:, 0], pairs[:, 1]
    if np.all(val == val[0]):
        return np.full(3, max(val[0], 0.0))
    sampled = np.interp(_LAMBDA, lam, val, left=val[0], right=val[-1])
    xyz = spd_to_xyz(sampled)
    if _Y_INTEGRAL > 0:
        xyz = xyz / _Y_INTEGRAL
    rgb = xyz_to_linear_srgb(xyz)
    return np.clip(rgb, 0.0, None)


def fresnel_conductor_rgb(eta, k) -> np.ndarray:
    """Per-channel normal-incidence Fresnel reflectance for a conductor."""
    eta = np.asarray(eta, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    return ((eta - 1.0) ** 2 + k**2) / ((eta + 1.0) ** 2 + k**2)


def named_metal_ior(name: str):
    """Return (eta_rgb, k_rgb) for a named conductor, or None if unknown."""
    key = _normalize_metal_key(name)
    val = NAMED_METAL_IOR.get(key)
    if val is None:
        return None
    return np.array(val[0]), np.array(val[1])


def named_metal_reflectance_rgb(name: str):
    """RGB normal-incidence reflectance of a named metal, or None."""
    ior = named_metal_ior(name)
    if ior is None:
        return None
    return fresnel_conductor_rgb(ior[0], ior[1])


def named_conductor_key(param):
    """Canonical metal key (``au``/``ag``/``al``/``cu``) for a named-conductor eta
    param, or ``None``.

    Spectral mode binds the exact vendored complex-IOR curve for a named metal;
    this recovers the identity from a pbrt ``"spectrum eta"`` whose value is a
    metal name (e.g. ``"metal-Au-eta"`` -> ``"au"``). Uses the same normalization
    as :func:`spectral_tables.named_metal_spectrum` (strip ``metal-``/``metal_``
    and ``-eta``/``-k``, lower-case, alias gold/silver/aluminium/copper). Returns
    ``None`` for a float eta, a non-metal spectrum, or an unknown name — so an
    RGB-only conductor authors no identity override.
    """
    if param is None or param.type != "spectrum":
        return None
    if not param.values or not isinstance(param.values[0], str):
        return None
    key = _normalize_metal_key(param.values[0])
    key = _CONDUCTOR_ALIASES.get(key, key)
    return key if key in _CONDUCTOR_CANON else None


def looks_like_spectrum_file(name: str) -> bool:
    """True if a named-spectrum string is really a file reference, not a name.

    pbrt falls back to ``readSpectrumFromFile`` when a name misses its built-in
    table (``paramdict.cpp`` ``GetNamedSpectrum``), so an unmatched string may be
    a path to a ``.spd``. skinny has no spectrum-file reader; distinguishing the
    two keeps the import report from calling a legitimate file reference an
    "unknown glass".
    """
    n = (name or "").strip()
    return "/" in n or "\\" in n or n.lower().endswith((".spd", ".txt", ".csv"))


def named_glass_key(param):
    """Normalized glass key for a named/dispersive dielectric eta param, or ``None``.

    A named ``"spectrum eta"`` (e.g. ``"glass-BK7"``) carries a dispersive IOR
    identity the spectral path binds via :func:`spectral_tables.named_glass_ior`;
    this normalizes it (strip ``glass-``/``glass_``, lower-case) to a key that
    function understands — a recognised glass (e.g. ``"bk7"``, ``"lasf9"``) or
    ``"default"`` for any other named glass. Returns ``None`` for a plain
    ``float eta`` (no dispersion identity) or a non-string spectrum, so a
    scalar-IOR dielectric authors nothing new.

    Every recognised glass resolves to its *own* key; ``"default"`` means "not
    recognised, substituting BK7". Callers that need to report the substitution
    use :func:`spectral_tables.glass_is_known` — a ``"default"`` return alone is
    ambiguous only in that it cannot say *which* unknown name produced it.
    """
    if param is None or param.type != "spectrum":
        return None
    if not param.values or not isinstance(param.values[0], str):
        return None
    key = _st.normalize_glass_key(str(param.values[0]))
    return key if _st.glass_is_known(key) else "default"


def named_illuminant_rgb(name: str):
    """Linear-RGB chromaticity of a pbrt named illuminant at unit luminance, or None.

    Mirrors :func:`blackbody_rgb`: the SPD is integrated to XYZ and normalised to
    Y = 1, so only *chromaticity* comes from the name and magnitude keeps coming
    from the light's ``L``/``scale``. This matches pbrt, which normalises named
    illuminants to luminance 1 at load (``FromInterleaved(..., normalize=true)``)
    and divides by ``CIE_Y_integral`` in its film — so pbrt's own film luminance
    for ``"spectrum L" "stdillum-A"`` is exactly 1.
    """
    spd = _st.named_illuminant_spectrum(name)
    if spd is None:
        return None
    xyz = spd_to_xyz(spd)
    if xyz[1] > 0:
        xyz = xyz / xyz[1]
    return np.clip(xyz_to_linear_srgb(xyz), 0.0, None)


def param_spectral_payload(param):
    """Raw spectral payload for a parsed :class:`Param`, or ``None`` for plain RGB.

    Spectral rendering (``--spectral``) consumes the exact authored spectrum
    rather than the RGB reduction. This preserves that source alongside the
    (unchanged) :func:`param_to_rgb` value, riding ``skinnyOverrides``:

    * ``blackbody [T]``      -> ``{"kind": "blackbody", "temperature": T}``
    * named ``spectrum``      -> ``{"kind": "spectrum_named", "name": <str>}``
      (named conductor eta/k, named glass — render-time binds the vendored curve)
    * inline ``spectrum``     -> ``{"kind": "spectrum_samples", "lambda": [...],
      "values": [...]}`` resampled onto the internal 5 nm grid

    Returns ``None`` for ``rgb``/``color``/``float`` params (and unknown types),
    so an RGB-only scene authors no new override — the import stays byte-identical.
    Reflectance spectra are intentionally not resampled here (RGB-only in v1).
    """
    if param is None:
        return None
    if param.type == "blackbody":
        return {"kind": "blackbody", "temperature": float(param.values[0])}
    if param.type == "spectrum":
        if isinstance(param.values[0], str):
            return {"kind": "spectrum_named", "name": str(param.values[0])}
        pairs = np.asarray(param.values, dtype=np.float64).reshape(-1, 2)
        lam, val = pairs[:, 0], pairs[:, 1]
        sampled = np.interp(_LAMBDA, lam, val, left=val[0], right=val[-1])
        return {
            "kind": "spectrum_samples",
            "lambda": [float(x) for x in _LAMBDA],
            "values": [float(x) for x in sampled],
        }
    return None


def param_to_rgb(param, *, illuminant: bool = False, default=None):
    """Reduce a parsed :class:`Param` (any spectral form) to a 3-float RGB list.

    Handles ``rgb``/``color`` passthrough, ``blackbody`` temperatures, named
    spectra (named metal reflectance, or — when *illuminant* — a named
    illuminant's unit-luminance chromaticity), and inline sampled spectra.

    The named-illuminant lookup is gated on *illuminant* so a reflectance-mode
    parameter (e.g. a medium's ``"spectrum sigma_a"``) can never reduce to an
    illuminant chromaticity.
    """
    if param is None:
        return default
    ptype = param.type
    if ptype in ("rgb", "color"):
        v = param.floats
        return v[:3] if len(v) >= 3 else [v[0], v[0], v[0]]
    if ptype == "blackbody":
        return list(blackbody_rgb(float(param.values[0])))
    if ptype == "float":
        v = param.floats
        return [v[0], v[0], v[0]]
    if ptype == "spectrum":
        if isinstance(param.values[0], str):
            name = param.values[0]
            if illuminant:
                illum = named_illuminant_rgb(name)
                if illum is not None:
                    return list(illum)
            refl = named_metal_reflectance_rgb(name)
            if refl is not None:
                return list(refl)
            return default
        return list(sampled_spectrum_to_rgb(param.values, illuminant=illuminant))
    return default
