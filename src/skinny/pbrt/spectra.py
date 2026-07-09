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
    spectra are integrated to XYZ: reflectance under an equal-energy whitepoint
    (documented simplification), illuminants directly.
    """
    pairs = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    lam, val = pairs[:, 0], pairs[:, 1]
    if np.all(val == val[0]):
        return np.full(3, max(val[0], 0.0))
    sampled = np.interp(_LAMBDA, lam, val, left=val[0], right=val[-1])
    xyz = spd_to_xyz(sampled)
    if not illuminant and _Y_INTEGRAL > 0:
        # reflectance under equal-energy E: normalise by the CMF integral
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
    spectra (returns the named reflectance), and inline sampled spectra.
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
            refl = named_metal_reflectance_rgb(param.values[0])
            if refl is not None:
                return list(refl)
            return default
        return list(sampled_spectrum_to_rgb(param.values, illuminant=illuminant))
    return default
