"""CPU reference for skinny's hero-wavelength spectral estimator.

This is the numpy mirror the GPU spectral kernels are validated against
(``spectrum.slang``): pbrt-exact visible-wavelength importance sampling, the
Wilkie et al. (2014) hero-wavelength rotation, secondary termination for
dispersion, and the film resolve back to linear sRGB. Colour-matching uses
skinny's existing Wyman-Sloan-Shirley fit (``spectra.cie_xyz_bar``) so the
importer, this mirror, and the shader all share one CMF definition.

RGB→spectrum upsampling and the vendored curves live in
:mod:`skinny.pbrt.data.spectral_tables`; this module composes them into a full
sample→resolve pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spectra
from .data import spectral_tables as st
from .spectra import _LAMBDA, _Y_INTEGRAL, cie_xyz_bar, planck, xyz_to_linear_srgb

# CIE Y / Rec.709 luminance weights — the Y row of the linear-sRGB→XYZ matrix
# under a D65 whitepoint, so ``dot(rgb, _REC709_Y)`` is exactly the CIE Y of a
# linear-sRGB colour.
_REC709_Y = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

N_SPECTRUM_SAMPLES = 4

# pbrt visible-wavelength importance-sampling constants (sampling.h / spectrum).
_LAMBDA_MIN = 360.0
_LAMBDA_MAX = 830.0


def sample_visible_wavelength(u: float) -> float:
    """pbrt ``SampleVisibleWavelengths`` — importance sample around 538 nm."""
    return 538.0 - 138.888889 * np.arctanh(0.85691062 - 1.82750197 * u)


def visible_wavelength_pdf(lam) -> np.ndarray:
    """pbrt ``VisibleWavelengthsPDF`` — 0 outside [360, 830] nm."""
    lam = np.asarray(lam, dtype=np.float64)
    pdf = 0.0039398042 / np.cosh(0.0072 * (lam - 538.0)) ** 2
    return np.where((lam < _LAMBDA_MIN) | (lam > _LAMBDA_MAX), 0.0, pdf)


@dataclass
class SampledWavelengths:
    """N hero-rotated wavelengths with their sampling pdfs."""

    lambda_: np.ndarray  # (N,)
    pdf: np.ndarray  # (N,)

    def secondary_terminated(self) -> bool:
        return bool(np.all(self.pdf[1:] == 0.0))


def sample_wavelengths(u: float, n: int = N_SPECTRUM_SAMPLES) -> SampledWavelengths:
    """Draw *n* hero-rotated wavelengths from one sample ``u`` (pbrt SampleVisible)."""
    lam = np.empty(n)
    for i in range(n):
        up = u + float(i) / n
        if up > 1.0:
            up -= 1.0
        lam[i] = sample_visible_wavelength(up)
    return SampledWavelengths(lambda_=lam, pdf=visible_wavelength_pdf(lam))


def terminate_secondary(sw: SampledWavelengths) -> SampledWavelengths:
    """Collapse to the hero wavelength (pbrt ``TerminateSecondary``) for dispersion."""
    if sw.secondary_terminated():
        return sw
    pdf = sw.pdf.copy()
    n = pdf.size
    pdf[1:] = 0.0
    pdf[0] /= n
    return SampledWavelengths(lambda_=sw.lambda_.copy(), pdf=pdf)


def upsample_reflectance(rgb, lam) -> np.ndarray:
    """Evaluate an sRGB reflectance as a spectrum at wavelengths *lam*."""
    coeffs = st.rgb_to_sigmoid_coeffs(rgb)
    return st.sigmoid_poly(coeffs, lam)


def _d65_luminance() -> float:
    """CMF luminance of the vendored (raw) D65 SPD, ΣD65·ȳ / Σȳ on the 5 nm grid."""
    d65 = st.d65_spd()
    yb = cie_xyz_bar(_LAMBDA)[1]
    return float(np.sum(d65 * yb) / np.sum(yb))


_D65_LUMINANCE = _d65_luminance()


def d65_normalized() -> np.ndarray:
    """D65 SPD scaled to unit luminance — pbrt's whitepoint illuminant, so a unit
    RGB illuminant resolves to unit radiance. This is the curve uploaded to the
    GPU (renderer) so the shader's upsampleIlluminant matches this mirror."""
    return st.d65_spd() / _D65_LUMINANCE


def upsample_illuminant(rgb, lam) -> np.ndarray:
    """Evaluate an sRGB emitter as a spectrum (pbrt ``RGBIlluminantSpectrum``):
    ``scale·sigmoid(rgb/scale)·D65_norm`` with ``scale = 2·max(rgb)``. The scale
    keeps HDR emitters (values > 1) in the sigmoid's [0,1] gamut and restores
    their magnitude — a plain clamp would collapse a bright light to white and
    lose its intensity. D65 is normalized to unit luminance so unit RGB → unit
    radiance."""
    rgb = np.asarray(rgb, dtype=np.float64)
    m = float(np.max(rgb))
    scale = 2.0 * m
    rgbn = rgb / scale if scale > 0.0 else np.zeros(3, dtype=np.float64)
    coeffs = st.rgb_to_sigmoid_coeffs(rgbn)
    shape = st.sigmoid_poly(coeffs, lam)
    d65 = np.interp(lam, _LAMBDA, d65_normalized())
    return scale * shape * d65


def spectrum_to_xyz(lam, values, pdf) -> np.ndarray:
    """Monte-Carlo estimate of XYZ from N spectral samples (pbrt ``ToXYZ``)."""
    lam = np.asarray(lam, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    pdf = np.asarray(pdf, dtype=np.float64)
    xb, yb, zb = cie_xyz_bar(lam)
    safe = np.where(pdf > 0.0, 1.0 / pdf, 0.0)
    n = lam.size
    x = np.sum(values * xb * safe) / n
    y = np.sum(values * yb * safe) / n
    z = np.sum(values * zb * safe) / n
    return np.array([x, y, z]) / _Y_INTEGRAL


def resolve_to_linear_srgb(lam, values, pdf) -> np.ndarray:
    """Full film resolve: N spectral samples → linear sRGB (clamped ≥ 0)."""
    return np.clip(xyz_to_linear_srgb(spectrum_to_xyz(lam, values, pdf)), 0.0, None)


# ── blackbody emitters (pbrt planckSpectrum mirror) ───────────────────────────


def blackbody_emission(sw: SampledWavelengths, temperature_k: float) -> np.ndarray:
    """Raw relative Planck SPD at the hero wavelengths ``sw.lambda_`` (N,).

    This is exactly what the GPU's ``planckSpectrum(sw, T)`` evaluates — the
    un-normalized Planck spectral radiance (``spectra.planck``); the per-emitter
    energy/chromaticity match is applied separately via :func:`blackbody_scale`,
    never here.
    """
    return planck(sw.lambda_, temperature_k)


def blackbody_scale(temperature_k: float, emission_rgb) -> float:
    """Scalar the renderer multiplies the Planck SPD by so the hero-λ film resolve
    of ``blackbody_emission(sw, T) * scale`` reproduces ``emission_rgb``.

    Deterministic (not Monte-Carlo). ``emission_rgb`` is the material's already
    computed linear-sRGB emission (``blackbody_rgb(T) × authored intensity``); the
    Planck SPD carries the blackbody *chromaticity* on its own, so this scale only
    has to fix the *luminance*::

        scale = Y_target / Y_planck_resolved

    * ``Y_target`` = the CIE Y of ``emission_rgb`` — the Rec.709 luminance
      ``dot(emission_rgb, [0.2126, 0.7152, 0.0722])``, which for a linear-sRGB
      colour equals its CIE Y under the D65 whitepoint (the Y row of the
      linear-sRGB→XYZ matrix). ``blackbody_rgb`` has unit CIE Y, so
      ``Y_target`` recovers the authored intensity.
    * ``Y_planck_resolved`` = the Planck SPD's luminance **under the same
      normalization the film resolve applies** — ``spectrum_to_xyz`` /
      ``resolve_to_linear_srgb`` divide the Monte-Carlo XYZ by ``_Y_INTEGRAL``
      (``∫ȳ dλ``), whereas ``spectra.spd_to_xyz`` returns the raw un-normalized
      trapz luminance. Dividing the raw ``Y_planck`` by ``_Y_INTEGRAL`` puts the
      denominator in the resolve's units, so the round-trip closes (a literal
      ``Y_target / spd_to_xyz(...)[1]`` would be off by exactly ``_Y_INTEGRAL``).

    Returns ``0.0`` for a non-positive temperature or a degenerate Planck SPD.
    """
    if temperature_k <= 0.0:
        return 0.0
    y_planck = float(spectra.spd_to_xyz(planck(_LAMBDA, temperature_k))[1])
    if y_planck <= 0.0:
        return 0.0
    y_target = float(np.dot(np.asarray(emission_rgb, dtype=np.float64), _REC709_Y))
    return y_target * _Y_INTEGRAL / y_planck


# ── conductor Fresnel (pbrt FrComplex mirror) ─────────────────────────────


def fresnel_conductor(cos_theta_i, eta, k) -> np.ndarray:
    """Unpolarized conductor Fresnel reflectance, per wavelength (pbrt ``FrComplex``).

    Port of pbrt-v4 ``FrComplex`` (spectrum.cpp): with complex index
    ``η̃ = eta + i·k`` and ``cosθ_i ∈ [0, 1]``,

        sin²θ_i = 1 − cos²θ_i,   sin²θ_t = sin²θ_i / η̃²,   cosθ_t = √(1 − sin²θ_t),
        r_∥ = (η̃·cosθ_i − cosθ_t) / (η̃·cosθ_i + cosθ_t),
        r_⊥ = (cosθ_i − η̃·cosθ_t) / (cosθ_i + η̃·cosθ_t),
        R  = (|r_∥|² + |r_⊥|²) / 2.

    ``eta``/``k`` are arrays over the sampled λ; ``cos_theta_i`` is a scalar or a
    broadcastable array. Computed in ``complex128``; the real reflectance is
    returned in [0, 1]. At normal incidence this equals
    ``spectra.fresnel_conductor_rgb`` per λ; at grazing (cosθ→0) it → 1.
    """
    cos_theta_i = np.clip(np.asarray(cos_theta_i, dtype=np.float64), 0.0, 1.0)
    eta_c = np.asarray(eta, dtype=np.float64) + 1j * np.asarray(k, dtype=np.float64)
    sin2_i = 1.0 - cos_theta_i * cos_theta_i
    sin2_t = sin2_i / (eta_c * eta_c)
    cos_t = np.sqrt(1.0 - sin2_t)
    r_parl = (eta_c * cos_theta_i - cos_t) / (eta_c * cos_theta_i + cos_t)
    r_perp = (cos_theta_i - eta_c * cos_t) / (cos_theta_i + eta_c * cos_t)
    r = 0.5 * (np.abs(r_parl) ** 2 + np.abs(r_perp) ** 2)
    return np.real(r)


def named_metal_eta_k(name: str, lam):
    """``(eta, k)`` for a named metal interpolated onto *lam*, or ``None`` if unknown.

    Interpolates :func:`spectral_tables.named_metal_spectrum` (the vendored 360–830
    nm / 5 nm eta/k grid) onto the sampled wavelengths ``lam`` with ``np.interp``.
    """
    spec = st.named_metal_spectrum(name)
    if spec is None:
        return None
    eta_grid, k_grid = spec
    lam = np.asarray(lam, dtype=np.float64)
    return np.interp(lam, _LAMBDA, eta_grid), np.interp(lam, _LAMBDA, k_grid)
