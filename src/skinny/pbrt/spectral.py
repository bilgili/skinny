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

from .data import spectral_tables as st
from .spectra import _LAMBDA, _Y_INTEGRAL, cie_xyz_bar, xyz_to_linear_srgb

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
