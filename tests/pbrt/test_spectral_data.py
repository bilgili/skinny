"""Vendored pbrt spectral-table sanity + RGB→spectrum round-trip (task 1.3).

The upsample table is pbrt-exact (extracted from pbrt-v4), but skinny resolves
spectra through its own Wyman-Sloan-Shirley CMF fit rather than pbrt's tabulated
CMFs, so the reflectance round-trip carries a small, systematic CMF-convention
bias. The tolerances below are the observed operational error of the
skinny pipeline (upsample with the pbrt table, resolve with the Wyman fit), not
a claim of bit-exact pbrt agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import spectra
from skinny.pbrt.data import spectral_tables as st

_LAMBDA = spectra._LAMBDA

# Observed operational round-trip error (pbrt table + Wyman CMFs); see docstring.
_ROUNDTRIP_TOL = 0.06


def _resolve_reflectance(rgb) -> np.ndarray:
    # pbrt's albedo table is fit so the spectrum reproduces the RGB under the
    # colour space's illuminant (D65 for sRGB): integrate S(λ)·D65 densely.
    coeffs = st.rgb_to_sigmoid_coeffs(rgb)
    spd = st.sigmoid_poly(coeffs, _LAMBDA)
    d65 = st.d65_spd()
    xyz = spectra.spd_to_xyz(spd * d65) / spectra._trapz(d65 * spectra._YBAR, _LAMBDA)
    return np.clip(spectra.xyz_to_linear_srgb(xyz), 0.0, None)


@pytest.mark.parametrize(
    "rgb",
    [
        (0.5, 0.5, 0.5),
        (0.8, 0.2, 0.2),
        (0.2, 0.7, 0.3),
        (0.1, 0.3, 0.9),
        (0.9, 0.9, 0.1),
        (0.05, 0.05, 0.05),
    ],
)
def test_upsample_roundtrip_reflectance(rgb):
    recovered = _resolve_reflectance(rgb)
    assert np.max(np.abs(recovered - np.array(rgb))) < _ROUNDTRIP_TOL


def test_uniform_rgb_is_flat_spectrum():
    coeffs = st.rgb_to_sigmoid_coeffs((0.5, 0.5, 0.5))
    assert coeffs[0] == 0.0 and coeffs[1] == 0.0  # pbrt closed-form branch
    spd = st.sigmoid_poly(coeffs, _LAMBDA)
    assert np.allclose(spd, spd[0])  # constant across wavelength
    assert abs(spd[0] - 0.5) < 1e-6


def test_sigmoid_range_bounded():
    for rgb in [(0.9, 0.1, 0.4), (0.3, 0.6, 0.2)]:
        spd = st.sigmoid_poly(st.rgb_to_sigmoid_coeffs(rgb), _LAMBDA)
        assert np.all(spd >= 0.0) and np.all(spd <= 1.0)


def test_d65_chromaticity_near_white():
    xyz = spectra.spd_to_xyz(st.d65_spd())
    x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
    # D65 reference chromaticity (0.3127, 0.3290); Wyman-fit CMFs shift it slightly.
    assert abs(x - 0.3127) < 0.015 and abs(y - 0.3290) < 0.015


def test_named_metal_curves():
    for name in ("metal-Au-eta", "Ag", "aluminium", "copper"):
        pair = st.named_metal_spectrum(name)
        assert pair is not None
        eta, k = pair
        assert eta.shape == _LAMBDA.shape and k.shape == _LAMBDA.shape
        assert np.all(np.isfinite(eta)) and np.all(np.isfinite(k))
    assert st.named_metal_spectrum("unobtanium") is None


def test_glass_dispersion_normal():
    n589 = st.named_glass_ior("glass-BK7", 589.0)
    assert abs(float(n589) - 1.5168) < 0.01
    n_blue = st.named_glass_ior("glass-BK7", 486.0)
    n_red = st.named_glass_ior("glass-BK7", 656.0)
    assert n_blue > n_red  # normal dispersion


def test_named_glass_cauchy_known_coeffs():
    for name in ("bk7", "default"):
        coeff = st.named_glass_cauchy(name)
        assert coeff is not None
        a, b = coeff
        assert isinstance(a, float) and isinstance(b, float)
        assert abs(a - 1.5046) < 1e-9 and abs(b - 0.00420) < 1e-9


def test_named_glass_cauchy_normalizes_prefix_and_case():
    # "glass-BK7" strips the prefix and lowercases to the bk7 entry.
    assert st.named_glass_cauchy("glass-BK7") == st.named_glass_cauchy("bk7")
    assert st.named_glass_cauchy("glass_bk7") == st.named_glass_cauchy("bk7")


def test_named_glass_cauchy_unknown_falls_back_to_default():
    # An unknown-but-present name rides the default (BK7-family) coefficients,
    # mirroring named_glass_ior's fallback.
    assert st.named_glass_cauchy("glass-SF11") == st.named_glass_cauchy("default")


def test_named_glass_cauchy_none_or_empty_is_none():
    assert st.named_glass_cauchy(None) is None
    assert st.named_glass_cauchy("") is None
    assert st.named_glass_cauchy("   ") is None


def test_named_glass_cauchy_matches_ior_evaluation():
    # The (A, B) accessor reproduces named_glass_ior when fed through n=A+B/λµm².
    a, b = st.named_glass_cauchy("glass-BK7")
    for lam_nm in (486.0, 589.0, 656.0):
        lam_um = lam_nm * 1e-3
        assert abs((a + b / lam_um**2) - float(st.named_glass_ior("bk7", lam_nm))) < 1e-12
