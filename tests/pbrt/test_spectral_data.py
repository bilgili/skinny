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
    # Refit from pbrt's own tabulated eta (change `pbrt-named-spectra`), superseding
    # the hand-entered catalogue coefficients (1.5046, 0.00420) — |Δn| ≈ 3e-4. pbrt is
    # the single source of truth for every glass, BK7 included. "default" is the
    # unrecognised-name fallback and is BK7 by definition.
    for name in ("bk7", "default"):
        coeff = st.named_glass_cauchy(name)
        assert coeff is not None
        a, b = coeff
        assert isinstance(a, float) and isinstance(b, float)
        assert abs(a - 1.50431) < 1e-9 and abs(b - 0.004267) < 1e-9


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


# --- named-spectrum coverage (change `pbrt-named-spectra`) -------------------
#
# The fits/curves are vendored from pbrt-v4 by `_extract_pbrt_spectra`; these
# tests are hostless because the raw glass eta curves ride in the .npz alongside
# the fitted literals, so the fit can be re-checked without a pbrt checkout.

_PBRT_GLASSES = ("bk7", "baf10", "fk51a", "lasf9", "f5", "f10", "f11")

#: Max |Δn| of the 2-term Cauchy fit vs pbrt's tabulated eta over 360-830 nm.
#: A 3rd term was measured and does not improve this (design D1) — the residual
#: is interpolation error in pbrt's own sparse table.
_FIT_RESIDUAL_TOL = 8e-3


@pytest.mark.parametrize("key", _PBRT_GLASSES)
def test_named_glass_cauchy_reproduces_pbrt_curve(key):
    curves = st._load_curves()
    pbrt_eta = curves[f"glass_{key}_eta"]
    a, b = st.named_glass_cauchy(key)
    lam_um = _LAMBDA * 1e-3
    fitted = a + b / lam_um**2
    assert np.max(np.abs(fitted - pbrt_eta)) < _FIT_RESIDUAL_TOL


def test_named_glasses_have_distinct_dispersion():
    # The bug this change fixes: every unrecognised glass silently rendered as BK7.
    coeffs = {k: st.named_glass_cauchy(k) for k in _PBRT_GLASSES}
    assert len(set(coeffs.values())) == len(_PBRT_GLASSES)
    assert st.named_glass_cauchy("glass-LASF9") != st.named_glass_cauchy("glass-BK7")


@pytest.mark.parametrize("key,expected", [
    ("bk7", 1.51673), ("baf10", 1.66988), ("fk51a", 1.48651),
    ("lasf9", 1.85004), ("f5", 1.67254), ("f10", 1.72806), ("f11", 1.78448),
])
def test_named_glass_ior_d_matches_pbrt(key, expected):
    assert st.named_glass_ior_d(key) == pytest.approx(expected, abs=1e-5)


def test_glass_is_known_distinguishes_fallback_from_real_glass():
    assert st.glass_is_known("glass-BK7") and st.glass_is_known("glass-LASF9")
    assert not st.glass_is_known("glass-NOSUCH")
    # "default" is the fallback itself, not a pbrt glass.
    assert not st.glass_is_known("default")


def test_unknown_glass_still_falls_back_to_bk7():
    # Best-effort translator: an unknown name must not hard-fail, it renders as
    # BK7 *and* gets reported (see test_named_spectra.py).
    assert st.named_glass_cauchy("glass-NOSUCH") == st.named_glass_cauchy("bk7")
    assert st.named_glass_ior_d("glass-NOSUCH") == st.named_glass_ior_d("bk7")


@pytest.mark.parametrize("name", ("metal-CuZn-eta", "metal-MgO-eta", "metal-TiO2-eta"))
def test_extended_metals_have_vendored_curves(name):
    ek = st.named_metal_spectrum(name)
    assert ek is not None
    eta, k = ek
    assert eta.shape == (95,) and k.shape == (95,)


def test_preexisting_metals_unchanged():
    # au/ag/al/cu keep their shipped RGB IOR: re-deriving them from the vendored
    # curves would move existing RGB baselines for no benefit (design m4).
    from skinny.pbrt.data import NAMED_METAL_IOR
    assert NAMED_METAL_IOR["au"] == ((0.143, 0.375, 1.442), (3.983, 2.386, 1.603))
    assert NAMED_METAL_IOR["cu"] == ((0.200, 0.924, 1.102), (3.910, 2.450, 2.140))


def test_named_illuminant_a_is_warm_and_d65_is_neutral():
    a_rgb = spectra.named_illuminant_rgb("stdillum-A")
    assert a_rgb[0] > a_rgb[2]  # tungsten: red-heavy
    d65 = spectra.named_illuminant_rgb("stdillum-D65")
    # D65 is the sRGB whitepoint, so unit luminance lands on ~[1,1,1]. Measured
    # deviation 6.0e-4 (the Wyman analytic CMF fit).
    assert np.allclose(d65, [1.0, 1.0, 1.0], atol=1e-3)


def test_named_illuminant_lookup_is_case_insensitive_and_bounded():
    assert spectra.named_illuminant_rgb("STDILLUM-F11") is not None
    assert spectra.named_illuminant_rgb("illum-acesD60") is not None
    assert spectra.named_illuminant_rgb("stdillum-NOPE") is None
    assert spectra.named_illuminant_rgb("") is None


def test_stdillum_d65_aliases_the_vendored_d65():
    # Same pbrt symbol (CIE_Illum_D6500); stored once, not twice.
    assert np.array_equal(st.named_illuminant_spectrum("stdillum-D65"), st.d65_spd())
