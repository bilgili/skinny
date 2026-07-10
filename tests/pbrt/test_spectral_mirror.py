"""CPU spectral-estimator mirror (task 1.4): sampling, termination, film resolve."""

from __future__ import annotations

import numpy as np

from skinny.pbrt import spectral
from skinny.pbrt.data import spectral_tables as st

_LAMBDA = spectral._LAMBDA


def test_visible_pdf_integrates_to_one():
    grid = np.arange(360.0, 830.0 + 0.5, 0.5)
    integral = np.trapezoid(spectral.visible_wavelength_pdf(grid), grid)
    assert abs(integral - 1.0) < 0.01


def test_pdf_matches_sampled_wavelengths():
    sw = spectral.sample_wavelengths(0.37)
    assert sw.lambda_.shape == (4,) and sw.pdf.shape == (4,)
    assert np.allclose(sw.pdf, spectral.visible_wavelength_pdf(sw.lambda_))
    assert np.all(sw.lambda_ >= 360.0) and np.all(sw.lambda_ <= 830.0)


def test_hero_rotation_spreads_samples():
    sw = spectral.sample_wavelengths(0.1)
    # 4 rotated samples are distinct and span a wide band.
    assert len(np.unique(np.round(sw.lambda_, 3))) == 4
    assert sw.lambda_.max() - sw.lambda_.min() > 100.0


def test_terminate_secondary():
    sw = spectral.sample_wavelengths(0.5)
    assert not sw.secondary_terminated()
    hero_pdf = sw.pdf[0]
    term = spectral.terminate_secondary(sw)
    assert term.secondary_terminated()
    assert np.all(term.pdf[1:] == 0.0)
    assert term.pdf[0] == hero_pdf / 4.0


def test_resolve_d65_is_neutral():
    # The sRGB white point is D65: a converged resolve of the D65 SPD is neutral.
    # A single 4-sample draw is pure MC noise — average many draws to converge.
    rng = np.random.default_rng(1)
    acc = np.zeros(3)
    trials = 6000
    for _ in range(trials):
        sw = spectral.sample_wavelengths(float(rng.random()))
        d65 = np.interp(sw.lambda_, _LAMBDA, st.d65_spd())
        acc += spectral.spectrum_to_xyz(sw.lambda_, d65, sw.pdf)
    rgb = np.clip(spectral.xyz_to_linear_srgb(acc / trials), 0.0, None)
    assert np.all(rgb > 0.0)
    assert (rgb.max() - rgb.min()) / rgb.max() < 0.05  # neutral within 5%


def test_resolve_matches_direct_reflectance_integration():
    # A dense N-sample estimate of a flat 0.5 reflectance ≈ the equal-energy
    # projection through the same CMFs (consistency of the MC estimator).
    rng = np.random.default_rng(0)
    acc = np.zeros(3)
    trials = 4000
    for _ in range(trials):
        sw = spectral.sample_wavelengths(float(rng.random()))
        vals = np.full(4, 0.5)
        acc += spectral.spectrum_to_xyz(sw.lambda_, vals, sw.pdf)
    xyz_mc = acc / trials
    xyz_ref = spectra_spd_xyz_flat(0.5)
    assert np.max(np.abs(xyz_mc - xyz_ref)) < 0.02


def spectra_spd_xyz_flat(v):
    from skinny.pbrt import spectra

    return spectra.spd_to_xyz(np.full(_LAMBDA.shape, v)) / spectra._Y_INTEGRAL


# ── conductor Fresnel mirror (pbrt FrComplex) ─────────────────────────────


def test_fresnel_normal_incidence_matches_rgb_formula():
    from skinny.pbrt import spectra

    # cosθ=1 must equal the closed-form ((η-1)²+k²)/((η+1)²+k²) per wavelength.
    eta = np.array([0.14, 0.95, 1.5, 2.7])
    k = np.array([3.9, 6.4, 0.0, 4.2])
    got = spectral.fresnel_conductor(1.0, eta, k)
    ref = spectra.fresnel_conductor_rgb(eta, k)
    assert np.allclose(got, ref, atol=1e-12)


def test_fresnel_gold_red_reflectance_high():
    # Gold is a strong red reflector: R climbs through the visible into the red.
    lam = np.array([580.0, 650.0])
    eta, k = spectral.named_metal_eta_k("au", lam)
    r = spectral.fresnel_conductor(1.0, eta, k)
    assert r.shape == (2,)
    assert r[0] > 0.88  # yellow-green already high
    assert r[1] > 0.92  # deep red near-total reflection
    # red (650) reflects at least as strongly as yellow-green (580)
    assert r[1] >= r[0]


def test_fresnel_reflectance_in_unit_range():
    lam = np.linspace(360.0, 830.0, 40)
    for name in ("au", "ag", "al", "cu"):
        eta, k = spectral.named_metal_eta_k(name, lam)
        for c in (1.0, 0.7, 0.3, 0.05):
            r = spectral.fresnel_conductor(c, eta, k)
            assert np.all(r >= 0.0) and np.all(r <= 1.0)


def test_fresnel_grazing_goes_to_one():
    lam = np.array([500.0, 600.0])
    eta, k = spectral.named_metal_eta_k("cu", lam)
    r = spectral.fresnel_conductor(1e-6, eta, k)
    assert np.all(r > 0.999)


def test_fresnel_rises_toward_grazing():
    # A conductor's reflectance may dip slightly (pseudo-Brewster) but grazing
    # incidence reflects far more than normal and tends to 1.
    lam = np.array([550.0])
    eta, k = spectral.named_metal_eta_k("ag", lam)
    r_normal = float(spectral.fresnel_conductor(1.0, eta, k)[0])
    r_graze = float(spectral.fresnel_conductor(0.02, eta, k)[0])
    assert r_graze > r_normal
    assert r_graze > 0.99


def test_named_metal_eta_k_unknown_is_none():
    assert spectral.named_metal_eta_k("unobtainium", np.array([500.0])) is None


# ── blackbody emitters (pbrt planckSpectrum mirror) ───────────────────────────

_REC709_Y = np.array([0.2126, 0.7152, 0.0722])


def _luminance(rgb):
    """CIE Y of a linear-sRGB colour (Rec.709 luminance, the resolve convention)."""
    return float(np.dot(np.asarray(rgb, dtype=np.float64), _REC709_Y))


def _mc_resolve_blackbody(temperature_k, emission_rgb, *, trials=8000, seed=0):
    """Monte-Carlo film resolve of the scaled Planck SPD over many hero draws."""
    from skinny.pbrt import spectral

    scale = spectral.blackbody_scale(temperature_k, emission_rgb)
    rng = np.random.default_rng(seed)
    acc = np.zeros(3)
    for _ in range(trials):
        sw = spectral.sample_wavelengths(float(rng.random()))
        vals = spectral.blackbody_emission(sw, temperature_k) * scale
        acc += spectral.resolve_to_linear_srgb(sw.lambda_, vals, sw.pdf)
    return acc / trials


def test_blackbody_emission_is_raw_planck():
    from skinny.pbrt import spectra

    sw = spectral.sample_wavelengths(0.31)
    got = spectral.blackbody_emission(sw, 5500.0)
    assert np.allclose(got, spectra.planck(sw.lambda_, 5500.0))


def test_blackbody_emission_all_positive():
    sw = spectral.sample_wavelengths(0.63)
    for temperature_k in (3000.0, 6500.0, 9000.0):
        vals = spectral.blackbody_emission(sw, temperature_k)
        assert np.all(vals > 0.0)


def test_blackbody_scale_guards():
    rgb = np.array([1.0, 1.0, 1.0])
    assert spectral.blackbody_scale(0.0, rgb) == 0.0
    assert spectral.blackbody_scale(-100.0, rgb) == 0.0


def test_blackbody_roundtrip_luminance_and_chromaticity():
    from skinny.pbrt import spectra

    intensity = 10.0
    for temperature_k in (3000.0, 6500.0, 9000.0):
        emission_rgb = spectra.blackbody_rgb(temperature_k) * intensity
        resolved = _mc_resolve_blackbody(temperature_k, emission_rgb)

        # Luminance reproduced within ~2%.
        lum_res = _luminance(resolved)
        lum_tgt = _luminance(emission_rgb)
        assert abs(lum_res - lum_tgt) / lum_tgt < 0.02

        # Chromaticity (rgb / sum) matches the blackbody chromaticity within ~2%.
        chroma_res = resolved / np.sum(resolved)
        bb = spectra.blackbody_rgb(temperature_k)
        chroma_ref = bb / np.sum(bb)
        assert np.max(np.abs(chroma_res - chroma_ref)) < 0.02


def test_blackbody_hotter_shifts_blue():
    from skinny.pbrt import spectra

    intensity = 5.0
    ratios = []
    for temperature_k in (3000.0, 6500.0, 9000.0):
        emission_rgb = spectra.blackbody_rgb(temperature_k) * intensity
        resolved = _mc_resolve_blackbody(temperature_k, emission_rgb, seed=7)
        ratios.append(resolved[2] / resolved[0])  # b / r
    # Cooler colour temperature (higher K) = bluer: b/r rises monotonically.
    assert ratios[0] < ratios[1] < ratios[2]
