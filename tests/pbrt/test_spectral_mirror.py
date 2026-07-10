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


# ── dispersion mirror (Cauchy IOR + secondary termination gate) ───────────────


def test_cauchy_ior_bk7_sodium_line():
    # n(589 nm) ≈ 1.5168 for the BK7-family fit (A + B / 0.589²).
    a, b = st.named_glass_cauchy("bk7")
    n589 = float(spectral.cauchy_ior(a, b, 589.0))
    assert abs(n589 - 1.5168) < 0.01


def test_cauchy_ior_normal_dispersion_monotonic():
    # Normal dispersion: index falls monotonically with wavelength (blue > red).
    a, b = st.named_glass_cauchy("bk7")
    lam = np.array([400.0, 486.0, 550.0, 589.0, 656.0, 700.0])
    n = spectral.cauchy_ior(a, b, lam)
    assert np.all(np.diff(n) < 0.0)
    # blue (486 nm) index exceeds red (656 nm) index.
    assert float(spectral.cauchy_ior(a, b, 486.0)) > float(spectral.cauchy_ior(a, b, 656.0))


def test_cauchy_ior_agrees_with_named_glass_ior_matched_wavelengths():
    # cauchy_ior takes nm; named_glass_ior takes µm — same physical λ ⇒ same n.
    for name in ("bk7", "default"):
        a, b = st.named_glass_cauchy(name)
        for lam_nm in (420.0, 486.0, 589.0, 656.0, 780.0):
            lam_um = lam_nm * 1e-3
            got = float(spectral.cauchy_ior(a, b, lam_nm))
            ref = float(st.named_glass_ior(name, lam_nm))
            # named_glass_ior evaluates a + b/λµm² directly:
            assert abs(got - (a + b / lam_um**2)) < 1e-12
            assert abs(got - ref) < 1e-12


def test_should_terminate_secondary_truth_table():
    # The exact GPU gate: delta pdf AND refraction (wo/wi opposite sides) AND
    # dispersive (b > 0). wo_z, wi_z are the tangent-space z of the outgoing /
    # sampled directions; opposite signs ⇒ the ray crossed the interface.
    # Entering refraction (wo above, wi below): terminates.
    assert spectral.should_terminate_secondary(0.0, 1.0, -1.0, 0.00420) is True
    # EXITING refraction (wo below, wi above) — also crosses ⇒ terminates. This is
    # the case a `transmitted` (wi_z < 0) flag would have missed.
    assert spectral.should_terminate_secondary(0.0, -1.0, 1.0, 0.00420) is True
    # Constant-IOR glass (b == 0) never terminates.
    assert spectral.should_terminate_secondary(0.0, 1.0, -1.0, 0.0) is False
    # Non-delta pdf (reflection lobe with a real pdf) never terminates.
    assert spectral.should_terminate_secondary(0.5, 1.0, -1.0, 0.00420) is False
    # Reflection (wi/wo on the SAME side) never terminates — achromatic.
    assert spectral.should_terminate_secondary(0.0, 1.0, 1.0, 0.00420) is False
    assert spectral.should_terminate_secondary(0.0, -1.0, -1.0, 0.00420) is False


def test_dispersion_gate_drives_terminate_secondary():
    # Constant-IOR glass (b = 0): gate is False ⇒ all 4 hero pdfs stay live.
    sw = spectral.sample_wavelengths(0.42)
    assert not spectral.should_terminate_secondary(0.0, 1.0, -1.0, 0.0)
    kept = (
        spectral.terminate_secondary(sw)
        if spectral.should_terminate_secondary(0.0, 1.0, -1.0, 0.0)
        else sw
    )
    assert not kept.secondary_terminated()
    assert np.all(kept.pdf > 0.0)  # 4 wavelengths carried

    # Dispersive glass (b > 0), a delta refraction (wo/wi opposite sides): gate is
    # True ⇒ collapse to the hero λ (pdf.x /= 4, y/z/w = 0).
    assert spectral.should_terminate_secondary(0.0, 1.0, -1.0, 0.00420)
    hero_pdf = sw.pdf[0]
    collapsed = (
        spectral.terminate_secondary(sw)
        if spectral.should_terminate_secondary(0.0, 1.0, -1.0, 0.00420)
        else sw
    )
    assert collapsed.secondary_terminated()
    assert np.all(collapsed.pdf[1:] == 0.0)
    assert collapsed.pdf[0] == hero_pdf / 4.0


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
