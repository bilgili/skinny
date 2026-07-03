"""Tests for spectrum -> RGB reduction (task 8.1)."""

from __future__ import annotations

import numpy as np

from skinny.pbrt import spectra
from skinny.pbrt.parser import parse_directives
from skinny.pbrt.tokenizer import tokenize


def _param(text, name):
    (d,) = parse_directives(tokenize(text))
    return d.params.get(name)


def test_blackbody_6500_is_near_white():
    rgb = spectra.blackbody_rgb(6500)
    # roughly neutral: channels within ~25% of each other, all positive
    assert np.all(rgb > 0)
    assert max(rgb) / min(rgb) < 1.3


def test_blackbody_warm_is_reddish():
    warm = spectra.blackbody_rgb(3000)
    assert warm[0] > warm[2]  # red dominates blue for a warm source


def test_named_gold_reflectance_is_golden():
    refl = spectra.named_metal_reflectance_rgb("metal-Au-eta")
    assert refl is not None
    assert refl[0] > refl[1] > refl[2]  # R > G > B -> gold


def test_rgb_param_passthrough():
    p = _param('Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]', "reflectance")
    assert spectra.param_to_rgb(p) == [0.2, 0.4, 0.6]


def test_blackbody_param_reduces():
    p = _param('LightSource "distant" "blackbody L" [5500]', "L")
    rgb = spectra.param_to_rgb(p, illuminant=True)
    assert len(rgb) == 3 and all(c >= 0 for c in rgb)


def test_fresnel_conductor_monotone():
    # higher k -> higher reflectance
    low = spectra.fresnel_conductor_rgb([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    high = spectra.fresnel_conductor_rgb([1.0, 1.0, 1.0], [5.0, 5.0, 5.0])
    assert np.all(high > low)


def test_sampled_reflectance_grey_is_neutral():
    # a flat 0.5 reflectance spectrum is achromatic: exactly [0.5, 0.5, 0.5]
    pairs = [400, 0.5, 500, 0.5, 600, 0.5, 700, 0.5]
    rgb = spectra.sampled_spectrum_to_rgb(pairs)
    assert np.array_equal(rgb, [0.5, 0.5, 0.5])


def test_constant_spectrum_is_achromatic():
    # pbrt "spectrum sigma_s" [200 10 900 10] -> exactly [10, 10, 10]
    rgb = spectra.sampled_spectrum_to_rgb([200, 10, 900, 10])
    assert np.array_equal(rgb, [10.0, 10.0, 10.0])


def test_constant_illuminant_is_achromatic():
    rgb = spectra.sampled_spectrum_to_rgb([200, 3, 900, 3], illuminant=True)
    assert np.array_equal(rgb, [3.0, 3.0, 3.0])


def _projected_rgb(pairs, *, illuminant=False):
    # the pre-shortcut projection path, spelled out
    pairs = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    lam, val = pairs[:, 0], pairs[:, 1]
    sampled = np.interp(spectra._LAMBDA, lam, val, left=val[0], right=val[-1])
    xyz = spectra.spd_to_xyz(sampled)
    if not illuminant:
        xyz = xyz / spectra._Y_INTEGRAL
    return np.clip(spectra.xyz_to_linear_srgb(xyz), 0.0, None)


def test_colored_spectrum_keeps_projection():
    # unequal values -> bit-identical to the XYZ->sRGB projection
    pairs = [400, 0.1, 700, 0.9]
    rgb = spectra.sampled_spectrum_to_rgb(pairs)
    assert np.array_equal(rgb, _projected_rgb(pairs))
    assert not np.array_equal(rgb, [rgb[0]] * 3)


def test_near_constant_spectrum_takes_projection_path():
    # any nonzero difference is a colored spectrum, not the shortcut
    pairs = [200, 10, 900, 10.000001]
    rgb = spectra.sampled_spectrum_to_rgb(pairs)
    assert np.array_equal(rgb, _projected_rgb(pairs))
