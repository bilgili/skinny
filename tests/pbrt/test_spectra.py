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
    # a flat 0.5 reflectance spectrum -> near-neutral grey
    pairs = [400, 0.5, 500, 0.5, 600, 0.5, 700, 0.5]
    rgb = spectra.sampled_spectrum_to_rgb(pairs)
    assert max(rgb) / max(min(rgb), 1e-6) < 1.5
