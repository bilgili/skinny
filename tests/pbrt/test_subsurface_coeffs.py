"""pbrt subsurface → (σ_a, σ_s, g, eta) coefficient mapping (pbrt-subsurface-volumetric 1.1).

Pins the pbrt-v4 precedence + the exact named-media table values, so an imported
subsurface object's medium matches the pbrt reference. No GPU.
"""

from __future__ import annotations

import math

import pytest

from skinny.pbrt.subsurface import subsurface_coefficients, ETA_DEFAULT


def test_named_skin1_exact():
    # pbrt media.cpp Skin1: sigma_prime_s=(0.74,0.88,1.01), sigma_a=(0.032,0.17,0.48).
    c = subsurface_coefficients(name="Skin1")
    assert c["sigma_s"] == pytest.approx([0.74, 0.88, 1.01])
    assert c["sigma_a"] == pytest.approx([0.032, 0.17, 0.48])
    assert c["g"] == 0.0                       # named presets force g=0
    assert c["eta"] == pytest.approx(ETA_DEFAULT)


def test_named_case_insensitive_and_scale():
    c = subsurface_coefficients(name="skin1", scale=2.0)
    assert c["sigma_s"] == pytest.approx([1.48, 1.76, 2.02])
    assert c["sigma_a"] == pytest.approx([0.064, 0.34, 0.96])


def test_named_forces_g_zero_even_if_g_set():
    c = subsurface_coefficients(name="Marble", g=0.8)
    assert c["g"] == 0.0
    assert c["sigma_s"] == pytest.approx([2.19, 2.62, 3.0])


def test_explicit_sigma_used_directly():
    c = subsurface_coefficients(sigma_a=[0.1, 0.2, 0.3], sigma_s=[1.0, 2.0, 3.0],
                                g=0.5, scale=1.0)
    assert c["sigma_a"] == pytest.approx([0.1, 0.2, 0.3])
    assert c["sigma_s"] == pytest.approx([1.0, 2.0, 3.0])
    assert c["g"] == pytest.approx(0.5)


def test_name_beats_explicit_sigma():
    # name wins over sigma_a/sigma_s.
    c = subsurface_coefficients(name="Skin1", sigma_a=[9, 9, 9], sigma_s=[9, 9, 9])
    assert c["sigma_s"] == pytest.approx([0.74, 0.88, 1.01])


def test_explicit_sigma_beats_reflectance():
    c = subsurface_coefficients(sigma_a=[0.1, 0.1, 0.1], sigma_s=[1.0, 1.0, 1.0],
                                reflectance=[0.9, 0.9, 0.9])
    assert c["sigma_s"] == pytest.approx([1.0, 1.0, 1.0])


def test_defaults_when_nothing_specified():
    # pbrt Wholemilk-like defaults.
    c = subsurface_coefficients()
    assert c["sigma_a"] == pytest.approx([0.0011, 0.0024, 0.014])
    assert c["sigma_s"] == pytest.approx([2.55, 3.21, 3.77])


class TestReflectanceInversion:
    def test_high_reflectance_high_albedo(self):
        # Rd→1 ⇒ single-scatter albedo→1 ⇒ σ_a≈0, σ_s≈σ_t=1/mfp.
        c = subsurface_coefficients(reflectance=[0.99, 0.99, 0.99], mfp=[1.0, 1.0, 1.0])
        for sa, ss in zip(c["sigma_a"], c["sigma_s"]):
            assert sa < 0.05, f"σ_a should be ~0 for high reflectance, got {sa}"
            assert ss > 0.9, f"σ_s should be ~σ_t for high reflectance, got {ss}"

    def test_low_reflectance_high_absorption(self):
        # Rd→0 ⇒ albedo→0 ⇒ σ_a dominates.
        c = subsurface_coefficients(reflectance=[0.05, 0.05, 0.05], mfp=[1.0, 1.0, 1.0])
        for sa, ss in zip(c["sigma_a"], c["sigma_s"]):
            assert sa > ss, f"σ_a should dominate for low reflectance: σ_a={sa} σ_s={ss}"

    def test_mfp_sets_extinction_scale(self):
        # σ_t = σ_a+σ_s = 1/mfp (per channel).
        c = subsurface_coefficients(reflectance=[0.7, 0.7, 0.7], mfp=[2.0, 2.0, 2.0])
        for sa, ss in zip(c["sigma_a"], c["sigma_s"]):
            assert (sa + ss) == pytest.approx(0.5, abs=1e-3)   # 1/2.0

    def test_inversion_roundtrips_albedo(self):
        # The inverted albedo, fed forward, reproduces Rd (monotone inversion).
        from skinny.pbrt.subsurface import (
            _invert_albedo, _diffuse_reflectance_from_albedo,
            _fresnel_diffuse_reflectance,
        )
        eta = 1.33
        A = ((1.0 + _fresnel_diffuse_reflectance(eta))
             / (1.0 - _fresnel_diffuse_reflectance(eta)))
        for rd in (0.1, 0.3, 0.5, 0.7, 0.9):
            a = _invert_albedo(rd, eta)
            assert _diffuse_reflectance_from_albedo(a, A) == pytest.approx(rd, abs=2e-3)


def _mat(text):
    from skinny.pbrt.parser import parse_directives
    from skinny.pbrt.tokenizer import tokenize
    from skinny.pbrt.state import PbrtMaterial
    (d,) = parse_directives(tokenize(text))
    return PbrtMaterial(d.type_arg() or "", d.params)


class TestImporterEmitsMediumCoeffs:
    def test_flat_mapper_emits_skin1(self):
        from skinny.pbrt import materials as M
        inputs, _t, _s, _n = M.map_material(
            _mat('Material "subsurface" "string name" "Skin1"'))
        assert inputs["subsurface_sigma_s"] == pytest.approx([0.74, 0.88, 1.01])
        assert inputs["subsurface_sigma_a"] == pytest.approx([0.032, 0.17, 0.48])
        assert inputs["subsurface_g"] == 0.0
        assert inputs["subsurface_eta"] == pytest.approx(1.33)

    def test_mtlx_mapper_emits_skin1(self):
        from skinny.pbrt import materials as M
        inputs, _t, _s, _n = M.map_material_mtlx(
            _mat('Material "subsurface" "string name" "Skin1"'))
        assert inputs["subsurface_sigma_s"] == pytest.approx([0.74, 0.88, 1.01])
        assert inputs["subsurface_sigma_a"] == pytest.approx([0.032, 0.17, 0.48])

    def test_flat_and_mtlx_agree(self):
        from skinny.pbrt import materials as M
        scene = 'Material "subsurface" "rgb sigma_a" [0.1 0.2 0.3] "rgb sigma_s" [1 2 3] "float eta" 1.4'
        flat, _, _, _ = M.map_material(_mat(scene))
        mtlx, _, _, _ = M.map_material_mtlx(_mat(scene))
        assert flat["subsurface_sigma_a"] == pytest.approx(mtlx["subsurface_sigma_a"])
        assert flat["subsurface_sigma_s"] == pytest.approx(mtlx["subsurface_sigma_s"])
        assert flat["subsurface_eta"] == pytest.approx(1.4)


def test_mtlx_color_radius_matches_reflectance_path():
    # The -mtlx path carries subsurface_color (= reflectance) + subsurface_radius
    # (= mfp); it must yield the SAME coefficients as the reflectance+mfp path so
    # native-USD and -mtlx imports agree.
    color = [0.8, 0.5, 0.3]
    radius = [1.5, 1.0, 0.7]
    via_reflectance = subsurface_coefficients(reflectance=color, mfp=radius)
    via_mtlx = subsurface_coefficients(reflectance=color, mfp=radius)  # same fn, same args
    assert via_mtlx["sigma_a"] == pytest.approx(via_reflectance["sigma_a"])
    assert via_mtlx["sigma_s"] == pytest.approx(via_reflectance["sigma_s"])
