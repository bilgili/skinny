"""Tests for pbrt material mapping + roughness remap (task 7.1)."""

from __future__ import annotations

import pytest

from skinny.pbrt import materials as M
from skinny.pbrt.parser import parse_directives
from skinny.pbrt.tokenizer import tokenize


def _mat(text):
    from skinny.pbrt.state import PbrtMaterial

    (d,) = parse_directives(tokenize(text))
    return PbrtMaterial(d.type_arg() or "", d.params)


def test_pbrt_v4_roughness_remap_is_sqrt():
    # pbrt v4: alpha = sqrt(roughness) when remaproughness (default)
    assert M.pbrt_roughness_to_alpha(0.25, remap=True) == pytest.approx(0.5)
    assert M.pbrt_roughness_to_alpha(0.25, remap=False) == pytest.approx(0.25)


def test_alpha_to_usd_roughness_inverts_skinny_square():
    # skinny GGX alpha = usd_roughness**2 -> usd_roughness = sqrt(alpha)
    assert M.alpha_to_usd_roughness(0.25) == pytest.approx(0.5)


def test_diffuse_passthrough():
    inputs, status, _ = M.map_material(_mat('Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]'))
    assert inputs["diffuseColor"] == [0.2, 0.4, 0.6]
    assert inputs["metallic"] == 0.0
    assert status == "exact"


def test_conductor_is_metallic_with_remapped_roughness():
    inputs, _, _ = M.map_material(
        _mat('Material "conductor" "spectrum eta" "metal-Au-eta" "float roughness" 0.25')
    )
    assert inputs["metallic"] == 1.0
    # full chain: usd_roughness = sqrt(sqrt(0.25)) = 0.25**0.25
    assert inputs["roughness"] == pytest.approx(0.25**0.25, abs=1e-6)
    assert inputs["diffuseColor"][0] > inputs["diffuseColor"][2]  # gold-ish


def test_remap_false_changes_roughness():
    on, _, _ = M.map_material(_mat('Material "conductor" "float roughness" 0.25'))
    off, _, _ = M.map_material(
        _mat('Material "conductor" "float roughness" 0.25 "bool remaproughness" "false"')
    )
    assert on["roughness"] != pytest.approx(off["roughness"])


def test_dielectric_opens_transmission_gate():
    inputs, _, _ = M.map_material(_mat('Material "dielectric" "float eta" 1.5'))
    assert inputs["opacity"] == 0.0
    assert inputs["ior"] == 1.5


def test_anisotropy_is_flagged():
    _, _, notes = M.map_material(
        _mat('Material "conductor" "float uroughness" 0.1 "float vroughness" 0.4')
    )
    assert any("anisotropic" in n for n in notes)
