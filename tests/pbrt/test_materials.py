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
    inputs, _tex, status, _ = M.map_material(_mat('Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]'))
    assert inputs["diffuseColor"] == [0.2, 0.4, 0.6]
    assert inputs["metallic"] == 0.0
    assert status == "exact"


def test_constant_spectrum_reflectance_is_achromatic():
    # "spectrum reflectance" [200 0.2 900 0.2] is constant -> exactly [0.2]*3
    inputs, _tex, _status, _ = M.map_material(
        _mat('Material "diffuse" "spectrum reflectance" [200 0.2 900 0.2]')
    )
    assert inputs["diffuseColor"] == pytest.approx([0.2, 0.2, 0.2])


def test_conductor_is_metallic_with_remapped_roughness():
    inputs, _tex, _, _ = M.map_material(
        _mat('Material "conductor" "spectrum eta" "metal-Au-eta" "float roughness" 0.25')
    )
    assert inputs["metallic"] == 1.0
    # full chain: usd_roughness = sqrt(sqrt(0.25)) = 0.25**0.25
    assert inputs["roughness"] == pytest.approx(0.25**0.25, abs=1e-6)
    assert inputs["diffuseColor"][0] > inputs["diffuseColor"][2]  # gold-ish


def test_remap_false_changes_roughness():
    on, _tex, _, _ = M.map_material(_mat('Material "conductor" "float roughness" 0.25'))
    off, _tex, _, _ = M.map_material(
        _mat('Material "conductor" "float roughness" 0.25 "bool remaproughness" "false"')
    )
    assert on["roughness"] != pytest.approx(off["roughness"])


def test_dielectric_opens_transmission_gate():
    inputs, _tex, _, _ = M.map_material(_mat('Material "dielectric" "float eta" 1.5'))
    assert inputs["opacity"] == 0.0
    assert inputs["ior"] == 1.5


def test_anisotropy_is_flagged():
    _, _, _, notes = M.map_material(
        _mat('Material "conductor" "float uroughness" 0.1 "float vroughness" 0.4')
    )
    assert any("anisotropic" in n for n in notes)


# --------------------------------------------------------------------------- #
# texture-valued (FloatTexture / SpectrumTexture) material parameters
# --------------------------------------------------------------------------- #
def _imagemap(name, datatype="float", filename="tex.png"):
    from skinny.pbrt.parser import Param, ParamSet
    from skinny.pbrt.state import PbrtTexture

    params = ParamSet({"filename": Param("string", "filename", (filename,))})
    return PbrtTexture(name, datatype, "imagemap", params)


def _scale_over(name, inner, datatype="float", scale=2.0):
    from skinny.pbrt.parser import Param, ParamSet
    from skinny.pbrt.state import PbrtTexture

    params = ParamSet(
        {
            "tex": Param("texture", "tex", (inner,)),
            "scale": Param("float", "scale", (scale,)),
        }
    )
    return PbrtTexture(name, datatype, "scale", params)


def _other_class(name, klass="checkerboard", datatype="float"):
    from skinny.pbrt.parser import ParamSet
    from skinny.pbrt.state import PbrtTexture

    return PbrtTexture(name, datatype, klass, ParamSet({}))


# --- promoting accessors (mirror pbrt GetFloatTexture / GetSpectrumTexture) --- #
def test_get_float_texture_promotes_constant():
    p = _mat('Material "conductor" "float roughness" 0.3').params
    pv = M.get_float_texture(p, "roughness", 0.0)
    assert pv.const == pytest.approx(0.3)
    assert not pv.is_tex


def test_get_float_texture_absent_uses_default():
    p = _mat('Material "conductor"').params
    pv = M.get_float_texture(p, "roughness", 0.7)
    assert pv.const == pytest.approx(0.7)
    assert not pv.is_tex


def test_get_float_texture_resolves_named_texture():
    p = _mat('Material "conductor" "texture roughness" "r"').params
    pv = M.get_float_texture(p, "roughness", 0.0, textures={"r": _imagemap("r")})
    assert pv.is_tex
    assert pv.tex[0].endswith("tex.png")


def test_get_spectrum_texture_resolves_named_texture():
    p = _mat('Material "diffuse" "texture reflectance" "kd"').params
    pv = M.get_spectrum_texture(
        p, "reflectance", [0.5, 0.5, 0.5], textures={"kd": _imagemap("kd", datatype="spectrum")}
    )
    assert pv.is_tex
    assert pv.tex[0].endswith("tex.png")


# --- map_material: texture-valued params never crash + map to their own input --- #
def test_texture_roughness_does_not_raise_and_connects():
    inputs, tex, status, _ = M.map_material(
        _mat('Material "conductor" "texture roughness" "r"'),
        textures={"r": _imagemap("r")},
    )
    assert "roughness" in tex
    path, _cs, value_type = tex["roughness"]
    assert path.endswith("tex.png")
    assert value_type == "float"
    assert isinstance(inputs["roughness"], float)  # scalar fallback present


def test_nested_scale_over_imagemap_roughness_resolves_inner_image():
    textures = {"rough": _scale_over("rough", "rough-img"), "rough-img": _imagemap("rough-img")}
    inputs, tex, _, notes = M.map_material(
        _mat('Material "conductor" "texture roughness" "rough"'), textures=textures
    )
    assert tex["roughness"][0].endswith("tex.png")
    assert any("approx" in n.lower() or "remap" in n.lower() for n in notes)


def test_unsupported_texture_class_falls_back_to_scalar():
    inputs, tex, status, notes = M.map_material(
        _mat('Material "conductor" "texture roughness" "ck"'),
        textures={"ck": _other_class("ck")},
    )
    assert "roughness" not in tex  # no connection
    assert isinstance(inputs["roughness"], float)  # scalar default
    assert status == "approx"
    assert any("ck" in n for n in notes)


def test_constant_roughness_authors_no_connection():
    inputs, tex, _, _ = M.map_material(_mat('Material "conductor" "float roughness" 0.25'))
    assert "roughness" not in tex
    assert inputs["roughness"] == pytest.approx(0.25**0.25, abs=1e-6)


def test_texture_eta_does_not_raise():
    inputs, tex, _, notes = M.map_material(
        _mat('Material "dielectric" "texture eta" "e"'),
        textures={"e": _imagemap("e")},
    )
    assert isinstance(inputs["ior"], float)  # USD ior has no texture input -> scalar
    assert any("eta" in n for n in notes)


def test_reflectance_texture_maps_to_diffusecolor_not_roughness():
    inputs, tex, _, _ = M.map_material(
        _mat('Material "diffuse" "texture reflectance" "kd"'),
        textures={"kd": _imagemap("kd", datatype="spectrum")},
    )
    assert "diffuseColor" in tex
    assert tex["diffuseColor"][2] == "color3f"
    assert "roughness" not in tex


def test_per_parameter_mapping_roughness_vs_reflectance():
    # roughness texture -> roughness (.r/float); reflectance texture -> diffuseColor (.rgb/color3f)
    r_inputs, r_tex, _, _ = M.map_material(
        _mat('Material "conductor" "texture roughness" "r"'),
        textures={"r": _imagemap("r")},
    )
    d_inputs, d_tex, _, _ = M.map_material(
        _mat('Material "diffuse" "texture reflectance" "kd"'),
        textures={"kd": _imagemap("kd", datatype="spectrum")},
    )
    assert r_tex["roughness"][2] == "float"
    assert d_tex["diffuseColor"][2] == "color3f"
