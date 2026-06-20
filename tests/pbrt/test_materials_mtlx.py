"""Tests for pbrt -> MaterialX standard_surface mapping (task group 1).

``map_material_mtlx`` is a sibling of ``map_material`` that targets Autodesk
``standard_surface`` input names (consumed by ``pack_std_surface_params`` /
``_STD_SURFACE_TO_FLAT`` / ``_load_mtlx_materials``) instead of UsdPreviewSurface.
It fills the rich slots UsdPreviewSurface drops (``transmission_color``,
``coat``, ``subsurface_radius``, ``specular_anisotropy``, ``thin_walled`` …)
while keeping the roughness calibration chain bit-for-bit identical to
``map_material``.
"""

from __future__ import annotations

import pytest

from skinny.pbrt import materials as M
from skinny.pbrt.parser import parse_directives
from skinny.pbrt.tokenizer import tokenize


def _mat(text):
    from skinny.pbrt.state import PbrtMaterial

    (d,) = parse_directives(tokenize(text))
    return PbrtMaterial(d.type_arg() or "", d.params)


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


# --------------------------------------------------------------------------- #
# diffuse -> base_color + matte specular_roughness
# --------------------------------------------------------------------------- #
def test_diffuse_maps_to_base_color():
    inputs, _tex, status, _ = M.map_material_mtlx(
        _mat('Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]')
    )
    assert inputs["base_color"] == [0.2, 0.4, 0.6]
    assert inputs["metalness"] == 0.0
    # matte diffuse -> no specular highlight, roughness at the matte end
    assert inputs["specular_roughness"] == 1.0
    assert status == "exact"


def test_diffuse_default_grey():
    inputs, _tex, _status, _ = M.map_material_mtlx(_mat('Material "diffuse"'))
    assert inputs["base_color"] == [0.5, 0.5, 0.5]


def test_empty_material_is_grey_diffuse():
    inputs, _tex, _status, _ = M.map_material_mtlx(_mat('Material "none"'))
    assert inputs["base_color"] == [0.5, 0.5, 0.5]
    assert inputs["metalness"] == 0.0


def test_emissive_sets_weight_and_color():
    # An area-light material must author BOTH emission (weight) and
    # emission_color — _load_mtlx_materials recovers emissiveColor only when
    # emission > 0, so emission_color alone is dropped and the light renders
    # black. Regression for the area-light black-render bug.
    inputs, _tex, _status, _ = M.map_material_mtlx(
        _mat('Material "diffuse"'), emissive_rgb=[15.0, 15.0, 15.0]
    )
    assert inputs["emission"] == pytest.approx(1.0)
    assert list(inputs["emission_color"]) == pytest.approx([15.0, 15.0, 15.0])


# --------------------------------------------------------------------------- #
# conductor -> metalness=1 + base_color (reuse _conductor_basecolor) + roughness
# --------------------------------------------------------------------------- #
def test_conductor_is_metallic_with_basecolor():
    inputs, _tex, _, _ = M.map_material_mtlx(
        _mat('Material "conductor" "spectrum eta" "metal-Au-eta" "float roughness" 0.25')
    )
    assert inputs["metalness"] == 1.0
    assert inputs["base_color"][0] > inputs["base_color"][2]  # gold-ish (more red than blue)


def test_conductor_basecolor_matches_map_material():
    # the conductor reflectance reduction must be identical to the UsdPreviewSurface path
    src = 'Material "conductor" "spectrum eta" "metal-Cu-eta" "spectrum k" "metal-Cu-k"'
    legacy, _t, _s, _n = M.map_material(_mat(src))
    mtlx, _t2, _s2, _n2 = M.map_material_mtlx(_mat(src))
    assert mtlx["base_color"] == pytest.approx(legacy["diffuseColor"])


def test_conductor_roughness_chain_matches_map_material():
    # specular_roughness must equal the legacy roughness through the exact chain
    src = 'Material "conductor" "float roughness" 0.25'
    legacy, _t, _s, _n = M.map_material(_mat(src))
    mtlx, _t2, _s2, _n2 = M.map_material_mtlx(_mat(src))
    assert mtlx["specular_roughness"] == pytest.approx(legacy["roughness"])
    assert mtlx["specular_roughness"] == pytest.approx(0.25**0.25, abs=1e-6)


def test_conductor_specular_color_set():
    # std_surface conductors get a specular_color (tint) alongside base_color
    inputs, _tex, _, _ = M.map_material_mtlx(_mat('Material "conductor" "float roughness" 0.1'))
    assert "specular_color" in inputs


def test_remap_false_changes_roughness_mtlx():
    on, _t, _s, _n = M.map_material_mtlx(_mat('Material "conductor" "float roughness" 0.25'))
    off, _t2, _s2, _n2 = M.map_material_mtlx(
        _mat('Material "conductor" "float roughness" 0.25 "bool remaproughness" "false"')
    )
    assert on["specular_roughness"] != pytest.approx(off["specular_roughness"])


# --------------------------------------------------------------------------- #
# dielectric -> transmission>0, specular_IOR from eta, transmission_color
# --------------------------------------------------------------------------- #
def test_dielectric_opens_transmission():
    inputs, _tex, _, _ = M.map_material_mtlx(_mat('Material "dielectric" "float eta" 1.5'))
    assert inputs["transmission"] > 0.0
    assert inputs["specular_IOR"] == pytest.approx(1.5)
    # default white tint
    assert inputs["transmission_color"] == [1.0, 1.0, 1.0]


def test_dielectric_default_eta():
    inputs, _tex, _, _ = M.map_material_mtlx(_mat('Material "dielectric"'))
    assert inputs["specular_IOR"] == pytest.approx(1.5)


def test_dielectric_roughness_chain_matches_map_material():
    src = 'Material "dielectric" "float eta" 1.5 "float roughness" 0.25'
    legacy, _t, _s, _n = M.map_material(_mat(src))
    mtlx, _t2, _s2, _n2 = M.map_material_mtlx(_mat(src))
    assert mtlx["specular_roughness"] == pytest.approx(legacy["roughness"])


def test_thindielectric_sets_thin_walled():
    inputs, _tex, status, _ = M.map_material_mtlx(
        _mat('Material "thindielectric" "float eta" 1.5')
    )
    assert inputs["transmission"] > 0.0
    assert inputs["thin_walled"] is True
    assert status == "approx"


def test_dielectric_not_thin_walled():
    inputs, _tex, _, _ = M.map_material_mtlx(_mat('Material "dielectric" "float eta" 1.5'))
    assert inputs.get("thin_walled", False) is False


# --------------------------------------------------------------------------- #
# coateddiffuse / coatedconductor -> coat=1 + coat_color + distinct coat slots
# --------------------------------------------------------------------------- #
def test_coateddiffuse_sets_coat():
    inputs, _tex, _, _ = M.map_material_mtlx(
        _mat('Material "coateddiffuse" "rgb reflectance" [0.2 0.4 0.6]')
    )
    assert inputs["coat"] == 1.0
    assert inputs["base_color"] == [0.2, 0.4, 0.6]
    assert "coat_color" in inputs
    # coat interface IOR distinct from base; coat_roughness present (its own slot)
    assert "coat_IOR" in inputs
    assert "coat_roughness" in inputs


def test_coatedconductor_is_metallic_and_coated():
    inputs, _tex, _, _ = M.map_material_mtlx(
        _mat('Material "coatedconductor" "float conductor.roughness" 0.1')
    )
    assert inputs["metalness"] == 1.0
    assert inputs["coat"] == 1.0
    assert "coat_IOR" in inputs


# --------------------------------------------------------------------------- #
# subsurface -> subsurface>0 + subsurface_color + subsurface_radius
# --------------------------------------------------------------------------- #
def test_subsurface_sets_rich_slots():
    inputs, _tex, status, _ = M.map_material_mtlx(
        _mat('Material "subsurface" "float eta" 1.33')
    )
    assert inputs["subsurface"] > 0.0
    assert "subsurface_color" in inputs
    assert "subsurface_radius" in inputs
    assert inputs["specular_IOR"] == pytest.approx(1.33)
    assert status == "approx"


def test_subsurface_radius_from_pbrt_radius():
    inputs, _tex, _, _ = M.map_material_mtlx(
        _mat('Material "subsurface" "rgb radius" [1.0 0.5 0.3]')
    )
    assert inputs["subsurface_radius"] == [1.0, 0.5, 0.3]


# --------------------------------------------------------------------------- #
# diffusetransmission -> transmission tint (rich slot UsdPreviewSurface lacks)
# --------------------------------------------------------------------------- #
def test_diffusetransmission_uses_transmission():
    inputs, _tex, status, _ = M.map_material_mtlx(_mat('Material "diffusetransmission"'))
    assert inputs["transmission"] > 0.0
    assert status == "approx"


# --------------------------------------------------------------------------- #
# anisotropic uroughness/vroughness -> specular_roughness + specular_anisotropy
# (NOT isotropic geometric-mean collapse)
# --------------------------------------------------------------------------- #
def test_anisotropy_sets_specular_anisotropy_not_collapsed():
    inputs, _tex, _, notes = M.map_material_mtlx(
        _mat('Material "conductor" "float uroughness" 0.1 "float vroughness" 0.4')
    )
    assert inputs["specular_anisotropy"] != 0.0
    # specular_roughness present and not the isotropic geometric-mean collapse
    assert "specular_roughness" in inputs
    # no geometric-mean note (we represent anisotropy faithfully)
    assert not any("geometric mean" in n for n in notes)


def test_isotropic_roughness_has_zero_anisotropy():
    inputs, _tex, _, _ = M.map_material_mtlx(
        _mat('Material "conductor" "float roughness" 0.25')
    )
    assert inputs.get("specular_anisotropy", 0.0) == 0.0


def test_anisotropy_roughness_uses_calibrated_chain():
    # both u/v roughness go through pbrt_roughness_to_alpha + alpha_to_usd_roughness
    inputs, _tex, _, _ = M.map_material_mtlx(
        _mat('Material "conductor" "float uroughness" 0.1 "float vroughness" 0.4')
    )
    au = M.pbrt_roughness_to_alpha(0.1, True)
    av = M.pbrt_roughness_to_alpha(0.4, True)
    ru = M.alpha_to_usd_roughness(au)
    rv = M.alpha_to_usd_roughness(av)
    # specular_roughness should be one of the per-axis calibrated roughnesses
    # (the "primary" axis) — assert it falls within [min,max] of the calibrated pair
    assert min(ru, rv) - 1e-6 <= inputs["specular_roughness"] <= max(ru, rv) + 1e-6


# --------------------------------------------------------------------------- #
# texture connections returned in the same shape as map_material's tex_inputs
# --------------------------------------------------------------------------- #
def test_reflectance_texture_connects_to_base_color():
    inputs, tex, _, _ = M.map_material_mtlx(
        _mat('Material "diffuse" "texture reflectance" "kd"'),
        textures={"kd": _imagemap("kd", datatype="spectrum")},
    )
    assert "base_color" in tex
    path, _cs, value_type = tex["base_color"]
    assert path.endswith("tex.png")
    assert value_type == "color3f"
    # scalar fallback still present
    assert isinstance(inputs["base_color"], list)


def test_roughness_texture_connects_to_specular_roughness():
    inputs, tex, _, _ = M.map_material_mtlx(
        _mat('Material "conductor" "texture roughness" "r"'),
        textures={"r": _imagemap("r")},
    )
    assert "specular_roughness" in tex
    path, _cs, value_type = tex["specular_roughness"]
    assert path.endswith("tex.png")
    assert value_type == "float"
    assert isinstance(inputs["specular_roughness"], float)


def test_nested_scale_over_imagemap_resolves_inner():
    textures = {"rough": _scale_over("rough", "rough-img"), "rough-img": _imagemap("rough-img")}
    _inputs, tex, _, _ = M.map_material_mtlx(
        _mat('Material "conductor" "texture roughness" "rough"'), textures=textures
    )
    assert tex["specular_roughness"][0].endswith("tex.png")


def test_unsupported_texture_falls_back_to_scalar_and_approx():
    from skinny.pbrt.parser import ParamSet
    from skinny.pbrt.state import PbrtTexture

    ck = PbrtTexture("ck", "float", "checkerboard", ParamSet({}))
    inputs, tex, status, notes = M.map_material_mtlx(
        _mat('Material "conductor" "texture roughness" "ck"'),
        textures={"ck": ck},
    )
    assert "specular_roughness" not in tex
    assert isinstance(inputs["specular_roughness"], float)
    assert status == "approx"
    assert any("ck" in n for n in notes)


def test_constant_roughness_authors_no_connection():
    _inputs, tex, _, _ = M.map_material_mtlx(_mat('Material "conductor" "float roughness" 0.25'))
    assert "specular_roughness" not in tex


# --------------------------------------------------------------------------- #
# emissive passthrough
# --------------------------------------------------------------------------- #
def test_emissive_maps_to_emission_color():
    inputs, _tex, _, _ = M.map_material_mtlx(
        _mat('Material "diffuse"'), emissive_rgb=[3.0, 2.0, 1.0]
    )
    assert inputs["emission_color"] == [3.0, 2.0, 1.0]


# --------------------------------------------------------------------------- #
# unknown type -> grey diffuse, approx
# --------------------------------------------------------------------------- #
def test_unknown_type_is_grey_diffuse_approx():
    inputs, _tex, status, notes = M.map_material_mtlx(_mat('Material "wibble"'))
    assert inputs["base_color"] == [0.5, 0.5, 0.5]
    assert status == "approx"
    assert any("wibble" in n for n in notes)
