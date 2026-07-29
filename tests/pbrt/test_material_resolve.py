"""Hostless tests for the shared pbrt material resolver (change:
pbrt-material-shared-resolver).

``resolve_material`` is the single owner of pbrt-param interpretation; these
assert the *resolved intermediate* directly, so a param read, default, note or
status escalation is pinned once instead of twice through the two mappers.
``test_materials.py`` / ``test_materials_mtlx.py`` remain the output-level lock.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from skinny.pbrt import materials as M
from skinny.pbrt.parser import parse_directives
from skinny.pbrt.report import APPROX, EXACT
from skinny.pbrt.tokenizer import tokenize


def _mat(text):
    from skinny.pbrt.state import PbrtMaterial

    (d,) = parse_directives(tokenize(text))
    return PbrtMaterial(d.type_arg() or "", d.params)


def _res(text, flavor=M.USD, **kw):
    return M.resolve_material(_mat(text), flavor=flavor, **kw)


class _Tex:
    """Minimal stand-in for a parsed pbrt Texture directive."""

    def __init__(self, klass="imagemap", filename="t.png", datatype="spectrum"):
        self.klass = klass
        self.datatype = datatype
        (d,) = parse_directives(tokenize(f'Material "x" "string filename" "{filename}"'))
        self.params = d.params


def _textures(**kw):
    return dict(kw)


# --------------------------------------------------------------------------- #
# per-material-type resolved form
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_empty_and_none_resolve_to_grey_base(flavor):
    for src in ('Material ""', 'Material "none"'):
        res = _res(src, flavor)
        assert res.lobes["base_color"].const == [0.5, 0.5, 0.5]
        assert res.status == EXACT
        assert res.notes == []


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_interface_is_a_lobeless_null_boundary(flavor):
    res = _res('Material "interface"', flavor)
    assert res.lobes["base_color"].const == [0.0, 0.0, 0.0]
    assert res.lobes["metallic"] == 0.0
    assert res.lobes["roughness"].pv.const == 1.0
    assert res.notes == [
        "interface -> null boundary material (no BSDF lobes); routes to volume path"]
    # a null boundary is an exact translation, not an approximation
    assert res.status == EXACT


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_diffuse_reads_reflectance_and_pins_roughness(flavor):
    res = _res('Material "diffuse" "rgb reflectance" [0.2 0.4 0.6]', flavor)
    assert res.lobes["base_color"].const == [0.2, 0.4, 0.6]
    assert res.lobes["roughness"].pv.const == 1.0
    assert "metallic" not in res.lobes


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_conductor_is_metallic_with_a_specular_tint_matching_base(flavor):
    res = _res('Material "conductor" "rgb reflectance" [0.9 0.7 0.3] "float roughness" 0.25',
               flavor)
    assert res.lobes["metallic"] == 1.0
    assert res.lobes["base_color"].const == [0.9, 0.7, 0.3]
    # roughness 0.25 -> alpha 0.5 -> usd roughness sqrt(0.5)
    assert res.lobes["roughness"].pv.const == pytest.approx(0.5 ** 0.5)
    # the tint mirrors the resolved conductor base colour (here the authored
    # reflectance, which short-circuits the eta/k chain)
    assert res.lobes["specular_color"] == pytest.approx([0.9, 0.7, 0.3])


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_conductor_without_reflectance_tints_from_the_resolved_ior(flavor):
    res = _res('Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k"',
               flavor)
    assert res.lobes["specular_color"] == pytest.approx(res.lobes["base_color"].const)


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_dielectric_resolves_full_transmission_and_eta(flavor):
    res = _res('Material "dielectric" "float eta" 1.7', flavor)
    assert res.lobes["base_color"].const == [1.0, 1.0, 1.0]
    assert res.lobes["transmission"] == 1.0
    assert res.lobes["transmission_color"] == [1.0, 1.0, 1.0]
    assert res.lobes["ior"] == pytest.approx(1.7)
    assert "thin_walled" not in res.lobes
    assert res.status == EXACT


def test_thindielectric_is_approx_with_flavor_worded_note():
    usd = _res('Material "thindielectric"', M.USD)
    mtlx = _res('Material "thindielectric"', M.MTLX)
    assert usd.lobes["thin_walled"] is True and mtlx.lobes["thin_walled"] is True
    assert usd.status == APPROX and mtlx.status == APPROX
    assert usd.notes == ["thindielectric approximated as thin dielectric"]
    assert mtlx.notes == ["thindielectric approximated as thin-walled transmissive surface"]


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_coateddiffuse_coat_roughness_comes_from_top_level_roughness(flavor):
    # pbrt spells the interface (coat) roughness as the top-level `roughness`;
    # both targets take it from the same shared calibration chain.
    res = _res('Material "coateddiffuse" "float roughness" 0.25', flavor)
    assert res.lobes["coat"] == 1.0
    assert res.lobes["coat_color"] == [1.0, 1.0, 1.0]
    assert res.lobes["coat_roughness"].pv.const == pytest.approx(0.5 ** 0.5)
    # the base lobe stays fully rough (the coat carries the specular)
    assert res.lobes["roughness"].pv.const == 1.0


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_coatedconductor_coat_roughness_reads_interface_roughness(flavor):
    res = _res('Material "coatedconductor" "float interface.roughness" 0.4', flavor)
    assert res.lobes["coat"] == 1.0
    # calibrated like every other roughness: pbrt remaps `interface.*` too
    # (pbrt-v4 materials.cpp:351), so 0.4 -> alpha sqrt(0.4) -> sqrt of that
    assert res.lobes["coat_roughness"] == pytest.approx((0.4 ** 0.5) ** 0.5)


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_coatedconductor_metal_roughness_reads_conductor_roughness(flavor):
    """Both flavours take the base metal from `conductor.roughness` alone.

    The top-level `roughness` below is a value pbrt itself never reads:
    `CoatedConductorMaterial::Create` does not define one, and pbrt does not
    merely ignore it — it REFUSES the scene ("roughness": unused parameter). A
    read of it could therefore only come from a scene pbrt would not render.
    """
    src = ('Material "coatedconductor" "float conductor.roughness" 0.04 '
           '"float interface.roughness" 0.36 "float roughness" 0.64')
    res = _res(src, flavor)
    assert res.lobes["roughness"].pv.const == pytest.approx(0.2 ** 0.5)  # sqrt(sqrt(0.04))
    assert res.lobes["coat_roughness"] == pytest.approx((0.36 ** 0.5) ** 0.5)
    # the two lobes must not collapse onto one value — that is what makes the
    # confirming-suite scene a discriminator rather than a passenger
    assert res.lobes["roughness"].pv.const != pytest.approx(res.lobes["coat_roughness"])


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_coatedconductor_metal_anisotropy_survives_as_an_unreduced_pair(flavor):
    """`conductor.uroughness`/`conductor.vroughness` reach the adapters unreduced.

    Both flavours dropped them silently before this change, so an anisotropic
    coated metal imported isotropic with no note.
    """
    res = _res('Material "coatedconductor" "float conductor.uroughness" 0.04 '
               '"float conductor.vroughness" 0.25', flavor)
    rr = res.lobes["roughness"]
    assert rr.is_aniso
    assert rr.alpha_u == pytest.approx(0.2)   # remap: alpha = sqrt(roughness)
    assert rr.alpha_v == pytest.approx(0.5)
    # only the UsdPreviewSurface target loses the axes in its collapse, so only
    # it reports the reduction
    note = "anisotropic roughness reduced to isotropic (geometric mean)"
    assert (note in res.notes) == (flavor == M.USD)


def test_the_committed_coated_metal_scene_still_discriminates():
    """The confirming-suite scene ON DISK, not an inline stand-in.

    The scene is the load-bearing artifact: if it is ever regenerated with the
    two roughnesses equal, every render gate passes whichever parameter is read
    and the fix ships unprotected. An inline assertion cannot catch that, so this
    one imports the committed `.pbrt` and pins the two properties the scene has
    to have — no top-level `roughness` (pbrt refuses such a scene), and a metal
    that resolves distinctly from the coat.
    """
    import os

    from skinny.pbrt.parser import parse_directives
    from skinny.pbrt.state import PbrtMaterial

    scene = os.path.join(os.path.dirname(__file__), "..", "assets", "suite",
                         "mat_coated_metal", "mat_coated_metal.pbrt")
    directives = parse_directives(tokenize(open(scene).read()))
    coated = [d for d in directives
              if d.name == "Material" and d.type_arg() == "coatedconductor"]
    assert len(coated) == 1, "the scene must author exactly one coatedconductor"
    params = coated[0].params

    assert "conductor.roughness" in params
    assert "interface.roughness" in params
    assert "roughness" not in params, (
        "the scene must NOT author a top-level `roughness` — pbrt refuses such a "
        "scene, so it could carry no reference EXR")

    for flavor in (M.USD, M.MTLX):
        res = M.resolve_material(PbrtMaterial("coatedconductor", params), flavor=flavor)
        metal = res.lobes["roughness"].pv.const
        coat = res.lobes["coat_roughness"]
        assert metal != pytest.approx(coat, abs=0.05), (
            f"{flavor}: the scene's metal ({metal:.4f}) and coat ({coat:.4f}) "
            "roughnesses are too close to discriminate which spelling is read")


def test_the_two_coated_types_read_asymmetric_spellings_like_pbrt():
    """pbrt is asymmetric here and the resolver mirrors it rather than unifying.

    `CoatedDiffuseMaterial::Create` reads the top-level `roughness` for its coat;
    `CoatedConductorMaterial::Create` reads none at all. A later "consistency"
    cleanup that unified the two would break one of them.
    """
    diffuse = _res('Material "coateddiffuse" "float roughness" 0.04 '
                   '"float conductor.roughness" 0.64')
    assert diffuse.lobes["coat_roughness"].pv.const == pytest.approx(0.2 ** 0.5)

    conductor = _res('Material "coatedconductor" "float conductor.roughness" 0.04')
    assert conductor.lobes["roughness"].pv.const == pytest.approx(0.2 ** 0.5)


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_diffusetransmission_is_half_transmissive_and_approx(flavor):
    res = _res('Material "diffusetransmission"', flavor)
    assert res.lobes["base_color"].const == [0.25, 0.25, 0.25]
    assert res.lobes["transmission"] == 0.5
    assert res.status == APPROX


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_unknown_material_degrades_to_diffuse_grey(flavor):
    res = _res('Material "wibble"', flavor)
    assert res.lobes == {}
    assert res.status == APPROX
    assert res.notes == ["unknown material 'wibble' best-effort as diffuse grey"]


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_emissive_rgb_rides_as_a_neutral_lobe_last(flavor):
    res = _res('Material "diffuse"', flavor, emissive_rgb=[7.0, 5.0, 3.0])
    assert res.lobes["emission_rgb"] == [7.0, 5.0, 3.0]
    assert list(res.lobes)[-1] == "emission_rgb"


def test_missing_material_resolves_as_diffuse_over_an_empty_paramset():
    # an emissive shape may carry no material at all
    res = M.resolve_material(None, emissive_rgb=[1.0, 1.0, 1.0], flavor=M.USD)
    assert res.lobes["base_color"].const == [0.5, 0.5, 0.5]
    assert res.status == EXACT


# --------------------------------------------------------------------------- #
# named spectra (7 metals / 7 glasses d-line)
# --------------------------------------------------------------------------- #

def test_named_glass_eta_resolves_to_its_d_line_ior():
    from skinny.pbrt.data import spectral_tables as st

    res = _res('Material "dielectric" "spectrum eta" "glass-BK7"')
    assert res.lobes["ior"] == pytest.approx(st.named_glass_ior_d("glass-BK7"))
    assert res.notes == []  # an exact substitution, not a fallback


def test_unrecognised_named_eta_degrades_with_a_note():
    res = _res('Material "dielectric" "spectrum eta" "glass-ZZ9"')
    assert res.lobes["ior"] == pytest.approx(1.5)
    assert res.notes == ["named spectrum 'glass-ZZ9' on eta unrecognised; used default 1.5"]


def test_named_metal_eta_drives_the_conductor_base_colour():
    from skinny.pbrt import spectra

    res = _res('Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k"')
    assert res.lobes["base_color"].const == pytest.approx(
        list(spectra.named_metal_reflectance_rgb("au")))
    assert res.notes == []


def test_unknown_named_metal_falls_back_to_copper_with_a_note():
    res = _res('Material "conductor" "spectrum eta" "metal-Xx-eta"')
    assert res.notes == [
        "named spectrum 'metal-Xx-eta' on conductor eta unrecognised; defaulted to copper"]


# --------------------------------------------------------------------------- #
# texture bindings
# --------------------------------------------------------------------------- #

def test_texture_bound_reflectance_rides_on_the_base_colour_lobe():
    res = _res('Material "diffuse" "texture reflectance" "t"', textures=_textures(t=_Tex()))
    pv = res.lobes["base_color"]
    assert pv.is_tex and pv.tex[0].endswith("t.png")
    assert pv.const == [0.5, 0.5, 0.5]  # the constant stays the fallback
    assert res.status == EXACT


def test_unresolvable_texture_notes_and_escalates_to_approx():
    res = _res('Material "diffuse" "texture reflectance" "t"',
               textures=_textures(t=_Tex(klass="checkerboard")))
    assert res.notes == ["texture 't' on reflectance unresolved/unsupported; used default"]
    assert res.status == APPROX


def test_texture_bound_roughness_uses_a_mid_fallback_and_flags_the_missing_remap():
    res = _res('Material "conductor" "texture roughness" "t"', textures=_textures(t=_Tex()))
    rr = res.lobes["roughness"]
    assert rr.pv.const == 0.5 and rr.pv.is_tex
    assert "roughness texture connected; perceptual remap not applied to texture (approx)" \
        in res.notes


@pytest.mark.parametrize("flavor,target", [(M.USD, "USD input"),
                                           (M.MTLX, "standard_surface input")])
def test_texture_on_a_scalar_only_input_is_worded_for_the_flavor(flavor, target):
    res = _res('Material "dielectric" "texture eta" "t"', flavor, textures=_textures(t=_Tex()))
    assert res.lobes["ior"] == pytest.approx(1.5)
    assert f"eta texture not supported on {target}; used scalar default" in res.notes


# --------------------------------------------------------------------------- #
# anisotropy: resolved unreduced, reduced by adapter policy
# --------------------------------------------------------------------------- #

def test_anisotropic_roughness_resolves_to_unreduced_alphas():
    res = _res('Material "conductor" "float uroughness" 0.04 "float vroughness" 0.36')
    rr = res.lobes["roughness"]
    assert rr.is_aniso and rr.pv is None
    assert rr.alpha_u == pytest.approx(0.2)  # sqrt(0.04)
    assert rr.alpha_v == pytest.approx(0.6)  # sqrt(0.36)


def test_usd_collapses_anisotropy_to_the_geometric_mean_mtlx_keeps_both_axes():
    src = 'Material "conductor" "float uroughness" 0.04 "float vroughness" 0.36'
    rr = _res(src).lobes["roughness"]
    usd = M._usd_roughness(rr)
    mtlx_pv, aniso = M._mtlx_roughness(rr)
    assert usd.const == pytest.approx(((0.2 * 0.6) ** 0.5) ** 0.5)
    ru, rv = 0.2 ** 0.5, 0.6 ** 0.5
    assert mtlx_pv.const == pytest.approx(0.5 * (ru + rv))
    assert aniso == pytest.approx(1.0 - ru / rv)


def test_only_the_usd_flavor_notes_the_anisotropy_collapse():
    src = 'Material "conductor" "float uroughness" 0.04 "float vroughness" 0.36'
    note = "anisotropic roughness reduced to isotropic (geometric mean)"
    assert note in _res(src, M.USD).notes
    assert note not in _res(src, M.MTLX).notes


def test_isotropic_roughness_reduces_to_zero_anisotropy():
    rr = _res('Material "conductor" "float roughness" 0.25').lobes["roughness"]
    assert not rr.is_aniso
    assert M._mtlx_roughness(rr)[1] == 0.0
    assert M._usd_roughness(rr) is rr.pv


# --------------------------------------------------------------------------- #
# subsurface coefficient precedence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_subsurface_carries_medium_coefficients_and_is_approx(flavor):
    res = _res('Material "subsurface" "rgb sigma_a" [0.02 0.08 0.2] '
               '"rgb sigma_s" [2.5 3.2 4.0] "float g" 0.4 "float eta" 1.4', flavor)
    assert res.lobes["subsurface"] == 1.0
    assert res.lobes["ior"] == pytest.approx(1.4)
    assert res.lobes["subsurface_sigma_a"] == pytest.approx([0.02, 0.08, 0.2])
    assert res.lobes["subsurface_sigma_s"] == pytest.approx([2.5, 3.2, 4.0])
    assert res.lobes["subsurface_g"] == pytest.approx(0.4)
    assert res.lobes["subsurface_eta"] == pytest.approx(1.4)
    assert res.status == APPROX


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_subsurface_named_preset_beats_the_defaults(flavor):
    named = _res('Material "subsurface" "string name" "skin1"', flavor)
    plain = _res('Material "subsurface"', flavor)
    assert named.lobes["subsurface_sigma_s"] != plain.lobes["subsurface_sigma_s"]


@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_subsurface_scale_multiplies_the_coefficients(flavor):
    one = _res('Material "subsurface" "rgb sigma_a" [1 2 3] "rgb sigma_s" [4 5 6]', flavor)
    two = _res('Material "subsurface" "rgb sigma_a" [1 2 3] "rgb sigma_s" [4 5 6] '
               '"float scale" 2', flavor)
    assert two.lobes["subsurface_sigma_a"] == pytest.approx(
        [2 * v for v in one.lobes["subsurface_sigma_a"]])


# --------------------------------------------------------------------------- #
# flavour gates: the mtlx-only reads must not happen under the usd flavour
# (no value, no note, no EXACT -> APPROX escalation)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("src,lobe,param", [
    ('Material "diffusetransmission" "texture transmittance" "t"',
     "transmission_color", "transmittance"),
    ('Material "coateddiffuse" "texture interface.eta" "t"', "coat_ior", "interface.eta"),
    ('Material "coatedconductor" "texture interface.eta" "t"', "coat_ior", "interface.eta"),
])
def test_mtlx_only_reads_are_absent_under_the_usd_flavor(src, lobe, param):
    textures = _textures(t=_Tex(klass="checkerboard"))  # unresolvable -> would note
    usd = _res(src, M.USD, textures=textures)
    mtlx = _res(src, M.MTLX, textures=textures)
    assert lobe not in usd.lobes
    assert lobe in mtlx.lobes
    assert not any(param in n for n in usd.notes)
    assert any(param in n for n in mtlx.notes)


def test_subsurface_colour_and_radius_are_mtlx_only():
    """The LOBES stay mtlx-only; `subsurface_radius` now comes from `mfp`.

    pbrt's `SubsurfaceMaterial::Create` has no `radius` parameter — only shapes
    have one — so reading it was skinny inventing behaviour (change
    subsurface-promoting-accessors). `subsurface_radius` IS the mean free path,
    which `mfp` already carries, so the lobe is derived from it and the phantom
    read is gone.
    """
    src = ('Material "subsurface" "rgb reflectance" [0.7 0.4 0.3] '
           '"rgb mfp" [0.9 0.6 0.4] "rgb radius" [9 9 9]')
    usd, mtlx = _res(src, M.USD), _res(src, M.MTLX)
    assert "subsurface_color" not in usd.lobes and "subsurface_radius" not in usd.lobes
    assert mtlx.lobes["subsurface_color"] == pytest.approx([0.7, 0.4, 0.3])
    assert mtlx.lobes["subsurface_radius"] == pytest.approx([0.9, 0.6, 0.4])


def test_usd_flavor_does_not_escalate_on_an_mtlx_only_unresolvable_texture():
    src = 'Material "diffusetransmission" "texture transmittance" "t"'
    textures = _textures(t=_Tex(klass="checkerboard"))
    # both are APPROX for their own reason (the diffusetransmission branch), but
    # only the mtlx flavour records the unresolved-texture note
    assert not any("unresolved/unsupported" in n for n in _res(src, M.USD, textures=textures).notes)
    assert any("unresolved/unsupported" in n for n in _res(src, M.MTLX, textures=textures).notes)


# --------------------------------------------------------------------------- #
# note ORDER (accessor notes interleave with branch notes in read order)
# --------------------------------------------------------------------------- #

def test_notes_are_in_read_order():
    res = _res('Material "conductor" "spectrum eta" "metal-Xx-eta" "texture roughness" "t"',
               M.USD, textures=_textures(t=_Tex(klass="checkerboard")))
    assert res.notes == [
        "named spectrum 'metal-Xx-eta' on conductor eta unrecognised; defaulted to copper",
        "texture 't' on roughness unresolved/unsupported; used default",
    ]


def test_mtlx_coatedconductor_note_order_follows_its_read_order():
    res = _res('Material "coatedconductor" "texture interface.eta" "e" '
               '"texture interface.roughness" "r"',
               M.MTLX, textures=_textures(e=_Tex(), r=_Tex()))
    assert res.notes == [
        "conductor IOR unresolved; defaulted to copper",
        "interface.eta texture not supported on standard_surface input; used scalar default",
        "interface.roughness texture not supported on standard_surface input; used scalar default",
    ]


# --------------------------------------------------------------------------- #
# the adapters are emit-only: no ParamSet reads survive outside the resolver
# --------------------------------------------------------------------------- #

#: the promoting accessors — calling either means a pbrt param was read.
_ACCESSORS = frozenset({"get_float_texture", "get_spectrum_texture", "resolve_texture"})
#: ``ParamSet`` reader methods, flagged only on a ParamSet-shaped receiver
#: (``p``/``params``/``<x>.params``) so a plain ``dict.get`` on a lobe map is not
#: mistaken for a param read.
_PARAMSET_METHODS = frozenset(
    {"get", "string", "rgb", "floats", "bool", "float", "int", "ints"})
_PARAMSET_NAMES = frozenset({"p", "params"})


def _is_paramset(node) -> bool:
    return ((isinstance(node, ast.Name) and node.id in _PARAMSET_NAMES)
            or (isinstance(node, ast.Attribute) and node.attr == "params"))


def _reads_in(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _ACCESSORS:
            found.add(fn.id)
        elif (isinstance(fn, ast.Attribute) and fn.attr in _PARAMSET_METHODS
                and _is_paramset(fn.value)):
            found.add(f"params.{fn.attr}")
    return found


@pytest.mark.parametrize("adapter", [M.map_material, M.map_material_mtlx])
def test_adapters_perform_zero_pbrt_param_reads(adapter):
    assert _reads_in(adapter) == set(), (
        f"{adapter.__name__} reads pbrt params directly; interpretation belongs "
        "in resolve_material")


@pytest.mark.parametrize("adapter", [M.map_material, M.map_material_mtlx])
def test_adapters_only_ever_hand_the_material_to_the_resolver(adapter):
    """The structural half of the gate above.

    ``_reads_in`` is syntactic, so it could be walked around by aliasing the
    ParamSet (``q = pbrt_material.params; q.get(...)``). This closes that: the
    adapter may mention ``pbrt_material`` *only* as an argument of the
    ``resolve_material`` call, so there is nothing to alias in the first place.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(adapter)))
    handed_over = {
        id(a) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_material"
        for a in list(node.args) + [k.value for k in node.keywords]
    }
    leaked = [n for n in ast.walk(tree)
              if isinstance(n, ast.Name) and n.id == "pbrt_material"
              and isinstance(n.ctx, ast.Load) and id(n) not in handed_over]
    assert not leaked, (
        f"{adapter.__name__} touches pbrt_material outside the resolve_material "
        f"call (lines {[n.lineno for n in leaked]})")


def test_media_subsurface_emission_holds_no_second_coefficient_chain():
    """`media.subsurface_overrides` emits; it does not interpret.

    It used to carry its own copy of the coefficient chain, which drifted from
    the resolver's in three ways — no named-spectrum `eta` guard (a
    `"spectrum eta" "glass-BK7"` crashed the import), the mm-per-unit division,
    and the `ior` key. It now takes the RESOLVED lobes and adds only the last two
    (change subsurface-eta-single-owner).
    """
    from skinny.pbrt import media

    assert _reads_in(media.subsurface_overrides) == set(), (
        "media.subsurface_overrides reads pbrt material params directly; "
        "interpretation belongs in materials.subsurface_medium_overrides")


def test_media_subsurface_emission_never_sees_a_paramset():
    """Structural half — there is no ParamSet in scope to read or alias.

    Stronger than the syntactic gate above, which a rename (`q = params`) or a
    reader method missing from `_PARAMSET_METHODS` would slip past. `resolved` is
    a plain dict of already-interpreted values, so the only way back to a pbrt
    param would be a new argument — which this fails on.
    """
    from skinny.pbrt import media

    sig = inspect.signature(media.subsurface_overrides)
    assert list(sig.parameters) == ["resolved"], (
        "media.subsurface_overrides grew a parameter; if it is a ParamSet the "
        "second coefficient chain is back")
    tree = ast.parse(textwrap.dedent(inspect.getsource(media.subsurface_overrides)))
    assert not [n for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id in _PARAMSET_NAMES], (
        "media.subsurface_overrides names a ParamSet-shaped local")


def test_subsurface_eta_is_resolved_exactly_once_per_material(tmp_path):
    """The whole point of the change: ONE `eta` read reaches BOTH lanes.

    `pack_flat_material` takes the boundary IOR from the `ior` lane and never
    reads `subsurface_eta` (see `material_pack.pack_flat_material`), so the two
    must agree. Asserting the two resolved lobes is not enough — `media.py`
    assigns one from the other, so that pair cannot diverge by construction. The
    lanes that CAN diverge are the shader input and the `skinnyOverrides`, which
    used to come from two independent reads of the same param. This counts the
    reads.
    """
    from skinny.pbrt import materials as MM
    from skinny.pbrt.api import import_pbrt

    src = tmp_path / "s.pbrt"
    src.write_text('WorldBegin\nMaterial "subsurface" "spectrum eta" "glass-LASF9"\n'
                   'Shape "sphere" "float radius" 1\n')
    calls = []
    real = MM.get_float_texture

    def counting(params, name, default, **kw):
        if name == "eta":
            calls.append(name)
        return real(params, name, default, **kw)

    MM.get_float_texture = counting
    try:
        stage, _ = import_pbrt(str(src))
    finally:
        MM.get_float_texture = real
    assert len(calls) == 1, f"`eta` resolved {len(calls)} times, expected once"

    from pxr import UsdShade

    shader_ior = next(
        s.GetInput("ior").Get() for s in
        (UsdShade.Shader(p) for p in stage.TraverseAll())
        if s and s.GetInput("ior"))
    ovr = next(dict(p.GetCustomDataByKey("skinnyOverrides")) for p in stage.TraverseAll()
               if p.GetCustomDataByKey("skinnyOverrides")
               and "subsurface_eta" in p.GetCustomDataByKey("skinnyOverrides"))
    assert shader_ior == pytest.approx(1.85004, abs=1e-5)
    assert ovr["ior"] == pytest.approx(shader_ior)
    assert ovr["subsurface_eta"] == pytest.approx(shader_ior)


def test_subsurface_eta_and_ior_lobes_agree_for_every_param_type():
    for src, expected in (
        ('Material "subsurface" "float eta" 1.42', 1.42),
        ('Material "subsurface" "spectrum eta" "glass-LASF9"', 1.85004),
        ('Material "subsurface" "spectrum eta" "not-a-glass"', 1.33),
        ('Material "subsurface" "texture eta" "sometex"', 1.33),
    ):
        res = _res(src)
        assert res.lobes["ior"] == pytest.approx(expected, abs=1e-5), src
        assert res.lobes["subsurface_eta"] == pytest.approx(res.lobes["ior"]), src


# --------------------------------------------------------------------------- #
# the roughness calibration chain has ONE implementation
# (change coatedconductor-roughness-spelling)
# --------------------------------------------------------------------------- #

#: the calibration arithmetic: pbrt roughness -> alpha -> perceptual roughness.
_CALIBRATION = frozenset({"pbrt_roughness_to_alpha", "alpha_to_usd_roughness"})
#: the only functions allowed to call it — `_calibrate_roughness` (the single
#: owner of the arithmetic), the chain that uses it, and the two target reduction
#: policies, which convert an already-calibrated alpha pair. A material branch
#: appearing here means a second copy exists, which is the arrangement that
#: silently dropped `conductor.uroughness`/`vroughness`.
_CALIBRATION_OWNERS = frozenset({"_calibrate_roughness", "_resolve_roughness",
                                 "_usd_roughness", "_mtlx_roughness"})


def _calibration_callers(source: str) -> dict[str, set[str]]:
    """Map calibration function -> names of the functions that call it.

    Structural (AST), not a text scan: a grep for the identifier would also match
    a docstring or a comment naming it, and would miss a call made through a
    renamed import.
    """
    out: dict[str, set[str]] = {}

    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            name = (child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else enclosing)
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id in _CALIBRATION):
                out.setdefault(child.func.id, set()).add(name)
            walk(child, name)

    walk(ast.parse(source), "<module>")
    return out


def _calibration_offenders(source: str) -> set[str]:
    callers = _calibration_callers(source)
    # never report "clean" because nothing was found at all
    assert set(callers) == set(_CALIBRATION), f"calibration call sites missing: {callers}"
    return {n for names in callers.values() for n in names} - _CALIBRATION_OWNERS


def test_the_roughness_calibration_has_no_second_implementation():
    offenders = _calibration_offenders(inspect.getsource(M))
    assert not offenders, (
        f"the roughness calibration is re-implemented in {sorted(offenders)}; "
        "a material branch must reach it through _resolve_roughness(prefix=...)")


def test_the_calibration_gate_is_sensitive():
    # negative control: the exact source shape this change deleted from the
    # `coatedconductor` branch must still be caught.
    hand_rolled = textwrap.dedent("""
        def _resolve_roughness(params, notes, *, prefix=""):
            return alpha_to_usd_roughness(pbrt_roughness_to_alpha(r, remap))

        def resolve_material(p):
            if mtype == "coatedconductor":
                lobes["roughness"] = alpha_to_usd_roughness(
                    pbrt_roughness_to_alpha(rv.const, remap))
    """)
    assert _calibration_offenders(hand_rolled) == {"resolve_material"}


def test_the_read_gate_is_sensitive():
    # negative control: the resolver obviously does read params, so the detector
    # is not vacuously passing.
    reads = _reads_in(M.resolve_material)
    assert "get_float_texture" in reads and "get_spectrum_texture" in reads
    # `resolve_material`'s own body no longer holds a bare ParamSet method call —
    # deleting the hand-rolled `conductor.roughness` block took the last one
    # (change coatedconductor-roughness-spelling), so every read there now goes
    # through a promoting accessor or a helper. The bare-call half of the control
    # therefore points at the helpers that still make one, which is what proves
    # the AST detector can see that shape at all.
    assert any(r.startswith("params.") for r in _reads_in(M._resolve_roughness))
    assert any(r.startswith("params.") for r in _reads_in(M._conductor_basecolor))


# --------------------------------------------------------------------------- #
# promoting-accessor coverage (change subsurface-promoting-accessors)
#
# The promoting accessors exist so an unusable pbrt binding DEGRADES with a note
# instead of raising. Two things must hold for that to be true, and a test for
# each: no reader may call `float()` on a raw token (it raises, or worse parses
# garbage), and no degradation may be silent.
# --------------------------------------------------------------------------- #

#: `ParamSet` readers that call `float()` on the raw token values. On a texture
#: or a named spectrum they raise; on a `blackbody` or an inline sampled spectrum
#: they silently return the tokens themselves. Neither is a legal way to read a
#: material parameter value — that is what the promoting accessors are for.
_RAISING_PARAMSET_METHODS = frozenset({"rgb", "floats", "float", "int", "ints"})


def _raising_reads_in(func):
    """`params.<m>` calls on a ParamSet-shaped receiver, for the raising `m`."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RAISING_PARAMSET_METHODS
                and _is_paramset(node.func.value)):
            arg = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else "?"
            found.append(f"{arg}:{node.func.attr}@{node.lineno}")
    return found


def test_resolver_reads_no_parameter_value_through_a_raising_accessor():
    """The structural gate. It names no parameter, so it cannot rot.

    A behavioural sweep over "every param a type reads" has no machine-readable
    source — it is a hand-written list, so the NEXT param added with a raw
    accessor passes it. This fails the build instead, whichever branch and
    whichever name.
    """
    leaked = []
    for fn in (M.resolve_material, M.subsurface_medium_overrides,
               M._resolve_medium_colour, M.get_float_texture, M.get_spectrum_texture):
        leaked += _raising_reads_in(fn)
    assert not leaked, (
        "pbrt parameter values read through a float()-on-token accessor: "
        f"{leaked}. Use get_float_texture / get_spectrum_texture.")


def test_the_raising_read_detector_can_actually_fail():
    """Sensitivity control: a gate that cannot fail is decoration.

    Reconstructs the pre-change read and asserts the detector flags it, so the
    green above means "no raw read", not "detector matched nothing".
    """
    def _reintroduced(p):
        sigma_a = p.rgb("sigma_a", None)      # noqa: F841 - the read being detected
        g_f = p.floats("g", [0.0])            # noqa: F841
        return sigma_a, g_f

    found = _raising_reads_in(_reintroduced)
    assert [f.split("@")[0] for f in found] == ["sigma_a:rgb", "g:floats"], found


def test_named_spectrum_on_a_spectrum_lane_is_not_silently_dropped():
    """An unrecognised name must degrade WITH a note, as the float side does."""
    res = _res('Material "diffuse" "spectrum reflectance" "glass-BK7"')
    assert res.lobes["base_color"].const == [0.5, 0.5, 0.5]
    assert len(res.notes) == 1, res.notes
    assert "glass-BK7" in res.notes[0] and "reflectance" in res.notes[0]
    assert res.status == APPROX


def test_spectrum_file_on_a_spectrum_lane_is_reported():
    res = _res('Material "diffuse" "spectrum reflectance" "spd/foo.spd"')
    assert len(res.notes) == 1, res.notes
    assert "spd/foo.spd" in res.notes[0]
    assert res.status == APPROX


def test_named_metal_substitutes_on_a_reflectance_lane_without_a_note():
    """A recognised name on a lane where it means something is EXACT."""
    res = _res('Material "diffuse" "spectrum reflectance" "metal-Au-eta"')
    assert res.lobes["base_color"].const != [0.5, 0.5, 0.5]
    assert res.notes == []
    assert res.status == EXACT


def test_named_metal_does_not_substitute_into_a_coefficient_lane():
    """The spectrum-side mirror of `_IOR_PARAM_NAMES`.

    A metal's reflectance RGB is a reflectance. Writing it into an absorption
    coefficient is the same defect class as writing a glass IOR into a roughness.
    """
    res = _res('Material "subsurface" "spectrum sigma_a" "metal-Au-eta" '
               '"rgb sigma_s" [1 1 1]')
    gold = M.get_spectrum_texture(
        _mat('Material "diffuse" "spectrum reflectance" "metal-Au-eta"').params,
        "reflectance", [0.5, 0.5, 0.5]).const
    assert res.lobes["subsurface_sigma_a"] != pytest.approx(list(gold))
    assert any("sigma_a" in n for n in res.notes), res.notes


@pytest.mark.parametrize("binding,expected", [
    ('"blackbody sigma_a" [6500]', [1.042, 0.984, 1.035]),
    ('"spectrum sigma_a" [400 .1 700 .9]', [0.869, 0.461, 0.181]),
])
def test_legal_non_numeric_bindings_reduce_instead_of_yielding_tokens(binding, expected):
    """These parse through `ParamSet.rgb` without raising — into garbage.

    `"spectrum sigma_a" [400 .1 700 .9]` yields [400.0, 0.1, 700.0]: the raw
    wavelength/value tokens read as an RGB triple, no crash and no note. The
    corpus contains neither form, so its hash gate cannot see this.
    """
    res = _res(f'Material "subsurface" {binding} "rgb sigma_s" [1 1 1]')
    assert res.lobes["subsurface_sigma_a"] == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("prm", ["sigma_a", "sigma_s", "reflectance", "mfp", "g", "scale"])
@pytest.mark.parametrize("binding", ["texture", "spectrum"])
@pytest.mark.parametrize("flavor", [M.USD, M.MTLX])
def test_no_subsurface_binding_raises(prm, binding, flavor):
    val = '"sometex"' if binding == "texture" else '"glass-BK7"'
    res = _res(f'Material "subsurface" "{binding} {prm}" {val}', flavor)
    assert any(prm in n for n in res.notes), res.notes


def test_unusable_sigma_pair_keeps_the_explicit_sigma_branch():
    """Presence selects the branch; readability only affects the value.

    pbrt branches on `GetSpectrumTextureOrNull`, which is non-null for a texture
    binding too, so an unreadable sigma must not fall through to the reflectance
    inversion or the Wholemilk defaults — either would swap the physical model.
    """
    from skinny.pbrt.subsurface import subsurface_coefficients

    res = _res('Material "subsurface" "texture sigma_a" "t" "rgb sigma_s" [4 5 6] '
               '"rgb reflectance" [0.9 0.9 0.9]')
    # NOT the reflectance-inversion branch, which reflectance would have selected
    inversion = subsurface_coefficients(reflectance=[0.9, 0.9, 0.9])
    assert res.lobes["subsurface_sigma_s"] != pytest.approx(inversion["sigma_s"])


def test_half_unusable_sigma_pair_degrades_as_a_unit():
    """pbrt ErrorExits on a half-authored pair; skinny must not mix two materials.

    Substituting a default for sigma_a only would pair Wholemilk's absorption
    with the author's scattering — with a dense authored sigma_s the albedo
    approaches 1, the mean free path collapses, and the interior walk saturates.
    """
    from skinny.pbrt.subsurface import subsurface_coefficients

    res = _res('Material "subsurface" "texture sigma_a" "t" "rgb sigma_s" [100 100 100]')
    both = subsurface_coefficients()          # the default pair, degraded together
    assert res.lobes["subsurface_sigma_a"] == pytest.approx(both["sigma_a"])
    assert res.lobes["subsurface_sigma_s"] == pytest.approx(both["sigma_s"])
    assert any("sigma_a" in n for n in res.notes), res.notes


def test_absent_sigma_pair_still_falls_through():
    """The mirror: a promoting default must not make an absent param look present."""
    from skinny.pbrt.subsurface import subsurface_coefficients

    res = _res('Material "subsurface" "rgb reflectance" [0.9 0.9 0.9]')
    inversion = subsurface_coefficients(reflectance=[0.9, 0.9, 0.9])
    assert res.lobes["subsurface_sigma_a"] == pytest.approx(inversion["sigma_a"])

    plain = _res('Material "subsurface"')
    wholemilk = subsurface_coefficients()
    assert plain.lobes["subsurface_sigma_a"] == pytest.approx(wholemilk["sigma_a"])


def test_reflectance_is_resolved_once_for_both_consumers():
    """One read feeds the subsurface_color lobe and the coefficient chain."""
    calls = []
    real = M.get_spectrum_texture

    def counting(params, name, default, **kw):
        if name == "reflectance":
            calls.append(name)
        return real(params, name, default, **kw)

    M.get_spectrum_texture = counting
    try:
        res = _res('Material "subsurface" "rgb reflectance" [0.7 0.4 0.3]', M.MTLX)
    finally:
        M.get_spectrum_texture = real
    assert len(calls) == 1, f"reflectance resolved {len(calls)} times"
    assert res.lobes["subsurface_color"] == pytest.approx([0.7, 0.4, 0.3])


def test_subsurface_does_not_read_a_parameter_pbrt_never_defines():
    """pbrt's SubsurfaceMaterial::Create has no `radius`; only shapes do."""
    src = textwrap.dedent(inspect.getsource(M.resolve_material))
    assert '"radius"' not in src, "resolver reads a `radius` pbrt would ignore"


def test_a_degrading_named_spectrum_escalates_on_every_material_type():
    """The status escalation is NOT subsurface-only, by design.

    It keys off the `used default` marker that every degradation note carries,
    including `_named_spectrum_scalar`'s on the float side — which serves `eta`
    on every material type. So an unrecognised named eta on a `dielectric` now
    reports APPROX where it reported EXACT with a note, which was a substituted
    value wearing a clean status. Only `all_mtypes.pbrt` carries such a name, so
    no corpus scene's report moves.
    """
    res = _res('Material "dielectric" "spectrum eta" "glass-ZZ9"')
    assert res.lobes["ior"] == pytest.approx(1.5)
    assert res.notes == ["named spectrum 'glass-ZZ9' on eta unrecognised; used default 1.5"]
    assert res.status == APPROX

    # a RECOGNISED name is an exact substitution and must NOT escalate
    exact = _res('Material "dielectric" "spectrum eta" "glass-BK7"')
    assert exact.notes == [] and exact.status == EXACT
