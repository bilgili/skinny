"""Import-side handling of pbrt named spectra (change `pbrt-named-spectra`).

pbrt addresses ~30 built-in spectra by name (`"spectrum eta" "glass-BK7"`,
`"spectrum L" "stdillum-A"`). These cover the import resolution of those names:
the scalar/RGB values they reduce to, the identity preserved for spectral mode,
and — the point of the change — that an *unrecognised* name is reported rather
than silently substituted.

Hostless: parser + importer only, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import materials as M
from skinny.pbrt import spectra
from skinny.pbrt.parser import Param, ParamSet


def _ps(**kw) -> ParamSet:
    return ParamSet(kw)


def _spectrum(name: str, value: str) -> Param:
    return Param("spectrum", name, [value])


class _Mat:
    """Minimal stand-in for a parsed pbrt material."""

    def __init__(self, mtype: str, **params):
        self.type = mtype
        self.params = _ps(**params)


# --- named glasses ----------------------------------------------------------


def test_named_glass_resolves_to_its_own_d_line_ior():
    # The headline bug: every named glass rendered at the generic 1.5 default, so
    # LASF9 (n=1.850) was indistinguishable from a plain crown.
    pv = M.get_float_texture(_ps(eta=_spectrum("eta", "glass-LASF9")), "eta", 1.5)
    assert pv.const == pytest.approx(1.85004, abs=1e-5)


def test_named_glass_bk7_resolves_to_its_own_ior():
    pv = M.get_float_texture(_ps(eta=_spectrum("eta", "glass-BK7")), "eta", 1.5)
    assert pv.const == pytest.approx(1.51673, abs=1e-5)


def test_recognised_glass_reports_nothing():
    # A recognised glass's d-line index is exact, not a fallback — no note.
    notes: list[str] = []
    M.get_float_texture(_ps(eta=_spectrum("eta", "glass-BK7")), "eta", 1.5, notes=notes)
    assert notes == []


def test_scalar_eta_dielectric_is_untouched():
    pv = M.get_float_texture(_ps(eta=Param("float", "eta", [1.33])), "eta", 1.5)
    assert pv.const == pytest.approx(1.33)


def test_named_glass_rides_its_own_dispersion_key():
    ov = M.material_spectral_overrides(_Mat("dielectric", eta=_spectrum("eta", "glass-LASF9")))
    assert ov["glass_dispersion"] == "lasf9"


def test_bk7_dispersion_key_unchanged():
    ov = M.material_spectral_overrides(_Mat("dielectric", eta=_spectrum("eta", "glass-BK7")))
    assert ov["glass_dispersion"] == "bk7"


def test_scalar_dielectric_authors_no_override():
    ov = M.material_spectral_overrides(_Mat("dielectric", eta=Param("float", "eta", [1.5])))
    assert ov == {}


# --- named metals -----------------------------------------------------------


@pytest.mark.parametrize("name,key", [
    ("metal-CuZn-eta", "cuzn"), ("metal-MgO-eta", "mgo"), ("metal-TiO2-eta", "tio2"),
    ("metal-Au-eta", "au"),
])
def test_named_conductor_key_covers_extended_metals(name, key):
    ov = M.material_spectral_overrides(_Mat("conductor", eta=_spectrum("eta", name)))
    assert ov["conductor_metal"] == key


def test_coatedconductor_basecolor_reads_conductor_eta():
    # `coatedconductor` spells it `conductor.eta`. Reading only `eta` gave an RGB
    # copper base colour while the spectral override named the real metal — the
    # two modes rendered different materials.
    mat = _Mat("coatedconductor", **{"conductor.eta": _spectrum("conductor.eta", "metal-CuZn-eta")})
    notes: list[str] = []
    rgb = M._conductor_basecolor(mat.params, notes)
    copper = list(spectra.named_metal_reflectance_rgb("copper"))
    assert rgb != copper
    assert rgb == pytest.approx(list(spectra.named_metal_reflectance_rgb("metal-CuZn-eta")))
    assert notes == []


def test_coatedconductor_rgb_and_spectral_agree_on_the_metal():
    mat = _Mat("coatedconductor", **{"conductor.eta": _spectrum("conductor.eta", "metal-CuZn-eta")})
    assert M.material_spectral_overrides(mat)["conductor_metal"] == "cuzn"
    assert M._conductor_basecolor(mat.params, []) == pytest.approx(
        list(spectra.named_metal_reflectance_rgb("cuzn")))


# --- named illuminants ------------------------------------------------------


def test_stdillum_a_is_warm_not_white():
    rgb = spectra.param_to_rgb(_spectrum("L", "stdillum-A"), illuminant=True)
    assert rgb[0] > rgb[2]


def test_named_illuminant_magnitude_is_unaffected_by_the_name():
    # Pre-change the name fell through to the caller's [1,1,1] (luminance 1);
    # post-change it is a unit-luminance chromaticity. Magnitude still comes from
    # the light's own L/scale.
    rgb = np.asarray(spectra.param_to_rgb(_spectrum("L", "stdillum-A"), illuminant=True))
    lum = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    assert lum == pytest.approx(1.0, abs=1e-3)


def test_named_illuminant_is_gated_on_illuminant_mode():
    # A medium's `"spectrum sigma_a"` resolves in reflectance mode and must never
    # come back as an illuminant chromaticity.
    assert spectra.param_to_rgb(_spectrum("sigma_a", "stdillum-A")) is None


# --- illuminant projection continuity (design D8) ---------------------------


def test_colored_illuminant_no_longer_jumps_two_orders_of_magnitude():
    # The illuminant branch used to skip the CMF-integral division, so a 1e-6
    # nudge to a constant illuminant jumped it ~107x ([10,10,10] -> [1283,...]).
    const = spectra.sampled_spectrum_to_rgb([400, 10, 700, 10], illuminant=True)
    near = spectra.sampled_spectrum_to_rgb([400, 10, 700, 10.000001], illuminant=True)
    assert np.allclose(const, [10.0, 10.0, 10.0])
    # What remains is the equal-energy-whitepoint tint (~20%), not a 107x jump:
    # an equal-energy SPD is chromaticity (1/3,1/3), not sRGB's D65 white, so the
    # projection tints it. That tint is inherent to the projection and is exactly
    # what the constant shortcut exists to sidestep.
    assert np.max(np.abs(near - const)) < 0.25 * 10.0


def test_illuminant_and_reflectance_branches_agree_up_to_scale():
    # The real invariant behind the fix: both branches divide by the CMF integral,
    # so they are the same projection and differ only by the authored magnitude.
    illum = spectra.sampled_spectrum_to_rgb([400, 10, 700, 10.000001], illuminant=True)
    refl = spectra.sampled_spectrum_to_rgb([400, 0.2, 700, 0.2000001])
    assert np.allclose(illum, refl * (10.0 / 0.2), rtol=1e-5)


def test_colored_reflectance_projection_unchanged():
    # The reflectance branch always divided; these values are the pre-change ones.
    refl = spectra.sampled_spectrum_to_rgb([400, 0.2, 700, 0.2000001])
    assert refl == pytest.approx([0.24004638, 0.18994093, 0.1816701], abs=1e-6)


# --- unknown names are reported, not silently substituted -------------------


def test_unknown_glass_reports_its_fallback():
    notes: list[str] = []
    pv = M.get_float_texture(_ps(eta=_spectrum("eta", "glass-NOSUCH")), "eta", 1.5, notes=notes)
    assert pv.const == pytest.approx(1.5)
    assert len(notes) == 1
    assert "glass-NOSUCH" in notes[0] and "unrecognised" in notes[0]


def test_unknown_metal_reports_its_copper_fallback():
    notes: list[str] = []
    M._conductor_basecolor(_ps(eta=_spectrum("eta", "metal-NOSUCH-eta")), notes)
    assert len(notes) == 1
    assert "metal-NOSUCH-eta" in notes[0] and "copper" in notes[0]


def test_spectrum_file_reference_is_reported_as_a_file():
    # pbrt reads a file when a name misses its table; skinny has no reader, so the
    # note must not call a legitimate .spd path an "unknown glass".
    notes: list[str] = []
    M.get_float_texture(_ps(eta=_spectrum("eta", "spds/custom.spd")), "eta", 1.5, notes=notes)
    assert len(notes) == 1
    assert "spectrum file" in notes[0] and "not read" in notes[0]


def test_looks_like_spectrum_file_discriminates():
    assert spectra.looks_like_spectrum_file("spds/glass.spd")
    assert spectra.looks_like_spectrum_file("my.spd")
    assert not spectra.looks_like_spectrum_file("glass-BK7")
    assert not spectra.looks_like_spectrum_file("metal-Au-eta")


# --- inline material spectra ------------------------------------------------


def test_inline_material_spectrum_is_preserved():
    mat = _Mat("diffuse", reflectance=Param("spectrum", "reflectance", [400, 0.1, 700, 0.9]))
    payload = M.material_spectral_overrides(mat)["spectral"]
    assert payload["kind"] == "spectrum_samples"
    assert len(payload["values"]) == 95


def test_constant_inline_spectrum_authors_nothing():
    # A constant spectrum is identical to its RGB reduction; preserving it would
    # author an override for a scene that needs none.
    mat = _Mat("diffuse", reflectance=Param("spectrum", "reflectance", [400, 0.2, 700, 0.2]))
    assert M.material_spectral_overrides(mat) == {}


def test_rgb_only_material_authors_no_override():
    mat = _Mat("diffuse", reflectance=Param("rgb", "reflectance", [0.2, 0.4, 0.6]))
    assert M.material_spectral_overrides(mat) == {}


# --- metal id <-> upload order invariant (design D7) ------------------------


def _renderer():
    # `skinny.renderer` imports `vulkan` at module load, which needs VULKAN_SDK +
    # DYLD_LIBRARY_PATH on the library path (see CLAUDE.md) — absent under a plain
    # `.venv/bin/pytest`. That surfaces as OSError("Cannot find Vulkan SDK version"),
    # NOT ImportError, so pytest.importorskip does not catch it.
    try:
        import skinny.renderer as r
    except (ImportError, OSError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"skinny.renderer unavailable (needs VULKAN_SDK on the lib path): {exc}")
    return r


def test_conductor_metal_ids_are_append_only():
    # An id is a byte offset into the spectralMetals upload ((id-1)*stride), so
    # renumbering an existing metal silently swaps materials in shipped scenes.
    r = _renderer()
    assert r._CONDUCTOR_METAL_ID["au"] == 1
    assert r._CONDUCTOR_METAL_ID["ag"] == 2
    assert r._CONDUCTOR_METAL_ID["al"] == 3
    assert r._CONDUCTOR_METAL_ID["cu"] == 4
    # The upload order is derived from the id map, so this can only fail if
    # someone re-hardcodes it (design D7).
    for i, name in enumerate(r._SPECTRAL_METAL_ORDER):
        assert r._CONDUCTOR_METAL_ID[name] == i + 1


def test_every_conductor_key_has_a_metal_id():
    # A key the importer can emit but the renderer can't pack would silently
    # drop to the RGB upsample.
    r = _renderer()
    for key in spectra._CONDUCTOR_CANON:
        assert key in r._CONDUCTOR_METAL_ID
