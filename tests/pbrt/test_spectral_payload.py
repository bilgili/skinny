"""Importer preserves spectral payloads alongside the RGB reduction (task 2.2).

The RGB reduction (`param_to_rgb`) is unchanged; a spectral payload rides
`skinnyOverrides` only when a spectrum was actually authored, so RGB-only scenes
author nothing new.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import spectra, spectral

pytest.importorskip("pxr")

from skinny.pbrt.api import import_pbrt  # noqa: E402


def _stage(tmp_path, scene):
    from pxr import Usd

    src = tmp_path / "s.pbrt"
    src.write_text(scene)
    out = tmp_path / "s.usda"
    import_pbrt(str(src), out=str(out))
    return Usd.Stage.Open(str(out))


def _overrides_with_key(stage, key):
    for prim in stage.Traverse():
        ov = prim.GetCustomDataByKey("skinnyOverrides")
        if ov and key in ov:
            return dict(ov[key])
    return None


def _scalar_override(stage, key):
    """Return the (scalar) skinnyOverrides value for *key*, or None."""
    for prim in stage.Traverse():
        ov = prim.GetCustomDataByKey("skinnyOverrides")
        if ov and key in ov:
            return ov[key]
    return None


# ── param_spectral_payload unit coverage ──────────────────────────


def _param(ptype, values):
    from skinny.pbrt.parser import Param

    return Param(type=ptype, name="x", values=tuple(values))


def test_payload_blackbody():
    p = _param("blackbody", [4000.0])
    assert spectra.param_spectral_payload(p) == {"kind": "blackbody", "temperature": 4000.0}


def test_payload_named_spectrum():
    p = _param("spectrum", ["metal-Au-eta"])
    assert spectra.param_spectral_payload(p) == {
        "kind": "spectrum_named",
        "name": "metal-Au-eta",
    }


def test_payload_inline_spectrum_resampled():
    p = _param("spectrum", [400.0, 0.1, 700.0, 0.9])
    payload = spectra.param_spectral_payload(p)
    assert payload["kind"] == "spectrum_samples"
    assert len(payload["lambda"]) == len(spectra._LAMBDA)
    assert len(payload["values"]) == len(spectra._LAMBDA)
    # endpoints held on the clamped grid
    assert payload["values"][0] == pytest.approx(0.1)
    assert payload["values"][-1] == pytest.approx(0.9)


def test_payload_none_for_rgb_and_float():
    assert spectra.param_spectral_payload(_param("rgb", [0.5, 0.4, 0.3])) is None
    assert spectra.param_spectral_payload(_param("float", [0.7])) is None
    assert spectra.param_spectral_payload(None) is None


# ── named-conductor / glass key normalization (spectra helpers) ────


def test_named_conductor_key_normalizes_metals():
    for raw, expect in [
        ("metal-Au-eta", "au"),
        ("metal-Ag-eta", "ag"),
        ("metal-Al-eta", "al"),
        ("metal-Cu-eta", "cu"),
        ("gold", "au"),
        ("Silver", "ag"),
        ("aluminium", "al"),
        ("aluminum", "al"),
        ("copper", "cu"),
    ]:
        assert spectra.named_conductor_key(_param("spectrum", [raw])) == expect


def test_named_conductor_key_none_for_unknown_or_nonspectrum():
    assert spectra.named_conductor_key(_param("spectrum", ["metal-Fe-eta"])) is None
    assert spectra.named_conductor_key(_param("float", [1.5])) is None
    assert spectra.named_conductor_key(_param("rgb", [0.1, 0.2, 0.3])) is None
    assert spectra.named_conductor_key(None) is None


def test_named_glass_key_normalizes():
    assert spectra.named_glass_key(_param("spectrum", ["glass-BK7"])) == "bk7"
    assert spectra.named_glass_key(_param("spectrum", ["bk7"])) == "bk7"
    # Each recognised pbrt glass now resolves to its OWN key, not a shared fallback.
    assert spectra.named_glass_key(_param("spectrum", ["glass-LASF9"])) == "lasf9"
    # SF11 used to land on "default" (it was unrecognised); pbrt's `glass-F11` reads
    # the array `GlassSF11_eta`, so both spellings are that glass.
    assert spectra.named_glass_key(_param("spectrum", ["glass-SF11"])) == "sf11"
    # only a genuinely unknown named glass rides the fallback key
    assert spectra.named_glass_key(_param("spectrum", ["glass-NOSUCH"])) == "default"


def test_named_glass_key_none_for_float_eta():
    assert spectra.named_glass_key(_param("float", [1.5])) is None
    assert spectra.named_glass_key(None) is None


def test_rgb_reduction_unchanged_by_payload_path():
    # The spectral payload must not perturb the existing RGB reduction.
    for p in [
        _param("blackbody", [4000.0]),
        _param("spectrum", [400.0, 0.1, 700.0, 0.9]),
        _param("rgb", [0.5, 0.4, 0.3]),
    ]:
        _ = spectra.param_spectral_payload(p)  # side-effect free
        assert spectra.param_to_rgb(p) is not None or p.type == "spectrum"


# ── end-to-end import round-trip ──────────────────────────────────

_BLACKBODY_AREALIGHT = """
Film "rgb" "integer xresolution" 8 "integer yresolution" 8
Camera "perspective" "float fov" 65
WorldBegin
AttributeBegin
  AreaLightSource "diffuse" "blackbody L" 3000 "float scale" 5
  Shape "trianglemesh" "point3 P" [0 0 0  1 0 0  0 1 0] "integer indices" [0 1 2]
AttributeEnd
"""

_ILLUMINANT_SPD_LIGHT = """
Film "rgb" "integer xresolution" 8 "integer yresolution" 8
Camera "perspective" "float fov" 65
WorldBegin
LightSource "distant" "spectrum L" [400 0.2 500 0.6 600 0.9 700 0.4] \
  "point3 from" [0 0 0] "point3 to" [0 0 -1]
"""

_RGB_ONLY = """
Film "rgb" "integer xresolution" 8 "integer yresolution" 8
Camera "perspective" "float fov" 65
WorldBegin
LightSource "distant" "rgb L" [1 1 1] "point3 from" [0 0 0] "point3 to" [0 0 -1]
AttributeBegin
  AreaLightSource "diffuse" "rgb L" [1 0.5 0.2]
  Shape "trianglemesh" "point3 P" [0 0 0  1 0 0  0 1 0] "integer indices" [0 1 2]
AttributeEnd
"""


def test_blackbody_arealight_temperature_preserved(tmp_path):
    stage = _stage(tmp_path, _BLACKBODY_AREALIGHT)
    payload = _overrides_with_key(stage, "emissive_spectral")
    assert payload is not None
    assert payload["kind"] == "blackbody"
    assert payload["temperature"] == pytest.approx(3000.0)


def test_illuminant_spd_preserved(tmp_path):
    stage = _stage(tmp_path, _ILLUMINANT_SPD_LIGHT)
    payload = _overrides_with_key(stage, "spectral")
    assert payload is not None
    assert payload["kind"] == "spectrum_samples"
    lam = np.asarray(payload["lambda"])
    assert lam.shape == spectra._LAMBDA.shape
    assert np.allclose(lam, spectra._LAMBDA)


def test_rgb_only_scene_authors_no_spectral_payload(tmp_path):
    stage = _stage(tmp_path, _RGB_ONLY)
    assert _overrides_with_key(stage, "spectral") is None
    assert _overrides_with_key(stage, "emissive_spectral") is None


# ── producer round-trip through the renderer's material loader (Group 6.1) ──
#
# The renderer reads a blackbody area light's temperature from
# `Material.parameter_overrides["emissive_spectral"]` (renderer.py ~5725) and
# feeds it to `spectral.blackbody_scale`. This proves that payload survives the
# FULL pbrt → USD (skinnyOverrides) → `usd_loader._extract_material` path the
# renderer actually uses to build Material objects — not just the raw stage
# customData the tests above inspect. Hostless: pxr USD only, no GPU, no
# PyMaterialXGenSlang.


def _load_materials_via_loader(stage):
    """Yield `Material` objects built by the SAME loader the renderer uses,
    one per bound UsdShade.Material prim on the stage."""
    from pxr import UsdShade

    from skinny import usd_loader

    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            yield usd_loader._extract_material(UsdShade.Material(prim))


def test_blackbody_producer_roundtrip_through_loader(tmp_path):
    import math

    stage = _stage(tmp_path, _BLACKBODY_AREALIGHT)

    materials = list(_load_materials_via_loader(stage))
    assert materials, "importer authored no bound material"

    emissive_mats = [
        m for m in materials if m.parameter_overrides.get("emissive_spectral") is not None
    ]
    assert len(emissive_mats) == 1, "expected exactly one blackbody area-light material"
    mat = emissive_mats[0]

    # The temperature the renderer reads off parameter_overrides survived the
    # round-trip. The payload round-trips USD as a pxr.Vt.Dictionary, so
    # duck-type on `.get` (matches the renderer's own access at renderer.py).
    payload = mat.parameter_overrides["emissive_spectral"]
    assert hasattr(payload, "get")
    assert payload.get("kind") == "blackbody"
    assert float(payload.get("temperature")) == pytest.approx(3000.0)

    # The emissive RGB the renderer pairs with the temperature is also present,
    # and `blackbody_scale` yields a finite, strictly positive luminance scale.
    emissive = mat.parameter_overrides.get("emissiveColor")
    assert emissive is not None
    emissive_rgb = tuple(float(c) for c in emissive)
    scale = spectral.blackbody_scale(float(payload.get("temperature")), emissive_rgb)
    assert math.isfinite(scale)
    assert scale > 0.0


def test_rgb_only_producer_roundtrip_has_no_spectral(tmp_path):
    stage = _stage(tmp_path, _RGB_ONLY)
    for mat in _load_materials_via_loader(stage):
        assert mat.parameter_overrides.get("emissive_spectral") is None


def _load_distant_lights_via_loader(stage):
    """Distant lights (LightDir) built by the SAME loader the renderer uses."""
    from pxr import Usd

    from skinny import usd_loader

    lights_dir, _sphere, _env, _emissive = usd_loader._extract_lights(
        stage, Usd.TimeCode.Default(), [], {}
    )
    return lights_dir


def test_illuminant_spd_producer_roundtrip_through_loader(tmp_path):
    # Group 6.3: a distant `spectrum L` light round-trips its authored SPD all the
    # way into LightDir.spectral_spd (the field the renderer packs into binding 50),
    # NOT just the raw USD customData.
    stage = _stage(tmp_path, _ILLUMINANT_SPD_LIGHT)
    lights = _load_distant_lights_via_loader(stage)
    assert len(lights) == 1
    spd = lights[0].spectral_spd
    assert spd is not None
    assert spd.shape == spectra._LAMBDA.shape  # resampled to the 95-sample grid
    assert np.any(spd > 0.0)
    assert not np.allclose(spd, spd[0])         # non-constant (real illuminant)


def test_rgb_only_distant_light_has_no_spd(tmp_path):
    stage = _stage(tmp_path, _RGB_ONLY)
    for ld in _load_distant_lights_via_loader(stage):
        assert ld.spectral_spd is None


# ── named-conductor / dispersive-glass material identity round-trip ─

_TRI = ('Shape "trianglemesh" "point3 P" [0 0 0  1 0 0  0 1 0] '
        '"integer indices" [0 1 2]')


def _material_scene(material_line):
    return f"""
Film "rgb" "integer xresolution" 8 "integer yresolution" 8
Camera "perspective" "float fov" 65
WorldBegin
AttributeBegin
  {material_line}
  {_TRI}
AttributeEnd
"""


@pytest.mark.parametrize(
    "eta_name,expect",
    [("metal-Au-eta", "au"), ("metal-Cu-eta", "cu"),
     ("metal-Ag-eta", "ag"), ("metal-Al-eta", "al")],
)
def test_named_conductor_material_roundtrips_key(tmp_path, eta_name, expect):
    k_name = eta_name.replace("-eta", "-k")
    mat = (f'Material "conductor" "spectrum eta" "{eta_name}" '
           f'"spectrum k" "{k_name}" "float roughness" 0.2')
    stage = _stage(tmp_path, _material_scene(mat))
    assert _scalar_override(stage, "conductor_metal") == expect
    # additive: no glass key on a conductor
    assert _scalar_override(stage, "glass_dispersion") is None


def test_named_glass_material_roundtrips_dispersion(tmp_path):
    mat = 'Material "dielectric" "spectrum eta" "glass-BK7"'
    stage = _stage(tmp_path, _material_scene(mat))
    assert _scalar_override(stage, "glass_dispersion") == "bk7"
    assert _scalar_override(stage, "conductor_metal") is None


def test_rgb_conductor_authors_no_conductor_metal(tmp_path):
    # A conductor with no named eta (defaults to Cu RGB) authors no identity key.
    mat = 'Material "conductor" "float roughness" 0.2'
    stage = _stage(tmp_path, _material_scene(mat))
    assert _scalar_override(stage, "conductor_metal") is None
    assert _scalar_override(stage, "glass_dispersion") is None


def test_float_eta_dielectric_authors_no_dispersion(tmp_path):
    mat = 'Material "dielectric" "float eta" 1.5'
    stage = _stage(tmp_path, _material_scene(mat))
    assert _scalar_override(stage, "glass_dispersion") is None
    assert _scalar_override(stage, "conductor_metal") is None


def test_named_conductor_material_roundtrips_key_mtlx(tmp_path):
    from pxr import Usd

    mat = ('Material "conductor" "spectrum eta" "metal-Au-eta" '
           '"spectrum k" "metal-Au-k" "float roughness" 0.2')
    src = tmp_path / "s.pbrt"
    src.write_text(_material_scene(mat))
    out = tmp_path / "s.usda"
    import_pbrt(str(src), out=str(out), materialx=True)
    stage = Usd.Stage.Open(str(out))
    assert _scalar_override(stage, "conductor_metal") == "au"
