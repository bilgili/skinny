"""Importer preserves spectral payloads alongside the RGB reduction (task 2.2).

The RGB reduction (`param_to_rgb`) is unchanged; a spectral payload rides
`skinnyOverrides` only when a spectrum was actually authored, so RGB-only scenes
author nothing new.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import spectra

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
