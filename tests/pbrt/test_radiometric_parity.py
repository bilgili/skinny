"""pbrt radiometric parity — live film params, .hdr scale, absolute gate.

Change: pbrt-radiometric-parity. Hostless (USD + pbrt parser + numpy only); the
GPU effect of the imaging ratio / light-type offset is gated by the parity matrix.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

from skinny.pbrt import metrics, parity  # noqa: E402
from skinny.pbrt.api import import_pbrt  # noqa: E402
from skinny.usd_loader import _extract_camera  # noqa: E402

_CAM = (
    'Camera "perspective" "float fov" 50 '
    '"float shutteropen" 0 "float shutterclose" {sc}\n'
)
_AREA_WORLD = (
    "WorldBegin\nAttributeBegin\n"
    'AreaLightSource "diffuse" "rgb L" [1 1 1]\n'
    'Shape "sphere" "float radius" 1\n'
    "AttributeEnd\n"
)


def _import(tmp_path, pbrt_text: str) -> Usd.Stage:
    src = tmp_path / "s.pbrt"
    src.write_text(pbrt_text)
    out = tmp_path / "s.usda"
    import_pbrt(str(src), out=str(out))
    return Usd.Stage.Open(str(out))


# ── 1. Film params authored on the camera (live, not baked) ──────────────────


def test_camera_authors_film_attrs(tmp_path):
    """ISO + exposure time land on the camera prim; standard `exposure` is the
    log2 of the imaging ratio (exposure_time·iso/100)."""
    stage = _import(
        tmp_path,
        _CAM.format(sc=2)
        + 'Film "rgb" "float iso" 400 "integer xresolution" 64 '
        '"integer yresolution" 64\n'
        + _AREA_WORLD,
    )
    cam = stage.GetPrimAtPath("/World/Camera")
    assert cam and cam.IsValid()
    assert cam.GetAttribute("skinny:film:iso").Get() == pytest.approx(400.0)
    assert cam.GetAttribute("skinny:film:exposureTime").Get() == pytest.approx(2.0)
    # imaging ratio = 2 * 400 / 100 = 8 ⇒ exposure stops = log2(8) = 3
    exp = UsdGeom.Camera(cam).GetExposureAttr().Get()
    assert exp == pytest.approx(math.log2(8.0))


def test_default_film_attrs(tmp_path):
    """A scene with no iso / shutter authors the neutral 100 / 1.0 (ratio 1)."""
    stage = _import(tmp_path, _CAM.format(sc=1) + _AREA_WORLD)
    cam = stage.GetPrimAtPath("/World/Camera")
    assert cam.GetAttribute("skinny:film:iso").Get() == pytest.approx(100.0)
    assert cam.GetAttribute("skinny:film:exposureTime").Get() == pytest.approx(1.0)
    assert UsdGeom.Camera(cam).GetExposureAttr().Get() == pytest.approx(0.0)


def _emissive_luma(stage) -> float:
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/shape_0_mat/Surface"))
    val = shader.GetInput("emissiveColor").Get()
    rgb = np.asarray([val[0], val[1], val[2]], np.float64)
    return float(rgb @ np.array([0.2126, 0.7152, 0.0722]))


def test_imaging_ratio_not_baked_into_emitter(tmp_path):
    """A large imaging ratio (iso 400, exposure 1 ⇒ ratio 4) is NOT baked into the
    area-light emission — the authored radiance stays at the raw pbrt L."""
    stage = _import(
        tmp_path,
        _CAM.format(sc=1)
        + 'Film "rgb" "float iso" 400 "integer xresolution" 64 '
        '"integer yresolution" 64\n'
        + _AREA_WORLD,
    )
    # L = [1,1,1] illuminant ⇒ luma ~1.0; if the ratio were baked it would be ~4.
    assert _emissive_luma(stage) == pytest.approx(1.0, abs=0.05)


def test_emitter_identical_across_iso(tmp_path):
    """The authored emission is independent of the film ISO (proves no baking)."""
    base = _CAM.format(sc=1) + _AREA_WORLD
    s100 = _import(
        tmp_path / "a" if False else tmp_path,
        _CAM.format(sc=1)
        + 'Film "rgb" "float iso" 100 "integer xresolution" 64 "integer yresolution" 64\n'
        + _AREA_WORLD,
    )
    luma100 = _emissive_luma(s100)
    tp2 = tmp_path / "b"
    tp2.mkdir()
    s400 = _import(
        tp2,
        _CAM.format(sc=1)
        + 'Film "rgb" "float iso" 400 "integer xresolution" 64 "integer yresolution" 64\n'
        + _AREA_WORLD,
    )
    assert _emissive_luma(s400) == pytest.approx(luma100)
    assert base  # silence flake8 on the unused local


# ── 2. Loader reads film params back into the CameraOverride ─────────────────


def _camera_stage(film_attrs: dict | None = None, exposure_stops=None):
    stage = Usd.Stage.CreateInMemory()
    cam = UsdGeom.Camera.Define(stage, "/Camera")
    cam.CreateFocalLengthAttr(50.0)
    cam.CreateVerticalApertureAttr(24.0)
    prim = cam.GetPrim()
    for k, v in (film_attrs or {}).items():
        prim.CreateAttribute(k, Sdf.ValueTypeNames.Float).Set(float(v))
    if exposure_stops is not None:
        cam.CreateExposureAttr(float(exposure_stops))
    return stage


def test_extract_camera_reads_film():
    stage = _camera_stage(
        {"skinny:film:iso": 800.0, "skinny:film:exposureTime": 0.5}
    )
    ov = _extract_camera(stage, Usd.TimeCode.Default())
    assert ov is not None
    assert ov.iso == pytest.approx(800.0)
    assert ov.exposure_time == pytest.approx(0.5)


def test_extract_camera_exposure_fallback_stops_to_seconds():
    """No skinny:film:exposureTime, only the standard `exposure` (stops): the
    loader folds 2^stops into exposure_time (iso default 100)."""
    stage = _camera_stage({"skinny:film:iso": 100.0}, exposure_stops=1.0)
    ov = _extract_camera(stage, Usd.TimeCode.Default())
    assert ov.exposure_time == pytest.approx(2.0)  # 2^1
    assert ov.iso == pytest.approx(100.0)


def test_extract_camera_defaults():
    ov = _extract_camera(_camera_stage(), Usd.TimeCode.Default())
    assert ov.iso == pytest.approx(100.0)
    assert ov.exposure_time == pytest.approx(1.0)


def test_round_trip_imaging_ratio(tmp_path):
    """import → load: ISO 200 with a 2 s shutter ⇒ imaging ratio 2·200/100 = 4."""
    stage = _import(
        tmp_path,
        _CAM.format(sc=2)
        + 'Film "rgb" "float iso" 200 "integer xresolution" 64 "integer yresolution" 64\n'
        + _AREA_WORLD,
    )
    ov = _extract_camera(stage, Usd.TimeCode.Default())
    ratio = ov.exposure_time * ov.iso / 100.0
    assert ratio == pytest.approx(4.0)


# ── 3. .hdr-direct env carries its pbrt `scale` on the DomeLight ─────────────


def _dome_intensity(stage) -> float:
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            attr = UsdLux.DomeLight(prim).GetIntensityAttr()
            return float(attr.Get()) if attr and attr.HasAuthoredValue() else 1.0
    raise AssertionError("no DomeLight in stage")


def test_hdr_direct_scale_to_dome_intensity(tmp_path):
    """A pbrt `infinite` light pointing at a .hdr with `scale 2` carries that
    scale on the DomeLight intensity (it used to be silently dropped)."""
    stage = _import(
        tmp_path,
        _CAM.format(sc=1)
        + 'WorldBegin\n'
        'LightSource "infinite" "string filename" "sky.hdr" "float scale" 2\n'
        'Shape "sphere" "float radius" 1\n',
    )
    assert _dome_intensity(stage) == pytest.approx(2.0)


def test_hdr_direct_unit_scale_default(tmp_path):
    """No `scale` ⇒ intensity 1.0 (byte-equivalent to the old behaviour)."""
    stage = _import(
        tmp_path,
        _CAM.format(sc=1)
        + 'WorldBegin\n'
        'LightSource "infinite" "string filename" "sky.hdr"\n'
        'Shape "sphere" "float radius" 1\n',
    )
    assert _dome_intensity(stage) == pytest.approx(1.0)


# ── 4. Absolute-radiance gate + mean-ratio metric ────────────────────────────


def _flat(val: float, shape=(8, 8, 3)) -> np.ndarray:
    return np.full(shape, val, np.float64)


def test_mean_ratio():
    assert metrics.mean_ratio(_flat(2.0), _flat(1.0)) == pytest.approx(2.0)
    assert metrics.mean_ratio(_flat(1.0), _flat(1.0)) == pytest.approx(1.0)
    assert math.isinf(metrics.mean_ratio(_flat(1.0), _flat(0.0)))


def _spec(**kw):
    base = dict(name="t", file="t.pbrt", ref="t.exr", width=8, height=8, spp=1,
                relmse_tol=0.04, flip_tol=0.04)
    base.update(kw)
    return parity.SceneSpec(**base)


def _combo():
    return parity.RenderCombo("path", "wavefront")


def test_absolute_gate_optout_returns_none():
    """No `absolute` config ⇒ the gate is skipped (returns None)."""
    r = parity.absolute_radiance_result(_spec(), _combo(), _flat(2.0), _flat(1.0))
    assert r is None


def test_absolute_gate_fails_on_brightness_drift():
    """A 2× global brightness offset fails the absolute gate even though the
    exposure-blind gate would align it away."""
    spec = _spec(absolute={"mean_ratio_tol": 0.1, "relmse_tol": 0.05})
    r = parity.absolute_radiance_result(spec, _combo(), _flat(2.0), _flat(1.0))
    assert r is not None and not r.passed
    assert r.flip == pytest.approx(2.0)  # mean ratio in the reused slot


def test_absolute_gate_passes_when_matched():
    spec = _spec(absolute={"mean_ratio_tol": 0.1, "relmse_tol": 0.05})
    r = parity.absolute_radiance_result(spec, _combo(), _flat(1.02), _flat(1.0))
    assert r is not None and r.passed


def test_absolute_gate_baseline_centers_window():
    """A recorded baseline offset relaxes the window around the known ratio."""
    spec = _spec(absolute={
        "mean_ratio_tol": 0.1, "relmse_tol": 0.05,
        # relmse baseline ≥ the un-aligned 1.62-vs-1.0 error (~0.38) so this test
        # isolates the mean-ratio window; the ratio check is the lever here.
        "baselines": {"path|wavefront": {"mean_ratio": 1.6, "relmse": 0.4}},
    })
    # 1.62 is within 10% of the 1.6 baseline ⇒ pass; 1.0 is not ⇒ fail.
    ok = parity.absolute_radiance_result(spec, _combo(), _flat(1.62), _flat(1.0))
    assert ok is not None and ok.passed and ok.baseline_used
    bad = parity.absolute_radiance_result(spec, _combo(), _flat(1.0), _flat(1.0))
    assert bad is not None and not bad.passed
