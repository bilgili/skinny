"""Light emission attributes resolve at the requested time code.

`Usd.Attribute.Get()` with no time code resolves at the default time code. An
attribute holding only time samples has no value there, so USD falls back to the
schema default (50000 for a DistantLight intensity, 1.0 for a SphereLight). A
time-sampled light must therefore be read at a time code, or its animation is
invisible and the render sits at the fallback.
"""

from __future__ import annotations

import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_usd(), reason="pxr/USD not installed")


def _stage_with_animated_lights():
    """Stage whose lights author intensity as time samples ONLY.

    No default value is authored, which is exactly the case where a
    time-code-free read collapses to the schema fallback.
    """
    from pxr import Gf, Sdf, Usd, UsdLux
    stage = Usd.Stage.CreateInMemory()
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(24)

    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    d_api = UsdLux.LightAPI(distant.GetPrim())
    d_api.GetIntensityAttr().Set(3.0, 0.0)
    d_api.GetIntensityAttr().Set(7.0, 24.0)

    sphere = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/Bulb"))
    s_api = UsdLux.LightAPI(sphere.GetPrim())
    s_api.GetIntensityAttr().Set(2.0, 0.0)
    s_api.GetIntensityAttr().Set(9.0, 24.0)
    s_api.GetColorAttr().Set(Gf.Vec3f(1.0, 0.5, 0.25), 0.0)
    s_api.GetColorAttr().Set(Gf.Vec3f(0.25, 0.5, 1.0), 24.0)
    return stage


class TestLightEmissionTimeSamples:
    def test_distant_light_intensity_varies_across_time_codes(self):
        from pxr import Usd
        from skinny.usd_loader import _extract_distant_light
        stage = _stage_with_animated_lights()
        prim = stage.GetPrimAtPath("/World/Sun")

        at_start = _extract_distant_light(prim, Usd.TimeCode(0.0))
        at_end = _extract_distant_light(prim, Usd.TimeCode(24.0))
        assert at_start is not None and at_end is not None

        # The authored samples, not the 50000 DistantLight schema fallback.
        assert at_start.radiance[0] == pytest.approx(3.0)
        assert at_end.radiance[0] == pytest.approx(7.0)
        assert at_start.radiance[0] != at_end.radiance[0]

    def test_sphere_light_intensity_and_color_vary_across_time_codes(self):
        from pxr import Usd
        from skinny.usd_loader import _extract_sphere_light
        stage = _stage_with_animated_lights()
        prim = stage.GetPrimAtPath("/World/Bulb")

        at_start = _extract_sphere_light(prim, Usd.TimeCode(0.0))
        at_end = _extract_sphere_light(prim, Usd.TimeCode(24.0))
        assert at_start is not None and at_end is not None

        # radiance == color * intensity * 2^exposure, per time code.
        assert at_start.radiance == pytest.approx([2.0, 1.0, 0.5])
        assert at_end.radiance == pytest.approx([2.25, 4.5, 9.0])

        # The stashed colour/intensity the scene-graph editor mutates are read
        # from the same time code as the combined radiance.
        assert at_start.intensity == pytest.approx(2.0)
        assert at_end.intensity == pytest.approx(9.0)
        assert at_start.color == pytest.approx([1.0, 0.5, 0.25])
        assert at_end.color == pytest.approx([0.25, 0.5, 1.0])

    def test_exposure_is_read_at_the_time_code(self):
        from pxr import Sdf, Usd, UsdLux
        from skinny.usd_loader import _extract_distant_light
        stage = Usd.Stage.CreateInMemory()
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
        api = UsdLux.LightAPI(light.GetPrim())
        api.GetIntensityAttr().Set(1.0)          # default value, not sampled
        api.GetExposureAttr().Set(0.0, 0.0)
        api.GetExposureAttr().Set(2.0, 24.0)

        at_start = _extract_distant_light(light.GetPrim(), Usd.TimeCode(0.0))
        at_end = _extract_distant_light(light.GetPrim(), Usd.TimeCode(24.0))
        assert at_start.radiance[0] == pytest.approx(1.0)   # 1 * 2^0
        assert at_end.radiance[0] == pytest.approx(4.0)     # 1 * 2^2

    def test_static_light_is_unchanged_by_the_time_code(self):
        """No time samples ⇒ the time code must not alter the result."""
        from pxr import Gf, Sdf, Usd, UsdLux
        from skinny.usd_loader import _extract_distant_light
        stage = Usd.Stage.CreateInMemory()
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
        api = UsdLux.LightAPI(light.GetPrim())
        api.GetIntensityAttr().Set(5.0)
        api.GetColorAttr().Set(Gf.Vec3f(1.0, 0.5, 0.0))

        for tc in (Usd.TimeCode.Default(), Usd.TimeCode(0.0), Usd.TimeCode(99.0)):
            ld = _extract_distant_light(light.GetPrim(), tc)
            assert ld.radiance == pytest.approx([5.0, 2.5, 0.0])
