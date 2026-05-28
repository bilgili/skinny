"""Detection of time-sampled USD prims for animation playback."""

from __future__ import annotations

import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_usd(), reason="pxr/USD not installed")


def _animated_xform_stage():
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    stage.SetTimeCodesPerSecond(30)

    anim = UsdGeom.Mesh.Define(stage, "/Anim")
    op = UsdGeom.Xformable(anim).AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0.0)
    op.Set(Gf.Vec3d(10, 0, 0), 48.0)

    static = UsdGeom.Mesh.Define(stage, "/Static")
    UsdGeom.Xformable(static).AddTranslateOp().Set(Gf.Vec3d(1, 1, 1))
    return stage


class TestAnimationIndex:
    def test_detects_animated_xform(self):
        from skinny.usd_loader import build_animation_index
        idx = build_animation_index(_animated_xform_stage())
        assert "/Anim" in idx.xform_paths
        assert idx.has_animation

    def test_excludes_static_prim(self):
        from skinny.usd_loader import build_animation_index
        idx = build_animation_index(_animated_xform_stage())
        assert "/Static" not in idx.xform_paths

    def test_static_only_stage_has_no_animation(self):
        from pxr import Gf, Usd, UsdGeom
        from skinny.usd_loader import build_animation_index
        stage = Usd.Stage.CreateInMemory()
        m = UsdGeom.Mesh.Define(stage, "/M")
        UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        idx = build_animation_index(stage)
        assert not idx.has_animation
        assert idx.xform_paths == []
        assert not idx.camera_animated

    def test_child_of_animated_parent_is_flagged(self):
        from pxr import Gf, Usd, UsdGeom
        from skinny.usd_loader import build_animation_index
        stage = Usd.Stage.CreateInMemory()
        parent = UsdGeom.Xform.Define(stage, "/Parent")
        op = UsdGeom.Xformable(parent).AddTranslateOp()
        op.Set(Gf.Vec3d(0, 0, 0), 0.0)
        op.Set(Gf.Vec3d(5, 0, 0), 24.0)
        child = UsdGeom.Mesh.Define(stage, "/Parent/Child")  # static local xform
        UsdGeom.Xformable(child).AddTranslateOp().Set(Gf.Vec3d(0, 1, 0))
        idx = build_animation_index(stage)
        assert "/Parent/Child" in idx.xform_paths

    def test_detects_animated_light(self):
        from pxr import Usd, UsdLux
        from skinny.usd_loader import build_animation_index
        stage = Usd.Stage.CreateInMemory()
        light = UsdLux.SphereLight.Define(stage, "/Light")
        light.GetIntensityAttr().Set(100.0, 0.0)
        light.GetIntensityAttr().Set(200.0, 24.0)
        idx = build_animation_index(stage)
        assert "/Light" in idx.light_paths
        assert idx.has_animation

    def test_detects_animated_camera(self):
        from pxr import Gf, Usd, UsdGeom
        from skinny.usd_loader import build_animation_index
        stage = Usd.Stage.CreateInMemory()
        cam = UsdGeom.Camera.Define(stage, "/Cam")
        op = UsdGeom.Xformable(cam).AddTranslateOp()
        op.Set(Gf.Vec3d(0, 0, 0), 0.0)
        op.Set(Gf.Vec3d(0, 0, 5), 24.0)
        idx = build_animation_index(stage)
        assert idx.camera_animated
        assert idx.has_animation


class TestPlaybackClockFromStage:
    def test_reads_time_range_and_fps(self):
        from skinny.usd_loader import build_animation_index, build_playback_clock
        stage = _animated_xform_stage()
        idx = build_animation_index(stage)
        clock = build_playback_clock(stage, idx)
        assert clock.start_time_code == pytest.approx(0.0)
        assert clock.end_time_code == pytest.approx(48.0)
        assert clock.time_codes_per_second == pytest.approx(30.0)
        assert clock.playback_fps == pytest.approx(30.0)
        assert clock.current_time_code == pytest.approx(0.0)
        assert clock.has_animation

    def test_default_fps_fallback(self):
        from skinny.usd_loader import build_animation_index, build_playback_clock
        stage = _animated_xform_stage()
        idx = build_animation_index(stage)
        clock = build_playback_clock(stage, idx, default_fps=12.0)
        # tcps is authored to 30, so fps follows the stage, not the fallback
        assert clock.playback_fps == pytest.approx(30.0)
