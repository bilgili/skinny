"""Hostless regression for `apply_dome_light_texture` under lighting authority.

Bug (fix-added-dome-texture-authority): setting the texture on a dome that had
NO environment (e.g. one just created via `add_light`) routed the edit to the
fallback default-lights library, which the authored authority never reads, so
the dome stayed dark until a full stage resync. The fix branches on the active
lighting authority and constructs the authored `LightEnvHDR` when absent.

GPU behavior was verified separately on Metal; these tests pin the branch logic
hostless (the GPU upload is stubbed) so they run in the default sweep.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from unittest.mock import MagicMock

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdLux

from skinny.renderer import Renderer
from skinny.scene import LightEnvHDR, Scene, environment_contribution_intensity
from skinny.scene_graph import build_scene_graph, inject_default_lights
from skinny.usd_loader import load_scene_from_stage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_HDR = (
    PROJECT_ROOT / "tests" / "assets" / "suite" / "mat_dielectric"
    / "light_infinite_0_const.hdr"
)
# A second, distinct HDR (different stem + content) for swap tests.
SECOND_HDR = PROJECT_ROOT / "hdrs" / "venice_sunset.hdr"


def _default_light_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Skinny")
    UsdLux.DistantLight.Define(stage, "/Skinny/DefaultLight")
    UsdLux.DomeLight.Define(stage, "/Skinny/DefaultDome")
    return stage


def _editor() -> tuple[Renderer, Usd.Stage]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/Geometry")

    r = Renderer.__new__(Renderer)
    r._usd_stage = stage
    r._usd_edit_layer = None
    r._usd_scene = Scene()
    r._usd_model_index = 0
    r.model_index = 0
    r._material_version = 0
    r._scene_graph_version = 0
    r._last_env_index = (-1, -1)
    r.env_index = 0
    r.environments = []
    r.env_image = MagicMock()  # stub the GPU upload
    r._scene_graph = build_scene_graph(stage, r._usd_scene)
    r._default_light_stage = _default_light_stage()
    r._attach_edit_layer()
    inject_default_lights(
        r._scene_graph, r._default_light_stage, enabled=r.uses_default_lights,
    )

    def resync(self) -> None:
        self._usd_scene = load_scene_from_stage(self._usd_stage)
        self._scene_graph = build_scene_graph(self._usd_stage, self._usd_scene)
        inject_default_lights(
            self._scene_graph, self._default_light_stage,
            enabled=self.uses_default_lights,
        )

    r._resync_geometry_from_stage = MethodType(resync, r)
    return r, stage


def test_texture_on_added_dome_becomes_authored_environment():
    """The core bug: a freshly added (textureless) dome must contribute after
    its texture is set, with no resync/toggle."""
    r, _stage = _editor()

    r.add_light("DomeLight")
    # Precondition mirrors the measured failing state: authored authority,
    # dome present, but no environment yet.
    assert r.uses_default_lights is False
    assert r._usd_scene.environment is None

    ok = r.apply_dome_light_texture(0, str(FIXTURE_HDR))

    assert ok is True
    env = r._usd_scene.environment
    assert env is not None, "added dome's texture must create an authored env"
    assert environment_contribution_intensity(env) > 0.0
    # The GPU texture was uploaded and the cache invalidated for the next frame.
    r.env_image.upload_sync.assert_called()


def test_added_dome_intensity_folds_color_intensity_exposure():
    """The constructed env's intensity must match `_extract_dome_light`:
    luminance(color × intensity × 2**exposure). A plain >0 check can't catch
    formula/unit drift, so pin non-default authored values."""
    r, stage = _editor()
    path = r.add_light("DomeLight")
    api = UsdLux.LightAPI(stage.GetPrimAtPath(path))
    api.CreateColorAttr().Set(Gf.Vec3f(0.8, 0.4, 0.2))
    api.CreateIntensityAttr().Set(3.0)
    api.CreateExposureAttr().Set(1.0)

    r.apply_dome_light_texture(0, str(FIXTURE_HDR))

    scaled = 3.0 * (2.0 ** 1.0)  # intensity × 2**exposure
    radiance = np.array([0.8, 0.4, 0.2], np.float32) * scaled
    expected = float(np.dot(radiance, np.array([0.2126, 0.7152, 0.0722], np.float32)))
    assert r._usd_scene.environment.intensity == pytest.approx(expected, rel=1e-5)


def test_texture_swap_on_existing_dome_replaces_in_place():
    """Regression: a dome that already had an environment keeps the same
    LightEnvHDR object, only its data/name replaced."""
    r, _stage = _editor()
    r.add_light("DomeLight")
    r.apply_dome_light_texture(0, str(FIXTURE_HDR))
    first = r._usd_scene.environment
    assert first is not None
    first_name = first.name
    first_data = first.data

    r.apply_dome_light_texture(0, str(SECOND_HDR))
    second = r._usd_scene.environment

    assert second is first  # mutated in place, not replaced
    assert second.name != first_name  # a distinct HDR was actually applied
    assert second.data is not first_data


def test_missing_file_is_reported_without_mutation():
    r, _stage = _editor()
    r.add_light("DomeLight")
    before = r._usd_scene.environment

    ok = r.apply_dome_light_texture(0, "/no/such/file.hdr")

    assert ok is False
    assert r._usd_scene.environment is before  # unchanged (still None)


def test_default_lights_authority_uses_fallback_library():
    """With the synthesized default-lights authority active, the edit targets
    the env library, not an authored scene environment."""
    r, _stage = _editor()
    # No authored lights → default-lights authority.
    assert r.uses_default_lights is True
    r.environments = []

    ok = r.apply_dome_light_texture(0, str(FIXTURE_HDR))

    assert ok is True
    assert len(r.environments) == 1
    assert r.env_index == 0
    assert r._usd_scene.environment is None  # authored env untouched


def test_fallback_authority_leaves_retained_authored_environment_intact():
    """An inactive retained USD scene may still carry an authored environment
    while the default-lights authority owns the frame (the USD model is not the
    active model). The fallback branch must not clobber that authored env."""
    r, _stage = _editor()
    sentinel = LightEnvHDR(name="retained", data=np.zeros((2, 2, 4), np.float32))
    r._usd_scene.environment = sentinel
    # Deactivate the USD scene: model_index no longer matches the USD model, so
    # `_is_usd_active()` is False and the default-lights authority owns lighting
    # even though the retained scene carries an authored environment.
    r.model_index = 1
    assert r._is_usd_active() is False
    assert r.uses_default_lights is True
    r.environments = []

    ok = r.apply_dome_light_texture(0, str(FIXTURE_HDR))

    assert ok is True
    assert r._usd_scene.environment is sentinel  # retained env untouched
    assert sentinel.name == "retained"  # not mutated in place either
    assert len(r.environments) == 1  # new env landed in the fallback library
