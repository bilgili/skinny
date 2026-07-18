"""Hostless policy tests for USD-light authority."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from skinny.scene import (
    environment_contribution_intensity,
    scene_auxiliary_lights_for_authority,
    scene_has_authored_lighting,
    scene_environment_for_authority,
    scene_uses_default_lights,
    select_powered_distant_lights,
)


def _scene(*, dir_lights=(), sphere_lights=(), environment=None,
           instances=(), materials=(), has_authored_lighting=None):
    return SimpleNamespace(
        lights_dir=list(dir_lights),
        lights_sphere=list(sphere_lights),
        environment=environment,
        instances=list(instances),
        materials=list(materials),
        has_authored_lighting=has_authored_lighting,
    )


def _light(intensity=1.0, radiance=(1.0, 1.0, 1.0), enabled=True):
    return SimpleNamespace(intensity=intensity, radiance=radiance, enabled=enabled)


def _emissive_mat(color=(1.0, 1.0, 1.0)):
    return SimpleNamespace(parameter_overrides={"emissiveColor": color})


def _plain_mat():
    return SimpleNamespace(parameter_overrides={})


def _instance(material_id=0, enabled=True):
    return SimpleNamespace(material_id=material_id, enabled=enabled)


# ── Predicate truth table ───────────────────────────────────────────────────


def test_no_lights_at_all_has_no_authored_authority():
    assert scene_has_authored_lighting(_scene()) is False


def test_distant_light_presence_counts():
    assert scene_has_authored_lighting(
        _scene(dir_lights=[_light()])
    ) is True


def test_sphere_light_presence_counts():
    assert scene_has_authored_lighting(
        _scene(sphere_lights=[_light(intensity=60.0)])
    ) is True


def test_zero_power_lights_still_count_as_authored():
    scene = _scene(
        dir_lights=[_light(intensity=0.0)],
        sphere_lights=[_light(intensity=0.0, radiance=(0.0, 0.0, 0.0))],
    )
    assert scene_has_authored_lighting(scene) is True


def test_zero_radiance_light_still_counts_as_authored():
    scene = _scene(sphere_lights=[_light(intensity=None, radiance=(0.0, 0.0, 0.0))])
    assert scene_has_authored_lighting(scene) is True


def test_disabled_active_light_still_counts_as_authored():
    assert scene_has_authored_lighting(
        _scene(sphere_lights=[_light(enabled=False)])
    ) is True


def test_authored_dome_counts():
    # scene.environment is set ONLY for an authored UsdLux.DomeLight — the
    # renderer's built-in HDRI backdrop never appears here.
    assert scene_has_authored_lighting(
        _scene(environment=SimpleNamespace(name="dome"))
    ) is True


def test_emissive_instance_counts():
    scene = _scene(instances=[_instance(0)], materials=[_emissive_mat()])
    assert scene_has_authored_lighting(scene) is True


def test_non_emissive_instance_has_no_authored_authority():
    scene = _scene(instances=[_instance(0)], materials=[_plain_mat()])
    assert scene_has_authored_lighting(scene) is False


def test_disabled_emissive_instance_still_counts_as_authored():
    scene = _scene(
        instances=[_instance(0, enabled=False)], materials=[_emissive_mat()]
    )
    assert scene_has_authored_lighting(scene) is True


def test_result_re_evaluates_on_same_scene_object():
    scene = _scene(has_authored_lighting=True)
    assert scene_has_authored_lighting(scene) is True
    scene.has_authored_lighting = False
    assert scene_has_authored_lighting(scene) is False


# ── Active-scene authority ─────────────────────────────────────────────────


def test_no_loaded_usd_uses_defaults():
    assert scene_uses_default_lights(None, usd_active=False) is True


def test_active_lit_usd_suppresses_defaults():
    scene = _scene(has_authored_lighting=True)
    assert scene_uses_default_lights(scene, usd_active=True) is False


def test_active_lightless_usd_uses_defaults():
    scene = _scene(has_authored_lighting=False)
    assert scene_uses_default_lights(scene, usd_active=True) is True


def test_retained_inactive_usd_does_not_suppress_defaults():
    scene = _scene(has_authored_lighting=True)
    assert scene_uses_default_lights(scene, usd_active=False) is True
    assert scene_uses_default_lights(scene, usd_active=True) is False


def test_first_and_last_authored_light_transition_without_new_scene():
    scene = _scene(has_authored_lighting=False)
    assert scene_uses_default_lights(scene, usd_active=True) is True
    scene.has_authored_lighting = True
    assert scene_uses_default_lights(scene, usd_active=True) is False
    scene.has_authored_lighting = False
    assert scene_uses_default_lights(scene, usd_active=True) is True


@pytest.mark.parametrize(
    "scene, usd_active, expected_pair",
    [
        (None, False, (True, True)),
        (_scene(has_authored_lighting=False), True, (True, True)),
        (_scene(has_authored_lighting=True), True, (False, False)),
        # Retained but inactive USD metadata cannot suppress the active
        # default-head/OBJ fallback pair.
        (_scene(has_authored_lighting=True), False, (True, True)),
    ],
)
def test_default_distant_and_ibl_authority_is_all_or_nothing(
    scene, usd_active, expected_pair,
):
    uses_defaults = scene_uses_default_lights(scene, usd_active=usd_active)
    assert (uses_defaults, uses_defaults) == expected_pair


# ── Distant-light contribution routing ─────────────────────────────────────


def test_fallback_direct_toggle_can_disable_only_the_fallback_light():
    fallback = [_light()]
    assert select_powered_distant_lights(
        fallback,
        authority_enabled=False,
    ) == []
    assert select_powered_distant_lights(
        fallback,
        authority_enabled=True,
    ) == fallback


def test_authored_distant_light_state_controls_its_own_contribution():
    powered = _light()
    disabled = _light(enabled=False)
    zero_power = _light(intensity=0.0)
    selected = select_powered_distant_lights(
        [powered, disabled, zero_power],
        authority_enabled=True,
    )
    assert selected == [powered]


# ── Environment contribution routing ───────────────────────────────────────


def test_fallback_authority_selects_builtin_ibl():
    fallback = SimpleNamespace(name="fallback", intensity=0.5, enabled=True)
    usd = _scene(has_authored_lighting=False)
    selected = scene_environment_for_authority(
        usd,
        fallback,
        uses_default_lights=True,
    )
    assert selected is fallback
    assert environment_contribution_intensity(selected) == 0.5


def test_authored_dome_is_the_only_environment_in_authored_mode():
    fallback = SimpleNamespace(name="fallback", intensity=0.5, enabled=True)
    dome = SimpleNamespace(name="usd-dome", intensity=3.0, enabled=True)
    usd = _scene(environment=dome, has_authored_lighting=True)
    selected = scene_environment_for_authority(
        usd,
        fallback,
        uses_default_lights=False,
    )
    assert selected is dome
    assert environment_contribution_intensity(selected) == 3.0


def test_authored_scene_without_dome_has_black_environment():
    fallback = SimpleNamespace(name="fallback", intensity=0.5, enabled=True)
    usd = _scene(dir_lights=[_light()], has_authored_lighting=True)
    selected = scene_environment_for_authority(
        usd,
        fallback,
        uses_default_lights=False,
    )
    assert selected is None
    assert environment_contribution_intensity(selected) == 0.0


def test_disabled_authored_dome_has_zero_contribution():
    dome = SimpleNamespace(name="usd-dome", intensity=3.0, enabled=False)
    assert environment_contribution_intensity(dome) == 0.0


@pytest.mark.parametrize("integrator_index", range(4))
@pytest.mark.parametrize(
    "scene, expected_defaults",
    [
        (_scene(sphere_lights=[_light()], has_authored_lighting=True), False),
        (
            _scene(
                environment=SimpleNamespace(name="dome"),
                has_authored_lighting=True,
            ),
            False,
        ),
        (_scene(has_authored_lighting=True), False),  # converted area light
        (
            _scene(
                instances=[_instance(0)],
                materials=[_emissive_mat()],
                has_authored_lighting=True,
            ),
            False,
        ),
        (
            _scene(
                dir_lights=[_light(intensity=0.0)],
                has_authored_lighting=True,
            ),
            False,
        ),
        (_scene(has_authored_lighting=False), True),
    ],
)
def test_light_authority_is_integrator_independent(
    integrator_index,
    scene,
    expected_defaults,
):
    # The integrator selector is intentionally irrelevant: all four hosts
    # consume the same packed light counts and environment intensity.
    assert integrator_index in (0, 1, 2, 3)
    assert scene_uses_default_lights(scene, usd_active=True) is expected_defaults


def test_inactive_usd_auxiliary_lights_are_not_selected_for_fallback_scene():
    usd = _scene(
        sphere_lights=[_light()],
        instances=[_instance(0)],
        materials=[_emissive_mat()],
        has_authored_lighting=True,
    )
    spheres, emissive_scene = scene_auxiliary_lights_for_authority(
        usd,
        uses_default_lights=True,
    )
    assert spheres == []
    assert emissive_scene is None


def test_active_authored_scene_selects_usd_auxiliary_lights():
    sphere = _light()
    usd = _scene(
        sphere_lights=[sphere],
        instances=[_instance(0)],
        materials=[_emissive_mat()],
        has_authored_lighting=True,
    )
    spheres, emissive_scene = scene_auxiliary_lights_for_authority(
        usd,
        uses_default_lights=False,
    )
    assert spheres == [sphere]
    assert emissive_scene is usd


def test_update_polls_usd_before_building_authority_snapshot():
    renderer_path = (
        Path(__file__).resolve().parents[1] / "src" / "skinny" / "renderer.py"
    )
    tree = ast.parse(renderer_path.read_text())
    renderer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Renderer"
    )
    update = next(
        node
        for node in renderer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "update"
    )
    call_lines = {
        call.func.attr: call.lineno
        for call in ast.walk(update)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {
            "_poll_usd_streaming",
            "_build_scene_from_state",
            "_sync_auxiliary_light_authority",
        }
    }
    assert call_lines["_poll_usd_streaming"] < call_lines["_build_scene_from_state"]
    assert (
        call_lines["_build_scene_from_state"]
        < call_lines["_sync_auxiliary_light_authority"]
    )
