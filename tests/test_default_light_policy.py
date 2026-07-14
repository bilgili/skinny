"""Default-light synthesis policy (change distant-light-caustic-parity).

The synthesized default DistantLight is injected only into scenes that author
no *powered* light at all. These tests exercise the predicate truth table and
the per-frame mirror outcome, hostless (no GPU) — the renderer module import
is skipped when the Vulkan SDK is absent, mirroring test_camera_placement.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _have_renderer() -> bool:
    # skinny.renderer imports `vulkan` at module scope, which raises without
    # the Vulkan SDK on the dynamic-library path. These policy tests are pure
    # CPU but can't import the module without it.
    try:
        import skinny.renderer  # noqa: F401
        return True
    except Exception:
        return False


needs_renderer = pytest.mark.skipif(
    not _have_renderer(), reason="skinny.renderer unimportable (no Vulkan SDK)"
)


def _host():
    """Minimal object exposing the two policy methods unbound from Renderer,
    so no GPU context is stood up."""
    from skinny.renderer import Renderer

    class _Host:
        _scene_authors_lights = Renderer._scene_authors_lights
        _scene_has_emissive_instances = staticmethod(
            Renderer._scene_has_emissive_instances
        )

    return _Host()


def _scene(*, dir_lights=(), sphere_lights=(), environment=None,
           instances=(), materials=()):
    return SimpleNamespace(
        lights_dir=list(dir_lights),
        lights_sphere=list(sphere_lights),
        environment=environment,
        instances=list(instances),
        materials=list(materials),
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


@needs_renderer
def test_no_lights_at_all_is_unlit():
    assert _host()._scene_authors_lights(_scene()) is False


@needs_renderer
def test_powered_distant_light_counts():
    assert _host()._scene_authors_lights(
        _scene(dir_lights=[_light()])
    ) is True


@needs_renderer
def test_powered_sphere_light_counts():
    assert _host()._scene_authors_lights(
        _scene(sphere_lights=[_light(intensity=60.0)])
    ) is True


@needs_renderer
def test_zero_power_lights_count_as_unlit():
    # A scene authoring only zero-intensity lights keeps the default light
    # (rendering it black instead would be a worse default).
    scene = _scene(
        dir_lights=[_light(intensity=0.0)],
        sphere_lights=[_light(intensity=0.0, radiance=(0.0, 0.0, 0.0))],
    )
    assert _host()._scene_authors_lights(scene) is False


@needs_renderer
def test_zero_radiance_light_counts_as_unlit():
    scene = _scene(sphere_lights=[_light(intensity=None, radiance=(0.0, 0.0, 0.0))])
    assert _host()._scene_authors_lights(scene) is False


@needs_renderer
def test_disabled_light_counts_as_unlit():
    assert _host()._scene_authors_lights(
        _scene(sphere_lights=[_light(enabled=False)])
    ) is False


@needs_renderer
def test_authored_dome_counts():
    # scene.environment is set ONLY for an authored UsdLux.DomeLight — the
    # renderer's built-in HDRI backdrop never appears here.
    assert _host()._scene_authors_lights(
        _scene(environment=SimpleNamespace(name="dome"))
    ) is True


@needs_renderer
def test_emissive_instance_counts():
    scene = _scene(instances=[_instance(0)], materials=[_emissive_mat()])
    assert _host()._scene_authors_lights(scene) is True


@needs_renderer
def test_non_emissive_instance_is_unlit():
    scene = _scene(instances=[_instance(0)], materials=[_plain_mat()])
    assert _host()._scene_authors_lights(scene) is False


@needs_renderer
def test_disabled_emissive_instance_is_unlit():
    scene = _scene(
        instances=[_instance(0, enabled=False)], materials=[_emissive_mat()]
    )
    assert _host()._scene_authors_lights(scene) is False


@needs_renderer
def test_result_cached_per_scene_object():
    host = _host()
    scene = _scene(sphere_lights=[_light()])
    assert host._scene_authors_lights(scene) is True
    # Mutating the scene does not re-derive (load-time authority) …
    scene.lights_sphere = []
    assert host._scene_authors_lights(scene) is True
    # … but a different Scene object does.
    assert host._scene_authors_lights(_scene()) is False


# ── Per-frame mirror outcome ────────────────────────────────────────────────
# The mirror branch (renderer.update) is exercised structurally: authored
# lights_dir → those records; authored-light scene without lights_dir → zero
# records; unlit scene → the slider default light. We reproduce the branch
# with the real predicate to lock the decision table.


@needs_renderer
@pytest.mark.parametrize(
    "scene_kwargs, expected",
    [
        # authored DistantLight → its own records
        (dict(dir_lights=[_light()]), "authored"),
        # authored SphereLight only → ZERO records (no phantom sun)
        (dict(sphere_lights=[_light(intensity=60.0)]), "zero"),
        # authored dome only → ZERO records
        (dict(environment=SimpleNamespace(name="dome")), "zero"),
        # truly unlit → slider default light
        (dict(), "slider"),
        # zero-power sphere only → still the slider default light
        (dict(sphere_lights=[_light(intensity=0.0, radiance=(0, 0, 0))]), "slider"),
    ],
)
def test_mirror_decision_table(scene_kwargs, expected):
    host = _host()
    usd_scene = _scene(**scene_kwargs)
    if usd_scene.lights_dir:
        outcome = "authored"
    elif host._scene_authors_lights(usd_scene):
        outcome = "zero"
    else:
        outcome = "slider"
    assert outcome == expected
