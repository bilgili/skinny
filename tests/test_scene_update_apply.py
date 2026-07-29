"""One application path: `Renderer.apply_scene_update` (scene-intake-interface).

Device-free, but **not** import-free: the subject is `Renderer`'s own method,
so `skinny.renderer` — and therefore `vulkan`, which it imports at module
scope — must be importable. What these tests never do is construct a GPU
context. They bind the unbound adoption methods to a stub whose GPU steps
record their name instead of running, which is what makes the *order* and the
*runtime-state carry-over* assertable at all: before this change the carry-over
lived inside one of three adoption paths and could only be observed as a side
effect of taking that path.

Do not read this as a hostless gate for the intake seam — that is
`tests/test_scene_intake.py`, which imports no renderer at all. The end-to-end
GPU behaviour is gate 6.2/6.3; this is the property gate in between.
"""

from __future__ import annotations

import types

import pytest

from skinny.scene import LightDir, Material, Scene


def _renderer_stub():
    """The smallest object `apply_scene_update` can run against.

    Bound as an unbound function so no GPU context is constructed; every step
    that would touch the device appends its name to `calls`.
    """
    from skinny.renderer import Renderer

    r = types.SimpleNamespace()
    r.calls = []

    def _record(name, *, then=None):
        def _fn(*args, **kwargs):
            r.calls.append(name)
            if then is not None:
                then(*args, **kwargs)
        return _fn

    r._usd_scene = None
    r._usd_stage = None
    r._usd_edit_layer = None
    r._scene_graph = None
    r._anim_index = None
    r._skeletal = None
    r._usd_controls = []
    r._usd_up_axis_rt = None
    r._usd_model_index = -1
    r.model_index = -1
    r.models = ["head"]
    r.clock = None
    r.mm_per_unit = 120.0
    r.film_max_component = 0.0
    r._material_version = 0
    r._scene_graph_version = 0
    r._scene_version = 0
    r.usd_camera = object()

    r._attach_edit_layer = _record("attach_edit_layer")
    r._sync_volume_grid = _record("sync_volume_grid")
    r._gen_scene_materials = _record("gen_scene_materials")
    r._frame_camera_to_scene = _record("frame_camera")
    r._override_to_orbit = _record("override_to_orbit")
    r._apply_control_defaults = _record("control_defaults")
    r._upload_usd_scene = _record("upload")
    r._inject_default_lights_into_scene_graph = _record("inject_lights")
    r._refresh_camera_node = _record("refresh_camera_node")
    r._carry_runtime_state_into = types.MethodType(
        Renderer._carry_runtime_state_into, r)

    r.apply = types.MethodType(Renderer.apply_scene_update, r)
    return r


def _scene(*, mm_per_unit=120.0, film=0.0, camera=None):
    return Scene(mm_per_unit=mm_per_unit, film_max_component=film,
                 camera_override=camera)


class TestAdoptionOrder:
    """The order is stated once, so it can be asserted once."""

    def test_full_load_runs_every_step_in_the_stated_order(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_controls = ["a control"]
        update = scene_intake.SceneUpdate.streamed(
            _scene(mm_per_unit=1000.0, camera=object()),
            stage=object(), controls=["a control"],
        )
        r.apply(update)
        assert r.calls == [
            "attach_edit_layer",
            "sync_volume_grid",
            "gen_scene_materials",
            "frame_camera",
            "override_to_orbit",
            "control_defaults",
            "upload",
            "inject_lights",
            "refresh_camera_node",
        ]

    def test_scale_is_final_before_the_volume_grid_is_synced(self):
        """The σ folds packed by the upload read both, so the grid must not be
        uploaded against the previous scene's scale."""
        from skinny import scene_intake
        r = _renderer_stub()
        seen = []
        r._sync_volume_grid = lambda scene: seen.append(r.mm_per_unit)
        r.apply(scene_intake.SceneUpdate.streamed(
            _scene(mm_per_unit=1000.0), stage=None))
        assert seen == [1000.0]

    def test_resync_does_not_move_the_camera(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = _scene()
        r.apply(scene_intake.SceneUpdate.resynced(
            _scene(camera=object()), stage=None))
        assert "frame_camera" not in r.calls
        assert "override_to_orbit" not in r.calls

    def test_resync_does_not_reapply_control_defaults(self):
        """Re-applying them on every edit would clobber the user's later
        changes to the very controls they declared."""
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = _scene()
        r._usd_controls = ["a control"]
        r.apply(scene_intake.SceneUpdate.resynced(_scene(), stage=None))
        assert "control_defaults" not in r.calls

    def test_adopted_frames_the_camera_only_on_the_first_call(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r.apply(scene_intake.SceneUpdate.adopted(_scene(), stage=None))
        assert r.calls.count("frame_camera") == 1
        r.calls.clear()
        r.apply(scene_intake.SceneUpdate.adopted(_scene(), stage=None))
        assert "frame_camera" not in r.calls

    def test_adopted_reframes_whenever_the_scene_authors_a_camera(self):
        """An animated authored camera must track across repeated calls."""
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = _scene()
        r.apply(scene_intake.SceneUpdate.adopted(
            _scene(camera=object()), stage=None))
        assert "frame_camera" in r.calls


class TestPerTriggerFields:
    def test_activation_label_is_appended_once(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r.apply(scene_intake.SceneUpdate.adopted(_scene(), stage=None))
        assert r.models[-1] == "USD: (headless)"
        assert r.model_index == r._usd_model_index == 1
        r.apply(scene_intake.SceneUpdate.adopted(_scene(), stage=None))
        assert r.models.count("USD: (headless)") == 1

    def test_a_resync_never_enters_the_usd_active_state(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = _scene()
        r.apply(scene_intake.SceneUpdate.resynced(_scene(), stage=None))
        assert r.models == ["head"]
        assert r._usd_model_index == -1

    def test_sentinel_scale_is_never_adopted(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r.mm_per_unit = 42.0
        r.apply(scene_intake.SceneUpdate.streamed(_scene(), stage=None))
        assert r.mm_per_unit == 42.0

    def test_film_clamp_is_adopted_on_every_trigger(self):
        """The post-edit path used to re-read the stage and drop the film
        clamp on the floor — it hand-copied eight fields and this was not
        one of them."""
        from skinny import scene_intake
        for update in (
            scene_intake.SceneUpdate.streamed(_scene(film=7.5), stage=None),
            scene_intake.SceneUpdate.adopted(_scene(film=7.5), stage=None),
            scene_intake.SceneUpdate.resynced(_scene(film=7.5), stage=None),
        ):
            r = _renderer_stub()
            r._usd_scene = _scene()
            r.apply(update)
            assert r.film_max_component == 7.5

    def test_a_partial_update_keeps_the_stage_state_it_did_not_read(self):
        """A resync re-reads geometry, not the animation index. `None` there
        means 'not read', not 'clear it'."""
        from skinny import scene_intake
        r = _renderer_stub()
        r._anim_index = "existing index"
        r._skeletal = "existing skeleton"
        r._usd_controls = ["existing control"]
        r.clock = "existing clock"
        update = scene_intake.SceneUpdate.resynced(_scene(), stage=None)
        assert update.replaces_stage_state is False
        r.apply(update)
        assert r._anim_index == "existing index"
        assert r._skeletal == "existing skeleton"
        assert r._usd_controls == ["existing control"]
        assert r.clock == "existing clock"

    def test_a_replacing_update_clears_the_state_it_has_none_of(self):
        """The opposite case, and the reason the two are distinguished: a
        stage whose index build failed carries `anim_index=None` as a real
        opinion. Keeping the previous value would leave an index installed
        that points at a stage the renderer no longer holds."""
        from skinny import scene_intake
        from skinny.playback import PlaybackClock
        for ctor in (
            lambda: scene_intake.SceneUpdate.streamed(_scene(), stage=None),
            lambda: scene_intake.SceneUpdate.replacing(
                _scene(), stage=None, label="USD: (empty)"),
        ):
            r = _renderer_stub()
            r._anim_index = "stale index"
            r._skeletal = "stale skeleton"
            r._usd_controls = ["stale control"]
            r._usd_up_axis_rt = "stale rotation"
            r.clock = "stale clock"
            r._last_eval_time_code = 7.0
            update = ctor()
            assert update.replaces_stage_state is True
            r.apply(update)
            assert r._anim_index is None
            assert r._skeletal is None
            assert r._usd_controls == []
            assert r._usd_up_axis_rt is None
            assert isinstance(r.clock, PlaybackClock)
            assert r._last_eval_time_code is None


class TestRuntimeStateSurvivesAStageReread:
    """Finding #7, now a property of applying an update rather than a rescue
    step inside one of three adoption paths."""

    def _live_scene(self):
        scene = _scene()
        scene.instances = [
            types.SimpleNamespace(prim_path="/A", name="/A", enabled=False),
            types.SimpleNamespace(prim_path="/B", name="/B", enabled=True),
        ]
        scene.lights_dir = [
            LightDir(direction=(0, 1, 0), radiance=(1, 1, 1),
                     prim_path="/Sun", enabled=False),
        ]
        scene.materials = [
            Material(name="Wall", parameter_overrides={"base_color": "edited"}),
        ]
        scene.materials[0].source_prim_path = "/Looks/Wall"
        return scene

    def _reread_scene(self):
        scene = _scene()
        scene.instances = [
            types.SimpleNamespace(prim_path="/A", name="/A", enabled=True),
            types.SimpleNamespace(prim_path="/B", name="/B", enabled=True),
            types.SimpleNamespace(prim_path="/C", name="/C", enabled=True),
        ]
        scene.lights_dir = [
            LightDir(direction=(0, 1, 0), radiance=(1, 1, 1),
                     prim_path="/Sun", enabled=True),
        ]
        scene.materials = [
            Material(name="Wall", parameter_overrides={
                "base_color": "from disk", "roughness": "from disk"}),
        ]
        scene.materials[0].source_prim_path = "/Looks/Wall"
        return scene

    def test_disabled_instances_stay_disabled(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = self._live_scene()
        new = self._reread_scene()
        r.apply(scene_intake.SceneUpdate.resynced(new, stage=None))
        assert [i.enabled for i in new.instances] == [False, True, True]

    def test_disabled_lights_stay_disabled(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = self._live_scene()
        new = self._reread_scene()
        r.apply(scene_intake.SceneUpdate.resynced(new, stage=None))
        assert new.lights_dir[0].enabled is False

    def test_live_material_overrides_win_over_the_reloaded_defaults(self):
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = self._live_scene()
        new = self._reread_scene()
        r.apply(scene_intake.SceneUpdate.resynced(new, stage=None))
        assert new.materials[0].parameter_overrides == {
            "base_color": "edited",       # the user's edit survived
            "roughness": "from disk",     # a newly authored key still arrives
        }

    def test_overrides_do_not_cross_apply_between_scopes(self):
        """Keyed by prim path, not leaf name: `/ScopeA/Foo` and `/ScopeB/Foo`
        are different materials that happen to share a name."""
        from skinny import scene_intake
        r = _renderer_stub()
        old = _scene()
        old.materials = [Material(name="Foo", parameter_overrides={"c": "A"})]
        old.materials[0].source_prim_path = "/ScopeA/Foo"
        r._usd_scene = old

        new = _scene()
        new.materials = [Material(name="Foo", parameter_overrides={})]
        new.materials[0].source_prim_path = "/ScopeB/Foo"
        r.apply(scene_intake.SceneUpdate.resynced(new, stage=None))
        assert new.materials[0].parameter_overrides == {}

    def test_a_replacing_update_keeps_no_runtime_state(self):
        """`create_empty_scene` used to finish through the post-edit path and
        so inherited its carry-over, merging the *previous* scene's material
        overrides onto a brand-new empty stage. A force-replace replaces."""
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = self._live_scene()
        new = self._reread_scene()
        r.apply(scene_intake.SceneUpdate.replacing(
            new, stage=None, label="USD: (empty)"))
        assert [i.enabled for i in new.instances] == [True, True, True]
        assert new.lights_dir[0].enabled is True
        assert new.materials[0].parameter_overrides["base_color"] == "from disk"

    def test_a_full_load_does_not_carry_state_over(self):
        """`parameter_overrides` mixes authored loader values with live edits,
        so carrying them onto a *different* scene would let the old authored
        value beat the newly authored one."""
        from skinny import scene_intake
        r = _renderer_stub()
        r._usd_scene = self._live_scene()
        new = self._reread_scene()
        r.apply(scene_intake.SceneUpdate.streamed(new, stage=None))
        assert [i.enabled for i in new.instances] == [True, True, True]
        assert new.materials[0].parameter_overrides["base_color"] == "from disk"


class TestSceneVersion:
    def test_every_applied_update_bumps_the_counter(self):
        from skinny import scene_intake
        r = _renderer_stub()
        assert r._scene_version == 0
        for n in (1, 2, 3):
            r.apply(scene_intake.SceneUpdate.adopted(_scene(), stage=None))
            assert r._scene_version == n

    def test_the_counter_moves_where_object_identity_would_not(self):
        """The whole point of replacing `id(renderer._usd_scene)`: an id only
        changes on a swap, so it went stale the moment a path mutated the
        scene in place."""
        from skinny import scene_intake
        r = _renderer_stub()
        scene = _scene()
        r.apply(scene_intake.SceneUpdate.adopted(scene, stage=None))
        before_id, before_version = id(r._usd_scene), r._scene_version
        r.apply(scene_intake.SceneUpdate.adopted(scene, stage=None))
        assert id(r._usd_scene) == before_id
        assert r._scene_version == before_version + 1

    def test_accumulation_resets_on_every_adoption(self):
        """`material_version` feeds the accumulation state hash, and no
        provider covers scene identity — so without this bump a swapped scene
        could keep accumulating over the previous one's samples."""
        from skinny import scene_intake
        r = _renderer_stub()
        r.apply(scene_intake.SceneUpdate.adopted(_scene(), stage=None))
        assert r._material_version == 1

    def test_clearing_the_model_state_is_itself_a_scene_change(self):
        """Regression: with `id(_usd_scene)` as the token, dropping the scene
        was noticed by accident — `id(None)` differs from the old scene's id.
        A counter has no such accident, so the clear must bump it, or every
        consumer keeps showing state from a stage that is gone."""
        from skinny.playback import PlaybackClock
        from skinny.renderer import Renderer

        r = _renderer_stub()
        r._mesh_sources = []
        r._displacement_cache = {}
        r._normal_cache = {}
        r._material_graph_ids = {}
        r._material_graph_overrides = {}
        r._mtlx_scene_materials = {}
        r._dummy_mesh = object()
        r.material_capacity = 4
        r._upload_mesh = lambda _m: None
        r._upload_detail_maps = lambda _d: None
        r._usd_scene = _scene()
        r._usd_controls = ["a control from the old stage"]
        r._anim_index = "old index"
        r._skeletal = "old skeleton"
        r._usd_up_axis_rt = "old rotation"
        r.clock = "old clock"
        before = r._scene_version

        types.MethodType(Renderer._clear_model_state, r)()

        assert r._scene_version == before + 1, "the clear must be observable"
        assert r._usd_controls == []
        assert r._anim_index is None
        assert r._skeletal is None
        assert r._usd_up_axis_rt is None
        assert isinstance(r.clock, PlaybackClock)

    def test_ui_sites_read_the_counter_not_the_object_id(self):
        import ast
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "src" / "skinny"
        offenders = []
        for path in sorted(src.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "id"
                        and node.args
                        and "_usd_scene" in ast.dump(node.args[0])):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], (
            "id(_usd_scene) is not a change token; use renderer.scene_version: "
            + ", ".join(offenders))


@pytest.mark.parametrize("method", [
    "apply_scene_update", "_carry_runtime_state_into", "scene_version",
])
def test_the_renderer_exposes_the_one_application_path(method):
    from skinny.renderer import Renderer
    assert hasattr(Renderer, method)
