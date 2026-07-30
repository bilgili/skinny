"""Persisted session settings: merge-on-write and the declared snapshot schema.

Hostless — no GPU, no window. Covers change session-settings-owner: the two
interactive front-ends used to author the whole settings dict each, so exiting
one erased five of the other's keys.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skinny import session_snapshot, settings
from tests.test_ui_spec import _StubRenderer


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Redirect settings.json to a tmp dir and reset the in-memory cache."""
    monkeypatch.setattr(settings, "SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(settings, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(settings, "MESH_CACHE_DIR", tmp_path / "mesh_cache")
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(settings, "_last_dirs_cache", None)
    return tmp_path / "settings.json"


# ── merge on write ──────────────────────────────────────────────────

# The key sets as the two front-ends wrote them before the change; the
# erasure ran in both directions.
_GLFW_ONLY = {
    "vulkan_window": {"x": 12, "y": 34},
    "neural_handoff": "interop",
    "neural_trainer": "mlx",
    "train_precision": "fp16",
    "online_training": True,
}
_QT_ONLY = {
    "open_docks": ["scene_graph"],
    "last_dirs": {"model": "/tmp/models"},
    "section_states": {"Skin": True},
    "qt_geometry": "Z2VvbQ==",
    "qt_dock_state": "ZG9jaw==",
}
_SHARED = {
    "params": {"env_intensity": 2.0},
    "camera": {"mode": "orbit"},
    "gizmo_mode": 1,
    "backend": "metal",
    "encoding": "E0",
    "sppm_glossy_roughness": 0.6,
}


def test_qt_exit_preserves_glfw_keys(tmp_settings):
    """`skinny` writes, then `skinny-gui` closes: the GLFW-only keys survive."""
    settings.save_settings({**_SHARED, **_GLFW_ONLY})
    settings.save_settings({**_SHARED, **_QT_ONLY})
    data = json.loads(tmp_settings.read_text())
    for key, value in _GLFW_ONLY.items():
        assert data[key] == value, key


def test_glfw_exit_preserves_qt_keys(tmp_settings):
    """The reverse direction: `skinny` closing keeps the Qt layout keys."""
    settings.save_settings({**_SHARED, **_QT_ONLY})
    settings.save_settings({**_SHARED, **_GLFW_ONLY})
    data = json.loads(tmp_settings.read_text())
    for key, value in _QT_ONLY.items():
        assert data[key] == value, key


def test_writer_updates_its_own_keys(tmp_settings):
    """Merge is not append: a rewritten key takes the new value."""
    settings.save_settings({"backend": "vulkan", "gizmo_mode": 0})
    settings.save_settings({"backend": "metal"})
    data = json.loads(tmp_settings.read_text())
    assert data["backend"] == "metal"
    assert data["gizmo_mode"] == 0


def test_nested_value_is_replaced_whole(tmp_settings):
    """The writer of a key owns its whole value — no deep merge."""
    settings.save_settings({"params": {"a": 1.0, "b": 2.0}})
    settings.save_settings({"params": {"a": 9.0}})
    assert json.loads(tmp_settings.read_text())["params"] == {"a": 9.0}


def test_corrupt_file_does_not_propagate(tmp_settings):
    """A corrupt file is replaced by the writer's keys, with no exception."""
    tmp_settings.parent.mkdir(parents=True, exist_ok=True)
    tmp_settings.write_text("{not json at all")
    settings.save_settings({"backend": "metal"})
    assert json.loads(tmp_settings.read_text()) == {"backend": "metal"}


def test_non_dict_file_does_not_propagate(tmp_settings):
    """A JSON scalar/list is as unusable as a syntax error — same recovery."""
    tmp_settings.parent.mkdir(parents=True, exist_ok=True)
    tmp_settings.write_text("[1, 2, 3]")
    settings.save_settings({"backend": "vulkan"})
    assert json.loads(tmp_settings.read_text()) == {"backend": "vulkan"}


# ── declared schema: capture / restore ──────────────────────────────


class _SnapshotStub(_StubRenderer):
    """The UI stub plus the state `capture_shared` reads. Real `OrbitCamera` /
    `FreeCamera` / `GizmoMode` / `NeuralBuildConfig` — all device-free.
    """

    def __init__(self) -> None:
        super().__init__()
        from skinny.camera import FreeCamera, OrbitCamera
        from skinny.gizmo import GizmoMode
        from skinny.sampling.neural_weights import NeuralBuildConfig

        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        self.camera_mode = "orbit"
        self.gizmo = SimpleNamespace(mode=GizmoMode(0))
        self._neural_config = NeuralBuildConfig()
        self._neural_handoff_kind = "file"
        self._neural_trainer_kind = "auto"
        self._train_precision = "fp32"
        self._online_training_requested = False

        from skinny.film_io import FilmParameters

        self.film = FilmParameters()

        # The shared UI stub predates some params (it is built for tree shape,
        # which reads specs not values). Default every declared path it lacks,
        # so a snapshot covers the whole set.
        from skinny.params import _get_nested, build_all_params

        for p in build_all_params(self):
            if p.path.startswith("mtlx."):
                continue          # served by mtlx_overrides, never AttributeError
            try:
                _get_nested(self, p.path)
            except AttributeError:
                assert "." not in p.path, f"stub lacks a holder for {p.path}"
                setattr(self, p.path, 0.0 if p.kind == "continuous" else 0)


def test_capture_shared_covers_exactly_the_shared_keys():
    captured = session_snapshot.capture_shared(_SnapshotStub(), backend="metal")
    assert set(captured) == set(session_snapshot.SHARED_KEYS)


def test_shared_and_contributed_sections_are_disjoint():
    assert not (session_snapshot.SHARED_KEYS & session_snapshot.CONTRIBUTED_KEYS)
    assert not (session_snapshot.GLFW_KEYS & session_snapshot.QT_KEYS)


def test_capture_restore_round_trip():
    src = _SnapshotStub()
    src.orbit_camera.yaw = 1.25
    src.orbit_camera.distance = 7.5
    src.free_camera.move_speed = 3.0
    src.camera_mode = "free"
    src.env_intensity = 2.5
    src.integrator_index = 1
    src._neural_handoff_kind = "interop"
    src._neural_trainer_kind = "mlx"
    src._train_precision = "fp16"
    src._online_training_requested = True
    src._sppm_glossy_roughness_override = 0.6
    from skinny.gizmo import GizmoMode
    src.gizmo.mode = GizmoMode(2)

    snapshot = session_snapshot.capture_shared(src, backend="metal")
    # Survives a JSON round-trip — this is what lands in settings.json.
    snapshot = json.loads(json.dumps(snapshot))

    dst = _SnapshotStub()
    session_snapshot.restore_shared(dst, snapshot)

    assert dst.orbit_camera.yaw == pytest.approx(1.25)
    assert dst.orbit_camera.distance == pytest.approx(7.5)
    assert dst.free_camera.move_speed == pytest.approx(3.0)
    assert dst.camera_mode == "free"
    assert dst.env_intensity == pytest.approx(2.5)
    assert dst.integrator_index == 1
    assert int(dst.gizmo.mode) == 2
    # Non-renderer-writable keys ride the dict for the front-ends to read.
    assert snapshot["backend"] == "metal"
    assert snapshot["neural_handoff"] == "interop"
    assert snapshot["neural_trainer"] == "mlx"
    assert snapshot["train_precision"] == "fp16"
    assert snapshot["online_training"] is True
    assert snapshot["sppm_glossy_roughness"] == pytest.approx(0.6)


def test_contributed_sections_are_preserved_verbatim():
    shared = session_snapshot.capture_shared(_SnapshotStub(), backend="vulkan")
    out = session_snapshot.contribute(
        shared, dict(_QT_ONLY), owned=session_snapshot.QT_KEYS)
    for key, value in _QT_ONLY.items():
        assert out[key] == value
    assert set(out) == set(session_snapshot.SHARED_KEYS) | set(_QT_ONLY)


def test_contribute_refuses_an_undeclared_key():
    """Accepting it would write a key the other front-end then erases."""
    with pytest.raises(ValueError, match="undeclared session-settings key"):
        session_snapshot.contribute(
            {}, {"my_new_dock": 1}, owned=session_snapshot.QT_KEYS)


def test_contribute_refuses_a_shared_key_as_contributed():
    """The shared section is the renderer's; a front-end may not restate it."""
    with pytest.raises(ValueError, match="undeclared session-settings key"):
        session_snapshot.contribute(
            {}, {"camera": {}}, owned=session_snapshot.QT_KEYS)


def test_contribute_refuses_the_other_frontend_s_key():
    """Per-front-end, not the union: `skinny` may not write a Qt-owned key, which
    `skinny-gui` would erase on its next exit.
    """
    with pytest.raises(ValueError, match="undeclared session-settings key"):
        session_snapshot.contribute(
            {}, {"qt_dock_state": "x"}, owned=session_snapshot.GLFW_KEYS)


def test_contribute_refuses_an_undeclared_owned_section():
    with pytest.raises(ValueError, match="not a declared contributed section"):
        session_snapshot.contribute({}, {}, owned=frozenset({"camera"}))


def test_restore_steps_are_fault_isolated():
    """A settings file that breaks one restore step must not cost the others —
    `skinny` had no isolation at all, `skinny-gui` had three try blocks.
    """
    dst = _SnapshotStub()

    class _Exploding(dict):
        def get(self, key, default=None):          # noqa: D102
            if key == "params":
                raise RuntimeError("boom")
            return super().get(key, default)

    data = _Exploding({
        "camera": {"orbit": {"distance": 9.0}},
        "gizmo_mode": 2,
    })
    session_snapshot.restore_shared(dst, data)
    assert dst.orbit_camera.distance == pytest.approx(9.0)
    assert int(dst.gizmo.mode) == 2


# ── the two reconciled divergences ──────────────────────────────────


def test_wide_orbit_distance_raises_the_cap():
    """One camera rule: a persisted distance past the cap moves the CAP, not
    the view. `skinny-gui` used to clamp it to 50 and lose the view.
    """
    dst = _SnapshotStub()
    session_snapshot.restore_camera(dst, {"orbit": {"distance": 250.0}})
    assert dst.orbit_camera.distance == pytest.approx(250.0)
    assert dst.orbit_camera.max_distance >= 250.0


def test_authored_lighting_does_not_discard_params():
    """Capture is unfiltered: a scene with authored lighting hides the
    fallback-light controls, but must not drop their persisted values.
    """
    from skinny.params import build_visible_params, is_fallback_light_param

    src = _SnapshotStub()
    src.uses_default_lights = False
    src.env_intensity = 3.0
    src.light_intensity = 4.0

    visible = {p.path for p in build_visible_params(src)}
    assert "env_intensity" not in visible, "fixture no longer exercises filtering"

    params = session_snapshot.capture_shared(src, backend="metal")["params"]
    assert params["env_intensity"] == pytest.approx(3.0)
    assert params["light_intensity"] == pytest.approx(4.0)
    assert any(is_fallback_light_param(p) for p in build_visible_params(_SnapshotStub()))

    # And they come back under a scene that does use the fallback pair.
    dst = _SnapshotStub()
    session_snapshot.restore_shared(dst, {"params": params})
    assert dst.env_intensity == pytest.approx(3.0)
    assert dst.light_intensity == pytest.approx(4.0)


# ── a pre-change settings file still starts both front-ends ─────────


def test_pre_change_settings_file_restores(tmp_settings):
    """A settings.json written by either front-end BEFORE the change loads and
    restores with no exception, and its persisted flags come back.
    """
    legacy = {
        # `skinny`'s historical shape: visibility-filtered params (no light_*),
        # a camera, and the neural keys `skinny-gui` used to erase.
        "params": {"env_intensity": 2.0, "mm_per_unit": 6.0},
        "camera": {"mode": "orbit", "orbit": {"yaw": 0.5, "distance": 120.0}},
        "gizmo_mode": 3,
        "backend": "vulkan",
        "encoding": "E0",
        "sppm_glossy_roughness": 0.5,
        "neural_handoff": "shared",
        "neural_trainer": "cpu",
        "train_precision": "fp16",
        "online_training": True,
        "vulkan_window": {"x": 5, "y": 6},
        # `skinny-gui`'s historical shape.
        "open_docks": ["bxdf"],
        "section_states": {"Skin": False},
        "qt_geometry": "Z2VvbQ==",
        "qt_dock_state": "ZG9jaw==",
    }
    settings.save_settings(legacy)
    loaded = settings.load_settings()

    dst = _SnapshotStub()
    session_snapshot.restore_shared(dst, loaded)
    assert dst.env_intensity == pytest.approx(2.0)
    assert dst.orbit_camera.distance == pytest.approx(120.0)   # past the 50 cap
    assert dst.orbit_camera.max_distance >= 120.0
    assert int(dst.gizmo.mode) == 3

    # No CLI flag, no env var: every persisted flag wins over the default.
    for key, cli_default, expected in (
        ("neural_handoff", "file", "shared"),
        ("neural_trainer", "auto", "cpu"),
        ("train_precision", "fp32", "fp16"),
        ("online_training", False, True),
        ("sppm_glossy_roughness", None, 0.5),
    ):
        assert session_snapshot.resolve_persisted_flag(
            key, cli_default, loaded, argv=[], environ={},
        ) == expected, key


def test_explicit_cli_flag_beats_the_persisted_value():
    saved = {"neural_trainer": "cpu"}
    assert session_snapshot.resolve_persisted_flag(
        "neural_trainer", "mlx", saved, argv=["--neural-trainer", "mlx"], environ={},
    ) == "mlx"
    assert session_snapshot.resolve_persisted_flag(
        "neural_trainer", "mlx", saved, argv=[], environ={"SKINNY_NEURAL_TRAINER": "mlx"},
    ) == "mlx"


def test_garbage_persisted_flag_falls_back_to_the_cli_value():
    saved = {"neural_trainer": "quantum", "sppm_glossy_roughness": "nope"}
    assert session_snapshot.resolve_persisted_flag(
        "neural_trainer", "auto", saved, argv=[], environ={},
    ) == "auto"
    assert session_snapshot.resolve_persisted_flag(
        "sppm_glossy_roughness", None, saved, argv=[], environ={},
    ) is None


# ── each front-end's contributed key set ────────────────────────────

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _contributed_keys(source: Path) -> set[str]:
    """String keys a front-end's `_contributed_session_state` writes.

    Source-level on purpose: neither front-end module imports hostlessly (both
    pull in `vulkan` at module scope), so an import-based check here would skip
    silently on a Metal-only host — and a skip reads as a pass.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_contributed_session_state"),
        None,
    )
    assert fn is not None, f"{source} has no _contributed_session_state"
    keys: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):  # return {"vulkan_window": …}
            keys.update(
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            )
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            # A computed key would read as "no key" here and blind the gate.
            assert isinstance(node.slice, ast.Constant), (
                f"{source}: computed contributed key at line {node.lineno} — "
                "write the key as a literal so this gate can see it"
            )
            keys.add(node.slice.value)  # out["open_docks"] = …
    return keys


def test_glfw_frontend_contributes_exactly_its_declared_keys():
    assert _contributed_keys(_SRC / "app.py") == set(session_snapshot.GLFW_KEYS)


def test_qt_frontend_contributes_exactly_its_declared_keys():
    assert _contributed_keys(_SRC / "ui" / "qt" / "app.py") == set(session_snapshot.QT_KEYS)


def test_each_frontend_declares_its_own_section_at_the_call_site():
    """`contribute(owned=…)` must name the front-end's OWN section — passing the
    other's would let it write a key the owner then erases.
    """
    glfw_src = (_SRC / "app.py").read_text(encoding="utf-8")
    qt_src = (_SRC / "ui" / "qt" / "app.py").read_text(encoding="utf-8")
    assert "owned=GLFW_KEYS" in glfw_src and "owned=QT_KEYS" not in glfw_src
    assert "owned=QT_KEYS" in qt_src and "owned=GLFW_KEYS" not in qt_src
