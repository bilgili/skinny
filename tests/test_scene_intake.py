"""Scene intake: stage → `SceneUpdate`, with no renderer anywhere in sight.

Hostless. Every assertion here runs in a process with no GPU device and no
`Renderer` instance — which is the point of the interface (change
`scene-intake-interface`).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src" / "skinny"


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="pxr/USD not installed")


# ─── Synthetic stages ─────────────────────────────────────────────────


def _stage(up_axis: str = "Y", *, meters_per_unit: float = 1.0):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(
        stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    return stage


def _sphere(stage, path: str, translate=(0, 0, 0), radius: float = 1.0):
    from pxr import Gf, UsdGeom
    s = UsdGeom.Sphere.Define(stage, path)
    s.GetRadiusAttr().Set(radius)
    UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*translate))
    return s


def _distant_light(stage, path: str, intensity: float = 3.0):
    from pxr import Gf, UsdLux
    lt = UsdLux.DistantLight.Define(stage, path)
    lt.GetIntensityAttr().Set(intensity)
    lt.GetColorAttr().Set(Gf.Vec3f(1.0, 0.5, 0.25))
    return lt


def _camera(stage, path: str = "/Cam", translate=(0, 0, 10)):
    from pxr import Gf, UsdGeom
    cam = UsdGeom.Camera.Define(stage, path)
    UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(*translate))
    return cam


def _populated_stage(up_axis: str = "Y"):
    stage = _stage(up_axis)
    _sphere(stage, "/Ball", translate=(2, 0, 0))
    _distant_light(stage, "/Sun")
    _camera(stage)
    return stage


# ─── The interface returns a value ────────────────────────────────────


@needs_usd
class TestSceneUpdateFromStage:
    def test_reads_a_synthetic_stage_with_no_renderer(self):
        from skinny import scene_intake
        update = scene_intake.read_open_stage(_populated_stage())
        assert update.scene.instances, "the sphere should have baked"
        assert len(update.scene.lights_dir) == 1
        assert update.scene.camera_override is not None
        assert update.stage is not None

    def test_empty_stage_is_allowed_explicitly(self):
        from skinny import scene_intake
        stage = _stage()
        with pytest.raises(ValueError):
            scene_intake.read_open_stage(stage)
        update = scene_intake.read_open_stage(stage, allow_empty=True)
        assert update.scene.instances == []

    def test_resync_update_carries_runtime_state_and_holds_the_camera(self):
        from skinny import scene_intake
        update = scene_intake.read_open_stage(_populated_stage())
        assert update.carry_runtime_state is True
        assert update.frame_camera == "never"
        assert update.adopt_mm_per_unit is False
        assert update.activate_label is None

    def test_replace_update_takes_the_new_scale_and_keeps_nothing(self):
        from skinny import scene_intake
        update = scene_intake.read_open_stage(
            _populated_stage(), replaces="USD: (empty)")
        assert update.carry_runtime_state is False
        assert update.adopt_mm_per_unit is True
        assert update.activate_label == "USD: (empty)"

    def test_adopted_update_frames_only_on_first_or_authored_camera(self):
        from skinny import scene_intake
        from skinny.scene import Scene
        update = scene_intake.adopt_scene(Scene())
        assert update.frame_camera == "if_first_or_authored"
        assert update.carry_runtime_state is False
        # A re-callable path must not adopt the scale on the caller's behalf.
        assert update.adopt_mm_per_unit is False

    def test_streamed_update_leaves_meshes_unbaked_for_the_caller(self):
        from skinny import scene_intake, usd_loader
        stage = _populated_stage()
        scene, prim_data, _ = usd_loader._read_open_stage(stage)
        update = scene_intake.SceneUpdate.streamed(
            scene, stage=stage, pending_prims=prim_data)
        assert update.pending_prims, "the sphere is still to bake"
        assert update.apply_control_defaults is True
        assert update.adopt_mm_per_unit is True
        assert update.frame_camera == "always"

    def test_bake_pending_produces_one_instance_per_prim(self):
        from skinny import scene_intake, usd_loader
        stage = _stage()
        _sphere(stage, "/A", translate=(0, 0, 0))
        _sphere(stage, "/B", translate=(4, 0, 0))
        _, prim_data, _ = usd_loader._read_open_stage(stage)
        baked = list(scene_intake.bake_pending(prim_data))
        assert sorted(i.prim_path for i in baked) == ["/A", "/B"]

    def test_sentinel_scale_is_not_adopted(self):
        """`120.0` is `Scene`'s default, not an opinion from the stage."""
        from skinny import scene_intake
        from skinny.scene import Scene
        scene = Scene()
        assert scene.mm_per_unit == 120.0, "the sentinel moved; update this test"
        update = scene_intake.SceneUpdate.streamed(scene, stage=None)
        assert update.adopt_mm_per_unit is True
        assert update.adopts_mm_per_unit is False
        scene.mm_per_unit = 1000.0
        assert update.adopts_mm_per_unit is True

    def test_stage_state_rides_the_update(self):
        """`_up_axis_rt`, the clock, the animation index, the skeletal handle
        and the UI controls are folded in, not fetched separately."""
        from skinny import scene_intake
        # `read_stage` opens a file; this exercises the same folding through
        # the helper it and `read_open_stage` share.
        index, clock, rt, skeletal, controls = scene_intake._stage_state(
            _populated_stage("Z"))
        assert rt is not None, "a Z-up stage needs an up-axis correction"
        assert index is not None and clock is not None
        assert controls == []
        assert skeletal is not None


# ─── Time-indexed re-read ─────────────────────────────────────────────


@needs_usd
class TestReadAtTime:
    def test_default_time_code_is_not_rounded_through_a_float(self):
        """`Usd.TimeCode.Default().GetValue()` is NaN — passing the sentinel
        straight through is the only way to ask for the default time."""
        from pxr import Usd
        from skinny import scene_intake
        stage = _populated_stage()
        sample = scene_intake.read_at_time(
            stage, Usd.TimeCode.Default(), xform_paths=["/Ball"])
        assert "/Ball" in sample.instance_transforms
        assert np.isfinite(sample.instance_transforms["/Ball"]).all()

    def test_skipping_lights_does_not_mean_no_lights(self):
        """`read_lights=False` must be distinguishable from an empty stage,
        or a caller would clear the scene's lights on every transform-only
        re-read."""
        from skinny import scene_intake
        stage = _populated_stage()
        skipped = scene_intake.read_at_time(stage, 0.0, want_lights=False)
        assert skipped.read_lights is False
        assert skipped.lights_dir == []
        read = scene_intake.read_at_time(stage, 0.0)
        assert read.read_lights is True
        assert len(read.lights_dir) == 1

    def test_up_axis_rotation_is_applied_to_every_kind(self):
        from skinny import scene_intake
        from skinny.usd_loader import _up_axis_rt
        stage = _populated_stage("Z")
        rt = _up_axis_rt("Z")
        plain = scene_intake.read_at_time(stage, 0.0, xform_paths=["/Ball"])
        turned = scene_intake.read_at_time(
            stage, 0.0, up_axis_rt=rt, xform_paths=["/Ball"])
        assert not np.allclose(
            plain.instance_transforms["/Ball"],
            turned.instance_transforms["/Ball"])
        assert not np.allclose(
            plain.lights_dir[0].direction, turned.lights_dir[0].direction)
        assert not np.allclose(
            plain.camera_override.position, turned.camera_override.position)

    def test_unknown_prim_path_is_skipped_not_raised(self):
        from skinny import scene_intake
        sample = scene_intake.read_at_time(
            _populated_stage(), 0.0, xform_paths=["/Ball", "/NotThere"])
        assert set(sample.instance_transforms) == {"/Ball"}


@needs_usd
class TestReadAtTimeMatchesPreChangeExtraction:
    """The recorded identity target (task 1.3).

    `tests/fixtures/anim_reread.json` was captured from a real headless
    renderer *before* this change, over a Z-up animated stage. `read_at_time`
    must reproduce it exactly.
    """

    @pytest.fixture(scope="class")
    def fixture(self):
        path = Path(__file__).resolve().parent / "fixtures" / "anim_reread.json"
        return json.loads(path.read_text())

    @pytest.fixture(scope="class")
    def stage(self, fixture, tmp_path_factory):
        from pxr import Usd
        p = tmp_path_factory.mktemp("anim") / "stage.usda"
        p.write_text(fixture["stage"])
        return Usd.Stage.Open(str(p))

    def test_every_time_code_reproduces_the_capture(self, fixture, stage):
        from skinny import scene_intake
        from skinny.usd_loader import _up_axis_rt
        rt = _up_axis_rt("Z")
        paths = sorted(fixture["frames"]["0"]["instances"])
        for tc in fixture["time_codes"]:
            want = fixture["frames"][f"{tc:g}"]
            got = scene_intake.read_at_time(
                stage, tc, up_axis_rt=rt, xform_paths=paths)

            for path in paths:
                np.testing.assert_allclose(
                    got.instance_transforms[path],
                    np.array(want["instances"][path]),
                    atol=1e-6, err_msg=f"{path} @ {tc}")

            assert len(got.lights_dir) == len(want["lights_dir"])
            for have, expect in zip(got.lights_dir, want["lights_dir"]):
                assert have.prim_path == expect["prim_path"]
                np.testing.assert_allclose(
                    have.direction, expect["direction"], atol=1e-6)
                np.testing.assert_allclose(
                    have.radiance, expect["radiance"], atol=1e-6)

            assert len(got.lights_sphere) == len(want["lights_sphere"])
            for have, expect in zip(got.lights_sphere, want["lights_sphere"]):
                assert have.prim_path == expect["prim_path"]
                np.testing.assert_allclose(
                    have.position, expect["position"], atol=1e-6)
                np.testing.assert_allclose(
                    have.radiance, expect["radiance"], atol=1e-6)

            cam = want["camera"]
            assert (got.camera_override is None) == (cam is None)
            if cam is not None:
                np.testing.assert_allclose(
                    got.camera_override.position, cam["position"], atol=1e-6)
                np.testing.assert_allclose(
                    got.camera_override.forward, cam["forward"], atol=1e-6)


# ─── Skinning ─────────────────────────────────────────────────────────


@needs_usd
def test_deform_skinned_mesh_returns_a_deformed_copy():
    """One intake call replaces the renderer's three-import CPU skinning
    block (joint matrices, LBS, smooth normals)."""
    from skinny import scene_intake
    from skinny.mesh import MeshSource
    from skinny.usd_loader import SkinnedMeshBinding

    from pxr import Gf

    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    tri = np.array([[0, 1, 2]], np.uint32)
    source = MeshSource(
        name="/Skinned", positions=rest.copy(), normals=np.zeros_like(rest),
        uvs=np.zeros((3, 2), np.float32), tri_idx=tri,
    )

    # One joint translating +x by 5, every vertex fully weighted to it. The
    # two queries are the pxr objects `compute_joint_matrices` calls; stubbing
    # them keeps this a test of the *composition* (matrices → LBS → normals),
    # which is what moved.
    class _SkelQuery:
        def ComputeSkinningTransforms(self, _time):
            return [Gf.Matrix4d(1).SetTranslateOnly(Gf.Vec3d(5, 0, 0))]

    class _SkinningQuery:
        def GetJointMapper(self):
            return None

        def GetGeomBindTransform(self, _time):
            return Gf.Matrix4d(1)

    binding = SkinnedMeshBinding(
        prim_path="/Skinned",
        rest_points=rest,
        rest_normals=np.zeros_like(rest),
        joint_indices=np.zeros((3, 1), np.int32),
        joint_weights=np.ones((3, 1), np.float32),
        influences=1,
        skel_query=_SkelQuery(),
        skinning_query=_SkinningQuery(),
    )
    out = scene_intake.deform_skinned_mesh(binding, source, 0.0)
    assert out is not source, "the rest source must not be mutated"
    np.testing.assert_allclose(source.positions, rest, atol=1e-6)
    np.testing.assert_allclose(out.positions[:, 0], rest[:, 0] + 5.0, atol=1e-5)
    assert out.normals.shape == rest.shape


# ─── The back-reference is gone ───────────────────────────────────────


@needs_usd
class TestIntakeHoldsNoRendererReference:
    def test_control_binding_resolves_from_values_only(self):
        """Intake is handed a scene and a stage, never a renderer."""
        import types
        from skinny import scene_intake
        from skinny.usd_loader import ControlSpec

        scene = types.SimpleNamespace(materials=[
            types.SimpleNamespace(name="default", mtlx_target_name=None),
            types.SimpleNamespace(name="Skin", mtlx_target_name=None),
        ])
        spec = ControlSpec(
            name="c", label="c", type="slider", target="material:Skin:roughness")
        binding = scene_intake.resolve_control_binding(spec, scene=scene)
        assert binding.kind == "material"
        assert binding.material_id == 1
        assert binding.input_name == "roughness"

    @pytest.mark.parametrize("target,kind,path", [
        ("renderer:env_intensity", "renderer", "env_intensity"),
        ("mtlx:base_color", "mtlx", "mtlx.base_color"),
    ])
    def test_param_targets_describe_a_path(self, target, kind, path):
        from skinny import scene_intake
        from skinny.usd_loader import ControlSpec
        spec = ControlSpec(name="c", label="c", type="slider", target=target)
        binding = scene_intake.resolve_control_binding(spec)
        assert (binding.kind, binding.param_path) == (kind, path)

    @pytest.mark.parametrize("target", [
        "material:Nope:roughness",       # no such material
        "material:Skin",                 # no input
        "usd:/Nope.inputs:intensity",    # no such prim
        "bogus:whatever",                # unknown prefix
    ])
    def test_unresolvable_targets_are_inert_not_fatal(self, target):
        import types
        from skinny import scene_intake
        from skinny.usd_loader import ControlSpec
        scene = types.SimpleNamespace(materials=[])
        spec = ControlSpec(name="c", label="c", type="slider", target=target)
        binding = scene_intake.resolve_control_binding(
            spec, scene=scene, stage=_stage())
        assert binding.kind == "inert" and binding.reason

    def test_usd_target_returns_the_live_attribute(self):
        from pxr import UsdLux
        from skinny import scene_intake
        from skinny.usd_loader import ControlSpec
        stage = _stage()
        light = UsdLux.SphereLight.Define(stage, "/Light")
        light.GetIntensityAttr().Set(100.0)
        spec = ControlSpec(
            name="c", label="c", type="slider",
            target="usd:/Light.inputs:intensity")
        binding = scene_intake.resolve_control_binding(spec, stage=stage)
        assert binding.kind == "usd"
        assert binding.attribute.Get() == pytest.approx(100.0)

    def test_intake_imports_nothing_from_the_renderer_or_params(self):
        """A back-reference reintroduced by import is caught here, not by a
        reviewer noticing it."""
        tree = ast.parse((SRC / "scene_intake.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not {m for m in imported if m.endswith("renderer")}
        assert "skinny.params" not in imported

    def test_loader_no_longer_reaches_into_the_renderer(self):
        """`usd_loader.resolve_control_binding` took a renderer, read its
        scene, called `apply_material_override`, set `_usd_live_dirty` and
        imported `skinny.params`. None of that may come back.

        Checked on the parsed tree, not the raw text: the loader's docstrings
        legitimately *mention* `renderer.apply_material_override` when
        explaining why a dict is unshared, and a substring gate would either
        fail on the prose or be loosened until it caught nothing.
        """
        tree = ast.parse((SRC / "usd_loader.py").read_text())

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                names = [
                    a.arg for a in
                    (*args.posonlyargs, *args.args, *args.kwonlyargs)
                ]
                assert "renderer" not in names, (
                    f"usd_loader.{node.name} still takes a renderer")
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.endswith("renderer")
                assert node.module != "skinny.params"
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert node.value.id != "renderer", (
                    f"usd_loader reads renderer.{node.attr}")

        assert "resolve_control_binding" not in {
            n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }


# ─── No lazy reach into intake internals ──────────────────────────────


def _function_local_imports(path: Path, module_prefix: str) -> list[str]:
    """Imports of `module_prefix` that sit inside a function body."""
    tree = ast.parse(path.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                if inner.module.startswith(module_prefix):
                    found.append(f"{path.name}:{inner.lineno} {inner.module}")
            elif isinstance(inner, ast.Import):
                for alias in inner.names:
                    if alias.name.startswith(module_prefix):
                        found.append(f"{path.name}:{inner.lineno} {alias.name}")
    return found


def test_no_function_local_imports_of_the_loader_anywhere_in_src():
    """The source gate for the 15 lazy imports this change deleted.

    A function-local import hides the dependency from the module-level import
    graph — which is exactly how the renderer came to depend on nine loader
    privates while appearing not to depend on the loader at all.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name in ("usd_loader.py", "scene_intake.py"):
            continue  # the loader itself, and its one legitimate consumer
        offenders += _function_local_imports(path, "skinny.usd_loader")
    assert offenders == [], "\n".join(offenders)


def test_renderer_declares_its_intake_dependency_at_module_level():
    tree = ast.parse((SRC / "renderer.py").read_text())
    top = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    top |= {
        alias.name
        for node in tree.body if isinstance(node, ast.Import)
        for alias in node.names
    }
    top |= {
        f"skinny.{alias.name}"
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "skinny"
        for alias in node.names
    }
    assert "skinny.scene_intake" in top
    assert "skinny.usd_loader" in top
