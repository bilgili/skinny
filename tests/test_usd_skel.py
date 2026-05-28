"""UsdSkel binding detection + extraction for skeletal animation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ELEPHANT = (
    PROJECT_ROOT / "assets" / "assets-main" / "full_assets"
    / "ElephantWithMonochord" / "SoC-ElephantWithMonochord.usdc"
)


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_usd(), reason="pxr/USD not installed")
needs_elephant = pytest.mark.skipif(
    not ELEPHANT.exists(), reason="ElephantWithMonochord asset not present"
)


@needs_elephant
class TestSkeletalExtraction:
    @pytest.fixture(scope="class")
    def scene(self):
        from pxr import Usd
        from skinny.usd_loader import extract_skeletal_bindings
        stage = Usd.Stage.Open(str(ELEPHANT))
        return extract_skeletal_bindings(stage)

    def test_has_skinning(self, scene):
        assert scene.has_skinning
        assert len(scene.meshes) == 2

    def test_joint_counts(self, scene):
        joint_counts = sorted(len(m.skel_query.GetJointOrder()) for m in scene.meshes)
        assert joint_counts == [2, 27]

    def test_influences_and_shapes(self, scene):
        by_name = {m.prim_path.split("/")[-1]: m for m in scene.meshes}
        ele = by_name["Elefant1"]
        assert ele.influences == 4
        assert ele.rest_points.shape == (1312, 3)
        assert ele.rest_normals.shape == (1312, 3)
        assert ele.joint_indices.shape == (1312, 4)
        assert ele.joint_weights.shape == (1312, 4)
        assert ele.rest_points.dtype == np.float32
        assert ele.joint_indices.dtype == np.int32

    def test_monochord_present(self, scene):
        by_name = {m.prim_path.split("/")[-1]: m for m in scene.meshes}
        assert "Monochord_Vibrator" in by_name
        assert by_name["Monochord_Vibrator"].rest_points.shape == (62, 3)


@needs_elephant
def test_animation_index_flags_skinned_stage():
    from pxr import Usd
    from skinny.usd_loader import build_animation_index
    stage = Usd.Stage.Open(str(ELEPHANT))
    idx = build_animation_index(stage)
    assert idx.has_animation
    assert len(idx.skinned_mesh_paths) == 2
    # This asset has no animated xform/light/camera tracks — only skinning.
    assert idx.xform_paths == []
    assert not idx.camera_animated


@needs_elephant
class TestJointMatrices:
    @pytest.fixture(scope="class")
    def elefant(self):
        from pxr import Usd
        from skinny.usd_loader import extract_skeletal_bindings
        stage = Usd.Stage.Open(str(ELEPHANT))
        scene = extract_skeletal_bindings(stage)
        ele = next(m for m in scene.meshes if m.prim_path.endswith("Elefant1"))
        # Keep the scene (cache + stage) alive for the class — pxr invalidates
        # skinning queries once their cache/stage is dropped.
        ele._keep_scene_alive = scene
        return ele

    def test_lbs_matches_pxr_reference(self, elefant):
        from pxr import Gf, Usd, Vt
        from skinny.usd_loader import compute_joint_matrices, lbs_points

        t = 300.0
        mats = compute_joint_matrices(elefant, t)
        mine = lbs_points(
            elefant.rest_points, elefant.joint_indices,
            elefant.joint_weights, mats,
        )
        # pxr reference in skel space
        xf = elefant.skel_query.ComputeSkinningTransforms(Usd.TimeCode(t))
        vt = Vt.Vec3fArray([Gf.Vec3f(*p) for p in elefant.rest_points.tolist()])
        assert elefant.skinning_query.ComputeSkinnedPoints(xf, vt, Usd.TimeCode(t))
        ref = np.array(vt, dtype=np.float32)

        assert np.abs(mine - ref).max() < 1e-4
        # sanity: the pose at t=300 actually differs from the rest pose
        assert np.abs(ref - elefant.rest_points).max() > 1e-3

    def test_world_transform_applied(self, elefant):
        from skinny.usd_loader import compute_joint_matrices, lbs_points

        skel_to_world = np.eye(4, dtype=np.float32)
        skel_to_world[3, :3] = (10.0, 0.0, 0.0)  # translate +10 in x (row-vector)
        mats = compute_joint_matrices(elefant, 300.0, skel_to_world=skel_to_world)
        moved = lbs_points(
            elefant.rest_points, elefant.joint_indices,
            elefant.joint_weights, mats,
        )
        base = lbs_points(
            elefant.rest_points, elefant.joint_indices, elefant.joint_weights,
            compute_joint_matrices(elefant, 300.0),
        )
        assert np.allclose(moved - base, np.array([10.0, 0.0, 0.0]), atol=1e-3)


def test_non_skel_stage_has_no_skinning():
    from pxr import Usd, UsdGeom
    from skinny.usd_loader import extract_skeletal_bindings
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/M")
    scene = extract_skeletal_bindings(stage)
    assert not scene.has_skinning
    assert scene.meshes == []
