"""Up-axis correction on USD load + camera hero-angle framing."""

from __future__ import annotations

import numpy as np
import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


class TestUpAxisRotation:
    def test_y_up_returns_none(self):
        from skinny.usd_loader import _up_axis_rt
        assert _up_axis_rt("Y") is None

    def test_z_up_matrix_maps_z_to_y(self):
        from skinny.usd_loader import _up_axis_rt
        rt = _up_axis_rt("Z")
        assert rt is not None
        assert rt.shape == (3, 3)
        # Row-vector right-multiply: +Z world axis maps to +Y.
        np.testing.assert_allclose(np.array([0, 0, 1], np.float32) @ rt,
                                   np.array([0, 1, 0], np.float32), atol=1e-6)
        # +Y maps to -Z.
        np.testing.assert_allclose(np.array([0, 1, 0], np.float32) @ rt,
                                   np.array([0, 0, -1], np.float32), atol=1e-6)
        # +X unchanged.
        np.testing.assert_allclose(np.array([1, 0, 0], np.float32) @ rt,
                                   np.array([1, 0, 0], np.float32), atol=1e-6)


@needs_usd
class TestUpAxisCorrectionOnLoad:
    def _z_up_stage_with_tall_mesh(self):
        """A stage where geometry is tall along +Z (Z-up), 1 unit thin in X/Y."""
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        mesh = UsdGeom.Mesh.Define(stage, "/m")
        # A thin column standing along +Z: spans z in [0, 4], x/y in [-0.5, 0.5].
        pts = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0),
               (-0.5, 0.5, 0.0), (-0.5, -0.5, 4.0), (0.5, -0.5, 4.0),
               (0.5, 0.5, 4.0), (-0.5, 0.5, 4.0)]
        mesh.GetPointsAttr().Set([(float(x), float(y), float(z)) for x, y, z in pts])
        mesh.GetFaceVertexCountsAttr().Set([4, 4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 6, 7])
        return stage

    def test_z_up_mesh_stands_along_y_after_load(self):
        from skinny.usd_loader import load_scene_from_stage
        stage = self._z_up_stage_with_tall_mesh()
        scene = load_scene_from_stage(stage)
        amin, amax = scene.world_bounds()
        ext = amax - amin
        # Before correction the long axis was Z (~4). After correction the
        # tall extent must be on Y, and Z must be the short (~1) axis.
        assert ext[1] > 3.0, f"expected tall Y extent, got {ext}"
        assert ext[2] < 1.5, f"expected short Z extent, got {ext}"

    def test_y_up_mesh_unchanged(self):
        from pxr import Usd, UsdGeom
        from skinny.usd_loader import load_scene_from_stage
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        mesh = UsdGeom.Mesh.Define(stage, "/m")
        pts = [(-0.5, 0.0, -0.5), (0.5, 0.0, -0.5), (0.5, 0.0, 0.5),
               (-0.5, 0.0, 0.5), (-0.5, 4.0, -0.5), (0.5, 4.0, -0.5),
               (0.5, 4.0, 0.5), (-0.5, 4.0, 0.5)]
        mesh.GetPointsAttr().Set([(float(x), float(y), float(z)) for x, y, z in pts])
        mesh.GetFaceVertexCountsAttr().Set([4, 4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 6, 7])
        scene = load_scene_from_stage(stage)
        amin, amax = scene.world_bounds()
        ext = amax - amin
        assert ext[1] > 3.0 and ext[2] < 1.5, f"Y-up scene should be untouched, got {ext}"


@needs_usd
class TestCameraOverrideCorrection:
    def test_z_up_camera_forward_corrected(self):
        from pxr import Usd, UsdGeom, Gf
        from skinny.usd_loader import _read_open_stage
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        # Mesh so the stage has geometry (loader requires usable geometry).
        mesh = UsdGeom.Mesh.Define(stage, "/m")
        mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 0, 1)])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
        # Camera with identity xform: USD forward is local -Z = world (0,0,-1)
        # in a Z-up stage. After correction it rotates by Rᵀ to (0,-1,0).
        UsdGeom.Camera.Define(stage, "/cam")
        partial_scene, _prim_data, _ = _read_open_stage(stage)
        ov = partial_scene.camera_override
        assert ov is not None
        # A Z-up camera with identity xform looks down local -Z = world (0,0,-1).
        # Correction maps it by Rᵀ: (0,0,-1) @ Rᵀ = -Rᵀ[2] = (0,-1,0).
        np.testing.assert_allclose(ov.forward,
                                   np.array([0, -1, 0], np.float32), atol=1e-5)
