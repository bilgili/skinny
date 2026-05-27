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


def _have_renderer() -> bool:
    # skinny.renderer imports `vulkan` at module scope, which raises without
    # the Vulkan SDK on the dynamic-library path. These camera-logic tests are
    # pure CPU but can't import the module without it.
    try:
        import skinny.renderer  # noqa: F401
        return True
    except Exception:
        return False


needs_renderer = pytest.mark.skipif(
    not _have_renderer(), reason="skinny.renderer import unavailable (no Vulkan SDK)"
)


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
        from pxr import Usd, UsdGeom
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


@needs_renderer
class TestHeroOrientation:
    def test_hero_angles_applied(self):
        from skinny.renderer import OrbitCamera, _hero_yaw_pitch
        yaw, pitch = _hero_yaw_pitch()
        np.testing.assert_allclose(yaw, np.radians(30.0), atol=1e-6)
        np.testing.assert_allclose(pitch, np.radians(15.0), atol=1e-6)
        # Camera sits above and to the side of the target, looking down.
        cam = OrbitCamera()
        cam.target = np.array([0.0, 0.0, 0.0], np.float32)
        cam.distance = 5.0
        cam.yaw, cam.pitch = yaw, pitch
        pos = cam.position
        assert pos[1] > 0.0, "camera should be elevated (pitch>0)"
        assert pos[0] > 0.0, "camera should be turned to +X side (yaw>0)"


@needs_renderer
class TestResetReframes:
    class _Stub:
        """Minimal stand-in exposing only what reset_camera reads/writes."""
        def __init__(self):
            from skinny.renderer import OrbitCamera, FreeCamera
            self.orbit_camera = OrbitCamera()
            self.free_camera = FreeCamera()
            self.camera_mode = "free"
            self._usd_scene = None
            self._mesh_sources = []
            self._framed = None
            self._refreshed = False

        def _frame_camera_to_scene(self, scene):
            self._framed = ("scene", scene)

        def _frame_camera_to_mesh(self, src):
            self._framed = ("mesh", src)

        def _apply_camera_override(self, scene):
            pass

        def _refresh_camera_node(self):
            self._refreshed = True

    def test_reset_frames_usd_scene(self):
        from skinny.renderer import Renderer
        stub = self._Stub()
        stub._usd_scene = object()
        Renderer.reset_camera(stub)
        assert stub._framed == ("scene", stub._usd_scene)
        assert stub.camera_mode == "orbit"
        assert stub._refreshed

    def test_reset_frames_obj_mesh(self):
        from skinny.renderer import Renderer
        stub = self._Stub()
        src = object()
        stub._mesh_sources = [src]
        Renderer.reset_camera(stub)
        assert stub._framed == ("mesh", src)
        assert stub.camera_mode == "orbit"

    def test_reset_default_when_nothing_loaded(self):
        from skinny.renderer import Renderer
        stub = self._Stub()
        Renderer.reset_camera(stub)
        assert stub._framed is None
        assert stub.camera_mode == "orbit"


@needs_renderer
class TestDistanceCap:
    def test_cap_floor_and_scaling(self):
        from skinny.renderer import _orbit_distance_cap
        assert _orbit_distance_cap(2.0) == 50.0       # small scene → 50 floor
        assert _orbit_distance_cap(5.0) == 50.0       # boundary: 10×5 == 50
        assert _orbit_distance_cap(20.0) == 200.0     # large scene → 10×longest

    def test_frame_mesh_sets_cap(self):
        import types
        from skinny.renderer import Renderer, OrbitCamera
        stub = types.SimpleNamespace(orbit_camera=OrbitCamera())
        # A 40-unit-tall mesh: longest dim 40 → cap = max(50, 400) = 400.
        positions = np.array(
            [[-1, 0, -1], [1, 0, 1], [-1, 40, -1], [1, 40, 1]], dtype=np.float32
        )
        source = types.SimpleNamespace(positions=positions)
        Renderer._frame_camera_to_mesh(stub, source)
        assert stub.orbit_camera.max_distance == 400.0
        assert stub.orbit_camera.distance <= 400.0

    def test_set_distance_grows_max(self):
        from skinny.renderer import OrbitCamera
        cam = OrbitCamera()                  # max_distance defaults to 50
        cam.set_distance(120.0)
        assert cam.distance == 120.0
        assert cam.max_distance == 120.0     # grew to fit the larger value
        cam.set_distance(10.0)               # within ceiling → does not shrink
        assert cam.distance == 10.0
        assert cam.max_distance == 120.0

    def test_set_distance_clamps_floor_and_guard(self):
        from skinny.renderer import OrbitCamera
        cam = OrbitCamera()
        cam.set_distance(-5.0)
        assert cam.distance == 0.5           # lower floor
        cam.set_distance(1e12)
        assert cam.distance == 1e9           # upper degeneracy guard
        assert cam.max_distance == 1e9

    def test_zoom_grows_past_ceiling(self):
        from skinny.renderer import OrbitCamera
        cam = OrbitCamera()
        cam.max_distance = 200.0
        cam.distance = 199.0
        for _ in range(50):                  # zoom-out: distance *= 1.1 each step
            cam.zoom(-1.0)
        assert cam.distance > 200.0          # grew past the old ceiling
        assert cam.max_distance == cam.distance

    def test_scene_graph_slider_uses_max_distance(self):
        from skinny.renderer import OrbitCamera
        from skinny.scene_graph import SceneGraphNode, inject_renderer_camera
        cam = OrbitCamera()
        cam.max_distance = 160.0
        root = SceneGraphNode(path="/", name="root", type_name="Scope")
        inject_renderer_camera(root, cam, "orbit")
        synth = next(c for c in root.children if c.path == "/Skinny/MainCamera")
        dist = next(p for p in synth.properties if p.name == "distance")
        assert dist.metadata["max"] == 160.0

    def test_apply_camera_param_distance_grows(self):
        import types
        from skinny.renderer import Renderer, OrbitCamera
        cam = OrbitCamera()                  # max_distance defaults to 50
        stub = types.SimpleNamespace(
            camera=cam, camera_mode="orbit", _material_version=0,
        )
        Renderer.apply_camera_param(stub, "distance", 333.0)
        assert cam.distance == 333.0
        assert cam.max_distance == 333.0     # grew via set_distance

    def test_restore_honors_large_distance(self):
        import types
        from skinny.renderer import OrbitCamera
        from skinny.app import _apply_saved_camera
        stub = types.SimpleNamespace(orbit_camera=OrbitCamera(), free_camera=None)
        _apply_saved_camera(stub, {"orbit": {"distance": 250.0}})
        assert stub.orbit_camera.distance == 250.0
        assert stub.orbit_camera.max_distance >= 250.0


@needs_renderer
class TestStreamingReframe:
    """The async USD streaming poll must re-frame the camera after geometry
    streams in — the metadata-phase frame runs before instances exist, so it
    early-returns on empty bounds and the camera keeps its defaults."""

    def _stub(self, scene):
        import queue
        import threading
        import types
        bake_done = threading.Event()
        bake_done.set()
        iq = queue.Queue()
        stub = types.SimpleNamespace(
            _usd_bake_done=bake_done,
            _usd_metadata_queue=queue.Queue(),   # empty; Phase 1 skipped (scene set)
            _usd_instance_queue=iq,
            _usd_scene=scene,
            _scene_graph=None,
            _usd_model_index=0,
            models=["USD: x"],
            _usd_uploaded_count=0,
            framed=[],
            refreshed=[],
        )
        stub._is_usd_active = lambda: True
        stub._upload_usd_scene = lambda: None
        stub._frame_camera_to_scene = lambda s: stub.framed.append(s)
        stub._refresh_camera_node = lambda: stub.refreshed.append(True)
        return stub, iq

    def test_reframe_on_streaming_complete(self):
        import types
        from skinny.renderer import Renderer
        from skinny.scene import Scene
        scene = Scene()
        stub, iq = self._stub(scene)
        iq.put(types.SimpleNamespace(name="m0", enabled=True))
        iq.put(types.SimpleNamespace(name="m1", enabled=True))
        Renderer._poll_usd_streaming(stub)
        assert len(scene.instances) == 2, "instances should be drained into the scene"
        assert stub.framed == [scene], "must re-frame with the populated scene"
        assert stub.refreshed == [True], "must refresh the camera node after re-framing"
