"""Authored-camera up/roll fidelity (pbrt-camera-up-axis).

A pbrt camera whose up vector is not ≈ +Y (the pbrt Z-up convention, e.g.
sssdragon) must reconstruct on the authored basis, not a hardcoded world-up of
(0,1,0). These are CPU/math tests — no GPU.
"""

from __future__ import annotations

import numpy as np

from skinny.renderer import _look_at, OrbitCamera, Renderer
from skinny.scene import CameraOverride


def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class TestLookAtHonorsWorldUp:
    def test_world_up_param_changes_basis(self):
        # Oblique forward so neither +Y nor +Z is parallel to it.
        fwd = _norm([0.7, 0.0, -0.7]).astype(np.float32)
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_y = _look_at(pos, fwd, world_up=np.array([0.0, 1.0, 0.0], np.float32))
        v_z = _look_at(pos, fwd, world_up=np.array([0.0, 0.0, 1.0], np.float32))
        up_y = v_y[:3, 1]
        up_z = v_z[:3, 1]
        # Different world-up ⇒ different camera up row.
        assert not np.allclose(up_y, up_z, atol=1e-4)
        # The Z-up basis' up row carries a larger |z| than the Y-up one.
        assert abs(up_z[2]) > abs(up_y[2])

    def test_default_world_up_is_y(self):
        # Back-compat: omitting world_up reproduces the prior (0,1,0) basis.
        fwd = _norm([0.3, -0.2, -0.9]).astype(np.float32)
        pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        v_default = _look_at(pos, fwd)
        v_y = _look_at(pos, fwd, world_up=np.array([0.0, 1.0, 0.0], np.float32))
        assert np.allclose(v_default, v_y, atol=1e-7)

    def test_degenerate_up_parallel_to_forward(self):
        # up ∥ forward must not yield NaN / zero rows.
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        pos = np.zeros(3, dtype=np.float32)
        v = _look_at(pos, fwd, world_up=np.array([0.0, 0.0, 1.0], np.float32))
        assert np.isfinite(v).all()
        right = v[:3, 0]
        assert np.linalg.norm(right) > 0.5  # non-degenerate right axis


class TestCameraOverrideCarriesUp:
    def test_override_has_up_field_default_y(self):
        ov = CameraOverride(
            position=np.array([0, 0, 5], np.float32),
            forward=np.array([0, 0, -1], np.float32),
            focal_length_mm=50.0,
            vertical_aperture_mm=24.0,
        )
        assert np.allclose(_norm(ov.up), [0.0, 1.0, 0.0], atol=1e-6)

    def test_override_to_orbit_propagates_up_into_view(self):
        # A Z-up authored camera: the resulting OrbitCamera view must honor +Z up.
        up = _norm([0.0, 0.0, 1.0]).astype(np.float32)
        fwd = _norm([0.7, 0.0, -0.2]).astype(np.float32)
        ov = CameraOverride(
            position=np.array([2.0, 0.0, 1.0], np.float32),
            forward=fwd,
            up=up,
            focal_length_mm=35.0,
            vertical_aperture_mm=24.0,
        )
        cam = OrbitCamera()
        scene = type("S", (), {"world_bounds": staticmethod(lambda: None)})()
        # _override_to_orbit uses no instance state — call unbound with self=None.
        Renderer._override_to_orbit(None, cam, ov, scene)
        assert np.allclose(_norm(cam.up), up, atol=1e-5)
        view = cam.view_matrix()
        up_row = view[:3, 1]
        # The authored +Z up dominates the camera up row, not +Y.
        assert abs(up_row[2]) > abs(up_row[1])

    def test_yup_override_unchanged(self):
        # Back-compat: a Y-up authored camera produces the same view as before
        # (up defaults to +Y; behavior byte-identical).
        fwd = _norm([0.0, -0.2, -0.98]).astype(np.float32)
        ov = CameraOverride(
            position=np.array([0.0, 1.0, 4.0], np.float32),
            forward=fwd,
            focal_length_mm=50.0,
            vertical_aperture_mm=24.0,
        )
        cam = OrbitCamera()
        scene = type("S", (), {"world_bounds": staticmethod(lambda: None)})()
        Renderer._override_to_orbit(None, cam, ov, scene)
        assert np.allclose(_norm(cam.up), [0.0, 1.0, 0.0], atol=1e-6)
        view = cam.view_matrix()
        # Up row is Y-dominant for a Y-up camera.
        assert abs(view[:3, 1][1]) > abs(view[:3, 1][2])
