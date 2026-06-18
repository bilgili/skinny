"""Tests for pbrt transform algebra + the handedness bridge (task 3.1/3.3)."""

from __future__ import annotations

import numpy as np

from skinny.pbrt import transform as T


def test_pbrt_array_is_column_major_translate():
    # column-major: translation lives in the last column -> values 12,13,14
    m = T.from_pbrt_array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1])
    assert np.allclose(T.transform_point(m, [0, 0, 0]), [5, 6, 7])


def test_rotate_z_90_maps_x_to_y():
    m = T.rotate(90, 0, 0, 1)
    assert np.allclose(T.transform_point(m, [1, 0, 0]), [0, 1, 0], atol=1e-9)


def test_lookat_world_to_camera_sends_eye_to_origin():
    c2w = T.look_at([0, 0, 5], [0, 0, 0], [0, 1, 0])
    world_to_cam = T.invert(c2w)
    assert np.allclose(T.transform_point(world_to_cam, [0, 0, 5]), [0, 0, 0], atol=1e-9)


def test_change_of_basis_flips_z_only():
    assert np.allclose(T.transform_point(T.B, [1, 2, 3]), [1, 2, -3])


def test_to_skinny_preserves_determinant_sign():
    # B @ M @ B is a similarity transform: det of the 3x3 is unchanged
    m = T.translate(1, 2, 3) @ T.scale(2, 1, 1)
    assert np.sign(np.linalg.det(T.to_skinny(m)[:3, :3])) == np.sign(np.linalg.det(m[:3, :3]))


def test_bake_through_B_is_orientation_reversing():
    # M_bake = B @ CTM flips winding for a normal (det>0) pbrt transform
    assert T.is_orientation_reversing(T.B @ T.identity())


def test_normal_transform_under_mirror_flips():
    n = T.transform_normals(T.B, [[0, 0, 1]])
    assert np.allclose(n[0], [0, 0, -1])
