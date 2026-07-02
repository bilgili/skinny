"""Equal-area octahedral chart + equirect reprojection (pbrt-env-equiarea-projection)."""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import equiarea


def _grid(n=33):
    t = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(t, t)
    return np.stack([u.ravel(), v.ravel()], axis=-1)


def test_square_to_sphere_unit_length():
    p = _grid()
    d = equiarea.equal_area_square_to_sphere(p)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-6)


def test_round_trip_square_sphere_square():
    p = _grid()
    d = equiarea.equal_area_square_to_sphere(p)
    p2 = equiarea.sphere_to_equal_area_square(d)
    assert np.allclose(p, p2, atol=1e-6)


def test_round_trip_sphere_square_sphere():
    # sample directions on the sphere via a Fibonacci-ish set
    rng = np.random.default_rng(0)
    d = rng.standard_normal((500, 3))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p = equiarea.sphere_to_equal_area_square(d)
    d2 = equiarea.equal_area_square_to_sphere(p)
    assert np.allclose(d, d2, atol=1e-6)


def test_apply_axis_matches_geometry_basis_B():
    # The env reprojection MUST use the same pbrt<->skinny change-of-basis the
    # geometry import uses (`transform.B = diag(1,1,-1,1)`), otherwise the baked
    # env is rotated relative to the world the camera/meshes live in (the
    # sssdragon ground reflected a neutral band instead of the blue sky). This
    # pins _apply_axis == B[:3,:3] and guards against a regression to the old
    # Rx(+90) rotation.
    from skinny.pbrt.transform import B

    rng = np.random.default_rng(0)
    d = rng.standard_normal((64, 3))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    expect = d @ B[:3, :3].T            # B is symmetric/diagonal, so M·dᵀ
    assert np.allclose(equiarea._apply_axis(d), expect, atol=1e-12)
    # up-axis preserved (Y stays Y), handedness flipped on Z:
    assert np.allclose(equiarea._apply_axis(np.array([[0.0, 1.0, 0.0]]))[0], [0, 1, 0])
    assert np.allclose(equiarea._apply_axis(np.array([[0.0, 0.0, 1.0]]))[0], [0, 0, -1])


def test_apply_axis_is_involution():
    # B is its own inverse, so the forward and inverse maps coincide.
    rng = np.random.default_rng(1)
    d = rng.standard_normal((32, 3))
    assert np.allclose(equiarea._apply_axis(d), equiarea._apply_axis_inv(d))
    assert np.allclose(equiarea._apply_axis(equiarea._apply_axis(d)), d)


def test_axis_directions_map_to_known_square_points():
    # center of the square is +z; the four corners are -z; edge midpoints are ±x/±y
    center = equiarea.equal_area_square_to_sphere(np.array([[0.5, 0.5]]))[0]
    assert np.allclose(center, [0, 0, 1], atol=1e-6)
    for corner in ([0, 0], [1, 0], [0, 1], [1, 1]):
        d = equiarea.equal_area_square_to_sphere(np.array([[float(corner[0]), float(corner[1])]]))[0]
        assert d[2] < -0.999  # corners are the -z pole


def test_equiarea_to_equirect_shape_and_uniform():
    src = np.ones((16, 16, 3), np.float64) * 0.37
    out = equiarea.equiarea_to_equirect(src, height=16)
    assert out.shape == (16, 32, 3)
    assert np.allclose(out, 0.37, atol=1e-6)


def test_directional_consistency_delta():
    # a bright texel in the source square should reappear at the equirect pixel
    # whose shader direction round-trips back into that texel.
    edge = 32
    src = np.zeros((edge, edge, 3), np.float64)
    si, sj = 20, 8
    src[si, sj] = [100.0, 0.0, 0.0]
    H = edge
    out = equiarea.equiarea_to_equirect(src, height=H)
    # locate the brightest equirect pixel
    bi, bj = np.unravel_index(np.argmax(out[..., 0]), out[..., 0].shape)
    # map that equirect pixel back to a direction, then to square-uv
    u = (bj + 0.5) / (2 * H)
    v = (bi + 0.5) / H
    d = equiarea.equirect_uv_to_direction(np.array([[u, v]]))
    sq = equiarea.sphere_to_equal_area_square(equiarea._apply_axis(d))[0]
    src_j = sq[0] * edge - 0.5
    src_i = sq[1] * edge - 0.5
    assert abs(src_i - si) <= 1.5 and abs(src_j - sj) <= 1.5


class TestLightCtmRotation:
    """equiarea_to_equirect(world_to_light=...) bakes an authored light CTM
    rotation (nanovdb-volume-rendering: bunny-cloud `Rotate 10 1 0 0`)."""

    def _hemisphere_src(self, edge=64):
        # Equal-area square colored by LIGHT-space hemisphere: +z light = red,
        # -z light = blue.
        js = (np.arange(edge) + 0.5) / edge
        uu, vv = np.meshgrid(js, js)
        d = equiarea.equal_area_square_to_sphere(np.stack([uu, vv], axis=-1))
        src = np.zeros((edge, edge, 3))
        src[..., 0] = (d[..., 2] > 0.0)
        src[..., 2] = (d[..., 2] <= 0.0)
        return src

    def test_identity_matches_none(self):
        src = self._hemisphere_src()
        a = equiarea.equiarea_to_equirect(src, height=32)
        b = equiarea.equiarea_to_equirect(src, height=32, world_to_light=np.eye(3))
        assert np.allclose(a, b)

    def test_rotation_moves_hemisphere(self):
        src = self._hemisphere_src()
        # light->world = Rx(90deg): light +z ends up along world -y (pbrt space).
        c, s = 0.0, 1.0
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
        out = equiarea.equiarea_to_equirect(src, height=64,
                                            world_to_light=np.linalg.inv(R))
        # Sample the output at the equirect pixel whose SKINNY direction maps to
        # pbrt-world R @ +z_light: that pixel must be red.
        d_world_pbrt = R @ np.array([0.0, 0.0, 1.0])
        d_skinny = equiarea._apply_axis_inv(d_world_pbrt[None, :])[0]
        # Find the equirect pixel whose direction best matches d_skinny by
        # brute force over the output grid (no inverse-uv helper exported).
        h, w = out.shape[0], out.shape[1]
        js = (np.arange(w) + 0.5) / w
        is_ = (np.arange(h) + 0.5) / h
        uu, vv = np.meshgrid(js, is_)
        dirs = equiarea.equirect_uv_to_direction(np.stack([uu, vv], axis=-1))
        dots = (dirs * d_skinny).sum(axis=-1)
        iy, ix = np.unravel_index(np.argmax(dots), dots.shape)
        px = out[iy, ix]
        assert px[0] > 0.9 and px[2] < 0.1, px
