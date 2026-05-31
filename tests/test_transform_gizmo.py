"""Unit tests for the transform gizmo's mode cycle and per-mode drag math.

Pure: no Vulkan, no display. A deterministic identity view/proj makes the
screen projection trivial — world X → +px, world Y → -py (screen y is down),
w = 1 — so translate/rotate drag results can be asserted by hand.
"""

from __future__ import annotations

import math

import numpy as np

from skinny.gizmo import (
    GizmoMode,
    TransformGizmo,
    _rodrigues,
    is_local,
    is_translate,
)
from skinny.scene_graph import compose_trs_matrix

# Identity view & proj: clip = world (w=1); see module docstring.
_I4 = np.eye(4, dtype=np.float64)
_W = _H = 100


def _fresh(mode: GizmoMode, basis=None) -> TransformGizmo:
    g = TransformGizmo()
    g.mode = mode
    g.set_target(0, np.zeros(3, dtype=np.float32), basis)
    return g


class TestModeCycle:
    def test_cycle_is_index_plus_one_mod_four_grouped_by_type(self):
        g = TransformGizmo()
        assert g.mode == GizmoMode.ROTATE_WORLD
        assert g.cycle_mode() == GizmoMode.ROTATE_LOCAL
        assert g.cycle_mode() == GizmoMode.TRANSLATE_WORLD
        assert g.cycle_mode() == GizmoMode.TRANSLATE_LOCAL
        assert g.cycle_mode() == GizmoMode.ROTATE_WORLD  # wraps

    def test_type_and_space_predicates(self):
        assert not is_translate(GizmoMode.ROTATE_WORLD)
        assert not is_translate(GizmoMode.ROTATE_LOCAL)
        assert is_translate(GizmoMode.TRANSLATE_WORLD)
        assert is_translate(GizmoMode.TRANSLATE_LOCAL)
        assert not is_local(GizmoMode.ROTATE_WORLD)
        assert is_local(GizmoMode.ROTATE_LOCAL)
        assert not is_local(GizmoMode.TRANSLATE_WORLD)
        assert is_local(GizmoMode.TRANSLATE_LOCAL)

    def test_cycle_is_noop_while_dragging(self):
        g = _fresh(GizmoMode.ROTATE_WORLD)
        g.begin_drag("x", 50, 50, _I4, _I4, _W, _H, _I4)
        assert g.is_dragging
        assert g.cycle_mode() == GizmoMode.ROTATE_WORLD  # unchanged
        g.end_drag()
        assert g.cycle_mode() == GizmoMode.ROTATE_LOCAL


class TestTranslateDrag:
    def test_world_translate_moves_only_dragged_axis(self):
        g = _fresh(GizmoMode.TRANSLATE_WORLD)
        g.begin_drag("x", 50, 50, _I4, _I4, _W, _H, _I4)
        # 1 world unit projects to 50 px (half-width); drag +25 px → +0.5 world.
        t, r, s = g.update_drag(75, 50, _I4, _I4, _W, _H)
        assert t == (0.5, 0.0, 0.0)
        assert r == (0.0, 0.0, 0.0)
        assert s == (1.0, 1.0, 1.0)

    def test_local_translate_moves_along_instance_axis(self):
        # Local X points along world +Y (basis row 0 = [0,1,0]).
        basis = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        g = _fresh(GizmoMode.TRANSLATE_LOCAL, basis)
        g.begin_drag("x", 50, 50, _I4, _I4, _W, _H, _I4)
        # Local +X projects to screen -y; dragging up 25 px → +0.5 along world Y.
        t, _, _ = g.update_drag(50, 25, _I4, _I4, _W, _H)
        assert t[0] == 0.0
        assert math.isclose(t[1], 0.5, abs_tol=1e-9)
        assert t[2] == 0.0


class TestRotateCompose:
    def test_local_basis_identity_matches_world(self):
        gw = _fresh(GizmoMode.ROTATE_WORLD)
        gl = _fresh(GizmoMode.ROTATE_LOCAL, np.eye(3))
        # With an identity local basis the drag axis is canonical in both.
        assert np.allclose(gw._axis_dir("x", drag=True), [1, 0, 0])
        assert np.allclose(gl._axis_dir("x", drag=True), [1, 0, 0])

    def test_local_axis_follows_basis_row(self):
        basis = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        g = _fresh(GizmoMode.ROTATE_LOCAL, basis)
        assert np.allclose(g._axis_dir("x"), [0, 1, 0])
        assert np.allclose(g._axis_dir("y"), [-1, 0, 0])

    def test_compose_builds_rotation_about_world_axis(self):
        g = TransformGizmo()
        n = np.array([0.0, 0.0, 1.0])
        delta = 0.5
        euler = g._compose_rotation((0.0, 0.0, 0.0), n, delta)
        r_row = compose_trs_matrix((0, 0, 0), euler, (1, 1, 1))[:3, :3].astype(np.float64)
        # Row-vector rotation transposes to the column-vector form, which must
        # equal a Rodrigues rotation about n (start orientation is identity).
        assert np.allclose(r_row.T, _rodrigues(n, delta), atol=1e-9)
