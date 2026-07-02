"""Unit tests for pbrt ``Shape "disk"`` tessellation and import (task 3.5, change
nanovdb-volume-rendering). Both target scenes' ground planes are pbrt disks;
before this task ``_shape_geometry`` skipped them as an unsupported shape type.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import emit
from skinny.pbrt.api import import_pbrt

pytest.importorskip("pxr")
from pxr import UsdGeom  # noqa: E402


# --------------------------------------------------------------------------- #
# tessellation kernel
# --------------------------------------------------------------------------- #
def _tri_normals(pts, idx):
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    return np.cross(b - a, c - a)


def test_disk_vertex_count_and_shape():
    segments = 16
    pts, idx, nrm, uvs = emit.tessellate_disk(2.0, segments=segments)
    # two rings (outer + inner) of segments+1 vertices each
    assert pts.shape == (2 * (segments + 1), 3)
    assert nrm.shape == pts.shape
    assert uvs.shape == (pts.shape[0], 2)
    assert idx.shape == (2 * segments, 3)


def test_disk_all_vertices_at_height_plane():
    pts, _idx, _nrm, _uvs = emit.tessellate_disk(3.0, height=5.0, segments=12)
    assert np.allclose(pts[:, 2], 5.0)


def test_disk_radius_bound():
    radius = 4.5
    pts, _idx, _nrm, _uvs = emit.tessellate_disk(radius, segments=20)
    r = np.linalg.norm(pts[:, :2], axis=1)
    assert r.max() <= radius + 1e-9
    # outer ring vertices sit exactly on the radius
    assert np.allclose(r[: 21], radius)


def test_disk_normals_are_plus_z():
    pts, _idx, nrm, _uvs = emit.tessellate_disk(1.0, segments=16)
    assert np.allclose(nrm, np.array([0.0, 0.0, 1.0]))


def test_disk_winding_faces_plus_z():
    """Full disk (no inner radius): every non-degenerate triangle winds CCW
    as seen from +z, matching pbrt's un-reversed disk normal (+z)."""
    pts, idx, _nrm, _uvs = emit.tessellate_disk(1.0, segments=16)
    tri_n = _tri_normals(pts, idx)
    areas = np.linalg.norm(tri_n, axis=1)
    nondegenerate = areas > 1e-9
    assert nondegenerate.any()
    assert np.all(tri_n[nondegenerate, 2] > 0.0)


def test_disk_inner_radius_annulus_has_no_degenerate_triangles():
    pts, idx, _nrm, _uvs = emit.tessellate_disk(2.0, inner_radius=1.0, segments=16)
    r = np.linalg.norm(pts[:, :2], axis=1)
    assert r.min() >= 1.0 - 1e-9
    assert r.max() <= 2.0 + 1e-9
    tri_n = _tri_normals(pts, idx)
    areas = np.linalg.norm(tri_n, axis=1)
    assert np.all(areas > 1e-9)  # annulus: no collapsed-centre triangles
    assert np.all(tri_n[:, 2] > 0.0)


def test_disk_phimax_wedge_bounds_angular_range():
    pts, _idx, _nrm, _uvs = emit.tessellate_disk(1.0, phi_max=90.0, segments=8)
    phi = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    phi = np.where(phi < -1e-6, phi + 360.0, phi)
    assert phi.min() >= -1e-6
    assert phi.max() <= 90.0 + 1e-6


def test_disk_uv_matches_pbrt_parametrization():
    """u = phi/phiMax, v = (radius - r) / (radius - innerRadius): v=0 at the
    outer rim, v=1 at the inner rim (shapes.h Disk::InteractionFromIntersection)."""
    segments = 8
    radius, inner = 2.0, 0.5
    pts, _idx, _nrm, uvs = emit.tessellate_disk(
        radius, inner_radius=inner, segments=segments
    )
    row = segments + 1
    for j in (0, 1, segments):
        outer = uvs[j]
        inner_uv = uvs[row + j]
        assert outer == pytest.approx([j / segments, 0.0])
        assert inner_uv == pytest.approx([j / segments, 1.0])


def test_disk_default_radius_and_height_and_innerradius():
    pts_a, idx_a, _n, _uv = emit.tessellate_disk(1.0)
    pts_b, idx_b, _n, _uv = emit.tessellate_disk(1.0, height=0.0, inner_radius=0.0,
                                                  phi_max=360.0, segments=64)
    assert np.allclose(pts_a, pts_b)
    assert np.array_equal(idx_a, idx_b)


# --------------------------------------------------------------------------- #
# importer integration
# --------------------------------------------------------------------------- #
def _import(tmp_path, scene_text):
    src = tmp_path / "scene.pbrt"
    src.write_text(scene_text)
    return import_pbrt(str(src))


def _first_mesh(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(prim)
    raise AssertionError("no Mesh in stage")


_GROUND_DISK_SCENE = """
WorldBegin
AttributeBegin
  Translate 0 -1000 0
  Scale 2000 2000 2000
  Rotate -90 1 0 0
  Material "diffuse"
  Shape "disk"
AttributeEnd
"""


def test_disk_scene_emits_mesh_exact_not_skipped(tmp_path):
    stage, report = _import(tmp_path, _GROUND_DISK_SCENE)
    assert any(e.construct.startswith("shape:disk") for e in report.entries)
    assert not any(
        e.construct.startswith("shape:disk") and "unsupported" in (e.detail or "")
        for e in report.entries
    )
    assert not any(e.construct.startswith("shape:disk") and e.status == "skipped"
                   for e in report.entries)
    mesh = _first_mesh(stage)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    assert len(counts) > 0
    assert all(c == 3 for c in counts)


def test_disk_scene_transform_baked_into_points(tmp_path):
    """The disk's CTM (Translate/Scale/Rotate) is baked into world-space
    points via the same bake_world_mesh path every other shape uses, so with
    the object-space z=0 plane rotated -90 about X and scaled 2000, the
    baked points should sit near y=-1000 (translation) with world radius
    ~2000 (object radius 1 x scale 2000)."""
    stage, _report = _import(tmp_path, _GROUND_DISK_SCENE)
    mesh = _first_mesh(stage)
    pts = np.array([list(p) for p in mesh.GetPointsAttr().Get()])
    assert pts.shape[0] > 0
    # Rotate -90 about X maps object z-plane to world y=const (here translated
    # -1000), so the baked ring should cluster near y ~= -1000.
    assert np.allclose(pts[:, 1], -1000.0, atol=1e-6)
    r = np.linalg.norm(pts[:, [0, 2]], axis=1)
    assert r.max() == pytest.approx(2000.0, rel=1e-6)


def test_disk_radius_param_respected(tmp_path):
    stage, _report = _import(
        tmp_path,
        'WorldBegin\nMaterial "diffuse"\nShape "disk" "float radius" 1000\n',
    )
    mesh = _first_mesh(stage)
    pts = np.array([list(p) for p in mesh.GetPointsAttr().Get()])
    r = np.linalg.norm(pts, axis=1)
    assert r.max() == pytest.approx(1000.0, rel=1e-6)


def test_disk_gets_vertex_st(tmp_path):
    stage, _report = _import(tmp_path, _GROUND_DISK_SCENE)
    mesh = _first_mesh(stage)
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    assert pv.IsDefined()
    assert pv.GetInterpolation() == UsdGeom.Tokens.vertex


# --------------------------------------------------------------------------- #
# target scenes (surfaced by 3.4 as a pre-existing gap; re-checked here)
# --------------------------------------------------------------------------- #
def test_disney_cloud_and_bunny_cloud_disk_ground_no_longer_skipped():
    import os

    disney = os.path.expanduser("~/projects/pbrt-v4-scenes/disney-cloud/disney-cloud.pbrt")
    bunny = os.path.expanduser("~/projects/pbrt-v4-scenes/bunny-cloud/bunny-cloud.pbrt")
    for path in (disney, bunny):
        if not os.path.exists(path):
            pytest.skip("pbrt-v4-scenes corpus not available")
        _stage, report = import_pbrt(path)
        assert any(e.construct.startswith("shape:disk") for e in report.entries)
        assert not any(
            e.construct.startswith("shape:disk") and "unsupported" in (e.detail or "")
            for e in report.entries
        )
