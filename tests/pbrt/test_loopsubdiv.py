"""Unit tests for the pbrt Loop subdivision tessellation kernel."""
import numpy as np
import pytest

from skinny.pbrt.loopsubdiv import subdivide


def _tetrahedron():
    """Closed manifold (no boundary): 4 verts, 4 faces, every vertex valence 3."""
    pts = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ])
    # outward-facing windings
    idx = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return pts, idx


def _hex_fan():
    """Valence-6 interior centre at origin, six boundary spokes, all coplanar z=0."""
    pts = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = 2.0 * np.pi * k / 6.0
        pts.append([np.cos(a), np.sin(a), 0.0])
    pts = np.array(pts)
    idx = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
    return pts, idx


def _unit_square():
    """Flat square, two triangles, all four corners on the boundary, z=0."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    idx = np.array([[0, 1, 2], [0, 2, 3]])
    return pts, idx


@pytest.mark.parametrize("levels", [0, 1, 2])
def test_triangle_count_quadruples_per_level(levels):
    pts, idx = _tetrahedron()
    _, out_idx, _ = subdivide(pts, idx, levels)
    assert out_idx.shape[0] == (4 ** levels) * idx.shape[0]


def test_levels_zero_preserves_topology_count():
    pts, idx = _tetrahedron()
    out_pts, out_idx, out_nrm = subdivide(pts, idx, 0)
    assert out_idx.shape[0] == idx.shape[0]
    assert out_pts.shape[0] == pts.shape[0]
    assert out_nrm.shape == out_pts.shape


def test_planar_patch_stays_planar_with_axis_normals():
    pts, idx = _hex_fan()
    out_pts, out_idx, out_nrm = subdivide(pts, idx, 3)
    # A planar control cage subdivides to a planar limit surface.
    assert np.allclose(out_pts[:, 2], 0.0, atol=1e-9)
    # Every limit normal is the patch normal (+/- Z) for a flat patch.
    assert np.allclose(np.abs(out_nrm[:, 2]), 1.0, atol=1e-6)
    assert np.allclose(out_nrm[:, :2], 0.0, atol=1e-6)


def test_regular_interior_limit_point_on_plane():
    # The valence-6 centre is the first vertex; its limit position stays on z=0
    # and (by symmetry of the regular hexagon) at the origin in x,y.
    pts, idx = _hex_fan()
    out_pts, _, _ = subdivide(pts, idx, 1)
    # locate the centre limit vertex (closest to origin)
    d = np.linalg.norm(out_pts, axis=1)
    centre = out_pts[np.argmin(d)]
    assert np.allclose(centre, [0.0, 0.0, 0.0], atol=1e-9)


def test_boundary_patch_stays_planar_and_bounded():
    pts, idx = _unit_square()
    out_pts, _, out_nrm = subdivide(pts, idx, 2)
    # planar -> planar limit, axis normals
    assert np.allclose(out_pts[:, 2], 0.0, atol=1e-9)
    assert np.allclose(np.abs(out_nrm[:, 2]), 1.0, atol=1e-6)
    # boundary rules pull inward, never outside the original convex hull
    assert out_pts[:, 0].min() >= -1e-9 and out_pts[:, 0].max() <= 1.0 + 1e-9
    assert out_pts[:, 1].min() >= -1e-9 and out_pts[:, 1].max() <= 1.0 + 1e-9


def test_malformed_indices_raise():
    pts, _ = _tetrahedron()
    with pytest.raises(ValueError):
        subdivide(pts, np.array([0, 1, 2, 3]), 1)  # length not a multiple of 3


def test_empty_input_raises():
    with pytest.raises(ValueError):
        subdivide(np.zeros((0, 3)), np.zeros((0, 3), dtype=int), 1)


# --------------------------------------------------------------------------- #
# importer integration
# --------------------------------------------------------------------------- #
_TETRA_LOOPSUBDIV = (
    'Camera "perspective" "float fov" 40\n'
    "WorldBegin\n"
    'Material "diffuse"\n'
    'Shape "loopsubdiv"\n'
    '    "integer levels" [ 1 ]\n'
    '    "integer indices" [ 0 1 2  0 3 1  0 2 3  1 3 2 ]\n'
    '    "point3 P" [ 1 1 1  -1 -1 1  -1 1 -1  1 -1 -1 ]\n'
)


def test_loopsubdiv_imported_as_exact(tmp_path):
    from pxr import UsdGeom

    from skinny.pbrt.api import import_pbrt

    scene = tmp_path / "s.pbrt"
    scene.write_text(_TETRA_LOOPSUBDIV)
    stage, report = import_pbrt(str(scene), out=str(tmp_path / "s.usda"))

    # reported exact, never skipped
    assert any(e.construct.startswith("shape:loopsubdiv") for e in report.entries)
    assert not any(
        e.construct.startswith("shape:loopsubdiv") and "unsupported" in (e.detail or "")
        for e in report.entries
    )
    # one Loop-refinement level -> 4x the 4 cage faces
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes
    counts = meshes[0].GetFaceVertexCountsAttr().Get()
    assert len(counts) == 16


def test_loopsubdiv_malformed_is_skipped_not_fatal(tmp_path):
    from skinny.pbrt.api import import_pbrt

    scene = tmp_path / "s.pbrt"
    scene.write_text(
        'Camera "perspective" "float fov" 40\n'
        "WorldBegin\n"
        'Material "diffuse"\n'
        'Shape "loopsubdiv" "point3 P" [ 0 0 0  1 0 0  0 1 0 ]\n'  # no indices
    )
    # must not raise
    _stage, report = import_pbrt(str(scene), out=str(tmp_path / "s.usda"))
    assert any(
        e.construct.startswith("shape:loopsubdiv") and "malformed" in (e.detail or "")
        for e in report.entries
    )


def test_killeroo_simple_bodies_not_skipped(tmp_path):
    """killeroo bodies are loopsubdiv shapes; the real scene must import them."""
    import os

    from skinny.pbrt.api import import_pbrt

    scene = os.path.expanduser("~/projects/pbrt-v4-scenes/killeroos/killeroo-simple.pbrt")
    if not os.path.exists(scene):
        pytest.skip("pbrt-v4-scenes corpus not available")
    stage, report = import_pbrt(scene, out=str(tmp_path / "killeroo.usda"))
    assert not any(
        e.construct.startswith("shape:loopsubdiv") and "unsupported" in (e.detail or "")
        for e in report.entries
    )
    assert any(e.construct.startswith("shape:loopsubdiv") for e in report.entries)
