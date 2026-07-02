"""USD authoring for the pbrt importer.

Translates the :class:`~skinny.pbrt.state.PbrtScene` IR into a USD stage that
skinny's existing ``usd_loader`` consumes unchanged:

* shapes  -> ``UsdGeom.Mesh`` with **baked world-space points** and identity
  transform (sidesteps USD handedness; winding flipped when the bake matrix
  ``B @ CTM`` is orientation-reversing, see :mod:`~skinny.pbrt.transform`),
* materials -> ``UsdShade`` UsdPreviewSurface networks (mapping in
  :mod:`~skinny.pbrt.materials`),
* lights  -> ``UsdLux`` lights / emissive meshes (mapping in
  :mod:`~skinny.pbrt.lights`),
* camera  -> ``UsdGeom.Camera`` (mapping in :mod:`~skinny.pbrt.camera`).

USD uses a row-vector convention (translation in the last *row*), so numpy
column-vector matrices are transposed by :func:`to_gf_matrix`.
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol

from . import transform as T

# Stage unit declared on every imported pbrt stage. The loader derives
# `mm_per_unit = metersPerUnit * 1000`, so this fixes `mm_per_unit = 1000`. It is
# the single source of truth for the stage unit: anything that must cancel that
# factor (e.g. the subsurface medium optical density in `media.subsurface_overrides`)
# multiplies by this constant rather than a bare literal.
PBRT_STAGE_METERS_PER_UNIT = 1.0


def new_stage(path: str | None = None) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(path) if path else Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, PBRT_STAGE_METERS_PER_UNIT)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    return stage


def to_gf_matrix(m: np.ndarray) -> Gf.Matrix4d:
    """numpy column-vector 4x4 -> USD row-vector ``Gf.Matrix4d`` (transposed)."""
    mt = np.asarray(m, dtype=np.float64).T
    return Gf.Matrix4d(*mt.flatten().tolist())


def sanitize(name: str) -> str:
    """Make a string a valid USD prim name."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    s = "".join(out)
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    return s


def bake_world_mesh(points_local: np.ndarray, indices: np.ndarray, ctm: np.ndarray,
                    normals_local=None, reverse: bool = False):
    """Bake a triangle mesh into world space through ``M = B @ CTM``.

    Returns (world_points, world_indices, world_normals_or_None). Winding is
    flipped when the bake matrix is orientation-reversing XOR *reverse*.
    """
    m_bake = T.B @ ctm
    pts = T.transform_points(m_bake, points_local)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    flip = T.is_orientation_reversing(m_bake) ^ bool(reverse)
    if flip:
        idx = idx[:, ::-1].copy()
    nrm = None
    if normals_local is not None:
        nrm = T.transform_normals(m_bake, normals_local)
        if reverse:
            nrm = -nrm
    return pts, idx, nrm


def add_mesh(stage, path, points, indices, *, normals=None, uvs=None,
             uv_interpolation="vertex") -> UsdGeom.Mesh:
    """Author a triangle mesh. ``uvs`` is emitted as ``primvars:st``; pass
    ``uv_interpolation="faceVarying"`` for per-face-vertex UVs (one row per entry
    of ``faceVertexIndices``) or ``"vertex"`` (default, one row per point)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts.tolist()])
    mesh.CreateFaceVertexCountsAttr([3] * idx.shape[0])
    mesh.CreateFaceVertexIndicesAttr(idx.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if normals is not None:
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in nrm.tolist()])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uvs is not None:
        uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        interp = (UsdGeom.Tokens.faceVarying if uv_interpolation == "faceVarying"
                  else UsdGeom.Tokens.vertex)
        primvars = UsdGeom.PrimvarsAPI(mesh)
        st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interp)
        st.Set([Gf.Vec2f(*t) for t in uv.tolist()])
    return mesh


# pbrt Triangle default UVs when the mesh carries no uv array: each triangle's
# three vertices map to (0,0),(1,0),(1,1) (shapes.h:897). Authored faceVarying.
_DEFAULT_TRI_UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])


def default_triangle_uvs(num_tris: int) -> np.ndarray:
    """faceVarying UV rows (3*num_tris, 2) of pbrt's default per-triangle UVs."""
    return np.tile(_DEFAULT_TRI_UV, (int(num_tris), 1))


def add_volume(stage, path, ctm: np.ndarray, *, grid_asset: str, field_name: str = "density"
               ) -> UsdVol.Volume:
    """Author a ``UsdVol.Volume`` + ``OpenVDBAsset`` field prim for a nanovdb medium.

    *ctm* is the medium's pbrt-space CTM (captured at ``MakeNamedMedium``,
    :attr:`~skinny.pbrt.state.PbrtMedium.ctm`); it is converted to skinny/USD
    space through :func:`skinny.pbrt.transform.to_skinny` (``B @ M @ B``), the
    same helper the camera transform uses — never hand-rolled (see the
    module docstring and :func:`bake_world_mesh`, which uses the point-bake
    variant ``B @ CTM`` since it transforms baked points, not a standalone
    matrix). *grid_asset* is the absolute ``.nvdb`` path (already resolved
    against the scene directory by
    :func:`skinny.pbrt.media.heterogeneous_overrides`).
    """
    volume = UsdVol.Volume.Define(stage, path)
    UsdGeom.Xformable(volume).AddTransformOp().Set(to_gf_matrix(T.to_skinny(ctm)))
    field_path = f"{path}/{field_name}"
    field = UsdVol.OpenVDBAsset.Define(stage, field_path)
    field.CreateFilePathAttr(Sdf.AssetPath(grid_asset))
    field.CreateFieldNameAttr(field_name)
    volume.CreateFieldRelationship(field_name, field.GetPath())
    return volume


def tessellate_sphere(radius: float, *, segments: int = 32, rings: int = 16):
    """UV-sphere -> (points (N,3), tri-indices (M,3), normals (N,3), uvs (N,2)).

    UVs match pbrt's sphere parametrization (``u = phi/phiMax``,
    ``v = (theta-thetaMin)/(thetaMax-thetaMin)``): vertex ``(i,j)`` gets
    ``u = j/segments`` and ``v = 1 - i/rings`` (theta = pi*i/rings)."""
    pts = []
    nrm = []
    uvs = []
    for i in range(rings + 1):
        theta = np.pi * i / rings  # 0..pi (polar)
        st, ct = np.sin(theta), np.cos(theta)
        for j in range(segments + 1):
            phi = 2.0 * np.pi * j / segments
            sp, cp = np.sin(phi), np.cos(phi)
            n = np.array([st * cp, st * sp, ct])
            pts.append(n * radius)
            nrm.append(n)
            uvs.append([j / segments, 1.0 - i / rings])
    idx = []
    row = segments + 1
    for i in range(rings):
        for j in range(segments):
            a = i * row + j
            b = a + row
            idx.append([a, b, a + 1])
            idx.append([a + 1, b, b + 1])
    return np.array(pts), np.array(idx, dtype=np.int64), np.array(nrm), np.array(uvs)


def tessellate_disk(radius: float, *, height: float = 0.0, inner_radius: float = 0.0,
                    phi_max: float = 360.0, segments: int = 64):
    """pbrt ``Shape "disk"`` -> (points (N,3), tri-indices (M,3), normals (N,3),
    uvs (N,2)), tessellated in the object-space ``z = height`` plane.

    Two concentric rings of ``segments + 1`` vertices (outer at *radius*, inner
    at *inner_radius*) are quad-stripped, matching pbrt's disk parametrization
    (shapes.h ``Disk::InteractionFromIntersection``): ``u = phi / phiMax``,
    ``v = (radius - r) / (radius - innerRadius)`` (``v = 0`` at the outer rim,
    ``v = 1`` at the inner rim / centre). The plain-disk case (``inner_radius ==
    0``) still emits two rings — the inner one collapsed to a single point
    repeated *segments + 1* times — so the winding/index pattern and per-vertex
    UVs stay uniform between the disk and annulus cases; the degenerate
    zero-area wedge triangles this creates are harmless. Object-space normal is
    ``+z`` (pbrt's convention; :func:`bake_world_mesh` applies
    ``reverseOrientation``/handedness flips downstream, same as every other
    shape here)."""
    phi_max_rad = np.radians(np.clip(phi_max, 0.0, 360.0))
    pts = []
    nrm = []
    uvs = []
    n = np.array([0.0, 0.0, 1.0])
    for ring_radius, v in ((radius, 0.0), (inner_radius, 1.0)):
        for j in range(segments + 1):
            phi = phi_max_rad * j / segments
            pts.append([ring_radius * np.cos(phi), ring_radius * np.sin(phi), height])
            nrm.append(n)
            uvs.append([j / segments, v])
    idx = []
    row = segments + 1
    for j in range(segments):
        a = j            # outer ring
        b = row + j      # inner ring
        # CCW as seen from +z (outward +z normal): (a, a+1, b) not (a, b, a+1).
        idx.append([a, a + 1, b])
        idx.append([a + 1, b + 1, b])
    return np.array(pts), np.array(idx, dtype=np.int64), np.array(nrm), np.array(uvs)
