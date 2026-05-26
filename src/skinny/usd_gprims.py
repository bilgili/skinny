"""Tessellate USD analytic gprims into triangle meshes.

UsdGeom defines implicit shapes (Sphere, Cube, Cylinder, Cone, Capsule, Plane)
that carry no point data — only parameters. The renderer only consumes
triangle meshes, so this module converts each shape into a `MeshSource` with
explicit positions, analytic normals, UVs, and triangle indices.

Local geometry for the axial shapes is built with the symmetry axis along +Z
(the UsdGeom default) and then rotated onto the prim's authored `axis`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pxr import Usd, UsdGeom

from skinny.mesh import MeshSource

# Tessellation resolution. Radial = segments around a circumference; rings =
# latitude bands on a sphere / stacks on a hemisphere cap.
_RADIAL = 64
_RINGS = 32


def _axis_matrix(axis: str) -> np.ndarray:
    """3x3 rotation mapping local +Z geometry onto the target symmetry axis."""
    if axis == "X":
        # +Z -> +X (rotate about Y by +90 deg)
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    if axis == "Y":
        # +Z -> +Y (rotate about X by -90 deg)
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32)  # "Z"


def _apply_axis(
    positions: np.ndarray, normals: np.ndarray, axis: str
) -> tuple[np.ndarray, np.ndarray]:
    if axis == "Z":
        return positions, normals
    m = _axis_matrix(axis)
    return positions @ m.T, normals @ m.T


def _finish(
    name: str,
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    tris: np.ndarray,
) -> MeshSource:
    # V-flip to match the image / OBJ-loader convention (USD puts V=0 at the
    # bottom of the texture; image loaders put it at the top).
    uvs = uvs.copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]
    return MeshSource(
        name=name,
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
        tri_idx=np.ascontiguousarray(tris, dtype=np.int32),
    )


def _grid_indices(rings: int, segs: int) -> np.ndarray:
    """Two triangles per quad over a (rings+1) x (segs+1) vertex grid.

    Winding is CCW when viewed from outside (normals point away from the
    grid's near face), matching UsdGeom rightHanded orientation.
    """
    tris: list[tuple[int, int, int]] = []
    stride = segs + 1
    for i in range(rings):
        for j in range(segs):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            tris.append((a, c, b))
            tris.append((b, c, d))
    return np.asarray(tris, dtype=np.int32)


def _sphere(radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rings, segs = _RINGS, _RADIAL
    i = np.arange(rings + 1)
    j = np.arange(segs + 1)
    theta = (math.pi * i / rings)[:, None]          # polar 0..pi
    phi = (2.0 * math.pi * j / segs)[None, :]       # azimuth 0..2pi
    st, ct = np.sin(theta), np.cos(theta)
    nx = st * np.cos(phi)
    ny = np.broadcast_to(ct, (rings + 1, segs + 1))
    nz = st * np.sin(phi)
    normals = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3).astype(np.float32)
    positions = normals * radius
    u = np.broadcast_to(j / segs, (rings + 1, segs + 1))
    v = np.broadcast_to((i / rings)[:, None], (rings + 1, segs + 1))
    uvs = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)
    return positions, normals, uvs, _grid_indices(rings, segs)


def _cube(size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = size * 0.5
    # 6 faces, each its own 4 verts so edges stay hard (faceted normals).
    faces = [
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),
        ((0, 0, -1), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)]),
        ((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)]),
        ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)]),
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),
    ]
    positions: list[tuple] = []
    normals: list[tuple] = []
    uvs: list[tuple] = []
    tris: list[tuple] = []
    quad_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for normal, corners in faces:
        base = len(positions)
        positions.extend(corners)
        normals.extend([normal] * 4)
        uvs.extend(quad_uv)
        tris.append((base, base + 1, base + 2))
        tris.append((base, base + 2, base + 3))
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(tris, dtype=np.int32),
    )


def _ring(segs: int, radius: float, z: float):
    j = np.arange(segs + 1)
    phi = 2.0 * math.pi * j / segs
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return np.stack([x, y, np.full(segs + 1, z)], axis=-1).astype(np.float32), phi


def _cap(
    segs: int, radius: float, z: float, nz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A flat disk cap: center + boundary ring, fan-triangulated."""
    ring, phi = _ring(segs, radius, z)
    center = np.array([[0.0, 0.0, z]], dtype=np.float32)
    positions = np.concatenate([center, ring], axis=0)
    normals = np.broadcast_to(
        np.array([0.0, 0.0, nz], dtype=np.float32), positions.shape
    ).copy()
    uvs = np.concatenate(
        [
            np.array([[0.5, 0.5]], dtype=np.float32),
            np.stack(
                [0.5 + 0.5 * np.cos(phi), 0.5 + 0.5 * np.sin(phi)], axis=-1
            ).astype(np.float32),
        ],
        axis=0,
    )
    tris = []
    for j in range(segs):
        # nz>0: CCW seen from +z; nz<0: flip so the cap faces -z.
        if nz >= 0:
            tris.append((0, 1 + j, 1 + j + 1))
        else:
            tris.append((0, 1 + j + 1, 1 + j))
    return positions, normals, uvs, np.asarray(tris, dtype=np.int32)


def _concat(parts):
    positions, normals, uvs, tris = [], [], [], []
    offset = 0
    for p, n, u, t in parts:
        positions.append(p)
        normals.append(n)
        uvs.append(u)
        tris.append(t + offset)
        offset += p.shape[0]
    return (
        np.concatenate(positions, axis=0),
        np.concatenate(normals, axis=0),
        np.concatenate(uvs, axis=0),
        np.concatenate(tris, axis=0),
    )


def _cylinder(radius: float, height: float):
    segs = _RADIAL
    hz = height * 0.5
    bottom, phi = _ring(segs, radius, -hz)
    top, _ = _ring(segs, radius, hz)
    side_pos = np.concatenate([bottom, top], axis=0)
    radial_n = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    side_n = np.concatenate([radial_n, radial_n], axis=0).astype(np.float32)
    u = (phi / (2.0 * math.pi)).astype(np.float32)
    side_uv = np.concatenate(
        [np.stack([u, np.zeros_like(u)], -1), np.stack([u, np.ones_like(u)], -1)],
        axis=0,
    ).astype(np.float32)
    side_tris = _grid_indices(1, segs)  # 1 ring band, segs segments
    side = (side_pos, side_n, side_uv, side_tris)
    cap_top = _cap(segs, radius, hz, 1.0)
    cap_bottom = _cap(segs, radius, -hz, -1.0)
    return _concat([side, cap_top, cap_bottom])


def _cone(radius: float, height: float):
    segs = _RADIAL
    hz = height * 0.5
    base, phi = _ring(segs, radius, -hz)
    # Outward cone-surface normal: slope component r/h toward the apex (+z).
    slope = radius / height if height != 0 else 0.0
    nrm = np.stack([np.cos(phi), np.sin(phi), np.full_like(phi, slope)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = nrm.astype(np.float32)
    # One apex vertex per segment (segs total), each with the midpoint-azimuth
    # slope normal so adjacent side triangles shade consistently.
    apex = np.tile(np.array([0.0, 0.0, hz], dtype=np.float32), (segs, 1))
    mid = 0.5 * (phi[:-1] + phi[1:])
    apex_n = np.stack([np.cos(mid), np.sin(mid), np.full_like(mid, slope)], axis=-1)
    apex_n /= np.linalg.norm(apex_n, axis=-1, keepdims=True)
    apex_n = apex_n.astype(np.float32)
    positions = np.concatenate([base, apex], axis=0)
    normals = np.concatenate([nrm, apex_n], axis=0)
    u = (phi / (2.0 * math.pi)).astype(np.float32)
    base_uv = np.stack([u, np.zeros_like(u)], -1)
    apex_uv = np.stack([(u[:-1] + u[1:]) * 0.5, np.ones(segs)], -1)
    uvs = np.concatenate([base_uv, apex_uv], axis=0).astype(np.float32)
    tris = [(j, j + 1, (segs + 1) + j) for j in range(segs)]
    side = (
        positions,
        normals,
        uvs,
        np.asarray(tris, dtype=np.int32),
    )
    cap_base = _cap(segs, radius, -hz, -1.0)
    return _concat([side, cap_base])


def _capsule(radius: float, height: float):
    segs = _RADIAL
    caps = max(_RINGS // 2, 2)  # latitude stacks per hemisphere
    hz = height * 0.5
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # Top hemisphere: lat 0 (pole) .. pi/2 (equator at z = +hz).
    for i in range(caps + 1):
        lat = (math.pi / 2.0) * (i / caps)
        sl, cl = math.sin(lat), math.cos(lat)
        ring, phi = _ring(segs, radius * sl, hz + radius * cl)
        n = np.stack(
            [sl * np.cos(phi), sl * np.sin(phi), np.full_like(phi, cl)], axis=-1
        ).astype(np.float32)
        rows.append((ring, n, i / (2 * caps + 1)))

    # Cylinder seam (equator at +hz down to -hz). Two rings keep it a band.
    ring, phi = _ring(segs, radius, hz)
    radial = np.stack(
        [np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1
    ).astype(np.float32)
    # +hz radial ring (distinct normals from the hemisphere equator row above).
    rows.append((ring, radial, 0.5))
    ring2, _ = _ring(segs, radius, -hz)
    rows.append((ring2, radial, 0.5))

    # Bottom hemisphere: lat pi/2 (equator at -hz) .. pi (pole).
    for i in range(caps + 1):
        lat = (math.pi / 2.0) + (math.pi / 2.0) * (i / caps)
        sl, cl = math.sin(lat), math.cos(lat)
        ring, phi = _ring(segs, radius * sl, -hz + radius * cl)
        n = np.stack(
            [sl * np.cos(phi), sl * np.sin(phi), np.full_like(phi, cl)], axis=-1
        ).astype(np.float32)
        rows.append((ring, n, 0.5 + (i + 1) / (2 * caps + 1)))

    positions = np.concatenate([r[0] for r in rows], axis=0).astype(np.float32)
    normals = np.concatenate([r[1] for r in rows], axis=0).astype(np.float32)
    nrows = len(rows)
    u = (phi / (2.0 * math.pi)).astype(np.float32)
    uvs = np.concatenate(
        [
            np.stack([u, np.full_like(u, rows[k][2])], axis=-1)
            for k in range(nrows)
        ],
        axis=0,
    ).astype(np.float32)
    tris = _grid_indices(nrows - 1, segs)
    return positions, normals, uvs, tris


def _plane(width: float, length: float):
    hw, hl = width * 0.5, length * 0.5
    positions = np.array(
        [(-hw, -hl, 0), (hw, -hl, 0), (hw, hl, 0), (-hw, hl, 0)], dtype=np.float32
    )
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1))
    uvs = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
    tris = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int32)
    return positions, normals, uvs, tris


def tessellate_gprim(prim: Usd.Prim, time: Usd.TimeCode) -> Optional[MeshSource]:
    """Convert a UsdGeom analytic shape prim into a `MeshSource`.

    Returns None if the prim is not one of the supported implicit shapes.
    """
    name = str(prim.GetPath())

    if prim.IsA(UsdGeom.Sphere):
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get(time))
        p, n, uv, t = _sphere(r)
        return _finish(name, p, n, uv, t)

    if prim.IsA(UsdGeom.Cube):
        s = float(UsdGeom.Cube(prim).GetSizeAttr().Get(time))
        p, n, uv, t = _cube(s)
        return _finish(name, p, n, uv, t)

    if prim.IsA(UsdGeom.Cylinder):
        g = UsdGeom.Cylinder(prim)
        r = float(g.GetRadiusAttr().Get(time))
        h = float(g.GetHeightAttr().Get(time))
        axis = str(g.GetAxisAttr().Get(time))
        p, n, uv, t = _cylinder(r, h)
        p, n = _apply_axis(p, n, axis)
        return _finish(name, p, n, uv, t)

    if prim.IsA(UsdGeom.Cone):
        g = UsdGeom.Cone(prim)
        r = float(g.GetRadiusAttr().Get(time))
        h = float(g.GetHeightAttr().Get(time))
        axis = str(g.GetAxisAttr().Get(time))
        p, n, uv, t = _cone(r, h)
        p, n = _apply_axis(p, n, axis)
        return _finish(name, p, n, uv, t)

    if prim.IsA(UsdGeom.Capsule):
        g = UsdGeom.Capsule(prim)
        r = float(g.GetRadiusAttr().Get(time))
        h = float(g.GetHeightAttr().Get(time))
        axis = str(g.GetAxisAttr().Get(time))
        p, n, uv, t = _capsule(r, h)
        p, n = _apply_axis(p, n, axis)
        return _finish(name, p, n, uv, t)

    if prim.IsA(UsdGeom.Plane):
        g = UsdGeom.Plane(prim)
        w = float(g.GetWidthAttr().Get(time))
        length = float(g.GetLengthAttr().Get(time))
        axis = str(g.GetAxisAttr().Get(time))
        p, n, uv, t = _plane(w, length)
        p, n = _apply_axis(p, n, axis)
        return _finish(name, p, n, uv, t)

    return None
