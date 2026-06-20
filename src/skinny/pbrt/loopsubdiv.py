"""pbrt-exact Loop subdivision tessellation.

Ports the algorithm of pbrt v4's ``loopsubdiv.cpp`` so an imported
``Shape "loopsubdiv"`` control cage becomes the same triangle mesh pbrt renders:
the Loop rules are applied ``levels`` times, then the vertices are pushed to the
Loop *limit* surface and per-vertex *limit normals* are evaluated from the Loop
tangent masks.

Public entry: :func:`subdivide(points, indices, levels) -> (points, indices, normals)`.
Topology is triangle-only and uncreased, matching pbrt's ``loopsubdiv``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["subdivide"]


def _next(i: int) -> int:
    return (i + 1) % 3


def _prev(i: int) -> int:
    return (i + 2) % 3


def _beta(valence: int) -> float:
    # Warren weights (pbrt LoopSubdiv::beta): regular interior valence 6 -> 1/16.
    return 3.0 / 16.0 if valence == 3 else 3.0 / (8.0 * valence)


def _loop_gamma(valence: int) -> float:
    return 1.0 / (valence + 3.0 / (8.0 * _beta(valence)))


class _SDVertex:
    __slots__ = ("id", "p", "start_face", "boundary", "regular", "child")

    def __init__(self, vid: int, p):
        self.id = vid
        self.p = p
        self.start_face: _SDFace | None = None
        self.boundary = False
        self.regular = False
        self.child: _SDVertex | None = None

    def valence(self) -> int:
        f = self.start_face
        if not self.boundary:
            nf = 1
            face = f.next_face(self)
            while face is not f:
                nf += 1
                face = face.next_face(self)
            return nf
        nf = 1
        face = self.start_face
        while True:
            nxt = face.next_face(self)
            if nxt is None:
                break
            nf += 1
            face = nxt
        face = self.start_face
        while True:
            prv = face.prev_face(self)
            if prv is None:
                break
            nf += 1
            face = prv
        return nf + 1

    def one_ring(self) -> np.ndarray:
        """Ordered one-ring positions; for boundary verts ring[0] and ring[-1]
        are the two boundary neighbours."""
        ring: list = []
        if not self.boundary:
            face = self.start_face
            while True:
                ring.append(face.next_vert(self).p)
                face = face.next_face(self)
                if face is self.start_face:
                    break
        else:
            face = self.start_face
            while True:
                nxt = face.next_face(self)
                if nxt is None:
                    break
                face = nxt
            ring.append(face.next_vert(self).p)
            while face is not None:
                ring.append(face.prev_vert(self).p)
                face = face.prev_face(self)
        return np.asarray(ring, dtype=np.float64)


class _SDFace:
    __slots__ = ("v", "f", "children")

    def __init__(self, v):
        self.v: list = list(v) if v is not None else [None, None, None]
        self.f: list = [None, None, None]
        self.children: list = []

    def vnum(self, vert: _SDVertex) -> int:
        for i in range(3):
            if self.v[i] is vert:
                return i
        raise ValueError("vertex not in face")

    def next_face(self, vert):
        return self.f[self.vnum(vert)]

    def prev_face(self, vert):
        return self.f[_prev(self.vnum(vert))]

    def next_vert(self, vert):
        return self.v[_next(self.vnum(vert))]

    def prev_vert(self, vert):
        return self.v[_prev(self.vnum(vert))]

    def other_vert(self, v0, v1):
        for vv in self.v:
            if vv is not v0 and vv is not v1:
                return vv
        raise ValueError("face has no third vertex")


def _edge_key(a: _SDVertex, b: _SDVertex):
    return (a.id, b.id) if a.id < b.id else (b.id, a.id)


def _build(points: np.ndarray, indices: np.ndarray):
    verts = [_SDVertex(i, points[i]) for i in range(len(points))]
    faces = []
    for tri in indices:
        face = _SDFace([verts[int(tri[0])], verts[int(tri[1])], verts[int(tri[2])]])
        faces.append(face)
        for vv in face.v:
            vv.start_face = face
    # Match shared edges -> face neighbour pointers; leftover edges are boundary.
    edge_map: dict = {}
    for face in faces:
        for k in range(3):
            v0, v1 = face.v[k], face.v[_next(k)]
            key = _edge_key(v0, v1)
            hit = edge_map.pop(key, None)
            if hit is None:
                edge_map[key] = (face, k)
            else:
                of, ok = hit
                of.f[ok] = face
                face.f[k] = of
    # Boundary / valence / regular classification.
    for v in verts:
        face = v.start_face
        while True:
            face = face.next_face(v)
            if face is None or face is v.start_face:
                break
        v.boundary = face is None
        val = v.valence()
        v.regular = (not v.boundary and val == 6) or (v.boundary and val == 4)
    return verts, faces


def _subdivide_once(verts, faces):
    # Even (existing) vertices -> child positions.
    for v in verts:
        child = _SDVertex(-1, None)
        child.regular = v.regular
        child.boundary = v.boundary
        if not v.boundary:
            b = 1.0 / 16.0 if v.regular else _beta(v.valence())
            child.p = _weight_one_ring(v, b)
        else:
            child.p = _weight_boundary(v, 1.0 / 8.0)
        v.child = child
    new_vertices = [v.child for v in verts]

    for face in faces:
        face.children = [_SDFace(None) for _ in range(4)]
    new_faces = [c for face in faces for c in face.children]

    # Odd (new edge) vertices.
    edge_verts: dict = {}
    for face in faces:
        for k in range(3):
            v0, v1 = face.v[k], face.v[_next(k)]
            key = _edge_key(v0, v1)
            if key in edge_verts:
                continue
            vert = _SDVertex(-1, None)
            vert.regular = True
            vert.boundary = face.f[k] is None
            vert.start_face = face.children[3]
            if vert.boundary:
                vert.p = 0.5 * v0.p + 0.5 * v1.p
            else:
                vert.p = (
                    0.375 * v0.p + 0.375 * v1.p
                    + 0.125 * face.other_vert(v0, v1).p
                    + 0.125 * face.f[k].other_vert(v0, v1).p
                )
            edge_verts[key] = vert
            new_vertices.append(vert)

    # New even vertices' start_face.
    for v in verts:
        vert_num = v.start_face.vnum(v)
        v.child.start_face = v.start_face.children[vert_num]

    # Child face neighbour pointers.
    for face in faces:
        for j in range(3):
            face.children[3].f[j] = face.children[_next(j)]
            face.children[j].f[_next(j)] = face.children[3]
            f2 = face.f[j]
            face.children[j].f[j] = (
                f2.children[f2.vnum(face.v[j])] if f2 is not None else None
            )
            f2 = face.f[_prev(j)]
            face.children[j].f[_prev(j)] = (
                f2.children[f2.vnum(face.v[j])] if f2 is not None else None
            )

    # Child face vertex pointers.
    for face in faces:
        for j in range(3):
            face.children[j].v[j] = face.v[j].child
            vert = edge_verts[_edge_key(face.v[j], face.v[_next(j)])]
            face.children[j].v[_next(j)] = vert
            face.children[_next(j)].v[j] = vert
            face.children[3].v[j] = vert

    for i, v in enumerate(new_vertices):
        v.id = i
    return new_vertices, new_faces


def _weight_one_ring(vert: _SDVertex, beta: float) -> np.ndarray:
    val = vert.valence()
    ring = vert.one_ring()
    return (1.0 - val * beta) * vert.p + beta * ring.sum(axis=0)


def _weight_boundary(vert: _SDVertex, beta: float) -> np.ndarray:
    ring = vert.one_ring()
    return (1.0 - 2.0 * beta) * vert.p + beta * ring[0] + beta * ring[-1]


def _limit(verts):
    n = len(verts)
    plimit = np.zeros((n, 3), dtype=np.float64)
    normals = np.zeros((n, 3), dtype=np.float64)
    for i, v in enumerate(verts):
        if v.boundary:
            plimit[i] = _weight_boundary(v, 1.0 / 5.0)
        else:
            plimit[i] = _weight_one_ring(v, _loop_gamma(v.valence()))
    # Limit normals from the Loop tangent masks (computed on pre-limit positions).
    for i, v in enumerate(verts):
        val = v.valence()
        ring = v.one_ring()
        if not v.boundary:
            k = np.arange(val)
            s = (np.cos(2.0 * np.pi * k / val)[:, None] * ring).sum(axis=0)
            t = (np.sin(2.0 * np.pi * k / val)[:, None] * ring).sum(axis=0)
        else:
            s = ring[-1] - ring[0]
            if val == 2:
                t = ring[0] + ring[1] - 2.0 * v.p
            elif val == 3:
                t = ring[1] - v.p
            elif val == 4:  # regular crease
                t = -1.0 * ring[0] + 2.0 * ring[1] + 2.0 * ring[2] - 1.0 * ring[3] - 2.0 * v.p
            else:
                theta = np.pi / float(val - 1)
                t = np.sin(theta) * (ring[0] + ring[-1])
                for k in range(1, val - 1):
                    wt = (2.0 * np.cos(theta) - 2.0) * np.sin(k * theta)
                    t = t + wt * ring[k]
                t = -t
        nrm = np.cross(s, t)
        ln = float(np.linalg.norm(nrm))
        normals[i] = nrm / ln if ln > 0.0 else np.array([0.0, 0.0, 1.0])
    return plimit, normals


def subdivide(points, indices, levels: int):
    """Tessellate a Loop subdivision control cage to its limit triangle mesh.

    Args:
        points: ``(N, 3)`` control-cage vertex positions.
        indices: ``(M, 3)`` (or flat length ``3M``) triangle vertex indices.
        levels: number of Loop refinement steps (``>= 0``).

    Returns:
        ``(positions (V,3), indices (T,3) int64, normals (V,3))`` on the Loop
        limit surface, with ``T = 4**levels * M``.

    Raises:
        ValueError: empty input, an ``indices`` length not a multiple of three,
            or malformed (non-manifold beyond Loop's domain) topology.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    flat = np.asarray(indices).reshape(-1)
    if points.size == 0 or flat.size == 0:
        raise ValueError("loopsubdiv: empty points or indices")
    if flat.size % 3 != 0:
        raise ValueError("loopsubdiv: indices length not a multiple of 3")
    idx = flat.astype(np.int64).reshape(-1, 3)
    if int(idx.max(initial=0)) >= len(points):
        raise ValueError("loopsubdiv: index out of range")

    verts, faces = _build(points, idx)
    for _ in range(int(levels)):
        verts, faces = _subdivide_once(verts, faces)
    plimit, normals = _limit(verts)
    out_idx = np.array(
        [[f.v[0].id, f.v[1].id, f.v[2].id] for f in faces], dtype=np.int64
    )
    return plimit, out_idx, normals
