"""Triangle mesh loading and linear BVH construction for ray traced heads.

The mesh pipeline in `main_pass.slang` consumes three flat storage buffers:

    vertices  : Vertex[] ─ position (vec3) + u + normal (vec3) + v
    indices   : uint[]   ─ triangle-index list in BVH-permuted order
    bvhNodes  : BvhNode[]─ 32-byte nodes, depth-first layout

See `build_bvh` for the node encoding. Mesh normalisation centres the model at
the origin and scales it so its Y-extent is roughly [-1, +1] — matching the
SDF head's frame so the orbit camera doesn't need adjusting when switching.

Per-model layout: each immediate subdirectory of `heads/` is one model. The
first `.obj` in the directory is the geometry; other images are attached as
detail maps, selected by filename keyword — `normal`, `roughness`, and
`displacement` (case-insensitive). A loose `.obj` at the top level is still
loaded (with no texture maps) for backward compatibility.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


VERTEX_STRIDE = 32          # float3 pos + u + float3 nrm + v
BVH_NODE_STRIDE = 32        # float3 min + int + float3 max + int
BVH_LEAF_SIZE = 4           # max triangles per leaf

_IMAGE_EXTS = {".tga", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".exr"}


@dataclass
class Mesh:
    """GPU-ready mesh buffers (vertices + indices + BVH) plus metadata."""

    name: str
    vertex_bytes: bytes
    index_bytes: bytes
    bvh_bytes: bytes
    num_vertices: int
    num_triangles: int
    num_nodes: int
    # Optional detail-map source files for this model. Absent maps are None.
    normal_map: Path | None = None
    roughness_map: Path | None = None
    displacement_map: Path | None = None
    # True when the normal map's tangent-space perturbation has been baked
    # into the stored vertex normals. The shader uses this (via a detailFlags
    # bit) to skip runtime normal-map sampling and avoid double-application.
    normals_baked: bool = False


@dataclass
class MeshSource:
    """Undisplaced base geometry loaded from an OBJ.

    Kept around so the renderer can rebake (subdivide + displace + rebuild BVH)
    when the user changes subdivision or displacement scale without paying the
    OBJ parse cost again. UVs are already V-flipped to image convention.
    """

    name: str
    positions: np.ndarray          # (N, 3) float32, normalised to head frame
    normals: np.ndarray            # (N, 3) float32, smooth per-vertex
    uvs: np.ndarray                # (N, 2) float32, V-flipped
    tri_idx: np.ndarray            # (T, 3) int32
    normal_map: Path | None = None
    roughness_map: Path | None = None
    displacement_map: Path | None = None


# ── OBJ loading ────────────────────────────────────────────────────

def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Wavefront OBJ.

    Returns (positions, normals, uvs, triangle_indices) where `uvs` is (N, 2),
    one per position. If the OBJ has no `vt` coordinates, UVs are all zero.
    Supports `f` with `v`, `v/vt`, `v/vt/vn`, or `v//vn` variants; polygons are
    fan-triangulated. Smooth per-vertex normals are synthesised when absent.
    """
    positions: list[tuple[float, float, float]] = []
    normals_file: list[tuple[float, float, float]] = []
    uvs_file: list[tuple[float, float]] = []

    tris: list[tuple[int, int, int]] = []          # 0-based position indices
    tris_n: list[tuple[int, int, int] | None] = [] # 0-based normal indices (or None)
    tris_t: list[tuple[int, int, int] | None] = [] # 0-based uv indices (or None)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vn" and len(parts) >= 4:
                normals_file.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vt" and len(parts) >= 3:
                uvs_file.append((float(parts[1]), float(parts[2])))
            elif tag == "f" and len(parts) >= 4:
                pv: list[int] = []
                nv: list[int | None] = []
                tv: list[int | None] = []
                for token in parts[1:]:
                    bits = token.split("/")
                    vi = int(bits[0])
                    ni: int | None = None
                    ti: int | None = None
                    if len(bits) >= 2 and bits[1]:
                        ti = int(bits[1])
                    if len(bits) >= 3 and bits[2]:
                        ni = int(bits[2])
                    # OBJ is 1-indexed; negatives count from the end.
                    vi = vi - 1 if vi > 0 else len(positions) + vi
                    if ni is not None:
                        ni = ni - 1 if ni > 0 else len(normals_file) + ni
                    if ti is not None:
                        ti = ti - 1 if ti > 0 else len(uvs_file) + ti
                    pv.append(vi)
                    nv.append(ni)
                    tv.append(ti)
                for k in range(1, len(pv) - 1):
                    tris.append((pv[0], pv[k], pv[k + 1]))
                    if nv[0] is not None and nv[k] is not None and nv[k + 1] is not None:
                        tris_n.append((nv[0], nv[k], nv[k + 1]))
                    else:
                        tris_n.append(None)
                    if tv[0] is not None and tv[k] is not None and tv[k + 1] is not None:
                        tris_t.append((tv[0], tv[k], tv[k + 1]))
                    else:
                        tris_t.append(None)

    if not positions or not tris:
        raise ValueError(f"{path}: no geometry")

    pos = np.asarray(positions, dtype=np.float32)
    tri_idx = np.asarray(tris, dtype=np.int32)

    nrm = np.zeros_like(pos)
    if normals_file and all(t is not None for t in tris_n):
        file_n = np.asarray(normals_file, dtype=np.float32)
        counts = np.zeros(pos.shape[0], dtype=np.int32)
        for (a, b, c), (na, nb, nc) in zip(tris, tris_n):  # type: ignore[misc]
            nrm[a] += file_n[na]; counts[a] += 1
            nrm[b] += file_n[nb]; counts[b] += 1
            nrm[c] += file_n[nc]; counts[c] += 1
        counts = np.maximum(counts, 1)
        nrm = nrm / counts[:, None]
    else:
        p0 = pos[tri_idx[:, 0]]
        p1 = pos[tri_idx[:, 1]]
        p2 = pos[tri_idx[:, 2]]
        face_n = np.cross(p1 - p0, p2 - p0)
        for i in range(3):
            np.add.at(nrm, tri_idx[:, i], face_n)

    lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(lengths, 1e-8)

    # UVs: one per position. We average the file UVs for every face corner
    # pointing at a given vertex. OBJ lets a single position be referenced
    # with multiple UVs across faces (UV seams); averaging collapses seams
    # into one value per vertex. This is coarse — proper seam support would
    # need to duplicate the position for each unique (pos, uv) pair — but
    # it's fine for face scans where seams are at the hairline/back of head.
    uv = np.zeros((pos.shape[0], 2), dtype=np.float32)
    if uvs_file and any(t is not None for t in tris_t):
        file_t = np.asarray(uvs_file, dtype=np.float32)
        # Flip V so image V=0 maps to the top of the texture (OBJ's convention
        # has V=0 at bottom; most image formats + Vulkan sampler put V=0 at top
        # without a flip). This is a no-op if the texture author baked with the
        # opposite convention — users can always invert by editing the map.
        uv_counts = np.zeros(pos.shape[0], dtype=np.int32)
        for (a, b, c), tris_t_entry in zip(tris, tris_t):
            if tris_t_entry is None:
                continue
            ta, tb, tc = tris_t_entry
            uv[a] += file_t[ta]; uv_counts[a] += 1
            uv[b] += file_t[tb]; uv_counts[b] += 1
            uv[c] += file_t[tc]; uv_counts[c] += 1
        nonzero = uv_counts > 0
        uv[nonzero] /= uv_counts[nonzero, None]
        uv[:, 1] = 1.0 - uv[:, 1]

    return pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32), tri_idx.astype(np.int32)


def normalise_to_head_frame(positions: np.ndarray) -> np.ndarray:
    """Recentre the mesh and scale it to fit roughly y∈[-1, +1.1].

    The SDF head spans about y∈[-0.97, +1.05]. Matching that frame means the
    existing orbit-camera pivot (0, 0.1, 0.05) still looks sensible when the
    user flips to the mesh pipeline.
    """
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    centre = (lo + hi) * 0.5
    y_extent = max(hi[1] - lo[1], 1e-6)
    scale = 2.0 / y_extent                  # target height = 2 world units
    return ((positions - centre) * scale).astype(np.float32)


# ── Linear BVH ──────────────────────────────────────────────────────

@dataclass
class _BuildNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    first_tri: int
    tri_count: int
    left: int = -1
    right: int = -1


def _triangle_aabbs(positions: np.ndarray, tri_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = positions[tri_idx[:, 0]]
    p1 = positions[tri_idx[:, 1]]
    p2 = positions[tri_idx[:, 2]]
    tmin = np.minimum(np.minimum(p0, p1), p2)
    tmax = np.maximum(np.maximum(p0, p1), p2)
    centroid = (tmin + tmax) * 0.5
    return tmin, tmax, centroid


def build_bvh(
    positions: np.ndarray, tri_idx: np.ndarray
) -> tuple[np.ndarray, list]:
    tmin, tmax, centroids = _triangle_aabbs(positions, tri_idx)
    tri_ids = np.arange(tri_idx.shape[0], dtype=np.int32)

    nodes: list[_BuildNode] = []

    def box_of(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return tmin[ids].min(axis=0), tmax[ids].max(axis=0)

    root_ids = tri_ids
    root_min, root_max = box_of(root_ids)
    nodes.append(_BuildNode(root_min, root_max, first_tri=0, tri_count=0))
    permuted_blocks: list[np.ndarray] = []
    running_first = 0

    stack: list[tuple[int, np.ndarray]] = [(0, root_ids)]
    while stack:
        node_index, ids = stack.pop()
        node = nodes[node_index]
        amin, amax = box_of(ids)
        node.aabb_min = amin
        node.aabb_max = amax

        if ids.shape[0] <= BVH_LEAF_SIZE:
            node.first_tri = running_first
            node.tri_count = int(ids.shape[0])
            permuted_blocks.append(ids)
            running_first += ids.shape[0]
            continue

        extent = amax - amin
        axis = int(np.argmax(extent))
        cs = centroids[ids, axis]
        order = np.argsort(cs, kind="stable")
        mid = ids.shape[0] // 2
        left_ids = ids[order[:mid]]
        right_ids = ids[order[mid:]]

        left_index = len(nodes)
        nodes.append(_BuildNode(
            aabb_min=np.zeros(3, np.float32),
            aabb_max=np.zeros(3, np.float32),
            first_tri=0, tri_count=0,
        ))
        right_index = len(nodes)
        nodes.append(_BuildNode(
            aabb_min=np.zeros(3, np.float32),
            aabb_max=np.zeros(3, np.float32),
            first_tri=0, tri_count=0,
        ))

        node.left = left_index
        node.right = right_index
        stack.append((right_index, right_ids))
        stack.append((left_index, left_ids))

    permuted = np.concatenate(permuted_blocks, axis=0) if permuted_blocks else np.zeros(0, np.int32)
    permuted_tri_idx = tri_idx[permuted]
    return permuted_tri_idx, nodes


# ── Buffer packing for GPU upload ───────────────────────────────────

def _pack_vertices(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray) -> bytes:
    """Pack (N, 3) positions + normals + (N, 2) uvs into 32-byte records.

    Layout (matches `MeshVertex` in mesh_head.slang):
        float3 position
        float  u
        float3 normal
        float  v
    """
    n = positions.shape[0]
    out = np.zeros((n, 8), dtype=np.float32)
    out[:, 0:3] = positions
    out[:, 3] = uvs[:, 0]
    out[:, 4:7] = normals
    out[:, 7] = uvs[:, 1]
    return out.tobytes()


def _pack_indices(tri_idx: np.ndarray) -> bytes:
    return tri_idx.astype(np.uint32).tobytes()


def _pack_bvh(nodes: list) -> bytes:
    buf = bytearray()
    for n in nodes:
        if n.tri_count > 0:
            left_or_count = -n.tri_count
            right_or_first = n.first_tri
        else:
            left_or_count = n.left
            right_or_first = n.right
        buf += struct.pack(
            "fff i fff i",
            float(n.aabb_min[0]), float(n.aabb_min[1]), float(n.aabb_min[2]),
            int(left_or_count),
            float(n.aabb_max[0]), float(n.aabb_max[1]), float(n.aabb_max[2]),
            int(right_or_first),
        )
    return bytes(buf)


# ── Texture-file detection ──────────────────────────────────────────

_NORMAL_KEYS       = ("normal", "nrm", "nor")
_ROUGHNESS_KEYS    = ("rough", "roughness")
_DISPLACEMENT_KEYS = ("displacement", "displace", "disp", "height", "bump")


def _pick_texture(files: list[Path], keys: tuple[str, ...]) -> Path | None:
    """First image file whose stem contains any keyword (case-insensitive)."""
    for f in files:
        stem = f.stem.lower()
        if any(k in stem for k in keys):
            return f
    return None


def _load_model_dir(model_dir: Path) -> MeshSource | None:
    objs = sorted(model_dir.glob("*.obj"))
    if not objs:
        return None
    images = sorted(p for p in model_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)

    try:
        source = load_head_source(objs[0])
    except Exception as exc:  # noqa: BLE001
        print(f"[skinny] failed to load {objs[0]}: {exc}")
        return None

    source.name = model_dir.name
    source.normal_map       = _pick_texture(images, _NORMAL_KEYS)
    source.roughness_map    = _pick_texture(images, _ROUGHNESS_KEYS)
    source.displacement_map = _pick_texture(images, _DISPLACEMENT_KEYS)

    attached: list[str] = []
    if source.normal_map:       attached.append(f"normal={source.normal_map.name}")
    if source.roughness_map:    attached.append(f"roughness={source.roughness_map.name}")
    if source.displacement_map: attached.append(f"displacement={source.displacement_map.name}")
    attached_str = " [" + ", ".join(attached) + "]" if attached else ""
    print(
        f"[skinny] loaded head model '{source.name}' from {objs[0].name} "
        f"({source.positions.shape[0]} verts, {source.tri_idx.shape[0]} tris){attached_str}"
    )
    return source


# ── Subdivision, displacement, and baking ───────────────────────────

def subdivide_midpoint(
    positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, tri_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One level of 1-to-4 midpoint triangle subdivision.

    Each triangle is replaced by four: three corner children plus a centre
    triangle made from the three edge midpoints. Midpoints are shared across
    adjacent triangles via a (min,max) edge key, so the mesh stays watertight
    — no cracks at shared edges. Positions, normals, and UVs are all linearly
    interpolated at midpoints; normals are then renormalised.
    """
    new_pos: list[np.ndarray] = [positions]
    new_nrm: list[np.ndarray] = [normals]
    new_uv:  list[np.ndarray] = [uvs]
    next_idx = positions.shape[0]

    mid_pos_rows: list[np.ndarray] = []
    mid_nrm_rows: list[np.ndarray] = []
    mid_uv_rows:  list[np.ndarray] = []
    edge_cache: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int) -> int:
        nonlocal next_idx
        key = (a, b) if a < b else (b, a)
        cached = edge_cache.get(key)
        if cached is not None:
            return cached
        mid_pos_rows.append((positions[a] + positions[b]) * 0.5)
        mid_nrm_rows.append((normals[a] + normals[b]) * 0.5)
        mid_uv_rows.append((uvs[a] + uvs[b]) * 0.5)
        edge_cache[key] = next_idx
        next_idx += 1
        return edge_cache[key]

    out_tris: list[tuple[int, int, int]] = []
    for a, b, c in tri_idx:
        ab = mid(int(a), int(b))
        bc = mid(int(b), int(c))
        ca = mid(int(c), int(a))
        out_tris.append((int(a), ab, ca))
        out_tris.append((ab, int(b), bc))
        out_tris.append((ca, bc, int(c)))
        out_tris.append((ab, bc, ca))

    if mid_pos_rows:
        new_pos.append(np.asarray(mid_pos_rows, dtype=np.float32))
        mid_n = np.asarray(mid_nrm_rows, dtype=np.float32)
        lengths = np.linalg.norm(mid_n, axis=1, keepdims=True)
        mid_n = mid_n / np.maximum(lengths, 1e-8)
        new_nrm.append(mid_n)
        new_uv.append(np.asarray(mid_uv_rows, dtype=np.float32))

    return (
        np.concatenate(new_pos, axis=0).astype(np.float32),
        np.concatenate(new_nrm, axis=0).astype(np.float32),
        np.concatenate(new_uv, axis=0).astype(np.float32),
        np.asarray(out_tris, dtype=np.int32),
    )


def _smooth_normals(positions: np.ndarray, tri_idx: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals. Same method as the OBJ loader fallback."""
    p0 = positions[tri_idx[:, 0]]
    p1 = positions[tri_idx[:, 1]]
    p2 = positions[tri_idx[:, 2]]
    face_n = np.cross(p1 - p0, p2 - p0)   # magnitude = 2·area, so area-weighted
    nrm = np.zeros_like(positions)
    for i in range(3):
        np.add.at(nrm, tri_idx[:, i], face_n)
    lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
    return (nrm / np.maximum(lengths, 1e-8)).astype(np.float32)


def _bilinear_sample_rg8(data: bytes, res: int, uvs: np.ndarray) -> np.ndarray:
    """Bilinear-sample R and G of an RGBA8 texture. Returns (N, 2) float32 in [0, 1]."""
    arr = np.frombuffer(data, dtype=np.uint8).reshape(res, res, 4)
    u = np.clip(uvs[:, 0].astype(np.float32), 0.0, 1.0) * (res - 1)
    v = np.clip(uvs[:, 1].astype(np.float32), 0.0, 1.0) * (res - 1)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, res - 1)
    y1 = np.minimum(y0 + 1, res - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    a = arr[y0, x0, :2].astype(np.float32) / 255.0
    b = arr[y0, x1, :2].astype(np.float32) / 255.0
    c = arr[y1, x0, :2].astype(np.float32) / 255.0
    d = arr[y1, x1, :2].astype(np.float32) / 255.0
    top = a * (1.0 - fx) + b * fx
    bot = c * (1.0 - fx) + d * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def _per_vertex_tangents(
    positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, tri_idx: np.ndarray
) -> np.ndarray:
    """Area-weighted per-vertex tangent, orthonormalised against the stored normals.

    Mirrors the shader-side TBN formula in mesh_head.slang:226-247 — triangle
    edge + UV-delta system solved for the +U axis, averaged per vertex by
    summing contributions, then Gram-Schmidt against N. Degenerate UVs
    (zero-area in texture space) fall back to an axis orthogonal to N.
    """
    p0 = positions[tri_idx[:, 0]]
    p1 = positions[tri_idx[:, 1]]
    p2 = positions[tri_idx[:, 2]]
    u0 = uvs[tri_idx[:, 0]]
    u1 = uvs[tri_idx[:, 1]]
    u2 = uvs[tri_idx[:, 2]]

    e1 = p1 - p0
    e2 = p2 - p0
    du1 = u1 - u0
    du2 = u2 - u0
    det = du1[:, 0] * du2[:, 1] - du2[:, 0] * du1[:, 1]
    safe = np.where(np.abs(det) > 1e-8, det, 1.0)
    inv_det = (1.0 / safe).astype(np.float32)
    face_tan = (e1 * du2[:, 1:2] - e2 * du1[:, 1:2]) * inv_det[:, None]
    # Zero out tangent contributions from degenerate-UV triangles so they
    # don't pollute neighbouring vertices.
    face_tan = np.where(np.abs(det)[:, None] > 1e-8, face_tan, 0.0).astype(np.float32)

    tan = np.zeros_like(positions)
    for i in range(3):
        np.add.at(tan, tri_idx[:, i], face_tan)

    # Orthogonalise against N and normalise. When |T| ≈ 0 (isolated vertex
    # with no valid-UV triangle neighbours), pick any axis perpendicular to N.
    tan = tan - normals * np.sum(tan * normals, axis=1, keepdims=True)
    lengths = np.linalg.norm(tan, axis=1, keepdims=True)
    fallback = np.where(np.abs(normals[:, :1]) < 0.9,
                        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                        np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
    fallback = fallback - normals * np.sum(fallback * normals, axis=1, keepdims=True)
    fallback_n = fallback / np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-8)
    tan = np.where(lengths > 1e-6, tan / np.maximum(lengths, 1e-8), fallback_n)
    return tan.astype(np.float32)


def _bake_normal_map_into_normals(
    normals: np.ndarray,
    tangents: np.ndarray,
    uvs: np.ndarray,
    normal_bytes: bytes,
    normal_res: int,
    strength: float,
) -> np.ndarray:
    """Rotate per-vertex N by the tangent-space perturbation of the normal map.

    Matches the shader-side decoding in main_pass.slang:138-143: nxy from
    the (R, G) channels in [-1, 1], scaled by strength, z rebuilt to unit
    length. Produces an output guaranteed unit-length and roughly consistent
    with the runtime path so disabling the bake (subdivision=Off) gives a
    similar look at coarser resolution.
    """
    rg = _bilinear_sample_rg8(normal_bytes, normal_res, uvs)
    nxy = (rg * 2.0 - 1.0) * float(max(0.0, min(4.0, strength)))
    nxy_len2 = np.sum(nxy * nxy, axis=1, keepdims=True)
    nz = np.sqrt(np.maximum(1.0 - nxy_len2, 1e-4))
    bitangents = np.cross(normals, tangents)
    out = (tangents * nxy[:, 0:1]
           + bitangents * nxy[:, 1:2]
           + normals * nz)
    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(lengths, 1e-8)).astype(np.float32)


def pick_auto_subdivision_level(
    disp_activity: float, normal_activity: float, max_level: int = 2
) -> int:
    """Pick a uniform subdivision level from per-map frequency summaries.

    Thresholds were picked from representative face-scan assets: typical
    8-bit normal maps have mean-grad ≈ 0.01–0.03; the Texturing.xyz TIF
    displacement used for the bundled `face` model has std ≈ 0.075. The
    combined score treats either map being active (> its weight-weighted
    threshold) as enough to warrant one level, both being strong as enough
    for two. Users can always override via the explicit modes.
    """
    score = max(disp_activity * 12.0, normal_activity * 30.0)
    if score < 0.35:
        return 0
    if score < 0.9:
        return min(1, max_level)
    return min(2, max_level)


def _bilinear_sample_r8(data: bytes, res: int, uvs: np.ndarray) -> np.ndarray:
    """Bilinear-sample the red channel of an RGBA8 square texture at UV coords.

    Returns a (N,) float32 array in [0, 1]. UVs are clamped — displacement
    maps shouldn't tile, and the face-scan UV layout already falls inside
    [0, 1] anyway. Matches the Vulkan sampler's linear+clamp mode used by
    the shader for the on-GPU path (even though we're no longer using it
    for displacement — still sensible to keep authoring parity).
    """
    arr = np.frombuffer(data, dtype=np.uint8).reshape(res, res, 4)
    u = np.clip(uvs[:, 0].astype(np.float32), 0.0, 1.0) * (res - 1)
    v = np.clip(uvs[:, 1].astype(np.float32), 0.0, 1.0) * (res - 1)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, res - 1)
    y1 = np.minimum(y0 + 1, res - 1)
    fx = u - x0
    fy = v - y0
    a = arr[y0, x0, 0].astype(np.float32) / 255.0
    b = arr[y0, x1, 0].astype(np.float32) / 255.0
    c = arr[y1, x0, 0].astype(np.float32) / 255.0
    d = arr[y1, x1, 0].astype(np.float32) / 255.0
    top = a * (1.0 - fx) + b * fx
    bot = c * (1.0 - fx) + d * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def bake_mesh(
    source: MeshSource,
    subdivision_levels: int,
    displacement_bytes: bytes | None,
    displacement_res: int,
    displacement_scale_world: float,
    normal_bytes: bytes | None = None,
    normal_res: int = 0,
    normal_map_strength: float = 1.0,
) -> Mesh:
    """Subdivide, displace, rebuild normals, (optionally) bake normal map, build BVH.

    Displacement offset: ``offset = (disp - 0.5) * scale_world``, evaluated
    along each vertex's (pre-displacement) smooth normal.  The (disp - 0.5)
    bias maps mid-grey to zero offset; scale_world is already in world
    units (displacement_scale_mm / mm_per_unit). Smooth normals are re-synthesised from
    the displaced geometry so specular / shading responds to the new surface
    relief. When `normal_bytes` is supplied *and* the mesh got subdivided or
    displaced (so the geometry is meaningfully denser than base), each vertex
    normal is additionally rotated by the tangent-space perturbation of the
    normal map at its UV. The Mesh is marked `normals_baked=True` so the
    shader knows to skip its own normal-map sample for mesh hits.
    """
    positions = source.positions
    normals   = source.normals
    uvs       = source.uvs
    tri_idx   = source.tri_idx

    levels = max(0, int(subdivision_levels))
    for _ in range(levels):
        positions, normals, uvs, tri_idx = subdivide_midpoint(
            positions, normals, uvs, tri_idx
        )

    displace_active = (
        displacement_bytes is not None
        and abs(displacement_scale_world) > 1e-9
    )
    if displace_active:
        h = _bilinear_sample_r8(displacement_bytes, displacement_res, uvs)
        offset = (h - 0.5) * displacement_scale_world
        positions = positions + normals * offset[:, None]

    if levels > 0 or displace_active:
        normals = _smooth_normals(positions, tri_idx)

    # Bake the normal map into vertex normals only when we actually added
    # detail to the mesh — baking on the base-res mesh would just smear the
    # map across large triangles and still leave the shader doing the work
    # at pixel resolution, so we keep the runtime path in that case.
    normals_baked = False
    if (
        normal_bytes is not None
        and normal_res > 0
        and (levels > 0 or displace_active)
    ):
        tangents = _per_vertex_tangents(positions, normals, uvs, tri_idx)
        normals = _bake_normal_map_into_normals(
            normals, tangents, uvs, normal_bytes, normal_res, normal_map_strength
        )
        normals_baked = True

    perm_idx, nodes = build_bvh(positions, tri_idx)

    return Mesh(
        name=source.name,
        vertex_bytes=_pack_vertices(positions, normals, uvs),
        index_bytes=_pack_indices(perm_idx),
        bvh_bytes=_pack_bvh(nodes),
        num_vertices=positions.shape[0],
        num_triangles=perm_idx.shape[0],
        num_nodes=len(nodes),
        normal_map=source.normal_map,
        roughness_map=source.roughness_map,
        displacement_map=source.displacement_map,
        normals_baked=normals_baked,
    )


# ── Public entry point ─────────────────────────────────────────────

def load_head_source(path: Path) -> MeshSource:
    """Load an OBJ and normalise to the head frame — no BVH yet."""
    positions, normals, uvs, tri_idx = load_obj(path)
    positions = normalise_to_head_frame(positions)
    return MeshSource(
        name=path.stem,
        positions=positions.astype(np.float32),
        normals=normals.astype(np.float32),
        uvs=uvs.astype(np.float32),
        tri_idx=tri_idx.astype(np.int32),
    )


def discover_mesh_sources(head_dir: Path | None) -> list[MeshSource]:
    """Scan `head_dir` and return undisplaced MeshSource objects.

    Scans two sources, in order:
      1. Each immediate subdirectory that contains at least one `.obj` is
         treated as a named model. Texture files inside the directory are
         attached by filename keyword (normal/roughness/displacement).
      2. Any loose `*.obj` directly under `head_dir` is loaded as a model
         with no texture maps — preserves existing behaviour for flat layouts.
    """
    if head_dir is None or not head_dir.exists():
        return []
    out: list[MeshSource] = []

    for sub in sorted(p for p in head_dir.iterdir() if p.is_dir()):
        src = _load_model_dir(sub)
        if src is not None:
            out.append(src)

    for p in sorted(head_dir.glob("*.obj")):
        try:
            src = load_head_source(p)
            print(
                f"[skinny] loaded head mesh: {p.name} "
                f"({src.positions.shape[0]} verts, {src.tri_idx.shape[0]} tris)"
            )
            out.append(src)
        except Exception as exc:  # noqa: BLE001
            print(f"[skinny] failed to load {p.name}: {exc}")
    return out


def dummy_mesh() -> Mesh:
    """A single far-offscreen triangle with a 1-node BVH.

    Used when no OBJ is supplied so the descriptor layout always has valid
    storage buffers bound, even when the shader path is disabled.
    """
    positions = np.array(
        [[1e4, 1e4, 1e4], [1e4 + 1, 1e4, 1e4], [1e4, 1e4 + 1, 1e4]], dtype=np.float32
    )
    normals = np.zeros_like(positions)
    normals[:, 1] = 1.0
    uvs = np.zeros((3, 2), dtype=np.float32)
    tri_idx = np.array([[0, 1, 2]], dtype=np.int32)
    perm_idx, nodes = build_bvh(positions, tri_idx)
    return Mesh(
        name="(none)",
        vertex_bytes=_pack_vertices(positions, normals, uvs),
        index_bytes=_pack_indices(perm_idx),
        bvh_bytes=_pack_bvh(nodes),
        num_vertices=positions.shape[0],
        num_triangles=perm_idx.shape[0],
        num_nodes=len(nodes),
    )
