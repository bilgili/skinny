"""Pure-Python NanoVDB (``.nvdb``) reader: float density grids to a dense array.

Parses the NanoVDB v32.x file layout (the ABI pbrt v4 pins — vendored headers at
``pbrt-v4/src/ext/openvdb/nanovdb``) for **Float / FogVolume** grids only and
decodes the sparse tree (root tiles → upper 32³ → lower 16³ → leaf 8³ nodes) into
a dense ``numpy.float32`` array plus the index→world transform and value range.
NONE and ZIP (zlib) codecs are supported; everything else fails loudly.

Decode follows NanoVDB ``getValue`` semantics: the dense array is pre-filled with
the root background (0 for fog volumes) and node/tile values are written verbatim
— for fog volumes inactive voxels hold the background, so this is exactly the
field pbrt's grid sampler sees. Not a general NanoVDB library.

Public entry: :func:`read_nanovdb(path, field="density") -> NanoVdbGrid`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["NanoVdbError", "NanoVdbUnsupportedError", "NanoVdbGrid", "read_nanovdb"]

# "NanoVDB0" little-endian (NANOVDB_MAGIC_NUMBER).
_MAGIC = 0x304244566F6E614E
# The file format follows the in-memory ABI, versioned by the major number
# (Version = major<<21 | minor<<10 | patch). pbrt v4 pins 32.3.3; the struct
# offsets below are the v32 layout and are wrong for other majors.
_SUPPORTED_MAJOR = 32

_CODEC_NONE, _CODEC_ZIP, _CODEC_BLOSC = 0, 1, 2
_CODEC_NAMES = {_CODEC_NONE: "none", _CODEC_ZIP: "zip", _CODEC_BLOSC: "blosc"}

_GRID_TYPE_FLOAT = 1
_GRID_TYPE_NAMES = {
    0: "Unknown", 1: "Float", 2: "Double", 3: "Int16", 4: "Int32", 5: "Int64",
    6: "Vec3f", 7: "Vec3d", 8: "Mask", 9: "Half", 10: "UInt32", 11: "Boolean",
    12: "RGBA8", 13: "Fp4", 14: "Fp8", 15: "Fp16", 16: "FpN", 17: "Vec4f", 18: "Vec4d",
}
_GRID_CLASS_UNKNOWN, _GRID_CLASS_FOG = 0, 2
_GRID_CLASS_NAMES = {
    0: "unknown", 1: "LevelSet", 2: "fog_volume", 3: "Staggered", 4: "PointIndex",
    5: "PointData", 6: "Topology", 7: "VoxelVolume",
}

# util/IO.h Header: u64 magic, u32 version, u16 gridCount, u16 codec.
_FILE_HEADER = struct.Struct("<QIHH")
# util/IO.h MetaData (176 B): gridSize, fileSize, nameKey, voxelCount (u64 each),
# gridType u32, gridClass u32, worldBBox 6d, indexBBox 6i, voxelSize 3d,
# nameSize u32, nodeCount 4u32, tileCount 3u32, codec u16, pad u16, version u32.
_FILE_META = struct.Struct("<4Q II 6d 6i 3d I 4I 3I HH I")

# GridData (672 B): magic u64, checksum u64, version u32, flags u32, gridIndex u32,
# gridCount u32, gridSize u64 @32, gridName[256] @40, Map @296 (264 B), worldBBox
# 6d @560, voxelSize 3d @608, gridClass u32 @632, gridType u32 @636,
# blindMetadataOffset i64 @640, blindMetadataCount u32 @648, pad to 672.
_GRID_DATA_SIZE = 672
# Map: mMatF 9f, mInvMatF 9f, mVecF 3f, mTaperF f (88 B) then mMatD 9d, mInvMatD 9d,
# mVecD 3d, mTaperD d. mMatD applies row-major: world = M @ ijk + mVecD (matMult).
_MAP_OFFSET = 296
_MAT_D_OFFSET = _MAP_OFFSET + 88
_VEC_D_OFFSET = _MAT_D_OFFSET + 9 * 8 + 9 * 8
# TreeData (64 B, right after GridData): mNodeOffset[4] u64 (byte offsets from the
# tree to the first leaf/lower/upper node and to the root), mNodeCount[3] u32,
# mTileCount[3] u32, mVoxelCount u64.
_TREE_DATA = struct.Struct("<4Q 3I 3I Q")

# RootData<float>: BBox<Coord> 6i, mTableSize u32, background/min/max/avg/stddev f32,
# padded to 32-byte alignment -> 64 B; tiles follow immediately.
_ROOT_DATA = struct.Struct("<6i I 5f")
_ROOT_DATA_SIZE = 64
# Root Tile (32 B): key u64 (USE_SINGLE_ROOT_KEY), child i64 (byte offset from the
# root, 0 = value tile), state u32, value f32.
_ROOT_TILE = struct.Struct("<Q q I f")
_ROOT_TILE_SIZE = 32

# InternalData<float, LOG2DIM>: BBox<Coord> 6i + flags u64 (32 B), value mask,
# child mask (1<<3*LOG2DIM bits each), min/max/avg/stddev f32, then the 32-aligned
# tile table (union {f32 value; i64 child}, 8 B per entry).
_UPPER_LOG2, _LOWER_LOG2, _LEAF_LOG2 = 5, 4, 3
_UPPER_SPAN = 1 << (_UPPER_LOG2 + _LOWER_LOG2 + _LEAF_LOG2)  # 4096: voxels per side
_LOWER_SPAN = 1 << (_LOWER_LOG2 + _LEAF_LOG2)  # 128
_LEAF_SPAN = 1 << _LEAF_LOG2  # 8


def _internal_dtype(log2dim: int) -> np.dtype:
    words = (1 << 3 * log2dim) // 64
    entries = 1 << 3 * log2dim
    stats_end = 32 + 2 * 8 * words + 16
    table_off = (stats_end + 31) // 32 * 32  # alignas(32) mTable
    return np.dtype({
        "names": ["bbox", "flags", "value_mask", "child_mask", "table_raw"],
        "formats": [("<i4", 6), "<u8", ("<u8", words), ("<u8", words), ("V8", entries)],
        "offsets": [0, 24, 32, 32 + 8 * words, table_off],
        "itemsize": table_off + 8 * entries,
    })


_UPPER_DTYPE = _internal_dtype(_UPPER_LOG2)  # itemsize 270400
_LOWER_DTYPE = _internal_dtype(_LOWER_LOG2)  # itemsize 33856

# LeafData<float>: mBBoxMin 3i, mBBoxDif 3u8, mFlags u8, value mask 64 B,
# min/max/avg/stddev f32, then alignas(32) f32 values[512] -> 2144 B.
_LEAF_DTYPE = np.dtype({
    "names": ["bbox_min", "bbox_dif", "flags", "value_mask", "values"],
    "formats": [("<i4", 3), ("u1", 3), "u1", ("<u8", 8), ("<f4", 512)],
    "offsets": [0, 12, 15, 16, 96],
    "itemsize": 2144,
})


class NanoVdbError(RuntimeError):
    """Malformed or unreadable NanoVDB file."""


class NanoVdbUnsupportedError(NanoVdbError):
    """Valid NanoVDB file using a feature this reader does not support."""


@dataclass
class NanoVdbGrid:
    density: np.ndarray  # float32, shape (nx, ny, nz), dense, origin at index_min
    index_min: tuple[int, int, int]  # ijk of density[0, 0, 0]
    index_to_world: np.ndarray  # 4x4 float64, maps index-space (i,j,k,1) voxel centers to world
    value_min: float
    value_max: float
    grid_name: str
    grid_class: str  # "fog_volume" | "unknown"
    codec: str  # "none" | "zip"


def _version_str(version: int) -> str:
    return f"{version >> 21}.{version >> 10 & 0x7FF}.{version & 0x3FF}"


def _root_key_to_coord(key: np.ndarray) -> np.ndarray:
    """Invert RootData::CoordToKey: 21 bits per axis (z low, y mid, x high), each
    holding uint32(coord) >> 12; recover the signed upper-node origin."""
    shift = _UPPER_LOG2 + _LOWER_LOG2 + _LEAF_LOG2
    mask = np.uint64((1 << 21) - 1)
    packed = np.stack([key >> np.uint64(42), key >> np.uint64(21), key], axis=-1)
    return ((packed & mask) << np.uint64(shift)).astype(np.uint32).view(np.int32)


def _offset_to_local(offsets: np.ndarray, log2dim: int) -> np.ndarray:
    """InternalNode::OffsetToLocalCoord: n = (x << 2*LOG2DIM) | (y << LOG2DIM) | z."""
    dim_mask = (1 << log2dim) - 1
    return np.stack(
        [offsets >> 2 * log2dim, offsets >> log2dim & dim_mask, offsets & dim_mask], axis=-1)


def _fill_box(density: np.ndarray, index_min: np.ndarray, origin: np.ndarray, span: int,
              value: float) -> None:
    """Broadcast a constant tile value over its span, clipped to the dense array."""
    lo = origin - index_min
    hi = lo + span
    lo_c = np.maximum(lo, 0)
    hi_c = np.minimum(hi, density.shape)
    if np.any(lo_c >= hi_c):
        return
    density[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]] = value


def _decode_internal_tiles(density: np.ndarray, index_min: np.ndarray, nodes: np.ndarray,
                           log2dim: int, child_span: int, background: float) -> None:
    """Write the constant (non-child) tile values of one internal-node level."""
    for node in nodes:
        # InternalNode::origin(): bbox min snapped down to the node span.
        origin = node["bbox"][:3] & ~(child_span * (1 << log2dim) - 1)
        table = node["table_raw"]
        values = np.frombuffer(table.tobytes(), dtype="<f4").reshape(-1, 2)[:, 0]
        child_bits = np.unpackbits(node["child_mask"].view(np.uint8), bitorder="little")
        tile_idx = np.nonzero((child_bits == 0) & (values != background))[0]
        if tile_idx.size == 0:
            continue
        locals_ = _offset_to_local(tile_idx, log2dim)
        for local, value in zip(locals_, values[tile_idx]):
            _fill_box(density, index_min, origin + local * child_span, child_span, float(value))


def read_nanovdb(path, field: str = "density") -> NanoVdbGrid:
    """Read the named float grid of a ``.nvdb`` file into a :class:`NanoVdbGrid`."""
    path = Path(path)
    with open(path, "rb") as fh:
        data = fh.read()

    seen_names: list[str] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < _FILE_HEADER.size:
            raise NanoVdbError(f"{path}: truncated NanoVDB segment header at byte {offset}")
        magic, version, grid_count, codec = _FILE_HEADER.unpack_from(data, offset)
        offset += _FILE_HEADER.size
        if magic != _MAGIC:
            raise NanoVdbError(f"{path}: bad NanoVDB magic {magic:#018x} at byte {offset - 16}"
                               f" (expected {_MAGIC:#018x} 'NanoVDB0')")
        if version >> 21 != _SUPPORTED_MAJOR:
            raise NanoVdbUnsupportedError(
                f"{path}: unsupported NanoVDB file version {_version_str(version)}"
                f" (this reader implements the v{_SUPPORTED_MAJOR}.x ABI pbrt v4 pins)")
        if codec == _CODEC_BLOSC:
            raise NanoVdbUnsupportedError(f"{path}: unsupported NanoVDB codec BLOSC"
                                          " (only NONE and ZIP are supported)")
        if codec not in (_CODEC_NONE, _CODEC_ZIP):
            raise NanoVdbError(f"{path}: unknown NanoVDB codec id {codec}")

        metas = []
        for _ in range(grid_count):
            fields = _FILE_META.unpack_from(data, offset)
            offset += _FILE_META.size
            # fields: 0-3 gridSize/fileSize/nameKey/voxelCount, 4 gridType,
            # 5 gridClass, 6-11 worldBBox, 12-17 indexBBox, 18-20 voxelSize,
            # 21 nameSize, 22-25 nodeCount, 26-28 tileCount, 29-31 codec/pad/version.
            grid_size, file_size, name_size = fields[0], fields[1], fields[21]
            name = data[offset:offset + name_size].split(b"\0")[0].decode("utf-8", "replace")
            offset += name_size
            metas.append((name, grid_size, file_size, fields[4], fields[5]))

        for name, grid_size, file_size, grid_type, grid_class in metas:
            if name != field:
                seen_names.append(name)
                # Skip this grid's blob: fileSize is its on-disk byte count
                # (including the ZIP size prefix).
                offset += file_size if codec != _CODEC_NONE else grid_size
                continue
            if grid_type != _GRID_TYPE_FLOAT:
                type_name = _GRID_TYPE_NAMES.get(grid_type, f"id {grid_type}")
                raise NanoVdbUnsupportedError(
                    f"{path}: grid {name!r} has unsupported type {type_name}"
                    " (only Float density grids are supported)")
            if grid_class not in (_GRID_CLASS_UNKNOWN, _GRID_CLASS_FOG):
                class_name = _GRID_CLASS_NAMES.get(grid_class, f"id {grid_class}")
                raise NanoVdbUnsupportedError(
                    f"{path}: grid {name!r} has unsupported class {class_name}"
                    " (only FogVolume/Unknown density grids are supported)")
            if codec == _CODEC_ZIP:
                # ZIP frames each grid blob as u64 compressed size + one zlib stream
                # of the whole grid (util/IO.h Internal::read).
                (comp_size,) = struct.unpack_from("<Q", data, offset)
                comp = data[offset + 8:offset + 8 + comp_size]
                if len(comp) != comp_size:
                    raise NanoVdbError(f"{path}: truncated ZIP blob for grid {name!r}")
                blob = zlib.decompress(comp, bufsize=grid_size)
                del comp
            else:
                blob = data[offset:offset + grid_size]
            if len(blob) != grid_size:
                raise NanoVdbError(f"{path}: grid {name!r} blob is {len(blob)} bytes,"
                                   f" metadata promises {grid_size}")
            del data
            return _decode_grid(blob, path, _CODEC_NAMES[codec])

    raise NanoVdbError(f"{path}: no grid named {field!r}"
                       f" (found: {', '.join(repr(n) for n in seen_names) or 'none'})")


def _decode_grid(blob: bytes, path: Path, codec: str) -> NanoVdbGrid:
    """Decode one uncompressed v32 grid blob (GridData + tree) to a dense array."""
    (grid_magic,) = struct.unpack_from("<Q", blob, 0)
    if grid_magic != _MAGIC:
        raise NanoVdbError(f"{path}: grid blob magic {grid_magic:#018x} is not 'NanoVDB0'")
    grid_name = blob[40:296].split(b"\0")[0].decode("utf-8", "replace")
    mat_d = np.array(struct.unpack_from("<9d", blob, _MAT_D_OFFSET)).reshape(3, 3)
    vec_d = np.array(struct.unpack_from("<3d", blob, _VEC_D_OFFSET))
    grid_class, grid_type = struct.unpack_from("<II", blob, 632)
    if grid_type != _GRID_TYPE_FLOAT:  # cross-check the blob against the file metadata
        raise NanoVdbError(f"{path}: grid blob type {grid_type} contradicts Float metadata")

    tree_off = _GRID_DATA_SIZE
    tree = _TREE_DATA.unpack_from(blob, tree_off)
    node_offsets, node_counts = tree[:4], tree[4:7]
    root_off = tree_off + node_offsets[3]
    root = _ROOT_DATA.unpack_from(blob, root_off)
    bbox_min = np.array(root[:3], np.int64)
    bbox_max = np.array(root[3:6], np.int64)
    table_size = root[6]
    background, value_min, value_max = root[7], root[8], root[9]
    if np.any(bbox_max < bbox_min):
        if table_size:
            raise NanoVdbError(f"{path}: grid {grid_name!r} has an empty index bbox"
                               f" but {table_size} root tiles")
        bbox_min = np.zeros(3, np.int64)
        bbox_max = -np.ones(3, np.int64)
    dims = tuple(int(n) for n in bbox_max - bbox_min + 1)

    density = np.full(dims, background, np.float32)

    # Root tiles: constant values span an entire upper node (4096 voxels/side).
    if table_size:
        tiles = np.frombuffer(blob, np.dtype([
            ("key", "<u8"), ("child", "<i8"), ("state", "<u4"), ("value", "<f4"),
            ("pad", "V8")]), count=table_size, offset=root_off + _ROOT_DATA_SIZE)
        for tile in tiles[(tiles["child"] == 0) & (tiles["value"] != background)]:
            origin = _root_key_to_coord(tile["key"])
            _fill_box(density, bbox_min, origin.astype(np.int64), _UPPER_SPAN,
                      float(tile["value"]))
        del tiles

    # Internal-node constant tiles, iterated over the contiguous per-level arrays
    # (mNodeOffset/mNodeCount): upper tiles span 128 voxels, lower tiles span 8.
    upper = np.frombuffer(blob, _UPPER_DTYPE, count=node_counts[2],
                          offset=tree_off + node_offsets[2])
    _decode_internal_tiles(density, bbox_min, upper, _UPPER_LOG2, _LOWER_SPAN, background)
    del upper
    lower = np.frombuffer(blob, _LOWER_DTYPE, count=node_counts[1],
                          offset=tree_off + node_offsets[1])
    _decode_internal_tiles(density, bbox_min, lower, _LOWER_LOG2, _LEAF_SPAN, background)
    del lower

    # Leaves: verbatim 8x8x8 value blocks (offset n = x<<6 | y<<3 | z matches the
    # (nx, ny, nz) C-order dense array), clipped to the active-value bbox.
    leaves = np.frombuffer(blob, _LEAF_DTYPE, count=node_counts[0],
                           offset=tree_off + node_offsets[0])
    origins = (leaves["bbox_min"] & ~(_LEAF_SPAN - 1)) - bbox_min[None, :]
    for origin, values in zip(origins, leaves["values"]):
        block = values.reshape(_LEAF_SPAN, _LEAF_SPAN, _LEAF_SPAN)
        lo_c = np.maximum(origin, 0)
        hi_c = np.minimum(origin + _LEAF_SPAN, dims)
        if np.any(lo_c >= hi_c):
            continue
        density[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]] = block[
            lo_c[0] - origin[0]:hi_c[0] - origin[0],
            lo_c[1] - origin[1]:hi_c[1] - origin[1],
            lo_c[2] - origin[2]:hi_c[2] - origin[2]]
    del leaves

    index_to_world = np.eye(4)
    index_to_world[:3, :3] = mat_d  # matMult applies row-major: world = M @ ijk + vec
    index_to_world[:3, 3] = vec_d
    return NanoVdbGrid(
        density=density,
        index_min=(int(bbox_min[0]), int(bbox_min[1]), int(bbox_min[2])),
        index_to_world=index_to_world,
        value_min=float(value_min),
        value_max=float(value_max),
        grid_name=grid_name,
        grid_class=_GRID_CLASS_NAMES[grid_class]
        if grid_class in (_GRID_CLASS_UNKNOWN, _GRID_CLASS_FOG)
        else _GRID_CLASS_NAMES.get(grid_class, f"id {grid_class}"),
        codec=codec,
    )
