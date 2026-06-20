"""Minimal PLY reader for pbrt ``plymesh`` (ascii + binary little/big endian).

Reads vertex ``x y z`` (plus optional ``nx ny nz`` and ``u/v`` or ``s/t``) and
a face list of vertex indices, triangulating polygons with a fan. Enough for
the meshes pbrt scenes ship; not a general PLY library.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass

import numpy as np

_NUMPY_FMT = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_STRUCT_FMT = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


@dataclass
class PlyMesh:
    points: np.ndarray  # (N,3)
    indices: np.ndarray  # (M,3)
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None


def read_ply(path: str) -> PlyMesh:
    with open(path, "rb") as fh:
        data = fh.read()
    # pbrt ships large meshes gzip-compressed (`*.ply.gz`) and gunzips them on
    # load; sniff the gzip magic (independent of the filename) and do the same.
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    nl = data.index(b"\n")
    if data[:nl].strip() != b"ply":
        raise ValueError(f"{path}: not a PLY file")
    header_end = data.index(b"end_header")
    header_end = data.index(b"\n", header_end) + 1
    header = data[:header_end].decode("ascii", "replace")
    body = data[header_end:]

    fmt = "ascii"
    elements: list[tuple[str, int, list]] = []  # (name, count, props)
    for line in header.splitlines():
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "format":
            fmt = toks[1]
        elif toks[0] == "element":
            elements.append((toks[1], int(toks[2]), []))
        elif toks[0] == "property" and elements:
            elements[-1][2].append(toks[1:])

    if fmt == "ascii":
        return _read_ascii(body, elements)
    little = fmt == "binary_little_endian"
    return _read_binary(body, elements, little)


def _vertex_columns(props):
    names = [pr[-1] for pr in props]
    return names


def _read_ascii(body: bytes, elements) -> PlyMesh:
    tokens = body.split()
    pos = 0
    points = normals = uvs = None
    indices: list = []
    for name, count, props in elements:
        names = _vertex_columns(props)
        if name == "vertex":
            ncol = len(props)
            vals = np.array(tokens[pos : pos + count * ncol], dtype=np.float64).reshape(count, ncol)
            pos += count * ncol
            points, normals, uvs = _split_vertex(vals, names)
        elif name == "face":
            for _ in range(count):
                k = int(tokens[pos])
                idx = [int(x) for x in tokens[pos + 1 : pos + 1 + k]]
                pos += 1 + k
                indices.extend(_fan(idx))
        else:
            pos += count * len(props)
    return PlyMesh(points, np.array(indices, dtype=np.int64), normals, uvs)


def _read_binary(body: bytes, elements, little: bool) -> PlyMesh:
    endian = "<" if little else ">"
    off = 0
    points = normals = uvs = None
    indices: list = []
    for name, count, props in elements:
        if name == "vertex" and all(pr[0] == "property" or True for pr in props):
            names = _vertex_columns(props)
            dtype = np.dtype([(pr[-1], endian + _NUMPY_FMT[pr[0]]) for pr in props])
            arr = np.frombuffer(body, dtype=dtype, count=count, offset=off)
            off += dtype.itemsize * count
            vals = np.stack([arr[n].astype(np.float64) for n in names], axis=1)
            points, normals, uvs = _split_vertex(vals, names)
        elif name == "face":
            for _ in range(count):
                ctype = props[0][1]  # list count type
                itype = props[0][2]  # list index type
                k = struct.unpack_from(endian + _STRUCT_FMT[ctype], body, off)[0]
                off += struct.calcsize(_STRUCT_FMT[ctype])
                isz = struct.calcsize(_STRUCT_FMT[itype])
                idx = list(struct.unpack_from(endian + _STRUCT_FMT[itype] * k, body, off))
                off += isz * k
                indices.extend(_fan(idx))
        else:
            # skip unknown fixed-size element
            dtype = np.dtype([(pr[-1], endian + _NUMPY_FMT[pr[0]]) for pr in props])
            off += dtype.itemsize * count
    return PlyMesh(points, np.array(indices, dtype=np.int64), normals, uvs)


def _split_vertex(vals: np.ndarray, names: list[str]):
    col = {n: i for i, n in enumerate(names)}
    points = vals[:, [col["x"], col["y"], col["z"]]]
    normals = None
    if all(n in col for n in ("nx", "ny", "nz")):
        normals = vals[:, [col["nx"], col["ny"], col["nz"]]]
    uvs = None
    for u, v in (("u", "v"), ("s", "t"), ("texture_u", "texture_v")):
        if u in col and v in col:
            uvs = vals[:, [col[u], col[v]]]
            break
    return points, normals, uvs


def _fan(idx: list[int]) -> list[list[int]]:
    return [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
