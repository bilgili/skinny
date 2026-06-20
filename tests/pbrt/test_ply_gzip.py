"""Transparent gzip decompression of `plymesh` PLY input in the pbrt importer.

pbrt v4 scenes ship large meshes gzip-compressed (`*.ply.gz`) and pbrt itself
gunzips them on load. `read_ply` must do the same, gated on the gzip magic bytes
(`0x1f 0x8b`) so `.ply` and `.ply.gz` parse identically.
"""

from __future__ import annotations

import gzip
import struct

import numpy as np
import pytest

from skinny.pbrt.ply import read_ply


def _ply_ascii():
    return (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0\n"
        "1 0 0\n"
        "1 1 0\n"
        "3 0 1 2\n"
    ).encode("ascii")


def _ply_binary_le():
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    verts = [(0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 1)]
    body = b"".join(struct.pack("<6f", *vt) for vt in verts)
    body += struct.pack("<B", 3) + struct.pack("<3i", 0, 1, 2)
    return header + body


def _assert_same_mesh(a, b):
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.indices, b.indices)
    assert (a.normals is None) == (b.normals is None)
    if a.normals is not None:
        assert np.array_equal(a.normals, b.normals)
    assert (a.uvs is None) == (b.uvs is None)
    if a.uvs is not None:
        assert np.array_equal(a.uvs, b.uvs)


@pytest.mark.parametrize("raw", [_ply_ascii(), _ply_binary_le()])
def test_gzip_ply_reads_like_plain(tmp_path, raw):
    plain = tmp_path / "m.ply"
    plain.write_bytes(raw)
    gz = tmp_path / "m.ply.gz"
    gz.write_bytes(gzip.compress(raw))

    _assert_same_mesh(read_ply(str(gz)), read_ply(str(plain)))


def test_uncompressed_ply_unaffected(tmp_path):
    raw = _ply_binary_le()
    p = tmp_path / "m.ply"
    p.write_bytes(raw)
    mesh = read_ply(str(p))
    assert mesh.points.shape == (3, 3)
    assert mesh.indices.shape == (1, 3)


def test_gzip_wrapping_non_ply_raises(tmp_path):
    p = tmp_path / "bad.ply.gz"
    p.write_bytes(gzip.compress(b"this is not a ply file\n"))
    with pytest.raises(ValueError):
        read_ply(str(p))
