"""Tests for the pure-Python NanoVDB reader (``skinny.pbrt.nanovdb``).

The synthetic-fixture writer below authors exactly the v32 subset the reader
parses (file header + metadata + GridData/TreeData/root/upper/lower/leaf blob)
so known voxel values round-trip end-to-end, including root/upper/lower constant
tiles and both codecs. Real-file smoke tests cover the two pbrt-v4-scenes clouds.
"""

import os
import struct
import zlib

import numpy as np
import pytest

from skinny.pbrt.nanovdb import (
    NanoVdbError,
    NanoVdbUnsupportedError,
    read_nanovdb,
)

_MAGIC = 0x304244566F6E614E
_VERSION_32_3_3 = (32 << 21) | (3 << 10) | 3

WDAS_CLOUD = os.path.expanduser(
    "~/projects/pbrt-v4-scenes/disney-cloud/wdas_cloud_quarter.nvdb")
BUNNY_CLOUD = os.path.expanduser(
    "~/projects/pbrt-v4-scenes/bunny-cloud/bunny_cloud.nvdb")


# ---------------------------------------------------------------------------
# Synthetic .nvdb writer (v32 layout, single float grid)
# ---------------------------------------------------------------------------

_UPPER_SIZE = 270400  # sizeof(InternalData<float, 5>)
_LOWER_SIZE = 33856  # sizeof(InternalData<float, 4>)
_LEAF_SIZE = 2144  # sizeof(LeafData<float>)


def _mask_words(bits, count):
    words = np.zeros(count // 64, np.uint64)
    for b in bits:
        words[b // 64] |= np.uint64(1) << np.uint64(b % 64)
    return words.tobytes()


def _internal_node(log2dim, bbox, children, value_tiles):
    """children: {n: byte offset rel. to this node}; value_tiles: {n: value}."""
    entries = 1 << 3 * log2dim
    words = 2 * (entries // 64) * 8
    stats_end = 32 + words + 16
    table_off = (stats_end + 31) // 32 * 32
    blob = bytearray(table_off + 8 * entries)
    struct.pack_into("<6iQ", blob, 0, *bbox, 0)  # bbox + flags
    blob[32:32 + words // 2] = _mask_words(value_tiles, entries)
    blob[32 + words // 2:32 + words] = _mask_words(children, entries)
    struct.pack_into("<4f", blob, 32 + words, 0.0, 0.0, 0.0, 0.0)
    for n, child in children.items():
        struct.pack_into("<q", blob, table_off + 8 * n, child)
    for n, value in value_tiles.items():
        struct.pack_into("<f", blob, table_off + 8 * n, value)
    return bytes(blob)


def _leaf_node(origin, values):
    blob = bytearray(_LEAF_SIZE)
    struct.pack_into("<3i3Bb", blob, 0, *origin, 7, 7, 7, 0)
    blob[16:80] = _mask_words(range(512), 512)
    struct.pack_into("<4f", blob, 80, 0.0, 0.0, 0.0, 0.0)
    blob[96:96 + 2048] = np.asarray(values, "<f4").tobytes()
    return bytes(blob)


def _root_key(ijk):
    x, y, z = (np.uint32(c) >> np.uint32(12) for c in ijk)
    return (int(x) << 42) | (int(y) << 21) | int(z)


def _synthetic_grid_blob(name="density", grid_class=2, grid_type=1, version=_VERSION_32_3_3,
                         grid_magic=_MAGIC):
    """One grid blob: two leaves + lower/upper/root constant tiles, bbox-clipped.

    Expected dense field (bbox min (0,0,0), max (4097, 15, 129)):
      * leaf A  x 0..7,       y 0..7,  z 0..7    -> arange(512)/511 pattern
      * leaf B  x 8..15,      y 0..7,  z 0..7    -> constant 2.5
      * lower tile (8^3)      x 0..7,  y 8..15, z 0..7   -> 0.25
      * upper tile (128^3)    x 0..127, y 0..15, z 128..129 (clipped) -> 0.75
      * root tile (4096^3)    x 4096..4097 (clipped), y 0..15, z 0..129 -> 0.5
    """
    bbox = (0, 0, 0, 4097, 15, 129)
    leaf_a = np.arange(512, dtype=np.float32) / 511.0
    leaf_b = np.full(512, 2.5, np.float32)

    # Tree layout (offsets relative to the tree): TreeData 64 B, root (64 B +
    # 2 tiles x 32 B), upper, lower, then the two leaves.
    root_off, upper_off = 64, 64 + 64 + 2 * 32
    lower_off = upper_off + _UPPER_SIZE
    leaf_off = lower_off + _LOWER_SIZE
    tree_size = leaf_off + 2 * _LEAF_SIZE
    grid_size = 672 + tree_size

    grid = bytearray(672)
    struct.pack_into("<QQIIIIQ", grid, 0, grid_magic, 0, version, 0, 0, 1, grid_size)
    grid[40:40 + len(name)] = name.encode()
    # Map: float then double affine; the reader consumes mMatD (row-major) + mVecD.
    struct.pack_into("<9f", grid, 296, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5)
    struct.pack_into("<3f", grid, 296 + 72, 1.0, 2.0, 3.0)
    struct.pack_into("<9d", grid, 296 + 88, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5)
    struct.pack_into("<3d", grid, 296 + 88 + 144, 1.0, 2.0, 3.0)
    struct.pack_into("<II", grid, 632, grid_class, grid_type)

    tree = bytearray(64)
    struct.pack_into("<4Q3I3IQ", tree, 0, leaf_off, lower_off, upper_off, root_off,
                     2, 1, 1, 1, 1, 1, 1024 + 512)

    root = bytearray(64)
    struct.pack_into("<6iI5f", root, 0, *bbox, 2, 0.0, 0.0, 2.5, 0.0, 0.0)
    tiles = struct.pack("<QqIf8x", _root_key((0, 0, 0)), upper_off - root_off, 0, 0.0)
    tiles += struct.pack("<QqIf8x", _root_key((4096, 0, 0)), 0, 1, 0.5)

    # Upper (32^3 table, children span 128): lower child at local (0,0,0) -> n=0;
    # constant tile at local (0,0,1) -> n=1 (n = x<<10 | y<<5 | z).
    upper = _internal_node(5, (0, 0, 0, 15, 15, 129),
                           {0: lower_off - upper_off}, {1: 0.75})
    # Lower (16^3 table, children span 8): leaves at local (0,0,0) -> n=0 and
    # (1,0,0) -> n=256 (n = x<<8 | y<<4 | z); constant tile at (0,1,0) -> n=16.
    lower = _internal_node(4, (0, 0, 0, 15, 15, 7),
                           {0: leaf_off - lower_off, 256: leaf_off + _LEAF_SIZE - lower_off},
                           {16: 0.25})
    blob = bytes(grid) + bytes(tree) + bytes(root) + tiles + upper + lower
    blob += _leaf_node((0, 0, 0), leaf_a) + _leaf_node((8, 0, 0), leaf_b)
    assert len(blob) == grid_size
    return blob


def _expected_synthetic_density():
    dense = np.zeros((4098, 16, 130), np.float32)
    dense[4096:4098, :, :] = 0.5  # root tile, clipped
    dense[0:128, 0:16, 128:130] = 0.75  # upper tile, clipped
    dense[0:8, 8:16, 0:8] = 0.25  # lower tile
    dense[0:8, 0:8, 0:8] = (np.arange(512, dtype=np.float32) / 511.0).reshape(8, 8, 8)
    dense[8:16, 0:8, 0:8] = 2.5
    return dense


def _write_nvdb(path, blobs, codec=0, file_magic=_MAGIC, version=_VERSION_32_3_3):
    """blobs: list of (name, grid_blob, grid_type, grid_class)."""
    out = bytearray(struct.pack("<QIHH", file_magic, version, len(blobs), codec))
    encoded = []
    for name, blob, gtype, gclass in blobs:
        if codec == 1:
            comp = zlib.compress(blob)
            disk = struct.pack("<Q", len(comp)) + comp
        else:
            disk = blob
        encoded.append(disk)
        name_b = name.encode() + b"\0"
        out += struct.pack("<4QII", len(blob), len(disk), 0, 0, gtype, gclass)
        out += struct.pack("<6d", *([0.0] * 6))
        out += struct.pack("<6i", 0, 0, 0, 4097, 15, 129)
        out += struct.pack("<3d", 0.5, 0.5, 0.5)
        out += struct.pack("<I", len(name_b))
        out += struct.pack("<4I3I", 2, 1, 1, 1, 1, 1, 1)
        out += struct.pack("<HHI", codec, 0, version)
        out += name_b
    for disk in encoded:
        out += disk
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def synthetic_nvdb(tmp_path):
    def make(codec=0, name="density", grid_class=2, grid_type=1, meta_type=None,
             meta_class=None, file_magic=_MAGIC, version=_VERSION_32_3_3, extra_first=False):
        blob = _synthetic_grid_blob(name=name, grid_class=grid_class, grid_type=grid_type,
                                    version=version)
        blobs = [(name, blob, meta_type if meta_type is not None else grid_type,
                  meta_class if meta_class is not None else grid_class)]
        if extra_first:
            blobs.insert(0, ("temperature", _synthetic_grid_blob(name="temperature"), 1, 2))
        return _write_nvdb(tmp_path / "grid.nvdb", blobs, codec=codec,
                           file_magic=file_magic, version=version)

    return make


# ---------------------------------------------------------------------------
# Synthetic round-trip
# ---------------------------------------------------------------------------

class TestSyntheticRoundTrip:
    @pytest.mark.parametrize("codec,codec_name", [(0, "none"), (1, "zip")])
    def test_dense_field_round_trips(self, synthetic_nvdb, codec, codec_name):
        grid = read_nanovdb(synthetic_nvdb(codec=codec))
        assert grid.density.dtype == np.float32
        assert grid.density.shape == (4098, 16, 130)
        assert grid.index_min == (0, 0, 0)
        np.testing.assert_array_equal(grid.density, _expected_synthetic_density())
        assert grid.codec == codec_name

    def test_metadata_round_trips(self, synthetic_nvdb):
        grid = read_nanovdb(synthetic_nvdb())
        assert grid.grid_name == "density"
        assert grid.grid_class == "fog_volume"
        assert grid.value_min == 0.0
        assert grid.value_max == 2.5
        expected = np.eye(4)
        expected[:3, :3] = np.diag([0.5, 0.5, 0.5])
        expected[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(grid.index_to_world, expected)

    def test_zip_and_none_decode_identically(self, synthetic_nvdb, tmp_path):
        plain = read_nanovdb(synthetic_nvdb(codec=0))
        (tmp_path / "grid.nvdb").unlink()
        zipped = read_nanovdb(synthetic_nvdb(codec=1))
        np.testing.assert_array_equal(plain.density, zipped.density)

    def test_skips_preceding_grid_to_reach_field(self, synthetic_nvdb):
        grid = read_nanovdb(synthetic_nvdb(extra_first=True), field="density")
        np.testing.assert_array_equal(grid.density, _expected_synthetic_density())

    def test_unknown_grid_class_accepted(self, synthetic_nvdb):
        grid = read_nanovdb(synthetic_nvdb(grid_class=0, meta_class=0))
        assert grid.grid_class == "unknown"
        np.testing.assert_array_equal(grid.density, _expected_synthetic_density())


# ---------------------------------------------------------------------------
# Fail-loud errors
# ---------------------------------------------------------------------------

class TestFailLoud:
    def test_blosc_codec_unsupported(self, synthetic_nvdb):
        with pytest.raises(NanoVdbUnsupportedError, match="BLOSC"):
            read_nanovdb(synthetic_nvdb(codec=2))

    def test_level_set_class_rejected(self, synthetic_nvdb):
        with pytest.raises(NanoVdbUnsupportedError, match="LevelSet"):
            read_nanovdb(synthetic_nvdb(meta_class=1))

    def test_non_float_type_rejected(self, synthetic_nvdb):
        with pytest.raises(NanoVdbUnsupportedError, match="Double"):
            read_nanovdb(synthetic_nvdb(meta_type=2))

    def test_wrong_major_version_rejected(self, synthetic_nvdb):
        with pytest.raises(NanoVdbUnsupportedError, match="33"):
            read_nanovdb(synthetic_nvdb(version=(33 << 21) | 3))

    def test_bad_file_magic_rejected(self, synthetic_nvdb):
        with pytest.raises(NanoVdbError, match="magic"):
            read_nanovdb(synthetic_nvdb(file_magic=0xDEADBEEF))

    def test_missing_field_names_available_grids(self, synthetic_nvdb):
        with pytest.raises(NanoVdbError, match="'density'"):
            read_nanovdb(synthetic_nvdb(), field="temperature")

    def test_error_names_the_file(self, synthetic_nvdb):
        path = synthetic_nvdb(codec=2)
        with pytest.raises(NanoVdbUnsupportedError, match="grid.nvdb"):
            read_nanovdb(path)

    def test_truncated_file_rejected(self, synthetic_nvdb, tmp_path):
        path = synthetic_nvdb()
        path.write_bytes(path.read_bytes()[:8])
        with pytest.raises(NanoVdbError):
            read_nanovdb(path)


# ---------------------------------------------------------------------------
# Real pbrt-v4-scenes files (header recon values pinned from task 1.1)
# ---------------------------------------------------------------------------

class TestRealFiles:
    def _check(self, path, dims, index_min, value_max, avg, active_count, voxel):
        if not os.path.exists(path):
            pytest.skip("pbrt-v4-scenes corpus not available")
        grid = read_nanovdb(path)
        assert grid.density.shape == dims
        assert grid.index_min == index_min
        assert grid.grid_name == "density"
        assert grid.grid_class == "fog_volume"
        assert grid.codec == "zip"
        assert grid.value_min == 0.0
        assert grid.value_max == pytest.approx(value_max)
        assert grid.density.max() == pytest.approx(value_max)
        assert not np.isnan(grid.density).any()
        nonzero = np.count_nonzero(grid.density)
        assert nonzero / grid.density.size > 0.1
        # Independent mass check: the dense integral must match the file's own
        # stored statistic (root mAverage x activeVoxelCount) — this exercises
        # the constant-tile broadcast, not just the leaf voxels.
        total = float(grid.density.astype(np.float64).sum())
        assert total == pytest.approx(avg * active_count, rel=1e-5)
        np.testing.assert_allclose(np.diag(grid.index_to_world)[:3], voxel, rtol=1e-6)

    def test_wdas_cloud_quarter(self):
        self._check(WDAS_CLOUD, (498, 338, 613), (-261, -81, -358), 1.0,
                    0.5018400549888611, 24063202, 0.8333333134651184)

    def test_bunny_cloud(self):
        self._check(BUNNY_CLOUD, (577, 572, 438), (-300, -47, -208), 2.7922983169555664,
                    0.30440956354141235, 19210271, 0.08)
