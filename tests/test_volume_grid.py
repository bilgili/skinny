"""Hostless unit tests for nanovdb-volume-rendering (tasks 4.3/4.4).

Covers, with no GPU:

* FlatMaterialParams volume packing — the appended worldToUvw rows at
  192/208/224, the σ × value_max / mm_per_unit folds at 160/176, mediumKind =
  MEDIUM_NANOVDB at 188, the index-matched ior = 1.0, and the invariant that a
  NON-volume material's bytes are only *extended* (identity rows), never
  shifted within the old 0..192 prefix.
* `compute_volume_world_to_uvw` — the folded world→[0,1]³ affine, pinned
  against the geometry-side pbrt→USD convention (`B @ ctm @ M_grid` point
  chain): grid center → (0.5, 0.5, 0.5) including a bunny-style rotated medium
  CTM, and exact voxel-center texel coordinates.
* `_material_is_volume` — keys off the importer's explicit `volume_interface`
  marker only (no lobe sniffing).
* `_current_state_hash` accumulation-reset coverage — source-inspection
  (same precedent as tests/test_sppm_selection.py) that the volume grid key is
  hashed.

The MSL stride half (Metal reads the 240 B record at the same stride) lives in
tests/test_metal_flat_material_layout.py (gpu-marked).
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

# skinny.renderer imports `vulkan` unconditionally; skip cleanly without it.
try:
    from skinny.renderer import (
        FLAT_MATERIAL_STRIDE,
        MEDIUM_HOMOGENEOUS,
        MEDIUM_NANOVDB,
        _material_is_volume,
        pack_flat_material,
    )
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}",
                allow_module_level=True)

from skinny.pbrt import transform as T
from skinny.usd_loader import _PBRT_TO_USD_B, compute_volume_world_to_uvw


def _volume_material(**extra):
    overrides = {
        "volume_interface": True,
        "volume_sigma_a": (0.0, 0.25, 0.5),
        "volume_sigma_s": (1.0, 2.0, 4.0),
        "volume_g": 0.877,
        "volume_grid_asset": "/abs/path/wdas_cloud_quarter.nvdb",
        "volume_grid_field": "density",
    }
    overrides.update(extra)
    return SimpleNamespace(parameter_overrides=overrides)


_IDENTITY_ROWS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


def _rows_at(data: bytes):
    return tuple(struct.unpack_from("ffff", data, 192 + 16 * r) for r in range(3))


class TestVolumePacking:
    def test_non_volume_prefix_only_extended(self):
        """A plain flat material's record is the old 192 bytes + identity rows —
        nothing inside the old prefix moved."""
        mat = SimpleNamespace(parameter_overrides={
            "diffuseColor": (0.1, 0.2, 0.3), "roughness": 0.4, "ior": 1.5,
        })
        data = pack_flat_material(mat)
        assert len(data) == FLAT_MATERIAL_STRIDE == 240
        # Spot-check documented prefix offsets are where they always were.
        assert struct.unpack_from("fff", data, 0) == pytest.approx((0.1, 0.2, 0.3))
        assert struct.unpack_from("f", data, 12)[0] == pytest.approx(0.4)
        assert struct.unpack_from("f", data, 60)[0] == pytest.approx(1.5)
        assert struct.unpack_from("I", data, 188)[0] == MEDIUM_HOMOGENEOUS
        for got, want in zip(_rows_at(data), _IDENTITY_ROWS):
            assert got == pytest.approx(want)

    def test_subsurface_material_keeps_identity_rows(self):
        mat = SimpleNamespace(parameter_overrides={
            "subsurface_sigma_a": (0.032, 0.17, 0.48),
            "subsurface_sigma_s": (0.74, 0.88, 1.01),
        })
        data = pack_flat_material(mat)
        assert struct.unpack_from("fff", data, 160) == pytest.approx((0.032, 0.17, 0.48))
        assert struct.unpack_from("I", data, 188)[0] == MEDIUM_HOMOGENEOUS
        for got, want in zip(_rows_at(data), _IDENTITY_ROWS):
            assert got == pytest.approx(want)

    def test_volume_sigma_folds(self):
        """σ_packed = σ_pbrt × value_max / mm_per_unit — BOTH folds: value_max
        so the normalized texel is the density multiplier, 1/mm_per_unit so the
        walk's mm⁻¹ · (world·mmPerUnit) convention recovers σ_pbrt · d_world."""
        value_max, mmu = 2.792, 1000.0
        data = pack_flat_material(
            _volume_material(), volume_value_max=value_max, mm_per_unit=mmu)
        fold = value_max / mmu
        assert struct.unpack_from("fff", data, 160) == pytest.approx(
            (0.0, 0.25 * fold, 0.5 * fold))
        assert struct.unpack_from("f", data, 172)[0] == pytest.approx(0.877)
        assert struct.unpack_from("fff", data, 176) == pytest.approx(
            (1.0 * fold, 2.0 * fold, 4.0 * fold))
        assert struct.unpack_from("I", data, 188)[0] == MEDIUM_NANOVDB
        # Index-matched pass-through boundary: eta (the reused ior slot) = 1.
        assert struct.unpack_from("f", data, 60)[0] == pytest.approx(1.0)

    def test_volume_world_to_uvw_rows_land_at_192(self):
        rows = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.5
        data = pack_flat_material(_volume_material(), volume_world_to_uvw=rows)
        for r, got in enumerate(_rows_at(data)):
            assert got == pytest.approx(tuple(rows[r]))

    def test_homogeneous_interface_keeps_kind_homogeneous(self):
        """An interface boundary with NO grid asset (homogeneous free-standing
        interior) packs MEDIUM_HOMOGENEOUS, so densityAt stays ≡ 1."""
        mat = _volume_material(volume_grid_asset="")
        data = pack_flat_material(mat, volume_value_max=1.0, mm_per_unit=1000.0)
        assert struct.unpack_from("I", data, 188)[0] == MEDIUM_HOMOGENEOUS


class TestMaterialIsVolume:
    def test_marker_true(self):
        assert _material_is_volume(_volume_material())

    def test_marker_absent(self):
        assert not _material_is_volume(SimpleNamespace(parameter_overrides={
            "volume_sigma_s": (1.0, 1.0, 1.0),  # homogeneous fog WITHOUT interface
        }))

    def test_no_lobe_sniffing(self):
        """Glass/cutout materials never match — only the explicit marker."""
        assert not _material_is_volume(SimpleNamespace(parameter_overrides={
            "opacity": 0.0, "ior": 1.5, "transmission": 1.0,
        }))
        assert not _material_is_volume(SimpleNamespace(parameter_overrides={}))


def _stored(math_m: np.ndarray) -> np.ndarray:
    """Math (column-vector) 4x4 → this codebase's stored (row-vector) form,
    the same transpose `_world_transform` / `to_gf_matrix` round-trip yields."""
    return np.asarray(math_m, np.float64).T


def _apply(rows: np.ndarray, p) -> np.ndarray:
    """Apply the packed (3,4) rows exactly as the shader does:
    uvw[r] = dot(rows[r,:3], p) + rows[r,3]."""
    p = np.asarray(p, np.float64)
    return rows[:, :3].astype(np.float64) @ p + rows[:, 3].astype(np.float64)


class TestWorldToUvw:
    def test_identity_grid_voxel_centers(self):
        """Identity prim xform + identity grid map: voxel (a,b,c) of the dense
        array (grid index index_min + (a,b,c)) samples at ((a,b,c)+0.5)/dims.
        NOTE the identity prim xform still carries the B axis flip (a Volume
        prim xform is `B ctm B`, so world = B @ medium-space)."""
        index_min = (-2, 1, -5)
        dims = (4, 8, 16)
        rows = compute_volume_world_to_uvw(
            np.eye(4), np.eye(4), index_min, dims)
        for a, b, c in [(0, 0, 0), (3, 7, 15), (1, 2, 3)]:
            ijk = np.array([index_min[0] + a, index_min[1] + b, index_min[2] + c, 1.0])
            p_world = (_PBRT_TO_USD_B @ ijk)[:3]  # world = prim(=I) @ B @ ijk
            uvw = _apply(rows, p_world)
            want = ((a + 0.5) / dims[0], (b + 0.5) / dims[1], (c + 0.5) / dims[2])
            assert uvw == pytest.approx(want, abs=1e-6)

    def test_grid_center_maps_to_half(self):
        """The grid's geometric center maps to uvw = (0.5, 0.5, 0.5) through a
        bunny-style medium CTM (rotation + translation + scale) — the full pbrt
        point chain `p_usd = B @ ctm @ M_grid @ ijk` with the prim transform
        authored as `to_skinny(ctm)` in stored form, exactly as emit/loader do."""
        # bunny_cloud-style grid map: uniform voxel scale, zero translation.
        m_grid = np.diag([0.08, 0.08, 0.08, 1.0])
        # Medium CTM: rotated + translated + scaled (exercises every block).
        ctm = (T.translate(1.5, -2.0, 3.0) @ T.rotate(37.0, 0.3, 1.0, 0.2)
               @ T.scale(2.0, 2.0, 2.0))
        index_min = (-300, -47, -208)
        dims = (577, 572, 438)
        stored_prim = _stored(T.to_skinny(ctm))
        rows = compute_volume_world_to_uvw(stored_prim, m_grid, index_min, dims)

        # Continuous grid index of the uvw = 0.5 point: index_min + dims/2 - 0.5.
        center_ijk = np.array([index_min[i] + dims[i] / 2.0 - 0.5 for i in range(3)] + [1.0])
        # Geometry-side convention: points bake as B @ CTM @ p_local, so the
        # voxel center in skinny world space is B @ ctm @ (M_grid @ ijk).
        p_world = (T.B @ ctm @ m_grid @ center_ijk)[:3]
        assert _apply(rows, p_world) == pytest.approx((0.5, 0.5, 0.5), abs=1e-6)

    def test_prim_xform_equals_geometry_chain(self):
        """Pin the composition X = primXform_math @ B @ M_grid against the
        equivalent geometry chain B @ ctm @ M_grid for a random-ish CTM: both
        must give the same uvw for arbitrary voxels (the convention proof)."""
        m_grid = np.eye(4)
        m_grid[:3, :3] = np.diag([0.8333, 0.8333, 0.8333])
        m_grid[:3, 3] = (4.0, -1.0, 2.5)
        ctm = T.rotate(-64.0, 1.0, 0.2, -0.5) @ T.translate(-3.0, 7.0, 0.25)
        index_min = (-261, -81, -358)
        dims = (498, 338, 613)
        rows = compute_volume_world_to_uvw(
            _stored(T.to_skinny(ctm)), m_grid, index_min, dims)
        rng = np.random.default_rng(7)
        for _ in range(5):
            abc = rng.uniform((0, 0, 0), dims)
            ijk = np.array([index_min[0] + abc[0], index_min[1] + abc[1],
                            index_min[2] + abc[2], 1.0])
            p_world = (T.B @ ctm @ m_grid @ ijk)[:3]
            want = (abc + 0.5) / np.asarray(dims)  # continuous: (i - min + 0.5)/dims
            assert _apply(rows, p_world) == pytest.approx(tuple(want), abs=1e-6)


class TestStateHashCoverage:
    def test_volume_grid_key_is_hashed(self):
        """`_current_state_hash` must include the volume-grid identity so a
        grid swap resets progressive accumulation (source-inspection, same
        precedent as tests/test_sppm_selection.py)."""
        import inspect

        import skinny.renderer as renderer_mod
        src = inspect.getsource(renderer_mod.Renderer)
        start = src.index("def _current_state_hash")
        body = src[start:start + 4000]
        assert "self._volume_grid_key" in body, (
            "_current_state_hash must hash _volume_grid_key "
            "(accumulation reset on density-grid swap)")
