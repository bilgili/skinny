"""Layout guard + compile check for the SPPM per-pixel state structs
(change photon-mapping-sppm, PM-1).

GPU-free. The host buffer allocator (skinny.wavefront_layout) and the Slang
``VisiblePoint`` / ``SppmAccum`` structs in
``shaders/integrators/sppm_state.slang`` must agree on field order and size, or
the SPPM stages read garbage — mirrors ``test_wavefront_state.py``'s discipline.

The final test additionally runs ``slangc`` on a compile-only harness so the
module is type-checked standalone; it skips when slangc is not installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from skinny.wavefront_layout import (
    SLANG_MSL_ALIGNS,
    SLANG_MSL_SIZES,
    SLANG_SCALAR_SIZES,
    SPPM_ACCUM_FIELDS,
    SPPM_ACCUM_STRIDE,
    VISIBLE_POINT_FIELDS,
    VISIBLE_POINT_STRIDE,
    VISIBLE_POINT_STRIDE_MSL,
    VP_ACTIVE,
    sppm_accum_size,
    sppm_buffer_sizes,
    sppm_grid_buffer_sizes,
    sppm_grid_cell_count,
    visible_point_size,
)

_ROOT = Path(__file__).resolve().parent.parent
_SHADERS = _ROOT / "src" / "skinny" / "shaders"
_SPPM_SLANG = _SHADERS / "integrators" / "sppm_state.slang"
_HARNESS = _ROOT / "tests" / "harnesses" / "test_sppm_state_harness.slang"


def _parse_struct_fields(src: str, struct_name: str) -> list[tuple[str, str]]:
    """Return [(slang_type, field_name), …] in declaration order for the named
    struct, ignoring comments + helper functions."""
    m = re.search(rf"struct\s+{struct_name}\s*\{{(.*?)\}}\s*;", src, re.DOTALL)
    assert m, f"struct {struct_name} not found"
    fields: list[tuple[str, str]] = []
    for raw in m.group(1).splitlines():
        line = raw.split("//", 1)[0].strip()
        fm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", line)
        if fm:
            fields.append((fm.group(1), fm.group(2)))
    return fields


# ── strides ──────────────────────────────────────────────────────────

def test_visible_point_scalar_stride_is_180():
    assert VISIBLE_POINT_STRIDE == 180


def test_visible_point_msl_stride_is_240():
    assert VISIBLE_POINT_STRIDE_MSL == 240


def test_sppm_accum_stride_is_16_both_layouts():
    assert SPPM_ACCUM_STRIDE == 16
    assert sppm_accum_size(msl=True) == 16  # all-uint struct: layout-invariant


def test_sizes_derived_from_fields_match_strides():
    assert visible_point_size() == VISIBLE_POINT_STRIDE
    assert sppm_accum_size() == SPPM_ACCUM_STRIDE


def test_all_field_types_have_known_sizes():
    for fields in (VISIBLE_POINT_FIELDS, SPPM_ACCUM_FIELDS):
        for _name, t in fields:
            assert t in SLANG_SCALAR_SIZES, f"no scalar size for {t}"
            assert t in SLANG_MSL_SIZES and t in SLANG_MSL_ALIGNS


# ── Slang ↔ Python lockstep ──────────────────────────────────────────

def test_slang_visible_point_matches_python_layout():
    src = _SPPM_SLANG.read_text(encoding="utf-8")
    slang_fields = _parse_struct_fields(src, "VisiblePoint")
    expected = [(t, n) for (n, t) in VISIBLE_POINT_FIELDS]
    assert slang_fields == expected


def test_slang_sppm_accum_matches_python_layout():
    src = _SPPM_SLANG.read_text(encoding="utf-8")
    slang_fields = _parse_struct_fields(src, "SppmAccum")
    expected = [(t, n) for (n, t) in SPPM_ACCUM_FIELDS]
    assert slang_fields == expected


def test_slang_visible_point_size_sums_to_scalar_stride():
    src = _SPPM_SLANG.read_text(encoding="utf-8")
    slang_fields = _parse_struct_fields(src, "VisiblePoint")
    total = sum(SLANG_SCALAR_SIZES[t] for t, _ in slang_fields)
    assert total == VISIBLE_POINT_STRIDE


def test_vp_active_flag_matches_slang():
    src = _SPPM_SLANG.read_text(encoding="utf-8")
    m = re.search(r"VP_ACTIVE\s*=\s*(\d+)u", src)
    assert m, "VP_ACTIVE constant not found in sppm_state.slang"
    assert int(m.group(1)) == VP_ACTIVE


# ── FlatHitMat ⊆ VisiblePoint completeness (fix-sppm-bathroom-black-walls) ──
# The photon deposit rebuilds the FlatMaterial from VP-stored fields
# (sppmLoadMaterial). A FlatHitMat field with no VP slot / no store / no
# rebuild feeds UNDEFINED values into evaluate() at deposit time — the Stage-2
# rich inputs did exactly that and zeroed the SPPM photon term scene-wide.
# `emission` is the one documented exception (direct, not BRDF; rebuilt as 0).

_FLAT_SHADING = _SHADERS / "materials" / "flat" / "flat_shading.slang"
_WF_SPPM = _SHADERS / "integrators" / "wavefront_sppm.slang"
_VP_EXEMPT_FLAT_FIELDS = {"emission"}


def _flat_hit_mat_fields() -> list[str]:
    src = _FLAT_SHADING.read_text(encoding="utf-8")
    return [name for _t, name in _parse_struct_fields(src, "FlatHitMat")]


def test_every_flat_hit_mat_field_has_a_visible_point_slot():
    vp_names = {n for n, _t in VISIBLE_POINT_FIELDS}
    missing = [f for f in _flat_hit_mat_fields()
               if f not in _VP_EXEMPT_FLAT_FIELDS and f not in vp_names]
    assert not missing, (
        f"FlatHitMat field(s) {missing} have no VisiblePoint slot — the SPPM "
        f"photon deposit would evaluate the BSDF with undefined values. Add "
        f"slots to sppm_state.slang + wavefront_layout.VISIBLE_POINT_FIELDS "
        f"and wire them through sppmStoreVisiblePoint/sppmLoadMaterial."
    )


def test_sppm_store_and_load_cover_every_flat_hit_mat_field():
    src = _WF_SPPM.read_text(encoding="utf-8")
    store = re.search(
        r"void\s+sppmStoreVisiblePoint\s*\([^)]*\)\s*\{(.*?)\n\}", src, re.DOTALL)
    load = re.search(
        r"FlatMaterial\s+sppmLoadMaterial\s*\([^)]*\)\s*\{(.*?)\n\}", src, re.DOTALL)
    assert store and load, "sppmStoreVisiblePoint / sppmLoadMaterial not found"
    for f in _flat_hit_mat_fields():
        if f in _VP_EXEMPT_FLAT_FIELDS:
            continue
        assert re.search(rf"vp\.{f}\s*=", store.group(1)), (
            f"sppmStoreVisiblePoint never writes vp.{f}")
        assert re.search(rf"m\.{f}\s*=\s*vp\.{f}", load.group(1)), (
            f"sppmLoadMaterial never rebuilds m.{f} from vp.{f}")


# ── buffer sizing ────────────────────────────────────────────────────

def test_sppm_buffer_sizes_scale_with_num_pixels():
    s = sppm_buffer_sizes(num_pixels=1024)
    assert s["visible_points"] == 1024 * VISIBLE_POINT_STRIDE
    assert s["sppm_accum"] == 1024 * SPPM_ACCUM_STRIDE


def test_sppm_buffer_sizes_msl_uses_metal_stride():
    s = sppm_buffer_sizes(num_pixels=256, msl=True)
    assert s["visible_points"] == 256 * VISIBLE_POINT_STRIDE_MSL
    assert s["sppm_accum"] == 256 * 16


def test_sppm_grid_cell_count_is_next_pow2_of_twice_pixels():
    # 1024 px -> 2*1024 = 2048 (already a power of two) -> 2048.
    assert sppm_grid_cell_count(1024) == 2048
    # 1000 px -> 2000 -> next pow2 = 2048.
    assert sppm_grid_cell_count(1000) == 2048
    # 1025 px -> 2050 -> next pow2 = 4096.
    assert sppm_grid_cell_count(1025) == 4096


def test_sppm_grid_cell_count_masking_invariant():
    # numCells is a power of two so cell index can mask with & (numCells - 1).
    for px in (1, 100, 1024, 1_000_000):
        nc = sppm_grid_cell_count(px)
        assert nc & (nc - 1) == 0
        assert nc >= 2 * px


def test_sppm_grid_buffer_sizes():
    px = 4096
    nc = sppm_grid_cell_count(px)
    s = sppm_grid_buffer_sizes(px)
    # gridCount | gridOffset | gridCursor (each nc) + sortedIdx (px), all uint.
    assert s["grid_combined"] == (3 * nc + px) * 4
    assert s["scan_scratch"] == ((nc + 255) // 256) * 4


# ── slangc compile check (skipped without slangc) ────────────────────

def test_sppm_state_module_compiles():
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc not on PATH")
    out = Path("/tmp/skinny_sppm_state_harness.spv")
    cmd = [
        slangc, str(_HARNESS),
        "-target", "spirv", "-entry", "computeMain", "-stage", "compute",
        "-I", str(_SHADERS), "-fvk-use-scalar-layout",
        "-o", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"slangc failed:\n{res.stderr}"
    assert out.exists() and out.stat().st_size > 0
