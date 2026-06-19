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

def test_visible_point_scalar_stride_is_76():
    assert VISIBLE_POINT_STRIDE == 76


def test_visible_point_msl_stride_is_96():
    assert VISIBLE_POINT_STRIDE_MSL == 96


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


# ── buffer sizing ────────────────────────────────────────────────────

def test_sppm_buffer_sizes_scale_with_stream_size():
    s = sppm_buffer_sizes(stream_size=1024)
    assert s["visible_points"] == 1024 * VISIBLE_POINT_STRIDE
    assert s["sppm_accum"] == 1024 * SPPM_ACCUM_STRIDE


def test_sppm_buffer_sizes_msl_uses_metal_stride():
    s = sppm_buffer_sizes(stream_size=256, msl=True)
    assert s["visible_points"] == 256 * VISIBLE_POINT_STRIDE_MSL
    assert s["sppm_accum"] == 256 * 16


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
