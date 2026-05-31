"""Layout guard for the wavefront path-state record (P1 §P1-A).

GPU-free. The Python buffer allocator and the Slang `WavefrontPathState` struct
must agree on field order and size, or the wavefront stages read garbage. This
derives the stride from the actual Slang struct fields and cross-checks the
Python layout table — the scalar-layout corruption guard, mirroring
test_struct_layout's discipline.
"""

from __future__ import annotations

import re
from pathlib import Path

from skinny.wavefront_layout import (
    FIELDS,
    INDIRECT_ARGS_STRIDE,
    PATH_STATE_STRIDE,
    SLANG_SCALAR_SIZES,
    path_state_size,
    queue_buffer_sizes,
)

_STATE_SLANG = (
    Path(__file__).resolve().parent.parent
    / "src" / "skinny" / "shaders" / "wavefront" / "wavefront_state.slang"
)


def _parse_struct_fields(src: str, struct_name: str) -> list[tuple[str, str]]:
    """Return [(slang_type, field_name), …] in declaration order for the named
    struct. Ignores comments and helper functions."""
    m = re.search(
        rf"struct\s+{struct_name}\s*\{{(.*?)\}}\s*;", src, re.DOTALL
    )
    assert m, f"struct {struct_name} not found"
    body = m.group(1)
    fields: list[tuple[str, str]] = []
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        fm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", line)
        if fm:
            fields.append((fm.group(1), fm.group(2)))
    return fields


def test_python_stride_is_68():
    assert PATH_STATE_STRIDE == 68


def test_size_derived_from_fields_matches_stride():
    assert path_state_size() == PATH_STATE_STRIDE


def test_all_field_types_have_known_scalar_size():
    for _name, slang_type in FIELDS:
        assert slang_type in SLANG_SCALAR_SIZES, f"no scalar size for {slang_type}"


def test_slang_struct_matches_python_layout():
    src = _STATE_SLANG.read_text(encoding="utf-8")
    slang_fields = _parse_struct_fields(src, "WavefrontPathState")
    # (type, name) order must match the Python layout exactly.
    expected = [(t, n) for (n, t) in FIELDS]
    assert slang_fields == expected


def test_slang_struct_size_sums_to_stride():
    src = _STATE_SLANG.read_text(encoding="utf-8")
    slang_fields = _parse_struct_fields(src, "WavefrontPathState")
    total = sum(SLANG_SCALAR_SIZES[t] for t, _ in slang_fields)
    assert total == PATH_STATE_STRIDE


# ── queue buffer sizing (P1 §P1-B / §P1-C) ─────────────────────────


def test_path_state_buffer_scales_with_stream_size():
    s = queue_buffer_sizes(stream_size=1024, num_materials=4)
    assert s["path_state"] == 1024 * PATH_STATE_STRIDE
    assert s["ray_queue"] == 1024 * 4
    assert s["material_queue"] == 1024 * 4


def test_per_material_buffers_scale_with_material_count():
    s = queue_buffer_sizes(stream_size=512, num_materials=7)
    assert s["material_count"] == 7 * 4
    assert s["material_offset"] == 7 * 4
    assert s["indirect_args"] == 7 * INDIRECT_ARGS_STRIDE


def test_indirect_args_stride_is_three_uints():
    assert INDIRECT_ARGS_STRIDE == 12


def test_ray_count_is_single_uint():
    s = queue_buffer_sizes(stream_size=256, num_materials=2)
    assert s["ray_count"] == 4


def test_zero_materials_yields_empty_per_material_buffers():
    s = queue_buffer_sizes(stream_size=256, num_materials=0)
    assert s["material_count"] == 0
    assert s["indirect_args"] == 0
    # the stream-sized buffers are independent of material count
    assert s["path_state"] == 256 * PATH_STATE_STRIDE
