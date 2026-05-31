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
    PATH_STATE_STRIDE,
    SLANG_SCALAR_SIZES,
    path_state_size,
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
