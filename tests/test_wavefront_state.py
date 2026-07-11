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

import pytest

from skinny.wavefront_layout import (
    FIELDS,
    INDIRECT_ARGS_STRIDE,
    PATH_STATE_STRIDE,
    PATH_STATE_STRIDE_MSL,
    PATH_STATE_STRIDE_SPECTRAL,
    REC_VERTEX_FIELDS,
    REC_VERTEX_STRIDE_MSL,
    SLANG_MSL_ALIGNS,
    SLANG_MSL_SIZES,
    SLANG_SCALAR_SIZES,
    path_state_size,
    queue_buffer_sizes,
    rec_vertex_size,
)

_SHADERS = Path(__file__).resolve().parent.parent / "src" / "skinny" / "shaders"
_STATE_SLANG = _SHADERS / "wavefront" / "wavefront_state.slang"
_RECORDS_SLANG = _SHADERS / "wavefront" / "wf_records.slang"


def _msl_offsets(fields: list[tuple[str, str]]) -> list[tuple[str, int]]:
    """Per-field (name, byte offset) of a record struct under the MSL layout —
    the same walk `wavefront_layout._struct_stride` uses, exposed for the
    reflection lock test."""
    out: list[tuple[str, int]] = []
    offset = 0
    for name, t in fields:
        align = SLANG_MSL_ALIGNS[t]
        offset = (offset + align - 1) // align * align
        out.append((name, offset))
        offset += SLANG_MSL_SIZES[t]
    return out


# Nested-struct field types (not primitive) → their scalar byte size + the
# (type, name) expansion the Python mirror uses. `Spectrum` is a typealias, not a
# struct, so it is handled by _norm_type instead.
_NESTED_SCALAR_SIZE = {"SampledWavelengths": 32}  # { float4 lambda; float4 pdf; }


def _norm_type(t: str, *, spectral: bool) -> str:
    """Normalize the `Spectrum` typealias to its concrete type for the variant
    (float3 in the RGB build, float4 under SKINNY_SPECTRAL)."""
    if t == "Spectrum":
        return "float4" if spectral else "float3"
    return t


def _parse_struct_fields(
    src: str, struct_name: str, *, spectral: bool = False
) -> list[tuple[str, str]]:
    """Return [(slang_type, field_name), …] in declaration order for the named
    struct. Ignores comments and helper functions. Resolves `#if
    defined(SKINNY_SPECTRAL)` blocks per ``spectral`` and normalizes the
    `Spectrum` typealias to its concrete type. Nested struct types (e.g.
    SampledWavelengths) are returned verbatim; callers size them via
    _NESTED_SCALAR_SIZE."""
    m = re.search(
        rf"struct\s+{struct_name}\s*\{{(.*?)\}}\s*;", src, re.DOTALL
    )
    assert m, f"struct {struct_name} not found"
    body = m.group(1)
    fields: list[tuple[str, str]] = []
    in_spectral_block = False
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if line.startswith("#if defined(SKINNY_SPECTRAL)"):
            in_spectral_block = True
            continue
        if line.startswith("#endif"):
            in_spectral_block = False
            continue
        if in_spectral_block and not spectral:
            continue
        fm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", line)
        if fm:
            fields.append((_norm_type(fm.group(1), spectral=spectral), fm.group(2)))
    return fields


def _scalar_size(slang_type: str) -> int:
    """Scalar-layout byte size of a field type, including nested structs."""
    if slang_type in _NESTED_SCALAR_SIZE:
        return _NESTED_SCALAR_SIZE[slang_type]
    return SLANG_SCALAR_SIZES[slang_type]


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
    total = sum(_scalar_size(t) for t, _ in slang_fields)
    assert total == PATH_STATE_STRIDE


def test_slang_spectral_struct_matches_python_layout():
    # Under SKINNY_SPECTRAL the color roles are float4 and the struct appends
    # SampledWavelengths (spectral-wavefront change). The RGB layout above stays
    # byte-identical; this locks the spectral variant's scalar stride.
    src = _STATE_SLANG.read_text(encoding="utf-8")
    slang_fields = _parse_struct_fields(src, "WavefrontPathState", spectral=True)
    total = sum(_scalar_size(t) for t, _ in slang_fields)
    assert total == PATH_STATE_STRIDE_SPECTRAL == 108
    # color roles are float4, sw present
    types = [t for t, _ in slang_fields]
    assert types.count("float4") >= 2
    assert ("SampledWavelengths", "sw") in slang_fields


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


# ── MSL (Metal) layout sizing (task 1.5 / design B) ────────────────
#
# The Metal in-process Slang→Metal compile pads `float3` to 16 B, so the
# GPU-internal record/queue buffers need a larger stride on Metal than the
# scalar Vulkan layout. These lock the mirror's MSL strides to the values Slang's
# Metal target actually reflects, so the Metal wavefront allocator (phase 3,
# `queue_buffer_sizes(..., msl=True)`) cannot undersize the buffers.


def test_msl_strides_are_padded():
    # 4× float3 → 16 B each + 5 scalars(20) = 84 → round to 16 → 96.
    assert path_state_size(msl=True) == PATH_STATE_STRIDE_MSL == 96
    # 6× float3 → 16 B each + uint(4) = 100 → round to 16 → 112.
    assert rec_vertex_size(msl=True) == REC_VERTEX_STRIDE_MSL == 112
    # MSL is never smaller than scalar (it only adds float3 padding).
    assert PATH_STATE_STRIDE_MSL >= PATH_STATE_STRIDE
    assert REC_VERTEX_STRIDE_MSL >= rec_vertex_size(msl=False)


def test_queue_buffer_sizes_msl_covers_scalar():
    scalar = queue_buffer_sizes(stream_size=1024, num_materials=4)
    msl = queue_buffer_sizes(stream_size=1024, num_materials=4, msl=True)
    # The struct-backed buffer grows; layout-agnostic uint queues are unchanged.
    assert msl["path_state"] == 1024 * PATH_STATE_STRIDE_MSL
    assert msl["path_state"] > scalar["path_state"]
    for key in ("ray_queue", "material_queue", "ray_count",
                "material_count", "material_offset", "indirect_args"):
        assert msl[key] == scalar[key]
    # Default (no msl kwarg) stays byte-identical to the scalar Vulkan sizing.
    assert queue_buffer_sizes(1024, 4) == scalar


# ── Spectral allocation-stride guard (change spectral-wavefront) ────────────
#
# Regression for the P1 memory-corruption bug codex found: the spectral shader
# structs grew (WavefrontPathState / WfBdptAux+BDPTVertex / VisiblePoint+SppmAccum)
# but the host buffer allocators sized against the RGB strides, so a
# `--spectral --execution-mode wavefront` run overran/corrupted the GPU buffers.
# These lock every allocator's spectral stride to the spectral layout size AND
# strictly above the RGB stride, and pin spectral=False byte-identical to RGB.


def test_spectral_path_state_alloc_stride():
    from skinny.wavefront_layout import PATH_STATE_STRIDE_SPECTRAL_MSL
    # scalar (Vulkan renderer path-state buffer, renderer.py)
    assert path_state_size(spectral=True) == PATH_STATE_STRIDE_SPECTRAL == 108
    assert path_state_size(spectral=True) > PATH_STATE_STRIDE
    assert path_state_size(spectral=False) == PATH_STATE_STRIDE  # RGB unchanged
    # MSL (Metal validator expected stride, metal_wavefront.py)
    assert path_state_size(msl=True, spectral=True) == PATH_STATE_STRIDE_SPECTRAL_MSL == 128
    assert path_state_size(msl=True, spectral=True) > PATH_STATE_STRIDE_MSL
    # queue_buffer_sizes threads spectral into the path_state region only.
    spec = queue_buffer_sizes(1024, 4, spectral=True)
    rgb = queue_buffer_sizes(1024, 4)
    assert spec["path_state"] == 1024 * path_state_size(spectral=True)
    assert spec["path_state"] > rgb["path_state"]
    for key in ("ray_queue", "material_queue", "ray_count",
                "material_count", "material_offset", "indirect_args"):
        assert spec[key] == rgb[key]


def test_spectral_bdpt_aux_and_vertex_alloc_stride():
    from skinny.wavefront_layout import bdpt_vertex_size, wf_bdpt_aux_size
    # WfBdptAux is the codex-flagged overrun: RGB alloc constant AUX_STRIDE=128,
    # spectral struct ≈136 B scalar → must exceed both the RGB struct and 128.
    assert wf_bdpt_aux_size() == 92
    assert wf_bdpt_aux_size(spectral=True) == 136
    assert wf_bdpt_aux_size(spectral=True) > wf_bdpt_aux_size()
    assert wf_bdpt_aux_size(spectral=True) > 128  # > RGB alloc constant AUX_STRIDE
    # BDPTVertex grows throughput/emission float3→float4 (+8 B scalar).
    assert bdpt_vertex_size() == 120
    assert bdpt_vertex_size(spectral=True) == 128
    assert bdpt_vertex_size(spectral=True) > bdpt_vertex_size()
    # The renderer floors both strides by the RGB alloc constants (128/128), so
    # RGB is byte-identical and spectral bumps aux to 136 (vertex fits at 128).
    assert max(128, bdpt_vertex_size(spectral=False)) == 128
    assert max(128, wf_bdpt_aux_size(spectral=False)) == 128
    assert max(128, wf_bdpt_aux_size(spectral=True)) == 136


def test_spectral_sppm_alloc_stride():
    from skinny.wavefront_layout import (
        SPPM_ACCUM_STRIDE,
        VISIBLE_POINT_STRIDE,
        sppm_accum_size,
        sppm_buffer_sizes,
        visible_point_size,
    )
    # VisiblePoint: +conductorMetalId (spectral); SppmAccum: +phiW 4th channel.
    assert visible_point_size(spectral=True) == 192 > VISIBLE_POINT_STRIDE == 180
    assert sppm_accum_size(spectral=True) == 20 > SPPM_ACCUM_STRIDE == 16
    assert visible_point_size(spectral=False) == VISIBLE_POINT_STRIDE  # RGB unchanged
    assert sppm_accum_size(spectral=False) == SPPM_ACCUM_STRIDE
    spec = sppm_buffer_sizes(4096, spectral=True)
    rgb = sppm_buffer_sizes(4096)
    assert spec["visible_points"] == 4096 * visible_point_size(spectral=True)
    assert spec["sppm_accum"] == 4096 * sppm_accum_size(spectral=True)
    assert spec["visible_points"] > rgb["visible_points"]
    assert spec["sppm_accum"] > rgb["sppm_accum"]


def _reflect_msl_layout(struct: str, import_mod: str):
    """Reflect (stride, [(field, offset), …]) of a struct under Slang's Metal
    target by importing the real shader module — the GPU-free truth the mirror is
    locked to. Returns None when no Metal device is available."""
    spy = pytest.importorskip("slangpy")
    from skinny.backend_select import metal_available

    ok, _reason = metal_available()
    if not ok:
        return None
    try:
        dev = spy.create_device(
            type=spy.DeviceType.metal,
            include_paths=[str(_SHADERS), str(_SHADERS.parent / "mtlx" / "genslang")],
        )
    except Exception:
        return None
    try:
        src = (f"import {import_mod};\n"
               f"StructuredBuffer<{struct}> probe_b;\n"
               f"[shader(\"compute\")] [numthreads(1, 1, 1)]\n"
               f"void m(uint3 t : SV_DispatchThreadID) {{}}\n")
        module = dev.load_module_from_source("wf_layout_refl", src)
        program = dev.link_program([module], [module.entry_point("m")])
        p = next(x for x in program.layout.parameters if x.name == "probe_b")
        etl = p.type_layout.element_type_layout
        stride = int(getattr(etl, "stride", 0) or etl.size)
        offsets = [(f.name, int(f.offset)) for f in etl.fields]
        return stride, offsets
    finally:
        dev.close()


def test_msl_path_state_matches_reflected_metal_layout():
    refl = _reflect_msl_layout("WavefrontPathState", "wavefront.wavefront_state")
    if refl is None:
        pytest.skip("no Metal device for MSL reflection")
    stride, offsets = refl
    assert stride == PATH_STATE_STRIDE_MSL
    assert offsets == _msl_offsets(FIELDS)


def test_msl_rec_vertex_matches_reflected_metal_layout():
    refl = _reflect_msl_layout("RecVertex", "wavefront.wf_records")
    if refl is None:
        pytest.skip("no Metal device for MSL reflection")
    stride, offsets = refl
    assert stride == REC_VERTEX_STRIDE_MSL
    assert offsets == _msl_offsets(REC_VERTEX_FIELDS)
