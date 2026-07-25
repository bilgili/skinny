"""Single derived authority for every host-mirrored GPU byte layout
(change ``reflection-owned-byte-layouts``).

The host↔GPU interface is a byte-offset agreement with the Slang structs. This
module derives each mirrored struct's ordered field list by **parsing the
authoritative ``.slang`` declaration**, then computes offsets/strides for both
layout dialects the renderer speaks:

* **scalar** (Vulkan, ``slangc -fvk-use-scalar-layout``) — every offset is a
  pure running sum of scalar field sizes, so the declared field order alone is
  the reflection equivalent, hostlessly.
* **MSL** (Metal, in-process Slang→Metal) — Slang pads ``float3``/``uint3`` to
  16 B (size *and* alignment) and rounds the struct up to its largest member
  alignment.

Compile-variant gates are resolved per query. Exactly three defines are
resolvable — ``SKINNY_SPECTRAL``, ``SKINNY_MLT``, ``SKINNY_METAL`` — and the
parser **raises** on any other gate, declaration form, or field type: an
unparseable struct is a hostless test failure, never a silently wrong offset.

``FrameConstants`` carries one wrinkle the other structs do not: **declaration
order is not host blob order**. ``tileOriginY`` is ``#if
defined(SKINNY_METAL)``-gated and sits *before* the ``SKINNY_MLT`` tail in the
declaration, but ``_pack_uniforms`` always appends it **last** — so under an
MLT pack ``mltSigma`` lands at 564, exactly where the Vulkan MLT SPIR-V (which
has no ``tileOriginY`` at all) expects it, and the trailing word is benign
filler inside the oversized UBO. :func:`fc_scalar_blob` applies that blob rule.

See ``docs/Architecture.md`` § Byte-layout ownership.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_SHADERS = Path(__file__).resolve().parent / "shaders"

# ── Type tables ──────────────────────────────────────────────────────
#
# Scalar-layout byte sizes. All alignments are <= 4, so a scalar struct packs
# tightly with no interior padding and every offset is a running sum.
SLANG_SCALAR_SIZES: dict[str, int] = {
    "float": 4,
    "float2": 8,
    "float3": 12,
    "float4": 16,
    "float4x4": 64,
    "uint": 4,
    "uint2": 8,
    "uint3": 12,
    "uint4": 16,
    "int": 4,
    "bool": 4,  # Slang stores bool as a 32-bit value in a buffer (scalar layout)
}

# MSL (Metal target) byte sizes + alignments. Slang pads 3-component vectors to
# 16 B (size *and* alignment) on Metal; every other field keeps its natural
# size == alignment. A struct's stride rounds up to its largest member alignment.
SLANG_MSL_SIZES: dict[str, int] = {
    "float": 4, "float2": 8, "float3": 16, "float4": 16, "float4x4": 64,
    "uint": 4, "uint2": 8, "uint3": 16, "uint4": 16, "int": 4, "bool": 1,
}
SLANG_MSL_ALIGNS: dict[str, int] = {
    "float": 4, "float2": 8, "float3": 16, "float4": 16, "float4x4": 16,
    "uint": 4, "uint2": 8, "uint3": 16, "uint4": 16, "int": 4, "bool": 1,
}

#: Preprocessor gates the parser can resolve. Anything else raises.
RESOLVABLE_DEFINES = ("SKINNY_SPECTRAL", "SKINNY_MLT", "SKINNY_METAL")

#: Registered structs → the ``.slang`` file that declares them, relative to the
#: shader root. Nested field types must be registered too (they are parsed
#: recursively for their own layout).
STRUCT_SOURCES: dict[str, str] = {
    "FrameConstants": "common.slang",
    "Camera": "common.slang",
    "FlatMaterialParams": "common.slang",
    "MltPrimarySample": "common.slang",
    "StdSurfaceParams": "mtlx_std_surface.slang",
    "SampledWavelengths": "spectrum.slang",
    "WavefrontPathState": "wavefront/wavefront_state.slang",
    "RecVertex": "wavefront/wf_records.slang",
    "VisiblePoint": "integrators/sppm_state.slang",
    "SppmAccum": "integrators/sppm_state.slang",
    "BDPTVertex": "integrators/bdpt.slang",
    "WfBdptAux": "wavefront/wavefront_bdpt.slang",
    "MltChainMeta": "wavefront/wavefront_mlt.slang",
    "MltRecord": "wavefront/wavefront_mlt.slang",
}

# Lines inside a struct body that are not field declarations. Anything else that
# is not a recognised declaration raises rather than being skipped silently.
# NOTE: `[` is deliberately NOT here — an attributed declaration
# (`[[vk::offset(16)]] float b;`) is a real field, and skipping it would drop a
# field silently and shift every offset after it. Attributes are stripped
# instead (`_ATTR_PREFIX`), and a line that is only an attribute (e.g.
# `[mutating]` on the next line's method) reduces to empty and is skipped.
_SKIP_PREFIXES = ("property", "static", "typealias", "{", "}", "#pragma")

#: Leading Slang/HLSL attribute on a declaration — `[[a::b(1)]]` or `[foo]`.
_ATTR_PREFIX = re.compile(r"^(?:\[\[.*?\]\]|\[[^\]]*\])\s*")


class SlangLayoutError(RuntimeError):
    """A mirrored struct could not be fully classified — never guess an offset."""


def _norm_type(t: str, *, spectral: bool) -> str:
    """Resolve the ``Spectrum`` typealias for the variant (``float3`` in the RGB
    build, ``float4`` under ``SKINNY_SPECTRAL``)."""
    if t == "Spectrum":
        return "float4" if spectral else "float3"
    return t


def _struct_body(src: str, struct_name: str) -> str:
    m = re.search(rf"struct\s+{struct_name}\s*\{{(.*?)\}}\s*;", src, re.DOTALL)
    if not m:
        raise SlangLayoutError(f"struct {struct_name} not found")
    return m.group(1)


def parse_struct_fields(
    src: str,
    struct_name: str,
    *,
    spectral: bool = False,
    mlt: bool = False,
    metal: bool = False,
) -> list[tuple[str, str]]:
    """Return ``[(slang_type, field_name), …]`` in declaration order for the named
    struct, with the variant's ``#if defined(…)`` gates resolved and the
    ``Spectrum`` typealias normalized. Nested struct types are returned verbatim
    (callers size them via :func:`struct_scalar_size` / the layout walkers).

    Raises :class:`SlangLayoutError` on an unresolvable gate, an unrecognised
    declaration form, or an unknown field type.
    """
    active = {
        "SKINNY_SPECTRAL": spectral,
        "SKINNY_MLT": mlt,
        "SKINNY_METAL": metal,
    }
    body = _struct_body(src, struct_name)
    fields: list[tuple[str, str]] = []
    gate_stack: list[bool] = []
    # An attribute may sit on its own line, above the declaration it applies to.
    pending_attr = False
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            gm = re.fullmatch(r"#if\s+defined\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", line)
            if gm:
                name = gm.group(1)
                if name not in active:
                    raise SlangLayoutError(
                        f"{struct_name}: unresolvable preprocessor gate "
                        f"{name!r} (resolvable: {', '.join(RESOLVABLE_DEFINES)})")
                gate_stack.append(active[name])
                continue
            if line.startswith("#endif"):
                if not gate_stack:
                    raise SlangLayoutError(
                        f"{struct_name}: unbalanced #endif in struct body")
                gate_stack.pop()
                continue
            raise SlangLayoutError(
                f"{struct_name}: unsupported preprocessor directive {line!r}")
        if not all(gate_stack):
            continue
        # Strip any leading attributes. A line that is nothing but an attribute
        # (e.g. `[mutating]` above a method) reduces to empty and is skipped;
        # an attribute ON a field declaration is NOT skipped and NOT ignored —
        # an attribute like `[[vk::offset(16)]]` changes where the field lands,
        # and honouring the declaration order while erasing the attribute would
        # emit a confidently wrong offset. Raise instead.
        had_attr = pending_attr
        while True:
            stripped = _ATTR_PREFIX.sub("", line)
            if stripped == line:
                break
            line, had_attr = stripped, True
        if not line:
            # Attribute on its own line — it applies to the NEXT declaration.
            pending_attr = True
            continue
        pending_attr = False
        if had_attr and line.endswith(";") and "(" not in line:
            raise SlangLayoutError(
                f"{struct_name}: attributed field declaration {line!r} — the "
                "attribute may change the field's offset and this module does "
                "not interpret attributes; add explicit support before "
                "mirroring this struct")
        if line.startswith(_SKIP_PREFIXES) or "(" in line:
            continue
        fm = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", line)
        if not fm:
            raise SlangLayoutError(
                f"{struct_name}: unrecognised declaration {line!r}")
        ftype = _norm_type(fm.group(1), spectral=spectral)
        if ftype not in SLANG_SCALAR_SIZES and ftype not in STRUCT_SOURCES:
            raise SlangLayoutError(
                f"{struct_name}: unknown field type {ftype!r} in {line!r}")
        fields.append((ftype, fm.group(2)))
    if gate_stack:
        raise SlangLayoutError(f"{struct_name}: unbalanced #if in struct body")
    return fields


def _source_text(struct: str) -> str:
    rel = STRUCT_SOURCES.get(struct)
    if rel is None:
        raise SlangLayoutError(f"struct {struct!r} is not registered in STRUCT_SOURCES")
    return (_SHADERS / rel).read_text(encoding="utf-8")


def _mtime(struct: str) -> float:
    return (_SHADERS / STRUCT_SOURCES[struct]).stat().st_mtime


@lru_cache(maxsize=None)
def _fields_cached(struct: str, spectral: bool, mlt: bool, metal: bool,
                   _mtime_key: float) -> tuple[tuple[str, str], ...]:
    return tuple(parse_struct_fields(_source_text(struct), struct,
                                     spectral=spectral, mlt=mlt, metal=metal))


def struct_fields(struct: str, *, spectral: bool = False, mlt: bool = False,
                  metal: bool = False) -> list[tuple[str, str]]:
    """``[(slang_type, field_name), …]`` for a registered struct, parsed lazily
    from its ``.slang`` source and cached per (file mtime, struct, variant)."""
    if struct not in STRUCT_SOURCES:
        raise SlangLayoutError(f"struct {struct!r} is not registered in STRUCT_SOURCES")
    return list(_fields_cached(struct, spectral, mlt, metal, _mtime(struct)))


# ── Layout math ──────────────────────────────────────────────────────


def _scalar_size(t: str, *, spectral: bool, mlt: bool, metal: bool) -> int:
    if t in SLANG_SCALAR_SIZES:
        return SLANG_SCALAR_SIZES[t]
    return sum(
        _scalar_size(ft, spectral=spectral, mlt=mlt, metal=metal)
        for ft, _ in struct_fields(t, spectral=spectral, mlt=mlt, metal=metal)
    )


def _msl_size_align(t: str, *, spectral: bool, mlt: bool,
                    metal: bool) -> tuple[int, int]:
    if t in SLANG_MSL_SIZES:
        return SLANG_MSL_SIZES[t], SLANG_MSL_ALIGNS[t]
    offset = 0
    struct_align = 1
    for ft, _ in struct_fields(t, spectral=spectral, mlt=mlt, metal=metal):
        size, align = _msl_size_align(ft, spectral=spectral, mlt=mlt, metal=metal)
        struct_align = max(struct_align, align)
        offset = (offset + align - 1) // align * align + size
    return (offset + struct_align - 1) // struct_align * struct_align, struct_align


class Layout:
    """Derived byte layout of one struct in one dialect.

    ``entries`` is the flattened ``[(name, size), …]`` walk in declaration order
    (nested struct fields appear as ``parent.child``); ``offsets`` maps each name
    — including the nested parents themselves, matching Slang's reflection — to
    ``(offset, size)``; ``stride`` is the struct's byte stride.
    """

    __slots__ = ("entries", "offsets", "stride")

    def __init__(self, entries: list[tuple[str, int]],
                 offsets: dict[str, tuple[int, int]], stride: int) -> None:
        self.entries = entries
        self.offsets = offsets
        self.stride = stride

    def offset(self, name: str) -> int:
        return self.offsets[name][0]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Layout(stride={self.stride}, fields={len(self.entries)})"


def _walk(struct: str, *, msl: bool, spectral: bool, mlt: bool, metal: bool,
          base: int, prefix: str, entries: list[tuple[str, int]],
          offsets: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """Append one struct's fields to ``entries``/``offsets``; return
    (end offset, struct alignment)."""
    offset = base
    struct_align = 1
    for ftype, fname in struct_fields(struct, spectral=spectral, mlt=mlt, metal=metal):
        name = f"{prefix}{fname}"
        if msl:
            size, align = _msl_size_align(ftype, spectral=spectral, mlt=mlt,
                                          metal=metal)
        else:
            size, align = _scalar_size(ftype, spectral=spectral, mlt=mlt,
                                       metal=metal), 1
        struct_align = max(struct_align, align)
        offset = (offset + align - 1) // align * align
        offsets[name] = (offset, size)
        if ftype in STRUCT_SOURCES:
            _walk(ftype, msl=msl, spectral=spectral, mlt=mlt, metal=metal,
                  base=offset, prefix=f"{name}.", entries=entries, offsets=offsets)
        else:
            entries.append((name, size))
        offset += size
    return offset, struct_align


def _layout(struct: str, *, msl: bool, spectral: bool, mlt: bool,
            metal: bool) -> Layout:
    entries: list[tuple[str, int]] = []
    offsets: dict[str, tuple[int, int]] = {}
    end, struct_align = _walk(struct, msl=msl, spectral=spectral, mlt=mlt,
                              metal=metal, base=0, prefix="", entries=entries,
                              offsets=offsets)
    stride = (end + struct_align - 1) // struct_align * struct_align if msl else end
    return Layout(entries, offsets, stride)


def scalar_layout(struct: str, *, spectral: bool = False, mlt: bool = False,
                  metal: bool = False) -> Layout:
    """Scalar (Vulkan ``-fvk-use-scalar-layout``) layout of a registered struct."""
    return _layout(struct, msl=False, spectral=spectral, mlt=mlt, metal=metal)


def msl_layout(struct: str, *, spectral: bool = False, mlt: bool = False,
               metal: bool = True) -> Layout:
    """MSL (Metal target) layout of a registered struct. ``metal`` defaults to
    True — an MSL query is by definition a ``SKINNY_METAL`` compile."""
    return _layout(struct, msl=True, spectral=spectral, mlt=mlt, metal=metal)


def scalar_stride(struct: str, *, spectral: bool = False, mlt: bool = False,
                  metal: bool = False) -> int:
    return scalar_layout(struct, spectral=spectral, mlt=mlt, metal=metal).stride


def msl_stride(struct: str, *, spectral: bool = False, mlt: bool = False,
               metal: bool = True) -> int:
    return msl_layout(struct, spectral=spectral, mlt=mlt, metal=metal).stride


# ── FrameConstants host scalar blob (the D1 blob rule) ───────────────


def fc_scalar_blob(*, mlt: bool = False) -> tuple[tuple[str, int], ...]:
    """Ordered ``(field-name, scalar-byte-size)`` of the host ``fc`` scalar blob
    that ``_pack_uniforms`` appends, flattened (the embedded ``Camera`` is
    ``camera.<field>``, matching Slang's reflected names).

    **Blob rule:** the variant's declared fields in order, with ``tileOriginY``
    always present and relocated to the tail (after the MLT tail when present).
    ``_pack_uniforms`` appends the trailing ``tileOriginY`` word on *both*
    backends: on Metal it lands at its reflected offset, and on Vulkan — whose
    ``FrameConstants`` has no such field — it is benign filler inside the
    oversized UBO, which is precisely what lets ``mltSigma`` sit at 564 where
    the Vulkan MLT SPIR-V expects it.
    """
    entries = scalar_layout("FrameConstants", mlt=mlt, metal=True).entries
    tail = [e for e in entries if e[0] == "tileOriginY"]
    if not tail:
        raise SlangLayoutError(
            "FrameConstants has no tileOriginY under SKINNY_METAL — the host blob "
            "rule relocates that field to the tail")
    return tuple([e for e in entries if e[0] != "tileOriginY"] + tail)


def fc_blob_size(*, mlt: bool = False) -> int:
    """Byte length of the host ``fc`` scalar blob for the variant."""
    return sum(sz for _, sz in fc_scalar_blob(mlt=mlt))


def fc_tile_origin_y_offset() -> int:
    """Byte offset of the trailing ``tileOriginY`` u32 in the base (non-MLT) host
    scalar blob, so the Metal band loop can patch it in place without a re-pack."""
    return fc_blob_size() - 4
