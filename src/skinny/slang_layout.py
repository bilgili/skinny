"""Derived authority for the host-mirrored GPU byte layouts registered below
(change ``reflection-owned-byte-layouts``).

Owns the ``FrameConstants`` uniform block, the material param records, and the
wavefront/SPPM/BDPT/MLT record family — see ``STRUCT_SOURCES``. NOT yet owned
(single-authored at their packers, deferred follow-up): ``SkinParameters``'
std140 UBO, ``INSTANCE_STRIDE``, and the light-buffer records.

The host↔GPU interface is a byte-offset agreement with the Slang structs. This
module derives each registered struct's ordered field list by **parsing the
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
import struct
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

#: A field declaration: `type name;`, optionally with an initializer, which may
#: itself contain parentheses (`float b = float(0);`). Matching this BEFORE the
#: "line contains `(` ⇒ method" heuristic is what keeps an initialized field
#: from being silently dropped. The initializer is captured so a multi-declarator
#: (`float x = float(0), y = float(0);`) can be rejected rather than silently
#: yielding only its first name.
_FIELD_DECL = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(=[^;]*)?;")


def _has_top_level_comma(text: str) -> bool:
    """True if ``text`` has a comma outside any bracket — i.e. it declares more
    than one name. Commas *inside* an initializer call (`float3(0, 0, 0)`) are
    part of one declarator and do not count."""
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth <= 0:
            return True
    return False


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
    # Depth of any nested block (a method or property body). Lines are otherwise
    # classified independently, so without this a LOCAL declaration inside a
    # multiline method body (`float local = float(0);`) would be appended as a
    # phantom struct field and shift every offset after it (codex round-5).
    block_depth = 0
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if block_depth:
            block_depth += line.count("{") - line.count("}")
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
        # Classify BEFORE the method heuristic below: a field may carry an
        # initializer that contains parentheses (`float b = float(0);`), which a
        # bare "line contains (" test would misread as a method and drop
        # silently, shifting every offset after it (codex round-4 finding).
        decl = _FIELD_DECL.fullmatch(line)
        if decl is None and line.endswith(";") and _has_top_level_comma(line):
            # Mixed multi-declarator (`float x, y = float(0);`): parens make the
            # method heuristic below swallow it, so catch it here. Commas inside
            # a parameter list sit at bracket depth > 0 and do not trip this.
            raise SlangLayoutError(
                f"{struct_name}: unrecognised declaration {line!r} — multiple "
                "declarators on one line are not supported; split them")
        if decl is None and "(" in line:
            # Method / constructor / operator — not layout. Its body (this line's
            # trailing `{`, or a `{` on the following line) is swallowed whole.
            block_depth += line.count("{") - line.count("}")
            continue
        if had_attr and decl is not None:
            raise SlangLayoutError(
                f"{struct_name}: attributed field declaration {line!r} — the "
                "attribute may change the field's offset and this module does "
                "not interpret attributes; add explicit support before "
                "mirroring this struct")
        if line.startswith(_SKIP_PREFIXES):
            # Non-field construct (property, static const, a bare brace opening a
            # method body on its own line, …) — swallow any block it opens.
            block_depth += line.count("{") - line.count("}")
            continue
        if decl is None:
            raise SlangLayoutError(
                f"{struct_name}: unrecognised declaration {line!r}")
        if _has_top_level_comma(decl.group(3) or ""):
            # `float x = float(0), y = float(0);` — the regex would yield only
            # `x` and drop `y`, shifting every offset after it. The bare form
            # (`float x, y;`) already fails to match at all; make both loud.
            raise SlangLayoutError(
                f"{struct_name}: unrecognised declaration {line!r} — multiple "
                "declarators on one line are not supported; split them")
        ftype = _norm_type(decl.group(1), spectral=spectral)
        if ftype not in SLANG_SCALAR_SIZES and ftype not in STRUCT_SOURCES:
            raise SlangLayoutError(
                f"{struct_name}: unknown field type {ftype!r} in {line!r}")
        fields.append((ftype, decl.group(2)))
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


# ── Material field table (change flat-material-field-table) ──────────
#
# The two material records are owned down to NAMED FIELDS, not only their
# strides. `FlatMaterialParams` declares 14 opaque `float4` rows, so the parser
# derives each ROW's offset but not which lane inside a row means `roughness`
# and which means `metallic`. That lane assignment is declared below and pinned
# by a permanent golden captured from the pre-change packer
# (`tests/fixtures/material_field_offsets.json`) — derive where possible, pin
# where not. `StdSurfaceParams` declares named scalars, so its half of the table
# is derived outright.
#
# The table is LOAD-BEARING, not documentation: `material_pack` emits both
# records by walking it, so transposing two same-typed fields moves their
# offsets and fails the golden. A positional `struct.pack` could not be gated
# this way — that is why packing is keyed by name.

#: Field kinds and their lane width inside a row.
MATERIAL_KIND_LANES: dict[str, int] = {
    "float": 1, "uint": 1, "color3": 3, "vec4": 4,
}


class MaterialField:
    """One named field of a material record.

    ``row`` is the Slang-declared field it lives in (a ``float4`` row, or a
    scalar field that is its own row); ``lane`` is its first 4-byte lane inside
    that row. The byte offset is DERIVED — ``row``'s parsed offset plus
    ``4 * lane`` — so a shader-side reorder moves the field without an edit here.

    ``key`` is the ``parameter_overrides`` key that feeds this field directly,
    when one does. A field whose value comes from a derivation instead (the
    medium coefficients, the spectral identity lanes, the texture indices the
    upload supplies) carries ``key=None`` — the derivation's INPUT keys are
    registered separately, in ``FLAT_DERIVED_KEYS``.
    """

    __slots__ = ("name", "row", "lane", "kind", "default", "key")

    def __init__(self, name: str, row: str, lane: int, kind: str,
                 default, key: str | None = None) -> None:
        if kind not in MATERIAL_KIND_LANES:
            raise SlangLayoutError(f"{name}: unknown material field kind {kind!r}")
        self.name = name
        self.row = row
        self.lane = lane
        self.kind = kind
        self.default = default
        self.key = key

    @property
    def lanes(self) -> int:
        return MATERIAL_KIND_LANES[self.kind]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MaterialField({self.name!r}, {self.row}+{self.lane}, {self.kind})"


_U32_SENTINEL = 0xFFFFFFFF

#: ``FlatMaterialParams`` (binding 13) field by field, in offset order. Lane
#: assignment is the ONE thing the `float4` declaration cannot express.
FLAT_MATERIAL_FIELDS: tuple[MaterialField, ...] = (
    MaterialField("diffuseColor", "_diffuseColorRoughness", 0, "color3",
                  (0.72, 0.72, 0.72), key="diffuseColor"),
    MaterialField("roughness", "_diffuseColorRoughness", 3, "float", 0.5,
                  key="roughness"),
    MaterialField("metallic", "metallic", 0, "float", 0.0, key="metallic"),
    MaterialField("specular", "specular", 0, "float", 0.5, key="specular"),
    MaterialField("opacity", "opacity", 0, "float", 1.0, key="opacity"),
    MaterialField("diffuseTextureIdx", "diffuseTextureIdx", 0, "uint", _U32_SENTINEL),
    MaterialField("roughnessTextureIdx", "roughnessTextureIdx", 0, "uint", _U32_SENTINEL),
    MaterialField("metallicTextureIdx", "metallicTextureIdx", 0, "uint", _U32_SENTINEL),
    MaterialField("normalTextureIdx", "normalTextureIdx", 0, "uint", _U32_SENTINEL),
    MaterialField("emissiveTextureIdx", "emissiveTextureIdx", 0, "uint", _U32_SENTINEL),
    MaterialField("emissiveColor", "_emissiveColorIor", 0, "color3", (0.0, 0.0, 0.0),
                  key="emissiveColor"),
    MaterialField("ior", "_emissiveColorIor", 3, "float", 1.5, key="ior"),
    MaterialField("coat", "coat", 0, "float", 0.0, key="coat"),
    MaterialField("coatRoughness", "coatRoughness", 0, "float", 0.0,
                  key="coat_roughness"),
    MaterialField("coatIOR", "coatIOR", 0, "float", 1.5, key="coat_IOR"),
    MaterialField("opacityTextureIdx", "opacityTextureIdx", 0, "uint", _U32_SENTINEL),
    MaterialField("coatColor", "_coatColorOpacityThreshold", 0, "color3",
                  (1.0, 1.0, 1.0), key="coat_color"),
    MaterialField("opacityThreshold", "_coatColorOpacityThreshold", 3, "float", 0.0,
                  key="opacityThreshold"),
    MaterialField("normalScale", "_normalScaleChannelMask", 0, "color3",
                  (2.0, 2.0, 2.0)),
    MaterialField("channelMask", "_normalScaleChannelMask", 3, "uint", 0),
    MaterialField("normalBias", "_normalBiasPad", 0, "color3", (-1.0, -1.0, -1.0)),
    # `_normalBiasPad.w` — spectral-only Cauchy B (0 = constant IOR), so the RGB
    # pack keeps the literal 0 the pad always carried.
    MaterialField("glassCauchyB", "_normalBiasPad", 3, "float", 0.0),
    # transmissionColor's real default is the material's own diffuse albedo; the
    # packer supplies it, so the delta-transmission weight is unchanged when the
    # override is absent. The table default is only the no-material fallback.
    MaterialField("transmissionColor", "_transmissionColorDiffuseRough", 0, "color3",
                  (0.72, 0.72, 0.72), key="transmission_color"),
    MaterialField("diffuseRoughness", "_transmissionColorDiffuseRough", 3, "float", 0.0,
                  key="diffuse_roughness"),
    MaterialField("specularColor", "_specularColorPad", 0, "color3", (1.0, 1.0, 1.0),
                  key="specular_color"),
    # `_specularColorPad.w` — spectral-only named-conductor id (0 = RGB Schlick).
    MaterialField("conductorMetalId", "_specularColorPad", 3, "uint", 0),
    MaterialField("mediumSigmaA", "_mediumSigmaA_g", 0, "color3", (0.0, 0.0, 0.0)),
    MaterialField("mediumG", "_mediumSigmaA_g", 3, "float", 0.0),
    MaterialField("mediumSigmaS", "_mediumSigmaS_kind", 0, "color3", (0.0, 0.0, 0.0)),
    MaterialField("mediumKind", "_mediumSigmaS_kind", 3, "uint", 0),
    MaterialField("worldToUvw0", "_worldToUvw0", 0, "vec4", (1.0, 0.0, 0.0, 0.0)),
    MaterialField("worldToUvw1", "_worldToUvw1", 0, "vec4", (0.0, 1.0, 0.0, 0.0)),
    MaterialField("worldToUvw2", "_worldToUvw2", 0, "vec4", (0.0, 0.0, 1.0, 0.0)),
    MaterialField("cloudDensity", "_cloudDensityWispinessFrequency", 0, "float", 0.0),
    MaterialField("cloudWispiness", "_cloudDensityWispinessFrequency", 1, "float", 0.0),
    MaterialField("cloudFrequency", "_cloudDensityWispinessFrequency", 2, "float", 0.0),
    MaterialField("_cloudPad", "_cloudDensityWispinessFrequency", 3, "float", 0.0),
)

#: ``StdSurfaceParams`` (binding 19) — declared as named scalars, so the whole
#: table is derived. Defaults come from ``pack_std_surface_params``'s reality.
_STD_SURFACE_DEFAULTS: dict[str, object] = {
    "base_color": (0.8, 0.8, 0.8), "base": 1.0, "specular": 1.0,
    "specular_roughness": 0.5, "specular_color": (1.0, 1.0, 1.0),
    "specular_IOR": 1.5, "transmission_color": (1.0, 1.0, 1.0),
    "subsurface_color": (1.0, 1.0, 1.0), "subsurface_radius": (1.0, 1.0, 1.0),
    "sheen_color": (1.0, 1.0, 1.0), "sheen_roughness": 0.3,
    "coat_roughness": 0.1, "coat_IOR": 1.5, "coat_color": (1.0, 1.0, 1.0),
    "thin_film_IOR": 1.5, "emission_color": (1.0, 1.0, 1.0),
    "opacity": (1.0, 1.0, 1.0), "subsurface_scale": 1.0,
}

_SLANG_TO_MATERIAL_KIND = {"float": "float", "float3": "color3", "uint": "uint",
                           "float4": "vec4"}


@lru_cache(maxsize=None)
def _std_surface_fields(_mtime_key: float) -> tuple[MaterialField, ...]:
    out = []
    for ftype, fname in struct_fields("StdSurfaceParams"):
        kind = _SLANG_TO_MATERIAL_KIND.get(ftype)
        if kind is None:
            raise SlangLayoutError(
                f"StdSurfaceParams: field {fname!r} has type {ftype!r}, which the "
                "material field table cannot express")
        zero = 0.0 if kind != "uint" else 0
        default = _STD_SURFACE_DEFAULTS.get(
            fname, (zero, zero, zero) if kind == "color3" else zero)
        out.append(MaterialField(fname, fname, 0, kind, default))
    return tuple(out)


def std_surface_fields() -> tuple[MaterialField, ...]:
    """``StdSurfaceParams`` fields in declaration order, derived."""
    return _std_surface_fields(_mtime("StdSurfaceParams"))


def material_fields(record: str) -> tuple[MaterialField, ...]:
    """The field table of a registered material record."""
    if record == "FlatMaterialParams":
        return FLAT_MATERIAL_FIELDS
    if record == "StdSurfaceParams":
        return std_surface_fields()
    raise SlangLayoutError(f"{record!r} is not a registered material record")


def material_field_offsets(record: str, *, msl: bool = False) -> dict[str, int]:
    """``{field name: byte offset}`` for a material record.

    Row offsets are derived from the Slang declaration; the lane adds
    ``4 * lane``. This is the transposition gate's subject: swapping two
    same-typed fields in the table moves both offsets.
    """
    layout = (msl_layout if msl else scalar_layout)(record)
    out: dict[str, int] = {}
    for f in material_fields(record):
        try:
            row_off, row_size = layout.offsets[f.row]
        except KeyError:
            raise SlangLayoutError(
                f"{record}: field {f.name!r} names row {f.row!r}, which the "
                f"declaration does not have") from None
        end = 4 * (f.lane + f.lanes)
        if end > row_size:
            raise SlangLayoutError(
                f"{record}: field {f.name!r} occupies lanes "
                f"{f.lane}..{f.lane + f.lanes} of {f.row!r}, which is only "
                f"{row_size} B")
        out[f.name] = row_off + 4 * f.lane
    return out


def check_material_record(record: str) -> None:
    """Raise unless the record's fields tile its SCALAR stride exactly, with no
    overlap and no gap. A lane left unclaimed is a byte the host never writes.

    Scalar only, deliberately: the MSL dialect pads ``float3`` to 16 B, so its
    gaps are the layout working as intended and a tiling check there would be
    guaranteed to fail. The MSL offsets are gated by the golden instead."""
    stride = scalar_stride(record)
    claimed = bytearray(stride)
    offsets = material_field_offsets(record)
    for f in material_fields(record):
        off = offsets[f.name]
        for b in range(off, off + 4 * f.lanes):
            if claimed[b]:
                raise SlangLayoutError(
                    f"{record}: byte {b} is claimed twice (at {f.name!r})")
            claimed[b] = 1
    missing = [b for b, c in enumerate(claimed) if not c]
    if missing:
        raise SlangLayoutError(
            f"{record}: {len(missing)} byte(s) claimed by no field, first at "
            f"{missing[0]}")


#: ``<`` — little-endian, STANDARD size, no native alignment padding. The
#: pre-change packers used native mode; every field here is 4-byte aligned, so
#: the two agree byte for byte, and the explicit form cannot acquire padding if
#: a field kind is ever added.
_PACK_FMT = {"float": "<f", "uint": "<I", "color3": "<3f", "vec4": "<4f"}


def pack_material_record(record: str, values: dict, *,
                         msl: bool = False) -> bytes:
    """Emit one material record from ``{field name: value}``.

    Every field lands at the offset :func:`material_field_offsets` derives, and
    a field the caller omits gets the table's declared default. A name the table
    does not have raises — a packer cannot silently drop a value.

    This is what makes the name→offset golden a real transposition gate: the
    bytes are produced BY the table, so swapping two same-typed fields in it
    moves both offsets and the golden fails.
    """
    fields = material_fields(record)
    stride = (msl_stride if msl else scalar_stride)(record)
    offsets = material_field_offsets(record, msl=msl)
    unknown = sorted(set(values) - {f.name for f in fields})
    if unknown:
        raise SlangLayoutError(
            f"{record}: no such field(s) {', '.join(unknown)}")
    buf = bytearray(stride)
    for f in fields:
        value = values.get(f.name, f.default)
        fmt = _PACK_FMT[f.kind]
        if f.kind == "uint":
            struct.pack_into(fmt, buf, offsets[f.name], int(value) & 0xFFFFFFFF)
        elif f.kind == "float":
            struct.pack_into(fmt, buf, offsets[f.name], float(value))
        else:
            struct.pack_into(fmt, buf, offsets[f.name],
                             *(float(c) for c in value))
    return bytes(buf)


# ── Material override-key vocabulary ─────────────────────────────────
#
# The strings that travel on `Material.parameter_overrides` from the pbrt
# authors (`pbrt/materials.py`, `pbrt/media.py`), through the intake merge and
# derivations (`usd_loader.py`), to the packers (`material_pack.py`) — and that
# `mtlx_synthesis.py` advertises as editable. Three tables used to restate parts
# of this vocabulary with "keep in sync" comments; they are projections now.
#
# A key outside the vocabulary is REJECTED at the packing seam, not silently
# dropped: a misspelled override used to be invisible until someone noticed a
# wrong-looking render.

#: Override keys ``pack_flat_material`` reads that do NOT bind to one record
#: field — each steers a derivation whose result lands in one or more fields.
FLAT_DERIVED_KEYS: frozenset = frozenset({
    # spectral-only identity lookups → conductorMetalId / ior + glassCauchyB
    "conductor_metal", "glass_dispersion",
    # subsurface interior medium → mediumSigmaA/S + mediumG
    "subsurface_sigma_a", "subsurface_sigma_s", "subsurface_g",
    # free-standing medium boundary: the kind dispatch, the σ folds, the
    # world→uvw rows → mediumKind / mediumSigmaA/S / mediumG / worldToUvw*
    "volume_interface", "volume_grid_asset", "volume_cloud",
    "volume_sigma_a", "volume_sigma_s", "volume_g", "volume_world_to_uvw",
    # procedural cloud scalars → cloudDensity / cloudWispiness / cloudFrequency
    "cloud_density", "cloud_wispiness", "cloud_frequency",
})

#: Every override key ``pack_flat_material`` reads — the direct field bindings
#: declared on the table plus the derivation inputs above.
FLAT_OVERRIDE_KEYS: frozenset = frozenset(
    f.key for f in FLAT_MATERIAL_FIELDS if f.key) | FLAT_DERIVED_KEYS

#: Override keys ``pack_std_surface_params`` reads under their canonical
#: MaterialX ``standard_surface`` names — derived from the declaration, since
#: every one of them is a declared field of the record.
STD_SURFACE_OVERRIDE_KEYS: frozenset = frozenset(
    f.name for f in std_surface_fields() if not f.name.startswith("_"))

#: MaterialX ``standard_surface`` input name → the ``FlatMaterialParams``
#: override key it feeds. Four genuine renames (a rename cannot be derived) plus
#: every std-surface key the flat packer reads under the SAME name AND the same
#: kind. The kind guard is load-bearing: std-surface ``opacity`` is a ``color3``
#: while the flat record's is a ``float``, so aliasing them would advertise an
#: editable that ``_override_float`` silently discards.
_STD_SURFACE_RENAMES: dict[str, str] = {
    "base_color": "diffuseColor",
    "specular_roughness": "roughness",
    "metalness": "metallic",
    "specular_IOR": "ior",
}


def _flat_kinds() -> dict[str, str]:
    """``{override key: field kind}`` for the directly-bound flat fields."""
    return {f.key: f.kind for f in FLAT_MATERIAL_FIELDS if f.key}


@lru_cache(maxsize=None)
def _std_surface_to_flat(_mtime_key: float) -> dict[str, str]:
    flat_kind = _flat_kinds()
    std_kind = {f.name: f.kind for f in std_surface_fields()}
    table = dict(_STD_SURFACE_RENAMES)
    for key in sorted(STD_SURFACE_OVERRIDE_KEYS & FLAT_OVERRIDE_KEYS):
        if key in table:
            continue
        if flat_kind.get(key) == std_kind.get(key):
            table[key] = key
    return table


def std_surface_to_flat() -> dict[str, str]:
    """The one std_surface→flat alias table. ``usd_loader`` and
    ``mtlx_synthesis`` both project from this; neither restates it."""
    return dict(_std_surface_to_flat(_mtime("StdSurfaceParams")))


#: Flat override keys the loader's own folds author under UsdPreviewSurface
#: names — reachable by no std_surface alias, so the editable surface has to name
#: them explicitly. ``emissiveColor`` comes from the emission fold,
#: ``opacity``/``opacityThreshold`` from the transmission and cutout folds.
PREVIEW_SURFACE_FLAT_KEYS: frozenset = frozenset({
    "opacity", "opacityThreshold", "emissiveColor",
})

#: Keys a real scene carries that no packer reads — recorded so the seam can
#: tell "recognised bookkeeping" from "misspelled override". Each names its
#: author and why the packer ignores it.
INTAKE_ONLY_KEYS: frozenset = frozenset({
    # UsdPreviewSurface coat spelling; `_canonicalize_coat` folds these onto
    # `coat`/`coat_roughness` and leaves the originals in place.
    "clearcoat", "clearcoatRoughness",
    # pbrt `Material "subsurface"` boundary IOR (pbrt/materials.py). No packer
    # reads it: the flat record's boundary eta is the `ior` lane, which the same
    # author writes from the same pbrt `eta` parameter. Across the corpus the two
    # agree exactly. The two readers are not the same code, though: `ior` comes
    # from `scalar("eta", …)` while this key comes from `subsurface_coefficients`,
    # which alone resolves a NAMED-spectrum eta (`glass-LASF9`) to its d-line
    # index — the unguarded reader raises on one today. That is a pbrt-import
    # defect, not a packing one; recorded here so the key reads as known-unread
    # rather than as a typo.
    "subsurface_eta",
    # pbrt bookkeeping: the medium's pbrt name (pbrt/media.py) and the density
    # field inside the .nvdb (consumed by pbrt/api.py at import, not at pack).
    "pbrt_medium", "volume_grid_field",
})

#: Keys the RENDERER reads off ``parameter_overrides`` without either packer
#: seeing them. ``emissive_spectral`` carries the pbrt area-light SPD payload
#: (pbrt/api.py) that the spectral emissive path evaluates.
RENDERER_OVERRIDE_KEYS: frozenset = frozenset({"emissive_spectral"})

#: Every override key this table recognises.
#:
#: NOTE the scope. ``parameter_overrides`` is a SHARED bag: besides the two
#: material records it also carries the MaterialX skin library's input names
#: (packed by ``_pack_mtlx_skin_array`` against a MaterialX uniform block) and a
#: material graph's own parameter names. Those vocabularies are DATA — the
#: inputs of whatever document the scene references — so no static table can
#: enumerate them. This set governs the materials whose overrides the field
#: table owns; see ``material_pack.check_material_vocabulary`` for the scope
#: rule that decides when a stray key is an error and when it is only reported.
MATERIAL_OVERRIDE_KEYS: frozenset = (
    FLAT_OVERRIDE_KEYS | STD_SURFACE_OVERRIDE_KEYS | PREVIEW_SURFACE_FLAT_KEYS
    | INTAKE_ONLY_KEYS | RENDERER_OVERRIDE_KEYS | frozenset(std_surface_to_flat()))


def unknown_override_keys(overrides) -> list[str]:
    """The keys of *overrides* outside :data:`MATERIAL_OVERRIDE_KEYS`, sorted."""
    return sorted(str(k) for k in overrides if str(k) not in MATERIAL_OVERRIDE_KEYS)
