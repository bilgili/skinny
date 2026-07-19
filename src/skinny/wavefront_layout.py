"""GPU-free layout descriptor for the wavefront path-state record (P1 §P1-A).

Mirrors the `WavefrontPathState` struct in
``shaders/wavefront/wavefront_state.slang``. The renderer allocates the
path-state StructuredBuffer as ``stream_size * PATH_STATE_STRIDE`` bytes; the
Slang struct and this table must stay in lockstep or the wavefront stages read
garbage (``tests/test_wavefront_state.py`` locks them together).

Two layouts (design B / change metal-wavefront-parity):

* **scalar** (Vulkan, ``-fvk-use-scalar-layout``): ``float3`` = 12 B with 4-byte
  alignment, fields tightly packed. This is the default everywhere.
* **MSL** (Metal in-process Slang→Metal): the Metal target pads ``float3`` to a
  16-byte size *and* 16-byte alignment, so the same struct has a larger stride.
  The wavefront record/queue buffers are GPU-internal scratch (the host never
  packs them — only allocates), so the **only** cross-backend artifact is the
  buffer size: on Metal the allocator must size against the MSL stride or the
  Metal kernels overrun. Pass ``msl=True`` to the size helpers on the Metal path;
  ``tests/test_wavefront_state.py`` locks the MSL strides to the reflected Metal
  layout (task 1.5).
"""

from __future__ import annotations

# Scalar-layout byte sizes for the field types the record uses. All are
# multiples of 4 with alignment <= 4, so the struct packs tightly with no
# interior padding.
SLANG_SCALAR_SIZES: dict[str, int] = {
    "float": 4,
    "float2": 8,
    "float3": 12,
    "float4": 16,
    "uint": 4,
    "int": 4,
    "bool": 4,  # Slang stores bool as a 32-bit value in a buffer (scalar layout)
}

# MSL (Metal target) byte sizes + alignments. Slang pads ``float3`` to 16 B
# (size *and* alignment) on Metal; every other field keeps its natural
# size==alignment. A struct's stride rounds up to its largest member alignment.
SLANG_MSL_SIZES: dict[str, int] = {
    "float": 4, "float2": 8, "float3": 16, "float4": 16, "uint": 4, "int": 4,
    "bool": 1,
}
SLANG_MSL_ALIGNS: dict[str, int] = {
    "float": 4, "float2": 8, "float3": 16, "float4": 16, "uint": 4, "int": 4,
    "bool": 1,
}


def _struct_stride(fields: list[tuple[str, str]], *, msl: bool) -> int:
    """Byte stride of a record struct under the scalar (Vulkan) or MSL (Metal)
    layout. Scalar packs tightly (all alignments <= 4); MSL walks per-field
    alignment and rounds the total up to the struct's largest member alignment.
    """
    if not msl:
        return sum(SLANG_SCALAR_SIZES[t] for _, t in fields)
    offset = 0
    struct_align = 1
    for _, t in fields:
        align = SLANG_MSL_ALIGNS[t]
        struct_align = max(struct_align, align)
        offset = (offset + align - 1) // align * align
        offset += SLANG_MSL_SIZES[t]
    return (offset + struct_align - 1) // struct_align * struct_align

# (field_name, slang_type) in declaration order — must match the Slang struct.
# In the spectral build (change spectral-wavefront) the color roles carry
# `Spectrum` = float4 (vs float3 RGB) and the struct appends `SampledWavelengths`
# (two float4: lambda, pdf) — see wavefront_state.slang. The two variants diverge
# by construction: scalar grows +4 B per retyped color field plus +32 B for sw;
# MSL already pads float3→16 B so the color retype is 0-byte on Metal and only sw
# (+32 B) grows the MSL stride.


def _path_state_fields(spectral: bool) -> list[tuple[str, str]]:
    col = "float4" if spectral else "float3"
    fields: list[tuple[str, str]] = [
        ("rayOrigin",  "float3"),
        ("rayDir",     "float3"),
        ("throughput", col),
        ("radiance",   col),
        ("pixelIndex", "uint"),
        ("rngState",   "uint"),
        ("depth",      "uint"),
        ("flags",      "uint"),
        ("bsdfPdf",    "float"),
    ]
    if spectral:
        # SampledWavelengths { float4 lambda; float4 pdf; }
        fields += [("sw_lambda", "float4"), ("sw_pdf", "float4")]
    return fields


# RGB field list (back-compat: existing call sites and tests reference FIELDS).
FIELDS: list[tuple[str, str]] = _path_state_fields(spectral=False)


def path_state_size(*, msl: bool = False, spectral: bool = False) -> int:
    """Byte stride of WavefrontPathState — scalar (default) or MSL (``msl=True``),
    RGB (default) or spectral (``spectral=True``)."""
    return _struct_stride(_path_state_fields(spectral), msl=msl)


PATH_STATE_STRIDE = path_state_size()              # 68 B (scalar / Vulkan)
PATH_STATE_STRIDE_MSL = path_state_size(msl=True)  # 96 B (Metal)
PATH_STATE_STRIDE_SPECTRAL = path_state_size(spectral=True)              # scalar
PATH_STATE_STRIDE_SPECTRAL_MSL = path_state_size(msl=True, spectral=True)  # Metal

# `flags` bit positions (mirror the static consts in wavefront_state.slang).
PATH_FLAG_ALIVE = 1 << 0     # lane still bouncing
PATH_FLAG_SPECULAR = 1 << 1  # last bounce was specular → skip NEE-MIS


# ── Queue / dispatch buffer sizing (P1 §P1-B, §P1-C) ───────────────

_UINT = 4
# One VkDispatchIndirectCommand per material: (x, y, z) group counts.
INDIRECT_ARGS_STRIDE = 3 * _UINT  # 12 B


# ── Wavefront-native path-record buffers (change wavefront-native-path-records) ─
# Mirror of RecVertex in shaders/wavefront/wf_records.slang — the per-lane
# vertex stack the wavefront record emitter carries in VRAM (NOT in
# WavefrontPathState). The path pass allocates the record-stack buffer as
# `stream_size * REC_MAX_BOUNCES * REC_VERTEX_STRIDE` bytes and a per-lane count
# buffer as `stream_size * 4` bytes — full-size only while recording, else
# 1-element dummies. `tests/test_wavefront_state.py` locks this to the Slang struct.
REC_MAX_BOUNCES = 6  # lockstep with REC_MAX_BOUNCES (path_record_common.slang)

REC_VERTEX_FIELDS: list[tuple[str, str]] = [
    ("pos",     "float3"),
    ("normal",  "float3"),
    ("wo",      "float3"),
    ("wiLocal", "float3"),
    ("L_k",     "float3"),
    ("beta_in", "float3"),
    ("depth",   "uint"),
]


def rec_vertex_size(*, msl: bool = False) -> int:
    """Byte stride of RecVertex — scalar (default) or MSL (``msl=True``)."""
    return _struct_stride(REC_VERTEX_FIELDS, msl=msl)


REC_VERTEX_STRIDE = rec_vertex_size()              # 76 B (scalar / Vulkan)
REC_VERTEX_STRIDE_MSL = rec_vertex_size(msl=True)  # 112 B (Metal)


def queue_buffer_sizes(stream_size: int, num_materials: int,
                       *, msl: bool = False, spectral: bool = False) -> dict[str, int]:
    """Byte sizes for the wavefront stage buffers, the single source of truth
    the `WavefrontPasses` allocator (vk_wavefront.py) sizes against.

    Stream-sized buffers scale with the lane count; per-material buffers scale
    with the active material count. The hit buffer is intentionally absent —
    its stride follows `HitData` (common.slang) and is pinned alongside the
    intersect stage that produces it.

    ``msl=True`` sizes the struct-backed buffers (``path_state``) against the
    Metal MSL stride so the Metal wavefront allocator does not undersize them;
    plain (uint-stride) queue buffers are layout-agnostic. Vulkan callers omit
    it and keep the scalar sizing byte-for-byte. ``spectral=True`` sizes
    ``path_state`` against the wider spectral WavefrontPathState (Spectrum
    throughput/radiance + SampledWavelengths); RGB stays byte-identical.
    """
    path_state_stride = path_state_size(msl=msl, spectral=spectral)
    return {
        "path_state":      stream_size * path_state_stride,
        "ray_queue":       stream_size * _UINT,
        "material_queue":  stream_size * _UINT,
        "ray_count":       _UINT,
        "material_count":  num_materials * _UINT,
        "material_offset": num_materials * _UINT,
        "indirect_args":   num_materials * INDIRECT_ARGS_STRIDE,
    }


# ── SPPM per-pixel state buffers (change photon-mapping-sppm, PM-1) ─────────
# Mirror of VisiblePoint / SppmAccum in
# shaders/integrators/sppm_state.slang. Both are one-element-per-pixel
# GPU-internal scratch the SPPM stages own; the host only SIZES them (never
# packs). The renderer allocates the visible-point buffer as
# `stream_size * VISIBLE_POINT_STRIDE` and the per-pass deposit accumulator as
# `stream_size * SPPM_ACCUM_STRIDE`. `tests/test_sppm_state.py` locks these to
# the Slang structs.

# VisiblePoint.flags bits (mirror the static consts in sppm_state.slang).
VP_ACTIVE = 1 << 0  # a valid visible point was stored this pass

# VisiblePoint / SppmAccum, RGB and spectral (change spectral-wavefront, D5).
# Spectral (SKINNY_SPECTRAL): beta/ld carry `Spectrum`=float4 (per-pass spectral)
# and VisiblePoint appends `conductorMetalId` (exact conductor Fresnel at deposit);
# SppmAccum grows to 4 hero-λ flux channels (phi0..phi3). tau STAYS float3 in both
# builds — the per-pass per-λ flux is resolved to 3-wide BEFORE it folds into the
# progressive estimator (D5), so tau is a spectral-invariant quantity. The RGB
# layout is byte-identical to before (phiR/G/B → phi0/1/2 is a name-only change).


def _visible_point_fields(spectral: bool) -> list[tuple[str, str]]:
    col = "float4" if spectral else "float3"
    fields: list[tuple[str, str]] = [
        ("pos",           "float3"),
        ("ns",            "float3"),
        ("wo",            "float3"),
        ("beta",          col),
        ("ld",            col),
        ("albedo",        "float3"),
        ("F0",            "float3"),
        ("coatColor",     "float3"),
        ("roughness",     "float"),
        ("metallic",      "float"),
        ("specular",      "float"),
        ("ior",           "float"),
        ("opacity",       "float"),
        ("coat",          "float"),
        ("coatRoughness", "float"),
        ("coatIOR",       "float"),
        ("transmissionColor", "float3"),
        ("specularColor",     "float3"),
        ("diffuseRoughness",  "float"),
    ]
    if spectral:
        fields.append(("conductorMetalId", "uint"))
    fields += [
        ("tau",           "float3"),
        ("flags",         "uint"),
        ("radius",        "float"),
        ("n",             "float"),
    ]
    return fields


def _sppm_accum_fields(spectral: bool) -> list[tuple[str, str]]:
    # phiR/G/B are the RGB channels (kept byte-identical); phiW is the 4th spectral
    # hero-λ slot (RGBA convention), present only under SKINNY_SPECTRAL.
    fields: list[tuple[str, str]] = [
        ("phiR", "uint"),
        ("phiG", "uint"),
        ("phiB", "uint"),
    ]
    if spectral:
        fields.append(("phiW", "uint"))
    fields.append(("m", "uint"))
    return fields


# RGB field lists (back-compat: existing call sites + tests reference these).
VISIBLE_POINT_FIELDS: list[tuple[str, str]] = _visible_point_fields(spectral=False)
SPPM_ACCUM_FIELDS: list[tuple[str, str]] = _sppm_accum_fields(spectral=False)


def visible_point_size(*, msl: bool = False, spectral: bool = False) -> int:
    """Byte stride of VisiblePoint — scalar (default) or MSL (``msl=True``),
    RGB (default) or spectral (``spectral=True``)."""
    return _struct_stride(_visible_point_fields(spectral), msl=msl)


def sppm_accum_size(*, msl: bool = False, spectral: bool = False) -> int:
    """Byte stride of SppmAccum — scalar (default) or MSL (``msl=True``). All
    fields are uint, so the scalar and MSL strides are identical (RGB 16 B /
    spectral 20 B)."""
    return _struct_stride(_sppm_accum_fields(spectral), msl=msl)


VISIBLE_POINT_STRIDE = visible_point_size()              # 180 B (scalar / Vulkan)
VISIBLE_POINT_STRIDE_MSL = visible_point_size(msl=True)  # 240 B (Metal)
VISIBLE_POINT_STRIDE_SPECTRAL = visible_point_size(spectral=True)              # scalar
VISIBLE_POINT_STRIDE_SPECTRAL_MSL = visible_point_size(msl=True, spectral=True)  # Metal
SPPM_ACCUM_STRIDE = sppm_accum_size()                    # 16 B (both layouts)
SPPM_ACCUM_STRIDE_MSL = sppm_accum_size(msl=True)        # 16 B
SPPM_ACCUM_STRIDE_SPECTRAL = sppm_accum_size(spectral=True)              # 20 B
SPPM_ACCUM_STRIDE_SPECTRAL_MSL = sppm_accum_size(msl=True, spectral=True)  # 20 B


def sppm_buffer_sizes(num_pixels: int, *, msl: bool = False,
                      spectral: bool = False) -> dict[str, int]:
    """Byte sizes for the SPPM per-pixel buffers, the source of truth the SPPM
    stage allocator sizes against.

    CRITICAL: the SPPM visible-point estimator (radius/N/tau) is **per pixel**
    and PERSISTS across passes, so these buffers MUST be sized by
    ``num_pixels = width*height`` — NOT by the wavefront ``stream_size`` (which
    aliases different pixels across tiles, and would undersize the buffer for any
    frame that tiles, corrupting state after tile 0). The SPPM stages index by
    the per-pixel lane, in ``[0, num_pixels)``.

    ``msl=True`` uses the Metal struct stride so the Metal allocator does not
    undersize the visible-point region. ``spectral=True`` sizes against the wider
    spectral structs (Spectrum beta/ld + conductorMetalId; 4-channel SppmAccum)."""
    vp_stride = visible_point_size(msl=msl, spectral=spectral)
    acc_stride = sppm_accum_size(msl=msl, spectral=spectral)
    return {
        "visible_points": num_pixels * vp_stride,
        "sppm_accum":     num_pixels * acc_stride,
    }


# ── Wavefront BDPT subpath-vertex + aux buffers (change spectral-wavefront) ──
# Mirror of BDPTVertex (shaders/integrators/bdpt.slang) and WfBdptAux
# (shaders/wavefront/wavefront_bdpt.slang) — the per-lane eye/light vertex stacks
# and per-lane aux record the wavefront BDPT pass carries in VRAM (GPU-internal
# scratch; the host only SIZES them). In the spectral build (SKINNY_SPECTRAL) the
# color roles carry `Spectrum`=float4 (vs float3 RGB) and WfBdptAux appends
# `SampledWavelengths` (two float4). These grow the scalar (Vulkan) stride, so the
# host allocator must size against the spectral stride or the Metal/Vulkan BDPT
# kernels overrun. The Metal pass reflects the real MSL stride and does not use
# these helpers; they exist to size the Vulkan (scalar) allocation and to lock the
# spectral stride against a test. RGB is byte-identical to the pre-spectral build.


def _bdpt_vertex_fields(spectral: bool) -> list[tuple[str, str]]:
    col = "float4" if spectral else "float3"
    return [
        ("kind",       "uint"),
        ("position",   "float3"),
        ("N",          "float3"),
        ("throughput", col),
        ("emission",   col),
        ("pdfFwd",     "float"),
        ("pdfRev",     "float"),
        ("isDelta",    "bool"),
        ("onLight",    "bool"),
        ("matId",      "uint"),
        ("uv",         "float2"),
        ("posObject",  "float3"),
        ("geoN",       "float3"),
        ("tangent",    "float3"),
        ("hasTangent", "bool"),
    ]


def _wf_bdpt_aux_fields(spectral: bool) -> list[tuple[str, str]]:
    col = "float4" if spectral else "float3"
    fields: list[tuple[str, str]] = [
        ("eyeLen",        "int"),
        ("lightLen",      "int"),
        ("rngState",      "uint"),
        ("lensWeight",    "float"),
        ("pixel",         "uint"),
        ("escaped",       col),
        ("radiance",      col),
        ("ewRayO",        "float3"),
        ("ewRayD",        "float3"),
        ("ewThroughput",  col),
        ("ewPdfFwdOmega", "float"),
        ("ewMisBsdfPdf",  "float"),
        ("ewFlags",       "uint"),
    ]
    if spectral:
        # SampledWavelengths { float4 lambda; float4 pdf; }
        fields += [("sw_lambda", "float4"), ("sw_pdf", "float4")]
    return fields


def bdpt_vertex_size(*, msl: bool = False, spectral: bool = False) -> int:
    """Byte stride of BDPTVertex — scalar (default) or MSL (``msl=True``),
    RGB (default) or spectral (``spectral=True``)."""
    return _struct_stride(_bdpt_vertex_fields(spectral), msl=msl)


def wf_bdpt_aux_size(*, msl: bool = False, spectral: bool = False) -> int:
    """Byte stride of WfBdptAux — scalar (default) or MSL (``msl=True``),
    RGB (default) or spectral (``spectral=True``)."""
    return _struct_stride(_wf_bdpt_aux_fields(spectral), msl=msl)


BDPT_VERTEX_STRIDE = bdpt_vertex_size()                             # 120 B (scalar)
BDPT_VERTEX_STRIDE_SPECTRAL = bdpt_vertex_size(spectral=True)       # 128 B (scalar)
WF_BDPT_AUX_STRIDE = wf_bdpt_aux_size()                            # 92 B (scalar)
WF_BDPT_AUX_STRIDE_SPECTRAL = wf_bdpt_aux_size(spectral=True)      # 136 B (scalar)


def sppm_grid_cell_count(num_pixels: int) -> int:
    """Number of spatial-hash cells for the SPPM grid: the next power of two
    >= 2*num_pixels, so the cell index masks with ``& (numCells - 1)``."""
    target = max(1, 2 * num_pixels)
    cells = 1
    while cells < target:
        cells <<= 1
    return cells


def sppm_grid_buffer_sizes(num_pixels: int) -> dict[str, int]:
    """Byte sizes for the SPPM counting-sort grid buffers (all uint → identical
    scalar/MSL layout, no ``msl`` param). ``grid_combined`` holds four uint
    sub-ranges — ``gridCount | gridOffset | gridCursor`` (each ``numCells``) and
    ``sortedIdx`` (``num_pixels``); ``scan_scratch`` holds the per-block sums for
    the two-level exclusive prefix sum (block size 256)."""
    num_cells = sppm_grid_cell_count(num_pixels)
    grid_uints = 3 * num_cells + num_pixels
    scan_uints = (num_cells + 255) // 256
    return {
        "grid_combined": grid_uints * _UINT,
        "scan_scratch":  scan_uints * _UINT,
    }


# ── MLT chain buffers (change mlt-integrator) ───────────────────────────────
#
# Kelemen full-sample PSSMLT chain state (shaders/wavefront/wavefront_mlt.slang
# + the SKINNY_MLT RNG override in common.slang). All fields are 4-byte
# scalars, so the Vulkan-scalar and Metal MSL strides are identical by
# construction — asserted below rather than assumed.

MLT_MAX_DIMS = 192          # must match common.slang MLT_MAX_DIMS
MLT_RECORD_SLOTS = 8        # eye record + up to BDPT_MAX_VERTS-1 light splats

_MLT_PRIMARY_SAMPLE_FIELDS: list[tuple[str, str]] = [
    ("value",       "float"),
    ("valueBackup", "float"),
    ("lastMod",     "uint"),
    ("modBackup",   "uint"),
]

_MLT_CHAIN_META_FIELDS: list[tuple[str, str]] = [
    ("rngState",               "uint"),
    ("currentIteration",       "uint"),
    ("lastLargeStepIteration", "uint"),
    ("seedIndex",              "uint"),
    ("cCurrent",               "float"),
    ("nRecords",               "uint"),
    ("pad0",                   "uint"),
    ("pad1",                   "uint"),
]

_MLT_RECORD_FIELDS: list[tuple[str, str]] = [
    ("pixel", "uint"),
    ("r",     "float"),
    ("g",     "float"),
    ("b",     "float"),
]


def mlt_primary_sample_size(*, msl: bool = False) -> int:
    """Byte stride of MltPrimarySample (16 B on both layouts)."""
    return _struct_stride(_MLT_PRIMARY_SAMPLE_FIELDS, msl=msl)


def mlt_chain_meta_size(*, msl: bool = False) -> int:
    """Byte stride of MltChainMeta (32 B on both layouts)."""
    return _struct_stride(_MLT_CHAIN_META_FIELDS, msl=msl)


def mlt_record_size(*, msl: bool = False) -> int:
    """Byte stride of MltRecord (16 B on both layouts)."""
    return _struct_stride(_MLT_RECORD_FIELDS, msl=msl)


def mlt_buffer_sizes(num_chains: int, bootstrap_samples: int, *,
                     msl: bool = False) -> dict[str, int]:
    """Byte sizes for the MLT chain buffers — the source of truth the MLT
    stage allocator sizes against.

    CRITICAL: sized by ``num_chains`` (chain state PERSISTS across accumulation
    frames), never by the wavefront ``stream_size``. The primary-sample buffer
    is also the bootstrap scratch: bootstrap dispatches are breadth-tiled to at
    most ``num_chains`` in-flight slots so each slot owns a distinct X slice.
    """
    return {
        "mlt_primary_samples":   num_chains * MLT_MAX_DIMS * mlt_primary_sample_size(msl=msl),
        "mlt_chain_meta":        num_chains * mlt_chain_meta_size(msl=msl),
        "mlt_current_records":   num_chains * MLT_RECORD_SLOTS * mlt_record_size(msl=msl),
        "mlt_bootstrap_weights": max(1, bootstrap_samples) * 4,
        "mlt_chain_seeds":       num_chains * 4,
        # Proposal-record scratch (binding 57): mltEvaluate writes captured
        # records here instead of a thread-local array — the array (on top of
        # the spectral estimator's live state) overflowed Metal's per-thread
        # budget and hung wfMltMutate (change spectral-mlt). Slot-indexed by
        # chain (bootstrap: in-flight slot; both < num_chains by tiling).
        "mlt_proposal_records":  num_chains * MLT_RECORD_SLOTS * mlt_record_size(msl=msl),
    }
