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
}

# MSL (Metal target) byte sizes + alignments. Slang pads ``float3`` to 16 B
# (size *and* alignment) on Metal; every other field keeps its natural
# size==alignment. A struct's stride rounds up to its largest member alignment.
SLANG_MSL_SIZES: dict[str, int] = {
    "float": 4, "float2": 8, "float3": 16, "float4": 16, "uint": 4, "int": 4,
}
SLANG_MSL_ALIGNS: dict[str, int] = {
    "float": 4, "float2": 8, "float3": 16, "float4": 16, "uint": 4, "int": 4,
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
FIELDS: list[tuple[str, str]] = [
    ("rayOrigin",  "float3"),
    ("rayDir",     "float3"),
    ("throughput", "float3"),
    ("radiance",   "float3"),
    ("pixelIndex", "uint"),
    ("rngState",   "uint"),
    ("depth",      "uint"),
    ("flags",      "uint"),
    ("bsdfPdf",    "float"),
]


def path_state_size(*, msl: bool = False) -> int:
    """Byte stride of WavefrontPathState — scalar (default) or MSL (``msl=True``)."""
    return _struct_stride(FIELDS, msl=msl)


PATH_STATE_STRIDE = path_state_size()              # 68 B (scalar / Vulkan)
PATH_STATE_STRIDE_MSL = path_state_size(msl=True)  # 96 B (Metal)

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
                       *, msl: bool = False) -> dict[str, int]:
    """Byte sizes for the wavefront stage buffers, the single source of truth
    the `WavefrontPasses` allocator (vk_wavefront.py) sizes against.

    Stream-sized buffers scale with the lane count; per-material buffers scale
    with the active material count. The hit buffer is intentionally absent —
    its stride follows `HitData` (common.slang) and is pinned alongside the
    intersect stage that produces it.

    ``msl=True`` sizes the struct-backed buffers (``path_state``) against the
    Metal MSL stride so the Metal wavefront allocator does not undersize them;
    plain (uint-stride) queue buffers are layout-agnostic. Vulkan callers omit
    it and keep the scalar sizing byte-for-byte.
    """
    path_state_stride = PATH_STATE_STRIDE_MSL if msl else PATH_STATE_STRIDE
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

VISIBLE_POINT_FIELDS: list[tuple[str, str]] = [
    ("pos",           "float3"),
    ("ns",            "float3"),
    ("wo",            "float3"),
    ("beta",          "float3"),
    ("ld",            "float3"),
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
    ("tau",           "float3"),
    ("flags",         "uint"),
    ("radius",        "float"),
    ("n",             "float"),
]

SPPM_ACCUM_FIELDS: list[tuple[str, str]] = [
    ("phiR", "uint"),
    ("phiG", "uint"),
    ("phiB", "uint"),
    ("m",    "uint"),
]


def visible_point_size(*, msl: bool = False) -> int:
    """Byte stride of VisiblePoint — scalar (default) or MSL (``msl=True``)."""
    return _struct_stride(VISIBLE_POINT_FIELDS, msl=msl)


def sppm_accum_size(*, msl: bool = False) -> int:
    """Byte stride of SppmAccum — scalar (default) or MSL (``msl=True``). All
    fields are uint, so the scalar and MSL strides are identical."""
    return _struct_stride(SPPM_ACCUM_FIELDS, msl=msl)


VISIBLE_POINT_STRIDE = visible_point_size()              # 180 B (scalar / Vulkan)
VISIBLE_POINT_STRIDE_MSL = visible_point_size(msl=True)  # 240 B (Metal)
SPPM_ACCUM_STRIDE = sppm_accum_size()                    # 16 B (both layouts)
SPPM_ACCUM_STRIDE_MSL = sppm_accum_size(msl=True)        # 16 B


def sppm_buffer_sizes(num_pixels: int, *, msl: bool = False) -> dict[str, int]:
    """Byte sizes for the SPPM per-pixel buffers, the source of truth the SPPM
    stage allocator sizes against.

    CRITICAL: the SPPM visible-point estimator (radius/N/tau) is **per pixel**
    and PERSISTS across passes, so these buffers MUST be sized by
    ``num_pixels = width*height`` — NOT by the wavefront ``stream_size`` (which
    aliases different pixels across tiles, and would undersize the buffer for any
    frame that tiles, corrupting state after tile 0). The SPPM stages index by
    the per-pixel lane, in ``[0, num_pixels)``.

    ``msl=True`` uses the Metal struct stride so the Metal allocator does not
    undersize the visible-point region."""
    vp_stride = VISIBLE_POINT_STRIDE_MSL if msl else VISIBLE_POINT_STRIDE
    acc_stride = SPPM_ACCUM_STRIDE_MSL if msl else SPPM_ACCUM_STRIDE
    return {
        "visible_points": num_pixels * vp_stride,
        "sppm_accum":     num_pixels * acc_stride,
    }


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
