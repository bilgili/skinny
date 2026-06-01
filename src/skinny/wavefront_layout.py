"""GPU-free layout descriptor for the wavefront path-state record (P1 §P1-A).

Mirrors the `WavefrontPathState` struct in
``shaders/wavefront/wavefront_state.slang``. The renderer allocates the
path-state StructuredBuffer as ``stream_size * PATH_STATE_STRIDE`` bytes; the
Slang struct and this table must stay in lockstep or the wavefront stages read
garbage (``tests/test_wavefront_state.py`` locks them together).

Scalar layout (``-fvk-use-scalar-layout``): ``float3`` = 12 B with 4-byte
alignment, fields tightly packed.
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


def path_state_size() -> int:
    """Scalar-layout byte size of WavefrontPathState (tight, 4-byte aligned)."""
    return sum(SLANG_SCALAR_SIZES[t] for _, t in FIELDS)


PATH_STATE_STRIDE = path_state_size()  # 68 B

# `flags` bit positions (mirror the static consts in wavefront_state.slang).
PATH_FLAG_ALIVE = 1 << 0     # lane still bouncing
PATH_FLAG_SPECULAR = 1 << 1  # last bounce was specular → skip NEE-MIS


# ── Queue / dispatch buffer sizing (P1 §P1-B, §P1-C) ───────────────

_UINT = 4
# One VkDispatchIndirectCommand per material: (x, y, z) group counts.
INDIRECT_ARGS_STRIDE = 3 * _UINT  # 12 B


def queue_buffer_sizes(stream_size: int, num_materials: int) -> dict[str, int]:
    """Byte sizes for the wavefront stage buffers, the single source of truth
    the `WavefrontPasses` allocator (vk_wavefront.py) sizes against.

    Stream-sized buffers scale with the lane count; per-material buffers scale
    with the active material count. The hit buffer is intentionally absent —
    its stride follows `HitData` (common.slang) and is pinned alongside the
    intersect stage that produces it.
    """
    return {
        "path_state":      stream_size * PATH_STATE_STRIDE,
        "ray_queue":       stream_size * _UINT,
        "material_queue":  stream_size * _UINT,
        "ray_count":       _UINT,
        "material_count":  num_materials * _UINT,
        "material_offset": num_materials * _UINT,
        "indirect_args":   num_materials * INDIRECT_ARGS_STRIDE,
    }
