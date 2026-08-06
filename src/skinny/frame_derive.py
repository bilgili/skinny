"""Pure frame-constant derivation (change renderer-module-carveout, Stage B).

Every derived value the frame-constant packing path computes — the detail-flag
bitfield, the lens FOV-framing sensor half-height, the exposure/imaging-ratio
fold, and the proposal-mask/reuse capability folding — lives here as a
device-free function taking scalars/arrays and returning scalars/tuples.
`renderer._pack_uniforms` calls each at its existing append site and consumes
the result as a plain value; the append order, offsets, and MSL relocation
(the `reflection-owned-byte-layouts` scope) are untouched by construction.

Byte serialization is NOT here — these functions return Python numbers, the
packer `struct.pack`s them exactly as before.
"""

from __future__ import annotations

import math

from skinny import choice_tables

# Execution-mode index for wavefront, projected from the choice_tables owner
# (dependency-free, so this module still imports no GPU-touching code).
EXECUTION_WAVEFRONT = choice_tables.index_by_token(choice_tables.EXECUTION_MODE)["wavefront"]


def detail_flags(master_on: bool, nrm_ok: bool, rgh_ok: bool, dsp_ok: bool,
                 baked_normals: bool) -> int:
    """The `detailFlags` bitfield. Bit 0 = master enable (mirror of the UI
    toggle); bits 1-3 = normal/roughness/displacement map available; bit 4 =
    normal map already baked into vertex normals. The master bit stands alone;
    each per-map bit is set only when that map is present, so a missing map
    reads as off even when the master toggle is on (the shader AND-s master
    with the per-map bit)."""
    return (
        (1 if master_on else 0)
        | ((1 if nrm_ok else 0) << 1)
        | ((1 if rgh_ok else 0) << 2)
        | ((1 if dsp_ok else 0) << 3)
        | ((1 if baked_normals else 0) << 4)
    )


def film_half_height_world(va_mm: float, focal_mm: float, mm_per_unit: float,
                           lens_active_count: int,
                           lens_film_distance_world: float) -> float:
    """Sensor half-height in world units. Base is `0.5·verticalAperture /
    mm_per_unit`. When a lens stack is active, scale by `filmDistance / F`
    (F = focal length) so a unit NDC through the realistic lens projects to the
    same world angle as the idealised pinhole — a thick lens images onto a
    plane at the back focal length BFL ≠ F, which would otherwise widen/narrow
    the frame on lens enable."""
    mm_per_unit = max(float(mm_per_unit), 1e-6)
    half = 0.5 * float(va_mm) / mm_per_unit
    if lens_active_count > 0 and float(focal_mm) > 1e-3:
        half *= float(lens_film_distance_world) / (float(focal_mm) / mm_per_unit)
    return half


def exposure_stops(exposure_ev: float, imaging_ratio: float) -> float:
    """Display exposure in EV stops with the pbrt film imaging ratio
    (exposure_time·iso/100) folded in as log2(ratio) stops — a linear output
    gain reproduced with no shader/UBO change. ratio ≤ 0 ⇒ +0 stops."""
    fold = math.log2(imaging_ratio) if imaging_ratio > 0.0 else 0.0
    return float(exposure_ev) + fold


def fold_sampling_capabilities(mask: int, alpha, reuse_mode: int,
                               execution_mode_index: int):
    """Fold the scene-sampling proposal mixture + reuse mode against the
    backend's capabilities. Returns `(mask, alpha, reuse_mode, neural_stripped)`
    where `neural_stripped` is True when the neural bit was requested on the
    megakernel and dropped (the caller warns once).

    - ReSTIR DI reuse is wavefront-only (multi-pass); on the megakernel the
      reuse mode folds to 0 (identity) so the shader's depth-0 reuse gate stays
      inert — stock NEE.
    - The neural proposal (bit 2) is a wavefront compute pre-pass, infeasible
      inline in the megakernel; on the megakernel strip the bit and renormalise
      the mixture over the analytic remainder. Neural-only → fall back to the
      `{bsdf}` fast path (mask 0x1, alpha (1,0,0,0)) rather than an empty
      mixture.
    """
    is_wavefront = execution_mode_index == EXECUTION_WAVEFRONT
    alpha = tuple(float(a) for a in alpha)
    if not is_wavefront:
        reuse_mode = 0
    neural_stripped = False
    if (mask & 0x4) and not is_wavefront:
        neural_stripped = True
        mask &= ~0x4
        a = list(alpha)
        a[2] = 0.0
        s = sum(a) or 1.0
        alpha = (a[0] / s, a[1] / s, a[2] / s, a[3] / s)
        if mask == 0:
            mask = 0x1
            alpha = (1.0, 0.0, 0.0, 0.0)
    return mask, alpha, int(reuse_mode), neural_stripped
