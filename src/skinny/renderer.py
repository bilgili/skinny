"""Core renderer — orchestrates Vulkan compute dispatch for skin ray tracing."""

from __future__ import annotations

import abc
import math
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import vulkan as vk

from PIL import Image, ImageDraw, ImageFont

from skinny.environment import Environment, load_environments
from skinny.pbrt.data import CONDUCTOR_METAL_ID
from skinny.scene import CameraOverride, LensSystem, Scene, build_default_scene
from skinny.head_textures import (
    DETAIL_TEX_RES,
    blank_displacement_bytes,
    blank_normal_bytes,
    blank_roughness_bytes,
    load_texture_bytes,
)
from skinny.mesh import (
    Mesh,
    MeshSource,
    _load_model_dir,
    bake_mesh,
    dummy_mesh,
    load_obj_source,
)
from skinny.mesh_cache import (
    load_cache_index,
    lookup_cached_mesh,
    make_cache_key,
    save_cached_mesh,
)
from skinny.params import (
    EXECUTION_MEGAKERNEL,
    EXECUTION_WAVEFRONT,
    clamp_mode_index,
    effective_execution_mode,
)
from skinny.cli_common import resolve_walk
from skinny.playback import PlaybackClock
from skinny.presets import PRESETS, Preset
from skinny.settings import load_user_presets
from skinny.tattoos import TATTOO_HEIGHT, TATTOO_WIDTH, Tattoo, blank_tattoo_data, load_tattoos
from skinny.vk_context import VulkanContext
# GPU-resource classes (StorageBuffer, ComputePipeline, …) are resolved per
# backend at runtime via `backend_select.resource_module(ctx)` → `self._gpu.*`
# (vk_compute on Vulkan, metal_compute on Metal) so this module never imports
# `vulkan` on a Metal-only host (task 2.3). The single remaining type annotation
# (`SampledImage`) is pulled in under TYPE_CHECKING — never executed at runtime,
# so it forces no import on any backend.
if TYPE_CHECKING:
    from skinny.vk_compute import SampledImage

WORKGROUP_SIZE = 8
MAX_FRAMES_IN_FLIGHT = 2

# SPPM glossy/near-specular eye-walk continuation: default roughness threshold
# (change sppm-glossy-final-gather). Tuned for polished metals — brass in
# assets/three_materials_demo has roughness ~0.15 (p95 0.23); the metallic guard in
# the shader keeps dielectrics on the gather side. This threshold is in perceptual
# (USD) roughness, so it also has to catch pbrt-imported metals: pbrt's perceptual
# roughness r maps through alpha=sqrt(r) then usd=sqrt(alpha)=r**0.25, so a polished
# pbrt-roughness-0.1 conductor lands at usd ~0.562 (GGX alpha ~0.316). 0.6 covers it
# — an alpha<=~0.36 "polished metal" cutoff — while still leaving a pbrt-roughness-0.3
# metal (usd ~0.740) on the photon-gather side. A glossy-continued vertex escaping to
# a distant/env light is MIS-weighted in wfSppmEye, so continuation matches the path
# tracer instead of double-counting env (conductor_infinite). Override per-render via
# renderer._sppm_glossy_roughness_override (None = use this default; 0.0 = PM-1 delta-only).
_SPPM_GLOSSY_ROUGHNESS_DEFAULT = 0.6
# Default per-dispatch photon breadth for the Metal SPPM phase-3 tiling
# (change sppm-photon-dispatch-tiling). The photon command buffer's work is
# `batch × visible-points-gathered-per-photon`; sized to the megakernel
# watchdog band budget (~2·10⁵ work units) so even a caustic focus cell stays
# watchdog-safe while photons/pass remains the full width*height. Env override
# SKINNY_SPPM_METAL_PHOTON_BATCH; 0 disables tiling (single full dispatch).
_SPPM_METAL_PHOTON_BATCH_DEFAULT = 65536

# Per-dispatch chain breadth for the Metal MLT mutation/bootstrap phases (change
# mlt-integrator, design D7). One MLT chain runs a COMPLETE BDPT sample per
# dispatch (heavier than one SPPM photon deposit), so a very large `--chains`
# could push a single command buffer past the macOS GPU watchdog. The recorder
# tiles each phase into flushed sub-batches of this many chains. At the default
# `nChains` (16384) this is exactly one batch — the GPU-validated path is
# unchanged — and only a larger chain count subdivides. Env override
# SKINNY_MLT_METAL_CHAIN_BATCH; 0 disables tiling (single full dispatch).
_MLT_METAL_CHAIN_BATCH_DEFAULT = 16384


def _spectral_analytic_proposal_token(
    token: str,
    *,
    allow_environment: bool,
) -> str:
    """Resolve a proposal preset to the analytic spectral subset.

    BSDF and environment importance sampling are wavelength-independent and
    supported by the spectral path. Stateful neural inference is not; remove it
    and fall back to BSDF if no analytic proposal remains.
    """
    supported = {"bsdf", "env"} if allow_environment else {"bsdf"}
    analytic = [
        part.strip()
        for part in str(token).split(",")
        if part.strip() in supported
    ]
    return ",".join(analytic) or "bsdf"


def _sppm_photon_group_pmf(
    powers: tuple[float, float, float, float],
    present: tuple[bool, bool, bool, bool],
) -> tuple[float, float, float, float]:
    """Photon-emission group selection pmf (emissive, sphere, distant, env),
    proportional to each group's emitted power (change
    sppm-power-proportional-photon-groups). Per-photon flux then equalises
    across groups (Φ_g / p_g ≈ Φ_total) — the pbrt light-power distribution —
    which kills the sparse huge env splats that uniform 1/G selection produced.

    Absent groups get 0. Non-finite or negative powers are treated as 0. When
    the total usable power is 0 (or every power was non-finite) the pmf falls
    back to uniform over the *present* groups — the pre-change behavior.
    """
    clean = [
        p if (present[i] and math.isfinite(p) and p > 0.0) else 0.0
        for i, p in enumerate(powers)
    ]
    total = sum(clean)
    if total > 0.0 and math.isfinite(total):
        return tuple(p / total for p in clean)
    n_present = sum(1 for b in present if b)
    if n_present == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple((1.0 / n_present) if b else 0.0 for b in present)


# Ceiling on the env-aware photon-budget multiplier (change
# sppm-env-photon-budget): pmfEnv → 1 would send the budget to infinity, and ×8
# already cuts the env noise component by √8 ≈ 2.8 (measured exact on
# glass_caustics_test.usda) at negligible photon-stage cost.
_SPPM_ENV_PHOTON_BUDGET_CAP = 8.0


def _sppm_photon_budget(pixels: int, pmf_env: float,
                        cap: float = _SPPM_ENV_PHOTON_BUDGET_CAP) -> int:
    """Env-aware per-pass photon count (change sppm-env-photon-budget).

    ``pixels / (1 - pmfEnv)`` keeps the EXPECTED non-env photon count at
    exactly ``pixels`` (the flat pre-env budget) and rides the env group's
    photons on top of it instead of diluting the local lights — env photons
    deposit only after ≥1 bounce from a disc covering the whole scene bounding
    sphere, so at one-per-pixel they are sparse fat splats (speckle). Capped at
    ``cap``× so an env-dominated pmf can't run away; ``pmfEnv == 0`` returns
    ``pixels`` exactly (env-free scenes stay bit-identical). ``pmf_env`` is
    clamped to [0, 1] and treated as 0 when non-finite — the pmf override hook
    is unvalidated.
    """
    if not math.isfinite(pmf_env):
        pmf_env = 0.0
    pmf_env = min(max(pmf_env, 0.0), 1.0)
    return int(round(int(pixels) / max(1.0 - pmf_env, 1.0 / cap)))

# Ordered (field-name, scalar-byte-size) of the `FrameConstants fc` uniform block,
# matching `_pack_uniforms`'s append order exactly. Used only by the Metal MSL
# packer (`_pack_uniforms_msl`, design D3) to relocate each field from the Vulkan
# scalar blob into the MSL struct at its reflected offset — `float3` fields keep
# 12 scalar bytes but land in a 16-byte MSL slot. Names match the compiled
# module's reflected field names (the embedded `Camera` is `camera.<field>`).
# A drift guard asserts the cumulative size equals `len(_pack_uniforms())`.
_FC_SCALAR_FIELDS: tuple[tuple[str, int], ...] = (
    ("camera.viewInverse", 64), ("camera.projInverse", 64),
    ("camera.view", 64), ("camera.proj", 64),
    ("camera.position", 12), ("camera.fov", 4),
    ("frameIndex", 4), ("accumFrame", 4), ("time", 4), ("width", 4), ("height", 4),
    ("numDistantLights", 4), ("useMesh", 4), ("tattooDensity", 4), ("envIntensity", 4),
    ("furnaceMode", 4), ("mmPerUnit", 4), ("detailFlags", 4), ("normalMapStrength", 4),
    ("displacementScaleMM", 4), ("numInstances", 4), ("numSphereLights", 4),
    ("numEmissiveTriangles", 4), ("integratorType", 4), ("numGizmoSegments", 4),
    ("numLensElements", 4), ("filmDistance", 4), ("rearZ", 4), ("rearAperture", 4),
    ("frontZ", 4), ("filmHalfH", 4), ("emissiveTotalPower", 4), ("numPupilBounds", 4),
    ("filmDiagRadiusW", 4), ("focusOverlay", 4), ("focusPlaneOrigin", 12),
    ("focusPlaneNormal", 12), ("zoomMin", 8), ("zoomMax", 8), ("lensVignetteDebug", 4),
    ("pickPixel", 8), ("pickArmed", 4), ("exposure", 4), ("tonemapMode", 4),
    ("proposalMask", 4), ("reuseMode", 4), ("proposalAlpha", 16), ("flatLobeSamplers", 4),
    ("sceneBoundsMin", 12), ("sceneBoundsExtent", 12), ("neuralNetworkVersion", 4),
    ("recordMode", 4), ("cameraMirror", 4),
    # SPPM per-pass photon-mapping tail (change photon-mapping-sppm).
    ("sppmInitialRadius", 4), ("sppmCellSize", 4), ("sppmGridRes", 12),
    ("sppmPhotonsEmitted", 4),
    # Glossy / near-specular eye-walk continuation threshold (change sppm-glossy-final-gather).
    ("sppmGlossyContinueRoughness", 4),
    # Film per-sample radiance clamp (pbrt `maxcomponentvalue`, change
    # film-maxcomponent-clamp). 0 = disabled.
    ("filmMaxComponent", 4),
    # SPPM photon-emission group selection pmf (change
    # sppm-power-proportional-photon-groups): P(emissive/sphere/distant/env),
    # power-proportional, normalized host-side. Zeros when integrator != SPPM.
    ("sppmGroupPmfE", 4), ("sppmGroupPmfS", 4), ("sppmGroupPmfD", 4),
    ("sppmGroupPmfEnv", 4),
    # Metal megakernel watchdog tiling (change metal-megakernel-watchdog-tiling):
    # row-band Y origin for this dispatch. 0 from the base packer (Vulkan always
    # dispatches the full frame); the Metal megakernel path patches it per band.
    ("tileOriginY", 4),
)

# The `#if defined(SKINNY_MLT)` FrameConstants tail (change mlt-integrator).
# `_pack_uniforms` appends these ONLY when the MLT wavefront pass is the
# consumer — every other pipeline's struct ends at `tileOriginY` above, so the
# RGB SPIR-V and the non-MLT MSL layouts are byte-unchanged. Field order
# matches common.slang's gated block exactly; the scalar blob carries the tail
# BEFORE the trailing `tileOriginY` word (see `_pack_uniforms`), which is why
# this is a separate table rather than an extension of the one above.
_FC_MLT_FIELDS: tuple[tuple[str, int], ...] = (
    ("mltSigma", 4), ("mltLargeStepProb", 4), ("mltB", 4), ("mltMppActual", 4),
    ("mltNumChains", 4), ("mltChainBase", 4), ("mltMaxDepth", 4), ("mltSeed", 4),
)

# Scalar-blob field order for an MLT pack: the MLT tail sits where the Vulkan
# filler word would be, and `tileOriginY` follows it. `_pack_uniforms_msl`
# relocates by NAME from reflection, so this order need not match the MSL
# struct's — only the scalar blob it walks.
_FC_SCALAR_FIELDS_MLT: tuple[tuple[str, int], ...] = (
    _FC_SCALAR_FIELDS[:-1] + _FC_MLT_FIELDS + _FC_SCALAR_FIELDS[-1:]
)

# Byte offset of the `tileOriginY` u32 at the tail of the `_pack_uniforms` scalar
# blob, so the Metal band loop can patch it in place without a full re-pack.
# MLT never uses this (the megakernel band loop is the only patcher, and MLT is
# wavefront-only) — under an MLT pack the word moves by the tail's 32 B.
_TILE_ORIGIN_Y_OFFSET = sum(sz for _, sz in _FC_SCALAR_FIELDS) - 4

# Target pixels per Metal megakernel command buffer, per integrator, before the
# frame is split into more row bands (change metal-megakernel-watchdog-tiling).
# BDPT does the widest per-pixel work (eye × light subpaths + full s×t connection
# matrix, each connection a BSDF eval at both ends), so it needs a far smaller
# budget than the path tracer to stay under the macOS GPU watchdog on heavy
# (graph-material) scenes. Path/SPPM are cheap enough to keep the single
# full-frame dispatch on ordinary scenes. `_metal_megakernel_bands` reads these.
_METAL_MEGAKERNEL_BAND_PIXELS_DEFAULT = 8_000_000
_METAL_MEGAKERNEL_BAND_PIXELS = {
    0: 8_000_000,   # Path — effectively one band until very large frames
    1: 200_000,     # BDPT — the wedging case; ~1280×720 → ~5 bands
    2: 8_000_000,   # SPPM eye pass — cheap per pixel
}

# Per-tile lane cap for the wavefront BDPT/SPPM eye stage when the scene has a
# non-terminal non-flat material (VOLUME / PYTHON): the non-flat first-hit path
# fallback runs a full multi-bounce PathTracer.estimateRadiance per lane, so the
# per-tile Metal submit (WavefrontRecorder.flush_heavy_eye) must also cap the TILE
# itself — otherwise one committed command buffer could run the heavy fallback
# over a full-frame stream (up to 1<<20 SPPM lanes) and trip the macOS GPU
# watchdog. Sized to the megakernel BDPT band budget above (the heaviest
# proven-safe per-command-buffer work; a VOLUME path-fallback lane's cost is
# comparable). Change wavefront-nonflat-tiled-fallback.
_METAL_WAVEFRONT_HEAVY_EYE_BAND_LANES = 200_000

# Watchdog-safe upper bound for the Metal material-preview dispatch (codex #2).
# The preview commits ONE command buffer over `size×size` and can run per-pixel
# MaterialX graph evaluation, so `size` (a public `render_material_preview`
# parameter) must be bounded on Metal — macOS cannot cancel an over-long GPU
# kernel. 512² = 262 144 px of single-bounce thumbnail shading is far lighter
# than a full BDPT-over-graph frame (200 000 px/band) and comfortably clears the
# UI's 256 preview; larger requests clamp (the returned `size` reflects the
# clamp, and the dock reshapes against it). Vulkan is unbounded — it can cancel.
_METAL_PREVIEW_MAX_SIZE = 512

# Size of the Vulkan FrameConstants UBO. Must be ≥ len(_pack_uniforms()) — the
# `UniformBuffer.upload` path memmoves min(len(data), size), so an undersized
# buffer SILENTLY TRUNCATES the scalar blob's tail (this bit cameraMirror: a
# 512 B buffer dropped the 513–516 B field on Vulkan while Metal, sized from
# reflection, was fine). Kept 16-aligned with headroom so adding a few more
# scalar-tail fields doesn't need another bump. The import-time assert below
# ties it to the field table so it can never fall behind unnoticed.
_VK_UNIFORM_BUFFER_BYTES = 768
assert _VK_UNIFORM_BUFFER_BYTES >= sum(sz for _, sz in _FC_SCALAR_FIELDS), (
    "Vulkan UBO too small for the FrameConstants scalar blob — bump "
    "_VK_UNIFORM_BUFFER_BYTES")

# Cap on lens elements packed into the binding-23 SSBO. PBRT lens designs
# in the wild peak around 11-13 surfaces (Canon FD 200/1.8, double-Gauss
# variants); 32 leaves headroom for compound zooms without bloating the
# fixed SSBO allocation.
MAX_LENS_ELEMENTS = 32

# Per-instance storage record consumed by mesh_head.slang::Instance.
# Layout (std430-compatible): worldFromLocal (mat4x4, 64 B), localFromWorld
# (mat4x4, 64 B), four uints (blasNodeOffset, blasIndexOffset,
# blasVertexOffset, materialId; 16 B). Total 144 B, naturally 16-byte
# aligned so consecutive instances don't need padding.
INSTANCE_STRIDE = 144

# Per-material flat-shading record consumed by main_pass.slang's
# non-skin BSDF dispatch. Layout (scalar/std430-compatible):
#    0: diffuseColor (vec3, 12) + roughness (float, packs into trailing 4)
#   16: metallic + specular + opacity + diffuseTextureIdx
#   32: roughnessTextureIdx + metallicTextureIdx + normalTextureIdx + emissiveTextureIdx
#   48: emissiveColor (vec3, 12) + ior (float)
#   64: coat + coatRoughness + coatIOR + opacityTextureIdx
#   80: coatColor (vec3, 12) + opacityThreshold
#   96: normalScale (vec3, 12) + channelMask (uint, packed channel selectors)
#  112: normalBias  (vec3, 12) + _pad (4 B)
#  128: transmissionColor (vec3, 12) + diffuseRoughness (float)  [Stage-2]
#  144: specularColor (vec3, 12) + _pad1 (4 B)                   [Stage-2]
#  160: medium σ_a (vec3, 12) + medium g (float)                 [subsurface/volume]
#  176: medium σ_s (vec3, 12) + mediumKind (uint)                [subsurface/volume]
#  192: worldToUvw row 0 (vec4)                                  [volume]
#  208: worldToUvw row 1 (vec4)                                  [volume]
#  224: worldToUvw row 2 (vec4)                                  [volume]
#  240: cloud density + wispiness + frequency + pad (vec4)       [MEDIUM_CLOUD]
# 256 B / record, naturally 16-byte aligned.
FLAT_MATERIAL_STRIDE = 256
FLAT_MATERIAL_CAPACITY_INIT = 16

# Channel-selector codes packed into FlatMaterialParams.channelMask. Five
# scalar texture inputs (diffuse, roughness, metallic, opacity, emissive)
# carry a 4-bit channel index each — 20 bits total, leaving room for
# future inputs without changing the buffer layout.
_CHANNEL_CODE = {"rgb": 0, "r": 1, "g": 2, "b": 3, "a": 4}
_CHANNEL_SHIFT = {
    "diffuseColor":  0,
    "roughness":     4,
    "metallic":      8,
    "opacity":      12,
    "emissiveColor": 16,
}


def _instance_local_basis(transform: np.ndarray) -> np.ndarray:
    """World-space directions of an instance's local X/Y/Z axes — the
    normalized rows of the row-vector-convention transform's upper 3x3.
    Used by the local-space transform gizmo. Falls back to the matching world
    axis for any degenerate (zero-length) row."""
    m = np.asarray(transform, dtype=np.float64)
    basis = np.eye(3, dtype=np.float64)
    for i in range(3):
        row = m[i, :3]
        n = float(np.linalg.norm(row))
        if n > 1e-9:
            basis[i] = row / n
    return basis


def _encode_channel_mask(channels: dict[str, str]) -> int:
    """Pack per-input channel selectors into the FlatMaterialParams uint.

    Each entry maps an UsdPreviewSurface input name to a channel string
    ("rgb"/"r"/"g"/"b"/"a"). Unknown channels fall back to "rgb" (0),
    which makes the shader read whatever the input's natural fetch already
    expected — i.e. zero is the "do nothing different" code.
    """
    mask = 0
    for input_name, ch in channels.items():
        shift = _CHANNEL_SHIFT.get(input_name)
        if shift is None:
            continue
        code = _CHANNEL_CODE.get(ch, 0)
        mask |= (code & 0xF) << shift
    return mask & 0xFFFFFFFF


def _hashable_value(v: object) -> object:
    """Coerce mtlx_overrides values into something hash()-friendly."""
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _light_value_to_vec3(value: object) -> np.ndarray:
    """Convert a color/vec3 value (tuple, list, Gf.Vec3f) to float32 array."""
    if hasattr(value, "asTuple"):
        value = value.asTuple()
    if isinstance(value, (list, tuple)):
        return np.array([float(value[0]), float(value[1]), float(value[2])], np.float32)
    return np.array([float(value)] * 3, np.float32)

# Material type codes consumed by main_pass.slang's dispatcher.
MATERIAL_TYPE_SKIN = 0  # any mtlx_target_name pointing at the layered-skin
                         # material — routes to the inline skin BSSRDF/specular
                         # path. Only active when explicitly authored.
MATERIAL_TYPE_FLAT = 1  # UsdPreviewSurface-style standard surface — routes
                         # to evalFlatMaterial's bounded path tracer.
MATERIAL_TYPE_PYTHON = 3  # Python-authored slangpile material (one of
                          # `python_materials/*.py`) — routes through the
                          # generated dispatcher in
                          # `shaders/python_materials_dispatcher.slang`.
                          # Python material index packed into upper byte of
                          # `materialTypes[matId]` (MATERIAL_PYMAT_SHIFT).
MATERIAL_TYPE_SUBSURFACE = 4  # pbrt `subsurface`: a smooth dielectric boundary +
                          # a homogeneous interior medium (σ_a, σ_s, HG g),
                          # transported by the interior random walk
                          # (`materials/subsurface/subsurface_walk.slang`). The
                          # medium coefficients are packed inline into
                          # FlatMaterialParams (binding 13) — no new buffer
                          # (Metal 31-buffer cap) — and read via `resolveMedium`.
                          # Detected from non-zero `subsurface_sigma_*` overrides.
MATERIAL_TYPE_VOLUME = 5  # Free-standing participating medium bounded by a pbrt
                          # `Material "interface"` shape (nanovdb-volume-rendering):
                          # index-matched pass-through boundary + the medium walk
                          # (`materials/subsurface/volume_walk.slang`). Detected
                          # from the importer's explicit `volume_interface` marker
                          # (`_material_is_volume`); σ/g/worldToUvw are packed
                          # inline into FlatMaterialParams (160..240), mediumKind
                          # = MEDIUM_NANOVDB (1) when a density grid is present
                          # (else MEDIUM_HOMOGENEOUS for a homogeneous interior).

# Medium source kinds (bindings.slang MEDIUM_*): the density-seam dispatch tag
# packed into FlatMaterialParams.mediumKind.
MEDIUM_HOMOGENEOUS = 0
MEDIUM_NANOVDB = 1
MEDIUM_CLOUD = 2  # pbrt procedural cloud: analytic fBm density, no texture

# Sphere-light record (binding 17): vec3 position, float radius, vec3
# radiance, float pad. 32 B / record, naturally 16-byte aligned.
SPHERE_LIGHT_STRIDE = 32
SPHERE_LIGHT_CAPACITY = 16

# Distant-light record (binding 25): vec3 direction, float pad, vec3
# radiance, float pad. 32 B / record. Matches the DistantLight struct in
# shaders/common.slang. The buffer holds every UsdLux.DistantLight in the
# scene so the integrators can iterate them all via DirectionalLightImpl
# (ILight) rather than the legacy single-uniform path.
DISTANT_LIGHT_STRIDE = 32
DISTANT_LIGHT_CAPACITY = 16

# Authored illuminant SPD per distant light (binding 50, spectral variant only):
# 95 floats (360-830/5 nm radiance), one slot per distant light, indexed by the
# SPD-slot stored in the DistantLight record's `_direction.w`. A light with no
# authored SPD stores index -1 and the shader upsamples its RGB radiance instead.
SPECTRAL_LIGHT_SPD_SAMPLES = 95
SPECTRAL_LIGHT_SPD_STRIDE = SPECTRAL_LIGHT_SPD_SAMPLES * 4  # bytes / light

# Emissive-triangle record (binding 18): vec3 v0 + pad, vec3 v1 + pad,
# vec3 v2 + pad, vec3 emission + float area. 64 B / record.
EMISSIVE_TRI_STRIDE = 64
EMISSIVE_TRI_CAPACITY = 256

# Spectral emitter record (binding 49, spectral variant only): float2
# (temperature_K, scale) per emissive triangle, parallel-indexed to binding 18.
# 8 B / record. A blackbody emitter carries (T, blackbody_scale(T, emission));
# a plain-RGB emitter carries (0, 0) so the shader falls back to the RGB upsample.
SPECTRAL_EMITTER_STRIDE = 8

# StdSurfaceParams record (binding 19): full MaterialX standard_surface
# parameters packed in scalar layout matching the Slang struct in
# mtlx_std_surface.slang.  256 B / record.
STD_SURFACE_STRIDE = 256
STD_SURFACE_CAPACITY = FLAT_MATERIAL_CAPACITY_INIT

# Tool-buffer (binding 30) dispatch modes. Slot 0.x of the tool buffer selects
# how main_pass.mainImage / runFrame behave for the frame; these mirror the
# constants in shaders/bindings.slang. TOOL_MODE_STRUCTURAL writes one float4
# per pixel — (hit-mask, instanceId, materialId, depth) — into the tool buffer
# starting at slot TOOL_STRUCT_AOV_BASE, used by the Metal↔Vulkan structural-
# parity test (6.1). The structural region overlaps the BXDF grid output, but
# the modes are mutually exclusive (only one is armed at a time).
TOOL_MODE_STRUCTURAL = 4
TOOL_STRUCT_AOV_BASE = 16  # float4 slots; past the 16-slot header/pick region

# Default diffuse for materials whose UsdPreviewSurface diffuseColor is
# texture-connected rather than constant — mid-grey keeps unbound prims
# visible until bindless textures (Phase C-4) actually sample the file.
_FLAT_DEFAULT_DIFFUSE = (0.72, 0.72, 0.72)


def _override_float(overrides: dict, key: str, default: float) -> float:
    val = overrides.get(key)
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _override_color3(overrides: dict, key: str, default: tuple) -> tuple:
    val = overrides.get(key)
    if val is None:
        return tuple(float(c) for c in default)
    # USD Gf.Vec3f exposes index access; numpy / tuple do too.
    try:
        return float(val[0]), float(val[1]), float(val[2])
    except (TypeError, IndexError, ValueError):
        return tuple(float(c) for c in default)


def _material_is_subsurface(material) -> bool:
    """True when a material carries a non-zero subsurface interior medium
    (`subsurface_sigma_a` / `subsurface_sigma_s`, mm⁻¹).

    Such materials route to MATERIAL_TYPE_SUBSURFACE so the GPU runs the
    volumetric interior random walk (`subsurface_walk.slang`) instead of the flat
    opacity=0 delta-refraction (clear-glass) fallback. A free-standing fog
    `MediumInterface` carries `volume_*` keys (not `subsurface_*`), so it is left
    on the flat/dielectric path — only a pbrt `Material "subsurface"` matches.
    """
    overrides = getattr(material, "parameter_overrides", None) or {}
    sa = _override_color3(overrides, "subsurface_sigma_a", (0.0, 0.0, 0.0))
    ss = _override_color3(overrides, "subsurface_sigma_s", (0.0, 0.0, 0.0))
    return any(c > 0.0 for c in sa) or any(c > 0.0 for c in ss)


def _material_is_volume(material) -> bool:
    """True for a free-standing medium boundary (pbrt ``Material "interface"``).

    Keys off the importer's explicit ``volume_interface: True`` marker
    (`pbrt/api.py` sets it only for interface-typed materials carrying a
    `MediumInterface`), never lobe-value sniffing — so genuine cutout/glass
    materials can't be captured. Such materials route to MATERIAL_TYPE_VOLUME:
    the index-matched pass-through medium walk (`volume_walk.slang`).
    """
    overrides = getattr(material, "parameter_overrides", None) or {}
    return bool(overrides.get("volume_interface"))


class TexturePool:
    """Bindless flat-material texture pool (binding 14 in main_pass.slang).

    Owns up to BINDLESS_TEXTURE_CAPACITY SampledImage slots. Materials
    point at slots by index; unused slots stay None and are gated off by
    PARTIALLY_BOUND on the descriptor binding plus a sentinel index in
    the material record.

    Deduplication is by file path: two materials referencing the same
    PNG share one slot. Allocation is monotonic; we don't free slots
    mid-session because materials don't change after scene load.
    """

    SENTINEL = 0xFFFFFFFF

    def __init__(self, ctx, gpu) -> None:
        self.ctx = ctx
        # GPU-resource module (vk_compute / metal_compute) — the pool's bindless
        # capacity follows the active backend's cap (Metal trims to fit its
        # 128-texture / 16-sampler argument limit, design D8).
        self._gpu = gpu
        self._capacity = int(gpu.BINDLESS_TEXTURE_CAPACITY)
        self._slots = [None] * self._capacity
        self._by_path: dict[str, int] = {}
        self._next_slot = 0

    # Backend-neutral wrap tokens (resolved per backend inside SampledImage).
    _WRAP_TOKENS = {
        "repeat": "repeat", "clamp": "clamp", "mirror": "mirror",
        "black": "black", "useMetadata": "repeat",
    }

    def add_or_get(
        self,
        path: Path,
        *,
        linear: bool = False,
        wrap_s: str = "repeat",
        wrap_t: str = "repeat",
    ) -> int:
        """Decode the file at `path` and return the array slot it lives in.

        Subsequent calls with the same (path, linear, wrap_s, wrap_t) tuple
        return the cached slot. Returns SENTINEL when the file can't be
        loaded (missing/corrupt).

        `linear=True` uploads as VK_FORMAT_R8G8B8A8_UNORM (no gamma decode) —
        use for normal, roughness, metallic, and other non-colour data textures.
        `wrap_s` / `wrap_t` come from USD's per-texture
        `inputs:wrapS` / `inputs:wrapT`. Two materials referencing the same
        file with different wrap modes get distinct slots (each owns its
        own sampler).
        """
        key = str(path.resolve()) if path.is_absolute() else str(path)
        if linear:
            key += ":linear"
        key += f":{wrap_s}/{wrap_t}"
        cached = self._by_path.get(key)
        if cached is not None:
            return cached
        try:
            img = Image.open(path).convert("RGBA")
        except (FileNotFoundError, OSError):
            return self.SENTINEL
        if self._next_slot >= self._capacity:
            return self.SENTINEL
        w, h = img.size
        fmt = "rgba8_unorm" if linear else "rgba8_srgb"
        addr_u = self._WRAP_TOKENS.get(wrap_s, "repeat")
        addr_v = self._WRAP_TOKENS.get(wrap_t, "repeat")
        slot = self._gpu.SampledImage(
            self.ctx, w, h,
            format=fmt,
            bytes_per_pixel=4,
            address_mode_u=addr_u,
            address_mode_v=addr_v,
        )
        slot.upload_sync(img.tobytes())
        idx = self._next_slot
        self._slots[idx] = slot
        self._by_path[key] = idx
        self._next_slot += 1
        return idx

    def filled_slots(self) -> list[tuple[int, SampledImage]]:
        """(slot_index, SampledImage) pairs for every populated slot."""
        return [(i, s) for i, s in enumerate(self._slots) if s is not None]

    def destroy(self) -> None:
        for slot in self._slots:
            if slot is not None:
                slot.destroy()
        self._slots = []


# Named-conductor id (Group 6.2). Defined in skinny.pbrt.data (a GPU-free module)
# so the importer, this upload, and the shader gate share one source of truth that
# a hostless test can pin — see CONDUCTOR_METAL_ID's docstring for the append-only
# rule. Aliased to the historical private name used throughout this module.
_CONDUCTOR_METAL_ID = CONDUCTOR_METAL_ID

#: spectralMetals upload order — index i holds the metal with id i+1. Derived, so
#: the id↔offset invariant is structural rather than two lists kept in sync by hand.
_SPECTRAL_METAL_ORDER = tuple(
    k for k, _ in sorted(_CONDUCTOR_METAL_ID.items(), key=lambda kv: kv[1])
)


def pack_flat_material(
    material,
    diffuse_texture_idx: int = 0xFFFFFFFF,
    roughness_texture_idx: int = 0xFFFFFFFF,
    metallic_texture_idx: int = 0xFFFFFFFF,
    normal_texture_idx: int = 0xFFFFFFFF,
    emissive_texture_idx: int = 0xFFFFFFFF,
    opacity_texture_idx: int = 0xFFFFFFFF,
    *,
    normal_scale: tuple[float, float, float] = (2.0, 2.0, 2.0),
    normal_bias: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    channel_mask: int = 0,
    volume_world_to_uvw=None,
    volume_value_max: float = 1.0,
    mm_per_unit: float = 1.0,
    spectral: bool = False,
) -> bytes:
    """Pack a Material's overrides into FLAT_MATERIAL_STRIDE bytes
    (FlatMaterialParams).

    Layout (scalar/std430 compatible — `float3` packs at 4-byte alignment):
       0: diffuseColor.r/g/b      (vec3 → 12 B)
      12: roughness               (float)
      16: metallic                (float)
      20: specular                (float)
      24: opacity                 (float)
      28: diffuseTextureIdx       (uint; 0xFFFFFFFF = use constant)
      32: roughnessTextureIdx     (uint; sentinel = use constant)
      36: metallicTextureIdx      (uint; sentinel = use constant)
      40: normalTextureIdx        (uint; sentinel = use mesh normal)
      44: emissiveTextureIdx      (uint; sentinel = use emissive const)
      48: emissiveColor.r/g/b     (vec3 → 12 B)
      60: ior                     (float; index of refraction, default 1.5)
      64: coat                    (float; clear coat weight 0..1)
      68: coatRoughness           (float)
      72: coatIOR                 (float)
      76: opacityTextureIdx       (uint; sentinel = use constant)
      80: coatColor.r/g/b         (vec3 → 12 B)
      92: opacityThreshold        (float; cutout alpha threshold)
      96: normalScale.x/y/z       (vec3 → 12 B; UsdUVTexture inputs:scale.xyz)
     108: channelMask             (uint; packed per-input channel selectors)
     112: normalBias.x/y/z        (vec3 → 12 B; UsdUVTexture inputs:bias.xyz)
     124: _pad                    (uint; reserved)
     128: transmissionColor.r/g/b (vec3 → 12 B; Stage-2; fallback diffuseColor)
     140: diffuseRoughness        (float; Stage-2 Oren-Nayar; 0 ⇒ Lambert)
     144: specularColor.r/g/b     (vec3 → 12 B; Stage-2; fallback white)
     156: _pad1                   (uint; reserved)
     160: medium σ_a.r/g/b        (vec3 → 12 B; subsurface/volume; mm⁻¹; else 0)
     172: medium g                (float; medium HG anisotropy)
     176: medium σ_s.r/g/b        (vec3 → 12 B; subsurface/volume; mm⁻¹; else 0)
     188: mediumKind              (uint; MEDIUM_HOMOGENEOUS=0 / MEDIUM_NANOVDB=1
                                   / MEDIUM_CLOUD=2; eta reuses ior)
     192: worldToUvw row 0        (vec4; volume; identity row (1,0,0,0) else)
     208: worldToUvw row 1        (vec4; volume; identity row (0,1,0,0) else)
     224: worldToUvw row 2        (vec4; volume; identity row (0,0,1,0) else)
     240: cloudDensity            (float; MEDIUM_CLOUD only; else 0)
     244: cloudWispiness          (float; MEDIUM_CLOUD only; else 0)
     248: cloudFrequency          (float; MEDIUM_CLOUD only; else 0)
     252: _pad2                   (float; reserved)

    Stage-2 rich inputs (flat-lobes-rich-inputs) are back-compatible: an absent
    override reproduces the prior behavior — transmissionColor defaults to
    diffuseColor (so the delta-transmission weight is unchanged), specularColor
    defaults to white, and diffuseRoughness defaults to 0 (exact Lambert).

    Volume materials (nanovdb-volume-rendering; `_material_is_volume`) pack the
    free-standing medium in the same 160..192 slots plus the world→uvw rows:
    `volume_world_to_uvw` is the loader's (3, 4) math-convention affine
    (VolumeGrid.world_to_uvw); non-volume materials get identity rows, so the
    pre-existing 0..192 prefix bytes are only ever *extended*, never shifted.

    Procedural cloud media (pbrt-cloud-procedural-medium; overrides carry
    `volume_cloud: True`) pack `mediumKind = MEDIUM_CLOUD` plus the appended
    240..256 float4 (density/wispiness/frequency — pbrt `CloudMedium` params,
    evaluated analytically in-shader); the world→uvw rows come from the
    material's own `volume_world_to_uvw` override (world→medium-local, folded
    by the importer from the medium CTM) rather than the scene grid, and no
    grid `value_max` fold applies (there is no density texture).
    """
    overrides = material.parameter_overrides
    diffuse = _override_color3(overrides, "diffuseColor", _FLAT_DEFAULT_DIFFUSE)
    roughness = _override_float(overrides, "roughness", 0.5)
    metallic = _override_float(overrides, "metallic", 0.0)
    specular = _override_float(overrides, "specular", 0.5)
    opacity = _override_float(overrides, "opacity", 1.0)
    emissive = _override_color3(overrides, "emissiveColor", (0.0, 0.0, 0.0))
    ior = _override_float(overrides, "ior", 1.5)
    coat = _override_float(overrides, "coat", 0.0)
    coat_roughness = _override_float(overrides, "coat_roughness", 0.0)
    coat_ior_raw = overrides.get("coat_IOR")
    coat_ior = float(coat_ior_raw) if coat_ior_raw is not None else 1.5
    coat_color = _override_color3(overrides, "coat_color", (1.0, 1.0, 1.0))
    opacity_threshold = _override_float(overrides, "opacityThreshold", 0.0)
    # Stage-2 rich inputs. transmission_color falls back to the diffuse albedo so
    # the delta-transmission weight (was `albedo`) is byte-unchanged when absent.
    transmission_color = _override_color3(overrides, "transmission_color", diffuse)
    specular_color = _override_color3(overrides, "specular_color", (1.0, 1.0, 1.0))
    diffuse_roughness = _override_float(overrides, "diffuse_roughness", 0.0)
    # Named-conductor identity (Group 6.2): the importer preserves the metal name
    # on skinnyOverrides["conductor_metal"]; map to the shader id (_CONDUCTOR_METAL_ID,
    # ids 1..N, else 0 = RGB Schlick F0). Packed into the spare _specularColorPad.w
    # (read as asuint by conductorMetalId). SPECTRAL-ONLY: only the spectral
    # conductor Fresnel reads it, so gate the id on `spectral` (like glassCauchyB)
    # — the RGB pack keeps the literal 0 in that lane, byte-identical to baseline.
    # The importer authors conductor_metal regardless of --spectral, so computing
    # it unconditionally would perturb the RGB material buffer for a named metal.
    conductor_metal_id = 0
    if spectral:
        conductor_metal_id = _CONDUCTOR_METAL_ID.get(
            str(overrides.get("conductor_metal", "")).strip().lower(), 0)
    # Named-glass dispersion (Group 6.4): the importer preserves the glass name on
    # skinnyOverrides["glass_dispersion"]; the Cauchy fit is n(λ)=A+B/λ_µm². The
    # base index A becomes the scalar `ior` lane (exact); B rides the spare
    # _normalBiasPad.w (glassCauchyB). 0 = constant-IOR (non-dispersive), so every
    # non-glass material keeps the old literal-0 pad → RGB pack byte-identical.
    # Only the spectral variant substitutes Cauchy A here; the RGB build keeps the
    # authored `ior`. That authored value is NOT the old generic 1.5 default any
    # more: since `pbrt-named-spectra`, the importer resolves a named glass to its
    # d-line index (materials._named_spectrum_scalar), so `glass-LASF9` arrives as
    # 1.850 in both builds. The two agree at the d-line and differ only by
    # dispersion — the RGB build has no wavelength to disperse over.
    glass_cauchy_b = 0.0
    _gd = overrides.get("glass_dispersion")
    if spectral and _gd is not None:
        from skinny.pbrt.data.spectral_tables import named_glass_cauchy
        _ab = named_glass_cauchy(_gd)
        if _ab is not None:
            ior = float(_ab[0])
            glass_cauchy_b = float(_ab[1])
    # Subsurface medium (pbrt-subsurface-volumetric), packed inline (no new SSBO —
    # Metal 31-buffer cap). σ in mm⁻¹; zero for non-medium materials. Boundary
    # eta reuses `ior`.
    medium_sigma_a = _override_color3(overrides, "subsurface_sigma_a", (0.0, 0.0, 0.0))
    medium_sigma_s = _override_color3(overrides, "subsurface_sigma_s", (0.0, 0.0, 0.0))
    medium_g = _override_float(overrides, "subsurface_g", 0.0)
    medium_kind = MEDIUM_HOMOGENEOUS
    # Cloud scalars (MEDIUM_CLOUD only; zeros keep the bytes inert elsewhere).
    cloud_density = cloud_wispiness = cloud_frequency = 0.0
    # World→uvw rows (volume; identity elsewhere so the bytes are inert).
    w2u = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    if _material_is_volume(material):
        # Free-standing medium (nanovdb-volume-rendering). TWO folds on σ:
        #  * `volume_value_max` — the density texture is normalized to [0,1] by
        #    dividing by the grid's value max at upload, so folding value_max
        #    into σ here makes the normalized texel exactly the density
        #    multiplier (and the global majorant exactly the packed σ_t).
        #  * `1 / mm_per_unit` — the walk's convention is σ in mm⁻¹ with world
        #    distances × mmPerUnit (traverseMediumSegment), while the importer
        #    carries pbrt σ per *scene unit*; pre-dividing makes the walk's
        #    optical depth σ_packed·d_world·mmPerUnit == σ_pbrt·d_world.
        # NOTE: σ is folded at pack time with the renderer's live mm_per_unit;
        # a later mm_per_unit change re-packs via the material upload path.
        mmu = max(float(mm_per_unit), 1e-6)
        # Grid-backed media dispatch to the density texture; the procedural
        # cloud evaluates pbrt's fBm density analytically in-shader; a
        # homogeneous free-standing interior (no grid asset) keeps densityAt ≡ 1.
        if overrides.get("volume_cloud"):
            medium_kind = MEDIUM_CLOUD
            cloud_density = _override_float(overrides, "cloud_density", 1.0)
            cloud_wispiness = _override_float(overrides, "cloud_wispiness", 1.0)
            cloud_frequency = _override_float(overrides, "cloud_frequency", 5.0)
        elif overrides.get("volume_grid_asset"):
            medium_kind = MEDIUM_NANOVDB
        # else: homogeneous free-standing interior (medium_kind already
        # MEDIUM_HOMOGENEOUS from the initializer above).
        # `volume_value_max` is a *grid* normalization fold (texels divided by
        # the grid max at upload) — it must not scale the analytic kinds.
        fold = (float(volume_value_max) if medium_kind == MEDIUM_NANOVDB
                else 1.0) / mmu
        vs_a = _override_color3(overrides, "volume_sigma_a", (0.0, 0.0, 0.0))
        vs_s = _override_color3(overrides, "volume_sigma_s", (0.0, 0.0, 0.0))
        medium_sigma_a = tuple(c * fold for c in vs_a)
        medium_sigma_s = tuple(c * fold for c in vs_s)
        medium_g = _override_float(overrides, "volume_g", 0.0)
        ior = 1.0  # index-matched pass-through boundary (eta reuses the ior slot)
        # World→[0,1]³ rows: the cloud carries its own importer-folded
        # medium-local affine on the material overrides; the grid kind uses the
        # loader's per-scene grid affine (`volume_world_to_uvw` arg).
        rows = overrides.get("volume_world_to_uvw") if medium_kind == MEDIUM_CLOUD \
            else volume_world_to_uvw
        if rows is not None:
            m = np.asarray([float(v) for v in np.ravel(rows)], np.float32).reshape(3, 4)
            w2u = tuple(tuple(float(v) for v in row) for row in m)
    return struct.pack(
        "fff f f f f I I I I I fff f  f f f I  fff f  fff I fff f  fff f  fff I  fff f fff I"
        " ffff ffff ffff ffff",
        diffuse[0], diffuse[1], diffuse[2],
        roughness, metallic, specular, opacity,
        int(diffuse_texture_idx) & 0xFFFFFFFF,
        int(roughness_texture_idx) & 0xFFFFFFFF,
        int(metallic_texture_idx) & 0xFFFFFFFF,
        int(normal_texture_idx) & 0xFFFFFFFF,
        int(emissive_texture_idx) & 0xFFFFFFFF,
        emissive[0], emissive[1], emissive[2],
        ior,
        coat, coat_roughness, coat_ior,
        int(opacity_texture_idx) & 0xFFFFFFFF,
        coat_color[0], coat_color[1], coat_color[2],
        opacity_threshold,
        float(normal_scale[0]), float(normal_scale[1]), float(normal_scale[2]),
        int(channel_mask) & 0xFFFFFFFF,
        float(normal_bias[0]), float(normal_bias[1]), float(normal_bias[2]),
        float(glass_cauchy_b),
        transmission_color[0], transmission_color[1], transmission_color[2],
        diffuse_roughness,
        specular_color[0], specular_color[1], specular_color[2],
        int(conductor_metal_id) & 0xFFFFFFFF,
        medium_sigma_a[0], medium_sigma_a[1], medium_sigma_a[2],
        medium_g,
        medium_sigma_s[0], medium_sigma_s[1], medium_sigma_s[2],
        int(medium_kind) & 0xFFFFFFFF,
        w2u[0][0], w2u[0][1], w2u[0][2], w2u[0][3],
        w2u[1][0], w2u[1][1], w2u[1][2], w2u[1][3],
        w2u[2][0], w2u[2][1], w2u[2][2], w2u[2][3],
        float(cloud_density), float(cloud_wispiness), float(cloud_frequency), 0.0,
    )


def pack_std_surface_params(material) -> bytes:
    """Pack a Material's overrides into 256 bytes (StdSurfaceParams).

    Layout matches the Slang struct in mtlx_std_surface.slang (scalar layout).
    UsdPreviewSurface names are mapped to standard_surface equivalents.
    """
    o = material.parameter_overrides

    def _f(key, usd_key=None, default=0.0):
        v = o.get(key)
        if v is None and usd_key:
            v = o.get(usd_key)
        if v is None:
            return float(default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _c3(key, usd_key=None, default=(0.0, 0.0, 0.0)):
        v = o.get(key)
        if v is None and usd_key:
            v = o.get(usd_key)
        if v is None:
            return tuple(float(c) for c in default)
        try:
            return float(v[0]), float(v[1]), float(v[2])
        except (TypeError, IndexError, ValueError):
            return tuple(float(c) for c in default)

    base_color = _c3("base_color", "diffuseColor", (0.8, 0.8, 0.8))
    base = _f("base", default=1.0)
    diffuse_roughness = _f("diffuse_roughness", default=0.0)
    metalness = _f("metalness", "metallic", 0.0)
    specular = _f("specular", default=1.0)
    specular_roughness = _f("specular_roughness", "roughness", 0.5)
    specular_color = _c3("specular_color", default=(1.0, 1.0, 1.0))
    specular_IOR = _f("specular_IOR", "ior", 1.5)
    specular_anisotropy = _f("specular_anisotropy", default=0.0)
    specular_rotation = _f("specular_rotation", default=0.0)
    transmission = _f("transmission", default=0.0)
    transmission_depth = _f("transmission_depth", default=0.0)
    transmission_color = _c3("transmission_color", default=(1.0, 1.0, 1.0))
    transmission_scatter_aniso = _f("transmission_scatter_anisotropy", default=0.0)
    transmission_scatter = _c3("transmission_scatter", default=(0.0, 0.0, 0.0))
    transmission_dispersion = _f("transmission_dispersion", default=0.0)
    transmission_extra_roughness = _f("transmission_extra_roughness", default=0.0)
    subsurface = _f("subsurface", default=0.0)
    subsurface_scale = _f("subsurface_scale", default=1.0)
    subsurface_anisotropy = _f("subsurface_anisotropy", default=0.0)
    subsurface_color = _c3("subsurface_color", default=(1.0, 1.0, 1.0))
    subsurface_radius = _c3("subsurface_radius", default=(1.0, 1.0, 1.0))
    sheen = _f("sheen", default=0.0)
    sheen_color = _c3("sheen_color", default=(1.0, 1.0, 1.0))
    sheen_roughness = _f("sheen_roughness", default=0.3)
    coat = _f("coat", default=0.0)
    coat_roughness = _f("coat_roughness", default=0.1)
    coat_anisotropy = _f("coat_anisotropy", default=0.0)
    coat_rotation = _f("coat_rotation", default=0.0)
    coat_IOR = _f("coat_IOR", default=1.5)
    coat_affect_color = _f("coat_affect_color", default=0.0)
    coat_affect_roughness = _f("coat_affect_roughness", default=0.0)
    coat_color = _c3("coat_color", default=(1.0, 1.0, 1.0))
    thin_film_thickness = _f("thin_film_thickness", default=0.0)
    thin_film_IOR = _f("thin_film_IOR", default=1.5)
    emission = _f("emission", default=0.0)
    emission_color = _c3("emission_color", "emissiveColor", (1.0, 1.0, 1.0))

    if emission == 0.0 and "emissiveColor" in o:
        ec = o["emissiveColor"]
        try:
            if float(ec[0]) > 0 or float(ec[1]) > 0 or float(ec[2]) > 0:
                emission = 1.0
        except (TypeError, IndexError, ValueError):
            pass

    opacity = _c3("opacity", default=(1.0, 1.0, 1.0))
    if "opacity" in o and not hasattr(o["opacity"], "__getitem__"):
        try:
            f = float(o["opacity"])
            opacity = (f, f, f)
        except (TypeError, ValueError):
            pass

    thin_walled = int(_f("thin_walled", default=0))

    return struct.pack(
        "ffffffff"      # 0-32:   base_color(3), base, diffuse_roughness, metalness, specular, specular_roughness
        "ffffffff"      # 32-64:  specular_color(3), specular_IOR, specular_anisotropy, specular_rotation, transmission, transmission_depth
        "ffffffff"      # 64-96:  transmission_color(3), scatter_aniso, transmission_scatter(3), dispersion
        "ffffffff"      # 96-128: extra_roughness, subsurface, subsurface_scale, subsurface_aniso, subsurface_color(3), _pad0
        "ffffffff"      # 128-160: subsurface_radius(3), sheen, sheen_color(3), sheen_roughness
        "ffffffff"      # 160-192: coat, coat_roughness, coat_aniso, coat_rotation, coat_IOR, coat_affect_color, coat_affect_roughness, _pad1
        "ffffffff"      # 192-224: coat_color(3), thin_film_thickness, thin_film_IOR, emission, emission_color.r, emission_color.g
        "fffffIff",     # 224-256: emission_color.b, _pad2, opacity(3), thin_walled, _pad3, _pad4
        base_color[0], base_color[1], base_color[2], base,
        diffuse_roughness, metalness, specular, specular_roughness,
        specular_color[0], specular_color[1], specular_color[2], specular_IOR,
        specular_anisotropy, specular_rotation, transmission, transmission_depth,
        transmission_color[0], transmission_color[1], transmission_color[2], transmission_scatter_aniso,
        transmission_scatter[0], transmission_scatter[1], transmission_scatter[2], transmission_dispersion,
        transmission_extra_roughness, subsurface, subsurface_scale, subsurface_anisotropy,
        subsurface_color[0], subsurface_color[1], subsurface_color[2], 0.0,
        subsurface_radius[0], subsurface_radius[1], subsurface_radius[2], sheen,
        sheen_color[0], sheen_color[1], sheen_color[2], sheen_roughness,
        coat, coat_roughness, coat_anisotropy, coat_rotation,
        coat_IOR, coat_affect_color, coat_affect_roughness, 0.0,
        coat_color[0], coat_color[1], coat_color[2], thin_film_thickness,
        thin_film_IOR, emission, emission_color[0], emission_color[1],
        emission_color[2], 0.0, opacity[0], opacity[1], opacity[2],
        thin_walled, 0.0, 0.0,
    )


# StdSurfaceParams fields in struct order as (name, float-count). Names match
# the Slang struct in mtlx_std_surface.slang and `pack_std_surface_params`'s
# scalar (std430) packing — scalar offset of each field is the running byte sum
# (float3 = 12 B, no 16-B promotion). Used by `pack_std_surface_params_msl` to
# relocate the scalar record into Metal's MSL layout (where float3 → 16 B), keyed
# by the reflected field names. Total must equal STD_SURFACE_STRIDE (256 B).
_STD_SURFACE_FIELDS: tuple[tuple[str, int], ...] = (
    ("base_color", 3), ("base", 1), ("diffuse_roughness", 1), ("metalness", 1),
    ("specular", 1), ("specular_roughness", 1), ("specular_color", 3),
    ("specular_IOR", 1), ("specular_anisotropy", 1), ("specular_rotation", 1),
    ("transmission", 1), ("transmission_depth", 1), ("transmission_color", 3),
    ("transmission_scatter_anisotropy", 1), ("transmission_scatter", 3),
    ("transmission_dispersion", 1), ("transmission_extra_roughness", 1),
    ("subsurface", 1), ("subsurface_scale", 1), ("subsurface_anisotropy", 1),
    ("subsurface_color", 3), ("_pad0", 1), ("subsurface_radius", 3), ("sheen", 1),
    ("sheen_color", 3), ("sheen_roughness", 1), ("coat", 1), ("coat_roughness", 1),
    ("coat_anisotropy", 1), ("coat_rotation", 1), ("coat_IOR", 1),
    ("coat_affect_color", 1), ("coat_affect_roughness", 1), ("_pad1", 1),
    ("coat_color", 3), ("thin_film_thickness", 1), ("thin_film_IOR", 1),
    ("emission", 1), ("emission_color", 3), ("_pad2", 1), ("opacity", 3),
    ("thin_walled", 1), ("_pad3", 1), ("_pad4", 1),
)
assert sum(n for _, n in _STD_SURFACE_FIELDS) * 4 == STD_SURFACE_STRIDE


def pack_std_surface_params_msl(
    scalar: bytes, layout: dict[str, tuple[int, int]], stride: int
) -> bytes:
    """Relocate a scalar-packed `pack_std_surface_params` record (256 B, float3 =
    12 B) into Metal's reflected MSL element layout for
    `StructuredBuffer<StdSurfaceParams>` (binding 19), where Slang pads every
    `float3` to 16 B and grows the element stride past 256 B (≈400). Each field's
    bytes move from its scalar offset (the running sum over `_STD_SURFACE_FIELDS`)
    to its reflected MSL offset (`layout[name]`). Same design-D3 repack the skin
    params (`_pack_mtlx_skin_array_msl`) get; without it every field after
    `base_color` is misread on Metal (metalness reads specular, specular reads
    specular_roughness, coat → 0, …). (Graph params no longer need this — change
    combine-graph-param-buffers reads them via `ByteAddressBuffer.Load<T>`, which
    is scalar on both targets.)

    FORWARD-LOOKING / currently inert: binding 19 is read only by
    `preview_pass.slang` (the BXDF/std_surface visualiser), which is a Vulkan-only
    `PreviewPipeline` — Vulkan reads the scalar layout directly, and the Metal
    megakernel dead-strips binding 19 entirely (`loadStdSurfaceParams` is
    uncalled), so on Metal this relocation only activates once a Metal pipeline
    actually references `stdSurfaceParams`. It is the layout-correct path for that
    future port, not a fix for any image today (the path-traced flat BSDF reads
    the float4-wrapped, MSL-safe FlatMaterialParams at binding 13)."""
    rec = bytearray(stride)
    off = 0
    for name, nfloats in _STD_SURFACE_FIELDS:
        size = nfloats * 4
        moff = layout.get(name)
        if moff is not None:
            rec[moff[0]:moff[0] + size] = scalar[off:off + size]
        off += size
    return bytes(rec)


@dataclass
class SkinParameters:
    """Physically-based skin parameters.

    Layered skin model: epidermis -> dermis -> subcutaneous fat.
    Absorption and scattering coefficients are spectral (RGB approximation).
    """

    # Epidermis
    melanin_fraction: float = 0.15
    epidermis_thickness_mm: float = 0.1

    # Dermis
    hemoglobin_fraction: float = 0.05
    blood_oxygenation: float = 0.75
    dermis_thickness_mm: float = 1.0

    # Subcutaneous
    subcut_thickness_mm: float = 3.0

    # Scattering
    scattering_coefficient: np.ndarray = field(
        default_factory=lambda: np.array([3.7, 4.4, 5.05], dtype=np.float32)
    )
    anisotropy_g: float = 0.8

    # Surface
    roughness: float = 0.35
    ior: float = 1.4

    # Sub-millimeter surface detail (pores + vellus hair). Defaults to 0 so
    # loading a pre-detail preset renders identically to pre-change output.
    pore_density: float = 0.0
    pore_depth: float = 0.0
    hair_density: float = 0.0
    hair_tilt: float = 0.0

    def pack(self) -> bytes:
        """Pack into std140-compatible bytes matching the Slang SkinParams struct.

        std140 layout (offsets in bytes):
          0: melaninFraction      (float)
          4: hemoglobinFraction   (float)
          8: bloodOxygenation     (float)
         12: epidermisThickness   (float)
         16: dermisThickness      (float)
         20: subcutThickness      (float)
         24: <8 bytes padding>    (align float3 to 16)
         32: scatteringCoeff      (float3, 12 bytes)
         44: anisotropy           (float, fills vec3 trailing slot)
         48: roughness            (float)
         52: ior                  (float)
         56: poreDensity          (float)
         60: poreDepth            (float)
         64: hairDensity          (float)
         68: hairTilt             (float)
         72: <8 bytes padding>    (struct rounds to 16)
        Total: 80 bytes
        """
        return struct.pack(
            "6f 2I 3f f 2f 4f 2I",
            self.melanin_fraction,
            self.hemoglobin_fraction,
            self.blood_oxygenation,
            self.epidermis_thickness_mm,
            self.dermis_thickness_mm,
            self.subcut_thickness_mm,
            0, 0,                                # 8 bytes padding
            *self.scattering_coefficient,
            self.anisotropy_g,
            self.roughness,
            self.ior,
            self.pore_density,
            self.pore_depth,
            self.hair_density,
            self.hair_tilt,
            0, 0,                                # 8 bytes struct tail padding
        )


def _perspective(
    fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0,
) -> np.ndarray:
    """Reverse-depth perspective projection matrix (stored transposed for GPU).

    Math (OpenGL/Vulkan infinite-far convention, z_near = 0.1, z_far = 100):
        f   = 1 / tan(fov/2)
        P   = [[f/a,  0,        0,          0],
               [0,    f,        0,          0],
               [0,    0,  far/(n-f), n·far/(n-f)],
               [0,    0,       -1,          0]]

    numpy stores row-major → GPU reads column-major → receives Pᵀ → correct.
    """
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = far / (near - far)
    proj[2, 3] = -1.0
    proj[3, 2] = (near * far) / (near - far)
    return proj


def _look_at(pos: np.ndarray, forward: np.ndarray,
             world_up: Optional[np.ndarray] = None) -> np.ndarray:
    """View matrix from camera position and forward direction (stored transposed).

    Math (camera basis via cross-product):
        r = normalize(forward × up)      (right axis)
        u = r × forward                  (up axis, re-orthogonalised)
        V = [[r.x,  r.y,  r.z, −r·pos],
             [u.x,  u.y,  u.z, −u·pos],
             [−d.x, −d.y, −d.z, d·pos],
             [0,    0,    0,    1     ]]

    where d = forward. Stored transposed for the same numpy/GPU convention as
    _perspective — the GPU reads column-major and recovers V.
    """
    # Returns V^T — numpy row-major, GPU reads column-major → cancels back to V.
    # `world_up` is the reference up; an authored camera (CameraOverride.up) feeds
    # its own up here so non-Y-up pbrt cameras keep their roll. Default +Y ⇒ the
    # prior basis, byte-identical for Y-up / interactive cameras.
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        world_up = np.asarray(world_up, dtype=np.float32).reshape(3)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # Degenerate: up ∥ forward. Fall back to a secondary world axis so the
        # basis stays finite and orthonormal (no zero `right`).
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(forward, alt))) > 0.9:
            alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, alt)
        rn = np.linalg.norm(right)
    right = right / max(rn, 1e-6)
    up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float32)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = -forward
    view[3, 0] = -np.dot(right, pos)
    view[3, 1] = -np.dot(up, pos)
    view[3, 2] = np.dot(forward, pos)
    return view


def _hero_yaw_pitch() -> tuple[float, float]:
    """Default 3/4 hero-view orbit angles (radians): yaw 30°, pitch 15°."""
    return float(np.radians(30.0)), float(np.radians(15.0))


def _orbit_distance_cap(longest_dim: float) -> float:
    """Initial max orbit distance for a scene whose longest AABB edge is
    ``longest_dim``.

    At least 10× the longest dimension so large scenes can be framed and
    zoomed out, never below the legacy 50-unit floor for small scenes. This is
    the *initial* ceiling only — ``OrbitCamera.set_distance`` raises
    ``max_distance`` past this when the user types or zooms further out.
    """
    return float(max(50.0, 10.0 * longest_dim))


class CameraBase(abc.ABC):
    """PBRT CameraBase analogue — abstract camera-model surface.

    Concrete subclasses (OrbitCamera, FreeCamera) are `@dataclass`es that
    own their controller state. The base contributes the methods that
    every camera shares: the projection matrix and the common slice of
    the change-detection signature (the fields the lens-buffer sync /
    accumulation-reset paths read off the camera).

    Subclasses must expose attribute `position: np.ndarray` and implement
    `forward`, `view_matrix`, and `state_signature`. `position` is not
    declared abstract here because dataclass subclasses use either a
    field (FreeCamera) or a computed `@property` (OrbitCamera), and an
    abstract `@property` in the base would collide with the field form.
    """

    # Attributes every subclass is expected to provide. Listed for typing
    # / readers; concrete declarations live in the @dataclass subclasses.
    fov: float
    near: float
    far: float
    fstop: float
    focus_distance: float
    focal_length_mm: float
    vertical_aperture_mm: float
    lens: Optional["LensSystem"]

    @abc.abstractmethod
    def forward(self) -> np.ndarray: ...

    @abc.abstractmethod
    def view_matrix(self) -> np.ndarray: ...

    @abc.abstractmethod
    def state_signature(self) -> tuple: ...

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fov, aspect, self.near, self.far)

    def _common_signature(self) -> tuple:
        """Camera-model slice of state_signature (lens + intrinsics).

        Subclasses concatenate their controller state with this tuple so
        the accumulation-reset path notices changes to either side.
        """
        return (
            float(self.fov), float(self.near), float(self.far),
            float(self.fstop), float(self.focus_distance),
            float(self.focal_length_mm), float(self.vertical_aperture_mm),
            self.lens.signature() if self.lens is not None else ("lens", "none"),
        )


@dataclass
class OrbitCamera(CameraBase):
    """Camera that rotates around a target point (default: centre of the SDF head).

    The head's y-extent is roughly [-0.94, +1.15] and its z-extent [-0.80, +0.97],
    so target=(0, 0.1, 0.05) pins the pivot to its visual centroid. If you orbit
    around the world origin instead, the head drifts noticeably around the frame
    because that origin sits near the jaw/throat rather than the head's middle.
    """

    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.1, 0.05], dtype=np.float32)
    )
    # World-space up used to build the view basis. Defaults to +Y (the prior
    # behavior); an authored camera (_override_to_orbit) sets this to its up so a
    # non-Y-up pbrt camera keeps its roll.
    up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    distance: float = 3.0
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    fstop: float = 0.0          # 0 ⇒ wide open; >0 closes the iris to f/N
    focus_distance: float = 0.0  # 0 ⇒ track orbit distance
    focal_length_mm: float = 50.0  # used with fstop to drive iris diameter
    vertical_aperture_mm: float = 24.0  # sensor height in mm; used by the lens path
    lens: Optional["LensSystem"] = None  # PBRT-style thick lens; None ⇒ pinhole
    # Upper clamp for `distance` (wheel zoom, UI slider, auto-frame). Scales
    # with scene size — the renderer raises it to ≥4× the longest scene
    # dimension when a model loads so large scenes can be framed/zoomed out.
    max_distance: float = 50.0

    @property
    def position(self) -> np.ndarray:
        """Camera world position from spherical orbit coordinates.

        Math (spherical → Cartesian):
            x = d · cos(pitch) · sin(yaw)
            y = d · sin(pitch)
            z = d · cos(pitch) · cos(yaw)

        where  d     = orbit distance
               yaw   = azimuth angle (radians)
               pitch = elevation angle (radians, clamped ±89°)
        """
        x = self.distance * np.cos(self.pitch) * np.sin(self.yaw)
        y = self.distance * np.sin(self.pitch)
        z = self.distance * np.cos(self.pitch) * np.cos(self.yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.005
        self.pitch += dy * 0.005
        self.pitch = float(np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))

    def set_distance(self, value: float) -> None:
        """Set the orbit distance to any value ≥ 0.5, growing ``max_distance``
        to fit.

        ``max_distance`` is the current ceiling (slider range + wheel-zoom
        limit). Writing a larger distance raises it so the UI stays consistent;
        it never shrinks here — only a re-frame/model-load resets it. The 1e9
        cap is a degeneracy guard (bounds inf and int-slider precision loss),
        effectively unbounded for real scenes.
        """
        v = float(np.clip(value, 0.5, 1e9))
        if v > self.max_distance:
            self.max_distance = v
        self.distance = v

    def zoom(self, delta: float) -> None:
        self.set_distance(self.distance * (1.0 - delta * 0.1))

    def pan(self, dx: float, dy: float) -> None:
        f = self.target - self.position
        f = f / np.linalg.norm(f)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(f, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, f)
        scale = self.distance * 0.002
        self.target = self.target + (-right * dx + up * dy) * scale

    def forward(self) -> np.ndarray:
        f = self.target - self.position
        return f / max(np.linalg.norm(f), 1e-6)

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.forward(), self.up)

    def state_signature(self) -> tuple:
        return (
            "orbit",
            float(self.yaw), float(self.pitch), float(self.distance),
            float(self.target[0]), float(self.target[1]), float(self.target[2]),
            float(self.up[0]), float(self.up[1]), float(self.up[2]),
        ) + self._common_signature()


@dataclass
class FreeCamera(CameraBase):
    """FPS-style camera: WASD translates, mouse look rotates yaw/pitch."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 3.0], dtype=np.float32)
    )
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0
    move_speed: float = 1.5   # world units / second
    near: float = 0.1
    far: float = 100.0
    fstop: float = 0.0
    focus_distance: float = 0.0
    focal_length_mm: float = 50.0
    vertical_aperture_mm: float = 24.0
    lens: Optional["LensSystem"] = None

    def forward(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        return np.array([
            np.sin(self.yaw) * cp,
            np.sin(self.pitch),
            -np.cos(self.yaw) * cp,
        ], dtype=np.float32)

    def _right_vec(self) -> np.ndarray:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(self.forward(), world_up)
        return r / max(np.linalg.norm(r), 1e-6)

    def look(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.005
        self.pitch -= dy * 0.005
        self.pitch = float(np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))

    def move(self, forward: float, right: float, up: float, dt: float) -> None:
        step = self.move_speed * dt
        self.position = (
            self.position
            + self.forward() * (forward * step)
            + self._right_vec() * (right * step)
            + np.array([0.0, 1.0, 0.0], dtype=np.float32) * (up * step)
        )

    def zoom(self, delta: float) -> None:
        # Scroll changes movement speed in free mode.
        self.move_speed = float(np.clip(self.move_speed * (1.0 + delta * 0.1), 0.05, 50.0))

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.forward())

    def state_signature(self) -> tuple:
        return (
            "free",
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.yaw), float(self.pitch),
        ) + self._common_signature()


def _write_exr(path: str, rgb: np.ndarray) -> None:
    """Write float32 RGB to a scanline EXR via the Academy OpenEXR bindings."""
    import OpenEXR

    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {
        "R": np.ascontiguousarray(rgb[..., 0]),
        "G": np.ascontiguousarray(rgb[..., 1]),
        "B": np.ascontiguousarray(rgb[..., 2]),
    }
    OpenEXR.File(header, channels).write(path)


def _write_hdr_rgbe(path: str, rgb: np.ndarray) -> None:
    """Write float32 RGB to a Radiance .hdr (RGBE) file. No external deps."""
    rgb = np.maximum(rgb, 0.0).astype(np.float32, copy=False)
    h, w, _ = rgb.shape
    max_c = rgb.max(axis=2)
    mantissa, exponent = np.frexp(max_c)
    safe = max_c > 1e-32
    scale = np.where(safe, mantissa * 256.0 / np.where(safe, max_c, 1.0), 0.0)
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(3):
        rgbe[..., i] = np.clip(
            np.round(rgb[..., i] * scale), 0.0, 255.0,
        ).astype(np.uint8)
    rgbe[..., 3] = np.where(
        safe, np.clip(exponent + 128, 0, 255), 0,
    ).astype(np.uint8)
    header = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n"
        b"\n"
        + f"-Y {h} +X {w}\n".encode("ascii")
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rgbe.tobytes())


@dataclass
class FilmParameters:
    """pbrt film exposure controls, live on the renderer (change
    pbrt-radiometric-parity).

    `iso` and `exposure_time` are read from the authored camera
    (`skinny:film:iso` / `skinny:film:exposureTime`) and define the imaging
    ratio `exposure_time · iso / 100`, a global linear output scale on the
    rendered radiance (applied to the linear-HDR readback and folded into the
    display exposure). Defaults (100 / 1.0) ⇒ ratio 1.0 ⇒ a byte-identical render.
    Exposed in the UI as `film.iso` / `film.exposure_time` so they retune live.
    """

    iso: float = 100.0
    exposure_time: float = 1.0

    def imaging_ratio(self) -> float:
        return float(self.exposure_time) * float(self.iso) / 100.0


class Renderer:
    """Sets up Vulkan resources and dispatches Slang compute shaders each frame."""

    # Wavefront bidirectional integrator (Phase 3). True now that the staged
    # wavefront bdpt (WavefrontBdptPass) reaches A/B parity with the megakernel
    # bdpt: selecting Wavefront + BDPT routes to the staged subpath-walk +
    # connection pipeline instead of falling back to the megakernel.
    WAVEFRONT_BDPT_SUPPORTED = True

    def __init__(
        self,
        vk_ctx: VulkanContext,
        shader_dir: Path,
        hdr_dir: Path | None = None,
        tattoo_dir: Path | None = None,
        usd_scene_path: Path | None = None,
        use_usd_mtlx_plugin: bool = False,
        execution_mode: str = "megakernel",
        bdpt_walk: str = "fused",
        neural_config=None,
        neural_handoff: str = "file",
        neural_trainer: str = "auto",
        train_precision: str = "fp32",
        spectral: bool = False,
    ) -> None:
        self.ctx = vk_ctx
        # Resolve the GPU-resource module once from the context (design D1): the
        # renderer builds every megakernel resource through `self._gpu.*`, which is
        # `vk_compute` on a VulkanContext (byte-identical to before) and
        # `metal_compute` on a MetalContext — no per-construction-site backend
        # branch. `is_metal` gates the few genuinely different paths (uniform pack,
        # frame dispatch).
        from skinny.backend_select import resource_module
        self._gpu = resource_module(self.ctx)
        self.is_metal = bool(getattr(self.ctx, "is_metal", False))
        # Shared sampler for the Metal bindless texture pool (binding 38, design
        # D8). One sampler instead of the 128 a combined Sampler2D[] would emit.
        # MUST be repeat/repeat to match the Vulkan per-slot samplers, whose
        # TexturePool default is wrap_s=wrap_t="repeat": `_make_sampler` defaults
        # address_v="clamp" (right for the equirect env map, wrong for tiling
        # material textures). With clamp-V a texture sampled past v=1 — e.g. a
        # MaterialX `tiledimage` at uvtiling=4 like the wood material — clamps to
        # the edge row on Metal while Vulkan tiles, leaving wood ~11% bright
        # (rel-MSE ≈0.03 on its region). Per-texture USD wrapS/wrapT still can't be
        # honoured per-slot under one shared sampler (D8); repeat/repeat is the
        # correct default and matches the pool default.
        self._metal_common_sampler = (
            self._gpu._make_sampler(self.ctx, address_v="repeat")
            if self.is_metal else None)
        # Neural size/precision build config (study change
        # neural-precision-size-study). Fixed for the renderer's lifetime — the
        # study harness builds a fresh headless renderer per grid cell. Falls
        # back to fp32 on devices lacking fp16 via _effective_neural_config().
        from skinny.sampling.neural_weights import NeuralBuildConfig
        self._neural_config = neural_config or NeuralBuildConfig()
        self._fp16_fallback_warned = False
        self._requested_execution_mode = str(execution_mode)
        # Hero-wavelength spectral megakernel variant (change spectral-rendering,
        # GPU transport). When set, the megakernel pipeline is compiled with
        # `-DSKINNY_SPECTRAL` and the three pbrt upsample / D65 storage buffers
        # (bindings 45/46/47) are uploaded + bound. Every spectral resource is
        # guarded on this flag so a non-spectral run is byte-identical to before.
        self._spectral = bool(spectral)
        # BDPT subpath-build strategy for wavefront+bdpt (CLI-fixed per session):
        #   fused      — one walk kernel (the S1 connect-compaction win, default)
        #   eye        — staged eye walk + fused light tail
        #   eye_light  — fully staged eye + light walks
        # All produce the identical image; this only trades dispatch overhead vs
        # occupancy. The deprecated `megakernel` alias resolves to `fused`.
        try:
            self.bdpt_walk_mode = resolve_walk(bdpt_walk)
        except ValueError:
            self.bdpt_walk_mode = "fused"
        self.width = vk_ctx.width
        self.height = vk_ctx.height
        self.shader_dir = shader_dir
        self.skin = SkinParameters()

        # Biological presets (Fitzpatrick I–VI × M/F) plus any user-saved
        # presets discovered under ~/.skinny/presets/. Selecting a preset
        # pushes its values into self.skin via apply_preset() which goes
        # through the same _set_nested path as the keyboard/slider UI.
        self.presets: list[Preset] = list(PRESETS) + load_user_presets()
        self.preset_index = 0

        # Two cameras kept in parallel; `camera` returns whichever is active.
        # Orbit is the default so the head is framed and dragging rotates
        # around it; press 'C' to switch to free-fly (WASD + mouse look).
        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        # Follower fed by the animated USD camera when camera_mode == "usd".
        self.usd_camera = OrbitCamera()
        self.camera_mode: str = "orbit"

        self.frame_index = 0
        self.time_elapsed = 0.0

        # USD animation playback. The clock stays inert (has_animation=False)
        # until a USD scene with authored animation loads, at which point
        # _load_usd_model replaces it and populates _anim_index. _usd_camera_override
        # holds the USD camera evaluated at the current time when camera_mode == "usd".
        self.clock = PlaybackClock()
        self._anim_index = None
        self._skeletal = None  # usd_loader.SkeletalScene when a skinned USD loads
        self._skinning_passes = None  # vk_skinning.SkinningPasses (Vulkan only)
        self._usd_camera_override = None
        # Improper (mirrored) pbrt camera flag, set from CameraOverride.mirrored
        # when an authored camera is applied; folded into FrameConstants.cameraMirror.
        self._camera_mirror = False
        self._last_eval_time_code: float | None = None
        # USD-declared UI controls (skinny:ui:* prims). _usd_live_dirty is set
        # when a `usd:` control writes a stage attribute, prompting update() to
        # refresh the live-applicable scene state (lights/transforms/camera).
        self._usd_controls = []
        self._usd_live_dirty = False
        # Z-up→Y-up correction (3x3 or None) matching what _apply_up_axis_correction
        # bakes into instance transforms at load, re-applied during per-frame re-eval.
        self._usd_up_axis_rt = None
        # (enabled_idx, blas_offsets, material_ids) cached by _upload_usd_scene
        # so animation re-uploads only the small instance buffer.
        self._usd_instance_layout = None
        # USD prim path → indices into `_usd_scene.instances`. Rebuilt on every
        # geometry upload; the runtime scene-graph editing API resolves
        # add/remove/transform targets through it.
        self._prim_to_instances: dict[str, list[int]] = {}
        # Non-destructive in-memory edit layer (the stage session layer) + its
        # default on-disk save path. Attached on USD load so runtime scene-graph
        # edits never touch the original file; persisted only via save_edits().
        self._usd_edit_layer = None
        self._edit_layer_default_path: str | None = None

        # HDR environment library — built-in presets + any .hdr files found
        # in hdr_dir. Switching is driven by `env_index`.
        self.environments: list[Environment] = load_environments(hdr_dir)
        self.env_index = 0
        self._last_env_index: object = (-1, -1)

        # Progressive accumulation — running sample count across frames.
        # Reset to 0 (via reset_accumulation) whenever camera/skin/light changes.
        self.accum_frame = 0
        self._last_state_hash: int | None = None

        # HUD overlay
        self.show_hud = True
        self.hud_text_lines: list[str] = []  # set by the input layer each frame
        self._fps_smooth = 0.0
        self._hud_font = self._load_hud_font()

        # Light (spherical representation + derived direction/radiance).
        # Color channels are user-tunable so direct light can be warmed,
        # cooled, or tinted to match a gel. Defaults are the old hardcoded
        # (3.0, 2.8, 2.5) tungsten-ish ratio L2-normalised, preserving both
        # chromaticity and total radiance magnitude from before the split.
        self.light_elevation = 35.0   # degrees
        self.light_azimuth = 45.0     # degrees
        self.light_intensity = 5.0
        self.light_color_r = 0.624
        self.light_color_g = 0.583
        self.light_color_b = 0.520
        # In-memory USD stage holding the synthesized default DistantLight.
        # Populated below; treat it as the canonical representation of the
        # built-in light so it lives alongside imported USD lights in the
        # scene graph editor.
        self._default_light_stage = None
        self._default_light_prim = None
        self._default_dome_prim = None
        # Set when a USD model is loaded so HDR edits can mutate the
        # source DomeLight prim's ``inputs:texture:file`` attribute.
        self._usd_stage = None
        self._init_default_light_stage()
        self._update_light()

        # Direct-light toggle. Exposed to the UI as a discrete choice so the
        # user can fall back to pure image-based lighting.
        self.direct_light_modes: list[str] = ["On", "Off"]
        self.direct_light_index = 0

        # Skin scattering model. Each entry is a bitmask consumed by
        # main_pass.slang (bit 0 = point-BSSRDF, bit 1 = volume march).
        # Defaults to both so the visual matches existing renders.
        self.scatter_modes: list[str] = [
            "BSSRDF + Volume",
            "BSSRDF only",
            "Volume only",
            "Off",
        ]
        self._scatter_mode_bits = [0b11, 0b01, 0b10, 0b00]
        self.scatter_index = 0
        self._last_scatter_index = -1  # forces first _upload_material_types

        # Integrator selector. Index 0 = existing unidirectional path tracer
        # (untouched). Index 1 = BDPT, which only engages when the camera's
        # first hit is a FlatMaterial; skin / debug-normal first hits silently
        # fall through to the path tracer in main_pass.slang. Index 2 = SPPM
        # (Stochastic Progressive Photon Mapping), wavefront-only and flat-material
        # only; under the megakernel it falls through to the path tracer.
        self.integrator_modes: list[str] = ["Path", "BDPT", "SPPM", "MLT"]
        self.integrator_index = 0

        # Pluggable scene-sampling seam (sampling/proposal.slang). The active
        # directional-proposal mixture + reuse mode are modeled as discrete
        # presets so the data-driven `_disc` UI surfaces them across every
        # front-end and the settings snapshot persists them — exactly like the
        # integrator selector. Each preset is (label, comma-separated proposal
        # tokens). `_active_proposals()` / `_active_reuse()` resolve the current
        # indices to plugin instances, folded into FrameConstants by
        # _pack_uniforms. Default index 0 ({bsdf}/none) is bit-identical to the
        # pre-seam renderer. Reuse has only the identity mode until ReSTIR lands.
        self._PROPOSAL_PRESETS: list[tuple[str, str]] = [
            ("BSDF",          "bsdf"),
            ("BSDF + Env",    "bsdf,env"),
            ("Env",           "env"),
            # Learned neural spline-flow proposal (bit2). Wavefront-only — the
            # megakernel capability-gate (in _pack_uniforms) strips the bit and
            # falls back to its analytic subset, like ReSTIR DI → identity.
            ("BSDF + Neural", "bsdf,neural"),
            # Neural-only guiding (no BSDF MIS partner). Wavefront-only; on the
            # megakernel the stripped mask is empty, so it folds back to {bsdf}
            # (see the empty-mask guard in _pack_uniforms).
            ("Neural",        "neural"),
        ]
        self.proposal_preset_modes: list[str] = [n for n, _ in self._PROPOSAL_PRESETS]
        self.proposal_preset_index = 0
        # Neural proposal (bit2) host state. `neural_network_version` is stamped
        # into FrameConstants (baseline 0 — the per-sample version hook for online
        # training). The GPU weight buffers + the WavefrontNeuralProposalPass are
        # owned renderer-side and built lazily in wavefront mode (mirrors the
        # ReSTIR pass); None until the neural preset is active. `_neural_warned`
        # latches the one-shot megakernel-unsupported notice.
        self._neural_network_version = 0
        self._neural_warned = False
        self._neural_pass = None
        # Online training (Stage 2, change neural-online-training). The replay
        # buffer + trainer + weight publisher are built lazily by
        # enable_online_training; `_neural_handoff_kind` selects the publisher
        # backend (--neural-handoff: 'file' | 'interop'). `_neural_trainer_kind`
        # selects the training-compute backend (--neural-trainer: 'cpu' numpy |
        # 'cuda' torch | 'mlx' Apple-MLX-on-Metal | 'auto') and `_train_precision` the optimizer
        # precision (--train-precision: 'fp32' | 'fp16'); both feed TrainerConfig
        # in enable_online_training (change neural-trainer-backends). Off until
        # enabled.
        self._neural_handoff_kind = str(neural_handoff)
        self._neural_trainer_kind = str(neural_trainer)
        self._train_precision = str(train_precision)
        self._online_training = False
        # Observability (change online-training-observability). The startup
        # configuration matrix is emitted from update() and dedup-reprinted on a
        # status flip; the front-ends set `_requested_backend` /
        # `_online_training_requested` for display (None ⇒ "auto" / current
        # state). `_train_summary_printed` guards the one-shot STOPPED summary
        # across the disable + atexit paths.
        self._requested_backend = None
        self._online_training_requested = None
        self._last_config_sig = None
        self._train_summary_printed = False
        self._atexit_registered = False
        self._neural_replay = None
        self._neural_trainer = None
        self._neural_publisher = None
        # Background trainer thread (change online-training-trigger): started by
        # enable_online_training, loops online_train_and_publish + a short sleep
        # so a slow cycle (numpy oracle ~seconds) never stalls the render thread.
        self._trainer_thread = None
        self._trainer_stop = None
        self._trainer_cadence_s = 0.05
        # Path-record source for the live online drain (change wavefront-native-
        # path-records): 'auto' picks the wavefront-native emitter when running
        # the wavefront path integrator (no megakernel dispatch, no 2 s-TDR), and
        # the megakernel `mainImageRecord` otherwise. 'megakernel'/'wavefront'
        # force a source. `_wf_record_active` is the resolved gate (set by
        # enable_online_training) that drives FrameConstants.recordMode + the
        # path pass's full-size record buffers.
        self._record_source = "auto"
        self._wf_record_active = False
        self._wf_record_capacity = 0
        # Per-scene baked weights file (NFW1). None ⇒ the renderer bakes a dummy
        # net for the 1a plumbing bring-up; a CLI/settings override threads here.
        self._neural_weights_path = None
        # Reuse modes: identity (stock NEE) + ReSTIR DI (wavefront-only;
        # falls back to identity on megakernel/Metal via the capability gate).
        self._REUSE_TOKENS: list[str] = ["none", "restir-di"]
        self.reuse_modes: list[str] = ["None", "ReSTIR DI"]
        self.reuse_index = 0
        # Per-lobe sampler selection for the flat/std_surface BSDF — runtime +
        # GUI + persisted, folded into FrameConstants.flatLobeSamplers by
        # _pack_uniforms and unpacked in flat_material.slang. Index 0 = native for
        # every lobe (the default; the all-native selection is bit-identical to the
        # pre-change renderer). Modes are data-driven from the registry so each
        # lobe offers only valid strategies (basis VNDF on coat/spec,
        # uniform-hemisphere on diffuse). Hashed into _current_state_hash so a
        # change resets accumulation.
        from skinny.sampling import (
            LOBE_COAT,
            LOBE_DIFFUSE,
            LOBE_SPEC,
            lobe_sampler_modes,
        )

        self.coat_sampler_modes: list[str] = lobe_sampler_modes(LOBE_COAT)
        self.spec_sampler_modes: list[str] = lobe_sampler_modes(LOBE_SPEC)
        self.diff_sampler_modes: list[str] = lobe_sampler_modes(LOBE_DIFFUSE)
        self.coat_sampler_index = 0
        self.spec_sampler_index = 0
        self.diff_sampler_index = 0
        # ReSTIR reuse regime (only meaningful when reuse = ReSTIR DI). Maps to
        # the RestirPC flags (bit0 spatial, bit1 temporal). Spatial reuse uses the
        # unbiased GRIS combination (converges to NEE). On the PROGRESSIVE
        # accumulator temporal reuse double-counts correlated history (it fights
        # the accumulator's own frame averaging) and biases glossy surfaces unless
        # M_cap is kept small — proper deep temporal needs the reprojected (P3)
        # regime. So "Spatial only" is the default; the temporal regimes are
        # selectable but progressive-limited. Surfaced via the data-driven _disc
        # UI + persisted.
        self.restir_regime_modes: list[str] = ["Spatial only", "Spatial + Temporal", "Temporal only"]
        self._RESTIR_REGIME_FLAGS = [0x1, 0x3, 0x2]
        self.restir_regime_index = 0
        # Biased ΣM combination: faster (skips the GRIS per-domain re-eval) but
        # biased; bounded on spatial-only, over-brightens with temporal on glossy.
        # Stored as an index (0/1) so the data-driven _disc selector drives it.
        self.restir_combination_modes: list[str] = ["Unbiased (GRIS)", "Biased (ΣM)"]
        self.restir_biased = 0
        # ReSTIR tuning (push-constant only — refreshed per frame, no pass
        # rebuild). Gated visible in the UI when ReSTIR DI is active; folded into
        # _current_state_hash so changes reset accumulation.
        self.restir_m_light = 8        # initial light-sampled candidates
        self.restir_m_bsdf = 1         # initial BSDF-sampled candidates
        self.restir_spatial_k = 5      # spatial neighbours
        self.restir_spatial_radius = 16.0   # screen px
        self.restir_m_cap = 20         # temporal history cap

        # Execution backend, orthogonal to the integrator and FIXED for the
        # session — selected on the command line (`--execution-mode`,
        # constructor arg), not a runtime GUI toggle. Megakernel (index 0) is
        # the single main_pass.slang compute dispatch (default); wavefront
        # (index 1) is the staged per-material backend on Vulkan AND native
        # Metal (change metal-wavefront-parity phase 3 — the Metal path
        # integrator runs through `metal_wavefront.MetalWavefrontPathPass`;
        # wavefront BDPT still pins Metal to the megakernel until phase 4).
        # The renderer compiles ONLY the selected backend
        # (see `_build_pipeline_for_current_graphs`).
        _wavefront_capable = hasattr(self.ctx, "compute_queue") or self.is_metal
        self.execution_modes: list[str] = (
            ["Megakernel", "Wavefront"] if _wavefront_capable else ["Megakernel"]
        )
        _mode_aliases = {"megakernel": EXECUTION_MEGAKERNEL, "wavefront": EXECUTION_WAVEFRONT}
        _requested = _mode_aliases.get(
            self._requested_execution_mode.strip().lower(), EXECUTION_MEGAKERNEL
        )
        # Clamp to the available modes (collapses wavefront → megakernel on a
        # non-Vulkan / Metal backend, which offers only ["Megakernel"]).
        self.execution_mode_index = clamp_mode_index(_requested, len(self.execution_modes))
        # Lazily built env-only wavefront pass (Phase-1 integration milestone),
        # constructed on first wavefront-mode dispatch. None until then / on
        # non-Vulkan backends.
        self._wavefront_env_pass = None
        # Test/debug override: when set, the wavefront gate dispatches this pass
        # instead of the staged path tracer (used to verify intermediate stage
        # kernels — e.g. primary visibility — against the live scene).
        self._wavefront_debug_pass = None
        # The real per-frame wavefront dispatch (staged path tracer) + its
        # per-lane path-state buffer. Built lazily; rebuilt when the megakernel
        # pipeline or the frame size changes (it reuses the megakernel set 0).
        self._wavefront_path_pass = None
        self._wf_path_state_buf = None
        self._wf_path_hit_buf = None
        self._wf_path_pass_dims = None
        # Staged wavefront SPPM (change photon-mapping-sppm): owns its own set-1
        # buffers, same lazy lifecycle as the path pass. Photon count for the
        # current frame is stashed by _pack_uniforms for record_dispatch.
        self._wavefront_sppm_pass = None
        self._wf_sppm_pass_dims = None
        self._sppm_photons_emitted = 0
        self._sppm_metal_photon_batch = 0  # set per-frame in _pack_uniforms (Metal SPPM tiling)
        # Staged wavefront MLT (change mlt-integrator): PSSMLT chains over the
        # BDPT estimator, Vulkan only (the Metal adapter is a follow-up —
        # _render_scene_metal refuses integrator 3 explicitly). The pass owns
        # the six chain buffers (bindings 52–57 of the wavefront scene set);
        # per accumulation reset the renderer runs the synchronous bootstrap
        # round-trip (_run_wavefront_mlt_bootstrap) before recording frames.
        self._wavefront_mlt_pass = None
        self._wf_mlt_pass_dims = None
        self.mlt_num_chains = 16384          # design D1 default (one lane = one chain)
        # Bootstrap budget per accumulation reset (design D3): interactive
        # default 8192 (quick reseed while dragging); parity / headless raise
        # it toward pbrt's 100000 via SKINNY_MLT_BOOTSTRAP or the attribute.
        import os as _os
        self.mlt_bootstrap_samples = int(
            _os.environ.get("SKINNY_MLT_BOOTSTRAP", "8192"))
        self.mlt_sigma = 0.01                # pbrt `sigma`
        self.mlt_large_step_prob = 0.3       # pbrt `largestepprobability`
        self.mlt_max_depth = 5               # pbrt `maxdepth`
        self._mlt_seed = 0                   # derived per accumulation reset
        # Staged wavefront bdpt (Phase 3): subpath-vertex + aux buffers + pass,
        # same lazy lifecycle as the path pass.
        self._wavefront_bdpt_pass = None
        self._wf_bdpt_eye_buf = None
        self._wf_bdpt_light_buf = None
        self._wf_bdpt_aux_buf = None
        self._wf_bdpt_pass_dims = None

        # Display exposure (EV stops) and tonemap operator applied at the
        # end of main_pass.slang after progressive accumulation. These are
        # post-process knobs so they do not invalidate the accumulation
        # buffer and are excluded from `_current_state_hash`.
        self.tonemap_modes: list[str] = ["ACES", "Reinhard", "Hable", "Linear"]
        self.tonemap_index = 0
        self.exposure = 0.0

        # pbrt film exposure controls (change pbrt-radiometric-parity). The imaging
        # ratio exposure_time·iso/100 is a live linear output scale: it multiplies
        # the linear-HDR readback (EXR/HDR save + the parity render_linear) and is
        # folded into the packed display exposure for the on-screen path. Set from
        # the authored camera by `_apply_camera_override`; retunable live via the
        # `film.iso` / `film.exposure_time` UI params (resets accumulation).
        self.film = FilmParameters()


        # Scalar applied to every sampleEnvironment() lookup. With many HDR
        # environments the raw luminance swamps skin albedo once multiplied
        # through the SSS estimator; this lets the user rebalance direct vs.
        # indirect contribution.
        self.env_intensity = 0.5
        # Now that env state is final, mirror it into the default dome prim
        # so the scene graph shows the current HDR + intensity.
        self._sync_default_dome_prim()

        # Furnace / energy-conservation probe. In this mode the shader swaps
        # the head for a unit sphere, clamps the environment to white (L=1)
        # in every direction, disables analytic direct light, and paints any
        # pixel whose accumulated radiance exceeds 1.0 per channel in a loud
        # pink — so energy violations are visible by eye. Exposed as a
        # discrete UI slider (On/Off) instead of a CLI flag so it can be
        # toggled during a session without restarting.
        self.furnace_modes: list[str] = ["Off", "On"]
        self.furnace_index: int = 0

        self.material_capacity = FLAT_MATERIAL_CAPACITY_INIT
        self._per_material_furnace: list[bool] = [False] * self.material_capacity

        # Scene-scale bridge between mm-valued skin params and world-unit
        # ray distances. 1 world unit = mm_per_unit millimetres. The SDF
        self.mm_per_unit = 1000.0

        # Heterogeneous-medium density grid state (nanovdb-volume-rendering).
        # `_volume_grid_key` identifies the uploaded grid ((asset, value_max);
        # None = the always-bound 1×1×1 zero fallback) and feeds
        # `_current_state_hash` so a grid swap resets accumulation.
        # `_volume_world_to_uvw` / `_volume_value_max` feed pack_flat_material.
        self._volume_grid_key: "Optional[tuple]" = None
        self._volume_world_to_uvw = None   # (3, 4) float32 or None (identity)
        self._volume_value_max: float = 1.0

        # Film per-sample radiance clamp (pbrt `maxcomponentvalue`, change
        # film-maxcomponent-clamp). 0 = disabled (no clamp; byte-identical render).
        # Set from the imported pbrt film by usd_loader; clamps each sample's
        # radiance proportionally before accumulation (FrameConstants.filmMaxComponent).
        self.film_max_component = 0.0

        import queue as _queue

        self.models: list[str] = []
        self._mesh_sources: list[MeshSource] = []
        self.model_index: int = -1

        # USD streaming state (populated by _load_usd_model / load_model_from_path)
        self._usd_instance_queue: _queue.Queue = _queue.Queue()
        self._usd_metadata_queue: _queue.Queue = _queue.Queue()
        self._usd_bake_done = None
        self._usd_uploaded_count: int = 0
        self._usd_model_index: int = -1
        self._use_usd_mtlx_plugin: bool = use_usd_mtlx_plugin

        if usd_scene_path is not None:
            self._load_usd_model(usd_scene_path)

        self._mesh_cache_index: dict = load_cache_index()
        # Geometry suballocator: per-mesh slabs over the shared vertex/index/BVH
        # buffers (stable offsets + free-list), so a USD add/remove touches only
        # the changed mesh instead of re-concatenating the whole scene. Keyed by
        # (prim_path, sub-index); content changes are detected via _slab_content_fp.
        # The legacy single-mesh OBJ path bypasses this and resets it (the OBJ
        # upload clobbers offset 0). Mode-independent — both backends read the
        # suballocated buffers.
        from skinny.slab_allocator import SlabAllocator
        self._slab_alloc = SlabAllocator()
        self._slab_content_fp: dict = {}
        # Rebake-tracking: each (source, displacement-scale) combination
        # produces a different GPU mesh, so we remember what we last baked and
        # rebuild when any input changes. -1 and NaN sentinels force an initial
        # bake on the first mesh selection.
        self._baked_source_idx: int = -1
        self._baked_scale_mm: float = float("nan")
        self._baked_scale_world: float = float("nan")
        self._baked_mm_per_unit: float = float("nan")
        self._baked_normals: bool = False      # tracks Mesh.normals_baked
        self._baked_normal_strength: float = float("nan")   # bake-time strength
        self._dirty_since: float | None = None      # monotonic wall-clock
        # Texture bytes cached per source index. Loading a 2K TIF/TGA takes
        # ~1 s; rebaking on slider drag would feel terrible without a cache.
        self._displacement_cache: dict[int, bytes | None] = {}
        self._normal_cache: dict[int, bytes | None] = {}

        # Tattoo library — procedural presets plus any PNG/JPG in tattoo_dir.
        # Index 0 ("None") is an all-zero-alpha image so "no tattoo" is just
        # another selection, no special-casing on the GPU side.
        self.tattoos: list[Tattoo] = load_tattoos(tattoo_dir)
        self.tattoo_index = 0
        self._last_tattoo_index = -1
        self.tattoo_density = 1.0

        # Per-model detail maps (normal / roughness / displacement). When the
        # active head model has a corresponding texture file, it's uploaded
        # into these three SampledImages on mesh switch. The per-map
        # availability flags feed the UBO so the shader only reads a map
        # when it's actually meaningful — and the enable toggle below lets
        # the user fall back to the slider values at will.
        self.detail_maps_modes: list[str] = ["On", "Off"]
        self.detail_maps_index = 0          # 0 = maps on, 1 = use sliders
        self.normal_map_strength = 1.0      # multiplies tangent-space XY offset
        # Default displacement ≈ 1 mm peak so a model shipping with a disp map
        # actually gets displaced on first load. Models without a map are
        # unaffected — bake_mesh skips the offset step when bytes are absent.
        self.displacement_scale_mm = 1.0    # mm offset at (disp - 0.5); 0 = off
        self._detail_available = (False, False, False)  # (normal, rough, disp)

        # Phase B-1: keep a CPU-side `Scene` that summarizes the renderer's
        # current selection (env, mesh, materials, lights). Today's UI
        # state still owns the source of truth — model_index, env_index,
        # skin sliders, etc. — but each update() materializes a Scene off
        # those fields and the GPU-upload paths read from it. The Scene
        # is the seam Phase B-3 will replace with TLAS-driven multi-mesh
        # state and Phase C will populate with MaterialX-driven materials.
        self.scene: Scene = Scene()

        # USD scene is loaded in the background; starts as None.
        # Metadata (lights/camera/mm_per_unit) arrives via _usd_metadata_queue
        # and is applied in _poll_usd_streaming(). Mesh instances stream in
        # via _usd_instance_queue.
        self._usd_scene: Scene | None = None
        self._scene_graph: object | None = None
        self._last_projected_default_lights: bool | None = None
        self._last_aux_light_authority_token: tuple | None = None
        # Bumped whenever the scene graph is (re)built so the UI panels, which
        # poll it, repaint. Always defined so observers can read it pre-edit.
        self._scene_graph_version = 0

        # Load the MaterialX library and generate Slang for the canonical
        # skin material. The CompiledMaterial drives the per-material
        # MtlxSkinParams buffer (binding 15) that the shader reads via
        # skinParamsFromMtlx().
        self._mtlx_library: object | None = None
        self._mtlx_skin_material: object | None = None
        # MaterialX nodegraph fragments built for this scene's materials.
        # `_scene_graph_fragments` is the distinct fragment list (passed to
        # ComputePipeline so it sizes descriptor bindings). `_material_graph_ids`
        # maps material slot index → graphId (0 ⇒ no graph) for materialTypes
        # upper-byte encoding. `_material_graph_overrides` carries per-material
        # uniform overrides that pack_uniform_block packs into the per-graph
        # SSBO at the material's slot.
        self._scene_graph_fragments: list = []
        self._material_graph_ids: dict[int, int] = {}
        self._material_graph_overrides: dict[int, dict] = {}
        # Directory holding session-synthesized `.mtlx` documents added via
        # `add_material` (mcp-material-authoring, design D2/D7). Set the first
        # time a synthesized material is added; used at save time to tell a
        # session document (copy into the saved bundle) from a curated preset
        # (keep an absolute assets reference — the texture carve-out).
        self._material_session_dir: "str | None" = None
        # The single combined graph-param buffer (all graphs share one
        # matId-major byte buffer at GRAPH_BINDING_BASE — change
        # combine-graph-param-buffers). None until the first graph upload.
        self._graph_params_combined = None
        # Signature (target_name, slang-content-hash) per fragment in the
        # currently-built pipeline. _gen_scene_materials compares against
        # `_graph_set_signature()` to decide whether
        # `_build_pipeline_for_current_graphs` needs to run.
        self._pipeline_built_for_targets: tuple = ()
        # Pipeline + descriptors are built lazily (see _init_gpu docstring),
        # but seed the attributes now so the trigger check in
        # `_gen_scene_materials` (which can fire from `_init_materialx_runtime`
        # before `_init_gpu` runs) reads them safely.
        self.pipeline = None
        self.descriptor_pool = None
        self.descriptor_sets = None
        # Material-graph editor preview pipeline / image / readback. All
        # three are created on first call to `render_material_preview` and
        # torn down + rebuilt whenever the main pipeline rebuilds (because
        # the preview shares descriptor set 0 with the main pipeline's
        # layout — see PreviewPipeline).
        self._preview_pipeline = None
        self._preview_image = None
        self._preview_readback = None
        self._preview_size = 0
        # MaterialX field overrides keyed by uniform field name
        # (e.g. "layer_top_melanin"). Seeded from SkinParameters defaults;
        # all skin sliders now write here directly via mtlx.* paths.
        self.mtlx_overrides: dict[str, object] = {}
        self._init_materialx_runtime()
        self.mtlx_overrides.update(self._mtlx_skin_overrides())

        self._init_gpu()

        # USD meshes are uploaded as they arrive via _poll_usd_streaming().
        # No blocking upload here — scene starts empty.

    @property
    def camera(self):
        if self.camera_mode == "usd":
            return self.usd_camera
        return self.orbit_camera if self.camera_mode == "orbit" else self.free_camera

    @property
    def has_usd_camera(self) -> bool:
        """True when the loaded USD scene exposes a camera to follow."""
        scene = self._usd_scene
        return scene is not None and scene.camera_override is not None

    def reset_camera(self) -> None:
        """Snap both cameras back to a known-good frame on the head.

        Re-applies the active scene's camera override afterwards so the
        authored thick lens / focus distance / fstop are not lost when
        the user hits F.
        """
        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        self.camera_mode = "orbit"
        if self._usd_scene is not None:
            # Re-frame the loaded scene; honors an authored camera override
            # internally and otherwise applies the hero-angle auto-frame.
            self._frame_camera_to_scene(self._usd_scene)
        elif self._mesh_sources:
            self._frame_camera_to_mesh(self._mesh_sources[0])
        self._refresh_camera_node()

    def toggle_camera_mode(self) -> None:
        """Flip between orbit and free while preserving the current viewpoint."""
        if self.camera_mode == "orbit":
            # Orbit -> Free: match position and look direction.
            o = self.orbit_camera
            self.free_camera.position = o.position.astype(np.float32).copy()
            self.free_camera.yaw = -o.yaw
            self.free_camera.pitch = -o.pitch
            self.camera_mode = "free"
        else:
            # Free -> Orbit: pivot around the head's visual centre.
            head_centre = np.array([0.0, 0.1, 0.05], dtype=np.float32)
            pos = self.free_camera.position.astype(np.float32)
            offset = pos - head_centre
            dist = float(max(np.linalg.norm(offset), 0.5))
            self.orbit_camera.target = head_centre
            self.orbit_camera.set_distance(dist)
            self.orbit_camera.pitch = float(np.arcsin(offset[1] / dist))
            self.orbit_camera.yaw = float(np.arctan2(offset[0], offset[2]))
            self.camera_mode = "orbit"
        self._refresh_camera_node()

    # ── Execution backend (megakernel | wavefront) ──────────────────
    # The mode is fixed at construction (CLI `--execution-mode`), so there is
    # no runtime setter / cycler and it is excluded from `_current_state_hash`.

    @property
    def _scene_set0_layout(self):
        """Set-0 descriptor-set layout for the active backend, or None before
        a scene is loaded. Owned by `self._scene_bindings` (which is the
        megakernel `ComputePipeline` in megakernel mode, or a standalone
        `scene_bindings_only` build in wavefront mode)."""
        return self._scene_bindings.descriptor_set_layout if self._scene_bindings else None

    @property
    def _msl_layout_source(self):
        """Object owning the reflected MSL layouts on Metal (``uniform_layout``,
        ``graph_param_layouts``, ``std_surface_layout``, ``mtlx_skin_layout``):
        the megakernel pipeline in megakernel mode, else the lazily-built Metal
        wavefront pass — path, or bdpt when the bidirectional integrator is
        the only compiled program (both reflect the same surface from their
        own programs). ``None`` before any exists — the MSL relocators then
        fall through to their scalar defaults.

        The Metal MLT pass comes FIRST (change mlt-integrator) when it is the
        ACTIVE consumer: it is the only program compiled with ``SKINNY_MLT``,
        so it is the only one whose reflected ``fc`` carries the MLT tail that
        ``_pack_uniforms`` emits — and the two must agree or the drift guard in
        ``_pack_uniforms_msl`` fires. Gating on `_mlt_uniform_tail_active()`
        (not merely "the pass exists") is what makes runtime integrator cycling
        safe: after an MLT frame the pass stays cached, but switching to
        path/BDPT/SPPM — or a megakernel-fallback MLT selection — must fall
        through to the ordinary layout source, whose `fc` has no tail (codex
        pre-merge review)."""
        if self._mlt_uniform_tail_active():
            return self._wavefront_mlt_pass
        if self.pipeline is not None:
            return self.pipeline
        if self._wavefront_sppm_pass is not None:
            return self._wavefront_sppm_pass
        if self._wavefront_path_pass is not None:
            return self._wavefront_path_pass
        return self._wavefront_bdpt_pass

    @property
    def _scene_graph_bindings(self) -> dict:
        """MaterialX nodegraph → descriptor-binding map for the active backend,
        independent of which backend is compiled."""
        if self._scene_bindings is None:
            return {}
        return getattr(self._scene_bindings, "graph_bindings", {}) or {}

    @property
    def effective_execution_mode_index(self) -> int:
        """Execution mode actually used to render this frame, after capability
        gating. Wavefront + BDPT runs the staged bidirectional pipeline on both
        backends (Vulkan since wavefront phase 3; Metal since
        metal-wavefront-parity phase 4)."""
        return effective_execution_mode(
            self.execution_mode_index,
            self.integrator_index,
            self.WAVEFRONT_BDPT_SUPPORTED,
        )

    @property
    def execution_mode_fallback_active(self) -> bool:
        """True when the capability gate is overriding the selected execution
        mode (front-ends surface this to the user)."""
        return self.effective_execution_mode_index != self.execution_mode_index

    @property
    def _backend_render_ready(self) -> bool:
        """True once the selected backend can dispatch a frame: the per-frame
        scene descriptor sets exist and the compiled backend is present. In
        megakernel mode that means the megakernel pipeline; in wavefront mode
        the scene bindings (the staged stage pipelines build lazily)."""
        if self.is_metal:
            # The Metal megakernel binds resources at dispatch (no Vulkan
            # descriptor sets); readiness is the compiled pipeline. In wavefront
            # mode no megakernel is compiled (`scene_bindings_only`) — readiness
            # is the scene bindings; the stage pipelines build lazily.
            if self.effective_execution_mode_index == EXECUTION_WAVEFRONT:
                return self._scene_bindings is not None
            return self.pipeline is not None
        if self.descriptor_sets is None:
            return False
        if self.effective_execution_mode_index == EXECUTION_MEGAKERNEL:
            return self.pipeline is not None
        return self._scene_bindings is not None

    def _ensure_wavefront_env_pass(self):
        """Build (once) the env-only wavefront pass. Returns it, or None on a
        non-Vulkan backend. Phase-1 integration milestone — superseded by the
        staged pipeline."""
        if not hasattr(self.ctx, "compute_queue"):
            return None
        if self._wavefront_env_pass is None:
            from skinny.vk_wavefront import WavefrontEnvPass
            self._wavefront_env_pass = WavefrontEnvPass(
                self.ctx, self.shader_dir,
                uniform_buffer=self.uniform_buffer.buffer,
                uniform_size=self.uniform_size,
                accum_view=self.accum_image.view,
                env_view=self.env_image.view,
                env_sampler=self.env_image.sampler,
                width=self.width, height=self.height,
            )
        return self._wavefront_env_pass

    def _destroy_wavefront_env_pass(self) -> None:
        if self._wavefront_env_pass is not None:
            self._wavefront_env_pass.destroy()
            self._wavefront_env_pass = None
        if self._wavefront_debug_pass is not None:
            self._wavefront_debug_pass.destroy()
            self._wavefront_debug_pass = None
        self._destroy_wavefront_path_pass()
        self._destroy_wavefront_bdpt_pass()

    def _restir_build_config(self) -> dict:
        """ReSTIR config for the wavefront pass: the active reuse plugin's tuning
        with `flags` set from the selected regime. A `_restir_config` override
        (tests) wins entirely."""
        if getattr(self, "_restir_config", None):
            return self._restir_config
        cfg = dict(getattr(self._active_reuse(), "config", None) or {})
        idx = max(0, min(int(self.restir_regime_index), len(self._RESTIR_REGIME_FLAGS) - 1))
        cfg["flags"] = self._RESTIR_REGIME_FLAGS[idx]
        if getattr(self, "restir_biased", False):
            cfg["flags"] |= 0x4          # RESTIR_FLAG_BIASED — faster ΣM combination
        # Tuning (push-constant only) from the UI-driven renderer attrs.
        cfg["mLight"] = max(1, int(self.restir_m_light))
        cfg["mBsdf"] = max(0, int(self.restir_m_bsdf))
        cfg["spatialK"] = max(0, int(self.restir_spatial_k))
        cfg["spatialRadius"] = max(1.0, float(self.restir_spatial_radius))
        cfg["mCap"] = max(1, int(self.restir_m_cap))
        return cfg

    def _ensure_wavefront_path_pass(self):
        """Build (once) the staged wavefront path tracer — the real per-frame
        wavefront dispatch. Returns it, or None on a non-Vulkan backend or
        before the scene bindings exist (it reuses the scene-bindings set-0
        layout + the renderer's per-frame scene descriptor sets). Rebuilt by
        `_destroy_wavefront_path_pass` when the layout or frame size changes."""
        if self.is_metal:
            return self._ensure_wavefront_path_pass_metal()
        if not hasattr(self.ctx, "compute_queue"):
            return None
        if self._scene_bindings is None or self.descriptor_sets is None:
            return None
        # Build the heavy catch-all shade kernel only when the scene has a
        # non-flat material (skin/python); flat-only scenes compile just the
        # small flat shade kernel. Part of the rebuild key so a material-set
        # change that introduces a non-flat type rebuilds the pass.
        has_nonflat = any(int(t) != MATERIAL_TYPE_FLAT for t in self._material_types)
        # Reuse mode is part of the key so switching none↔ReSTIR rebuilds the
        # pass (and its ReSTIR sub-pass) — the seam's pass-structural contract.
        reuse_mode = int(self._active_reuse().reuse_mode)
        # Rebuild when the reuse mode or the ReSTIR regime/config changes.
        _rcfg = getattr(self, "_restir_config", None)
        # Record mode is part of the key so enabling/disabling the wavefront
        # record drain rebuilds the pass with full-size vs dummy record buffers.
        wf_record = bool(getattr(self, "_wf_record_active", False))
        key = (self.width, self.height, has_nonflat, reuse_mode,
               int(self.restir_regime_index) if reuse_mode == 1 else None,
               tuple(sorted(_rcfg.items())) if _rcfg else None,
               self._neural_active(), wf_record)
        if self._wavefront_path_pass is not None and self._wf_path_pass_dims == key:
            # Live ReSTIR tuning: refresh the push-constant config each frame so
            # slider changes (mLight/mBsdf/k/radius/mCap/biased) take effect
            # without a pass rebuild (recompile). Pass structure is unchanged.
            if self._restir_pass is not None:
                self._restir_pass.config = self._restir_build_config()
            return self._wavefront_path_pass
        self._destroy_wavefront_path_pass()
        from skinny.vk_wavefront import WavefrontPathPass
        from skinny.wavefront_layout import path_state_size
        # Tiled streaming: the path-state buffer holds a fixed-size stream
        # (capped at STREAM_CAP), not one slot per pixel — so VRAM does not grow
        # with resolution. The frame is processed in ceil(num_pixels/stream)
        # tiles. Allow a renderer override (`_wf_stream_cap`) for tests.
        num_pixels = self.width * self.height
        cap = int(getattr(self, "_wf_stream_cap", None) or WavefrontPathPass.STREAM_CAP)
        stream_size = max(1, min(num_pixels, cap))
        # Spectral path state is wider (Spectrum throughput/radiance +
        # SampledWavelengths); RGB (spectral=False) keeps the 68 B scalar stride.
        path_state_stride = path_state_size(spectral=self._spectral)
        self._wf_path_state_buf = self._gpu.StorageBuffer(self.ctx, stream_size * path_state_stride)
        self._wf_path_hit_buf = self._gpu.StorageBuffer(
            self.ctx, stream_size * WavefrontPathPass.HIT_STRIDE)
        self._wavefront_path_pass = WavefrontPathPass(
            self.ctx, self.shader_dir, self._scene_set0_layout,
            self._wf_path_state_buf.buffer, self._wf_path_state_buf.size,
            self._wf_path_hit_buf.buffer, self._wf_path_hit_buf.size,
            stream_size, num_pixels, build_catchall=has_nonflat,
            record_capacity=(stream_size if wf_record else 0),
            neural_config=self._effective_neural_config(),
            spectral=self._spectral,
        )
        # ReSTIR DI reuse plugin: build the primary-direct pass over the same
        # path-state + hit buffers and hook it at bounce 0. We are in wavefront
        # here, so constructing it only on reuse_mode == RESTIR_DI IS the
        # capability gate (megakernel/Metal never reach this builder).
        self._restir_pass = None
        if reuse_mode == 1:  # RESTIR_DI
            from skinny.vk_wavefront import RestirDiPass
            self._restir_pass = RestirDiPass(
                self.ctx, self.shader_dir, self._scene_set0_layout,
                self._wf_path_state_buf.buffer, self._wf_path_state_buf.size,
                self._wf_path_hit_buf.buffer, self._wf_path_hit_buf.size,
                stream_size, config=self._restir_build_config(),
            )
            self._wavefront_path_pass.set_restir(self._restir_pass)
        # Neural directional proposal (bit2): build the wavefront pre-pass over
        # the same path-state + hit buffers + the path pass's per-lane neural
        # buffer, and hook it every bounce. Constructing it only when neural is
        # active (always wavefront here) IS the capability gate.
        self._neural_pass = None
        if self._neural_active():
            self._sync_neural_weights()
            from skinny.vk_wavefront import WavefrontNeuralProposalPass
            self._neural_pass = WavefrontNeuralProposalPass(
                self.ctx, self.shader_dir, self._scene_set0_layout,
                self._wf_path_state_buf.buffer, self._wf_path_state_buf.size,
                self._wf_path_hit_buf.buffer, self._wf_path_hit_buf.size,
                self._wavefront_path_pass.neural_buf.buffer,
                self._wavefront_path_pass.neural_buf.size,
                stream_size, network_version=self._neural_network_version,
                neural_config=self._effective_neural_config(),
            )
            self._wavefront_path_pass.set_neural(self._neural_pass)
        self._wf_path_pass_dims = key
        return self._wavefront_path_pass

    def _ensure_wavefront_sppm_pass(self):
        """Build (once) the staged wavefront SPPM pass (change
        photon-mapping-sppm). Vulkan only for now; the native-Metal SPPM pass is
        a follow-up, so this returns None on Metal and the caller falls back to
        the path tracer. Reuses the megakernel scene set-0 layout + the
        per-frame scene descriptor sets like the path pass."""
        if self.is_metal:
            return self._ensure_wavefront_sppm_pass_metal()
        if not hasattr(self.ctx, "compute_queue"):
            return None
        if self._scene_bindings is None or self.descriptor_sets is None:
            return None
        key = (self.width, self.height)
        if self._wavefront_sppm_pass is not None and self._wf_sppm_pass_dims == key:
            return self._wavefront_sppm_pass
        self._destroy_wavefront_sppm_pass()
        from skinny.vk_wavefront import WavefrontSppmPass
        num_pixels = self.width * self.height
        cap = int(getattr(self, "_wf_stream_cap", None) or WavefrontSppmPass.STREAM_CAP)
        stream_size = max(1, min(num_pixels, cap))
        self._wavefront_sppm_pass = WavefrontSppmPass(
            self.ctx, self.shader_dir, self._scene_set0_layout, stream_size, num_pixels,
            spectral=self._spectral)
        self._wf_sppm_pass_dims = key
        return self._wavefront_sppm_pass

    def _ensure_wavefront_sppm_pass_metal(self):
        """Build (once) the native-Metal staged SPPM pass (change
        photon-mapping-sppm). Returns it, or None before the scene bindings
        exist (caller falls back to the path tracer). Mirrors
        `_ensure_wavefront_path_pass_metal`; the graph-set signature is in the
        rebuild key so a material-set change recompiles the pass."""
        if self._scene_bindings is None:
            return None
        heavy = self._has_heavy_nonflat()
        key = (self.width, self.height, self._graph_set_signature(), heavy)
        if self._wavefront_sppm_pass is not None and self._wf_sppm_pass_dims == key:
            return self._wavefront_sppm_pass
        self._destroy_wavefront_sppm_pass()
        from skinny.metal_wavefront import MetalWavefrontSppmPass
        num_pixels = self.width * self.height
        cap = int(getattr(self, "_wf_stream_cap", None) or MetalWavefrontSppmPass.STREAM_CAP)
        if heavy:  # bound the heavy per-tile eye submit (see the band constant)
            cap = min(cap, _METAL_WAVEFRONT_HEAVY_EYE_BAND_LANES)
        stream_size = max(1, min(num_pixels, cap))
        self._wavefront_sppm_pass = MetalWavefrontSppmPass(
            self.ctx, self.shader_dir, stream_size, num_pixels,
            graph_fragments=list(self._scene_graph_fragments),
            neural_config=self._effective_neural_config(),
            spectral=self._spectral)
        self._wf_sppm_pass_dims = key
        return self._wavefront_sppm_pass

    def _destroy_wavefront_sppm_pass(self):
        if self._wavefront_sppm_pass is not None:
            self._wavefront_sppm_pass.destroy()
            self._wavefront_sppm_pass = None
        self._wf_sppm_pass_dims = None

    def _mlt_pass_key(self):
        return (self.width, self.height,
                int(self.mlt_num_chains), int(self.mlt_bootstrap_samples))

    def _mlt_uniform_tail_active(self) -> bool:
        """Whether ``_pack_uniforms`` must emit the ``#if defined(SKINNY_MLT)``
        FrameConstants tail — i.e. the dispatched shader's ``fc`` actually has
        those fields (codex pre-merge review).

        Vulkan uses one oversized shared UBO, so appending the tail whenever
        MLT is the integrator is harmless — only the MLT ``.spv`` reads the
        offsets. Metal packs the blob per-dispatch and the drift guard asserts
        the blob length equals the reflected ``fc`` size, so the tail is packed
        ONLY when the Metal MLT wavefront pass is the real consumer: integrator
        3, wavefront mode, and the pass built. A megakernel-fallback MLT
        selection (execution mode != wavefront) or any non-MLT integrator gets
        the base layout, no tail — otherwise runtime integrator cycling crashes
        uniform packing."""
        if self.integrator_index != 3:
            return False
        if not self.is_metal:
            return True
        return (self.effective_execution_mode_index == EXECUTION_WAVEFRONT
                and self._wavefront_mlt_pass is not None)

    def _next_mlt_seed(self) -> int:
        """Per-reset MLT replay seed (design D3): stable across an accumulation
        run, decorrelated between consecutive resets, and REPRODUCIBLE ACROSS
        PROCESSES — the parity gate re-renders in a fresh interpreter and must
        get the same chains (design D6's deterministic budget mapping).

        Deliberately NOT derived from `_current_state_hash()`. That hash exists
        for change detection, where only equality *within* one process matters,
        and it hashes tuples containing str (`state_signature()` leads with
        "orbit"/"free") — so PYTHONHASHSEED randomizes it per process. Seeding
        MLT from it made every render irreproducible: the same scene scored
        self-consistency relMSE 0.17 / 0.25 / 1.10 across three runs, which is
        pass-or-fail by luck. `frame_index` alone already decorrelates resets
        (it advances between them) and is deterministic in a headless render,
        so it is both necessary and sufficient here.
        """
        # frame_index is a monotonic counter and mltSeed is a u32 shader field,
        # so mask to 32 bits — a signed "<i" pack raises struct.error past 2**31
        # (codex pre-merge review).
        return zlib.crc32(
            struct.pack("<I", int(self.frame_index) & 0xFFFFFFFF)) & 0xFFFFFFFF

    def _ensure_wavefront_mlt_pass_metal(self):
        """Build (once) the native-Metal staged MLT pass (change
        mlt-integrator, task 5.6) — the Metal sibling of
        `_ensure_wavefront_mlt_pass`. Metal binds by name, so there are no
        scene-set slots 52–57 to rebind: the pass merges its chain buffers into
        the per-dispatch bind map itself."""
        key = self._mlt_pass_key()
        if self._wavefront_mlt_pass is not None and self._wf_mlt_pass_dims == key:
            return self._wavefront_mlt_pass
        self._destroy_wavefront_mlt_pass()
        from skinny.metal_wavefront import MetalWavefrontMltPass
        self._wavefront_mlt_pass = MetalWavefrontMltPass(
            self.ctx, self.shader_dir,
            num_pixels=self.width * self.height,
            num_chains=int(self.mlt_num_chains),
            bootstrap_samples=int(self.mlt_bootstrap_samples),
            spectral=self._spectral)
        self._wf_mlt_pass_dims = key
        return self._wavefront_mlt_pass

    def _ensure_wavefront_mlt_pass(self):
        """Build (once) the staged wavefront MLT pass (change mlt-integrator).
        Vulkan path — the Metal sibling is `_ensure_wavefront_mlt_pass_metal`
        (`_render_scene_metal` routes there). Returns None when the scene set-0
        layout lacks the MLT bindings 52–57 (a megakernel-mode session — the
        `scene_bindings_only` wavefront layout always carries them); the
        caller then falls back to the path tracer like SPPM does."""
        if self.is_metal:
            return None
        if not hasattr(self.ctx, "compute_queue"):
            return None
        if self._scene_bindings is None or self.descriptor_sets is None:
            return None
        if not getattr(self._scene_bindings, "mlt_bindings", False):
            return None
        key = self._mlt_pass_key()
        if self._wavefront_mlt_pass is not None and self._wf_mlt_pass_dims == key:
            return self._wavefront_mlt_pass
        self._destroy_wavefront_mlt_pass()
        from skinny.vk_wavefront import WavefrontMltPass
        self._wavefront_mlt_pass = WavefrontMltPass(
            self.ctx, self.shader_dir, self._scene_set0_layout,
            num_pixels=self.width * self.height,
            num_chains=int(self.mlt_num_chains),
            bootstrap_samples=int(self.mlt_bootstrap_samples),
            spectral=self._spectral)
        # Rebind the scene descriptor sets' MLT slots (52–57) from the
        # creation-time dummies to this pass's chain buffers.
        for ds in self.descriptor_sets:
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vk.VkDescriptorBufferInfo(
                        buffer=buf.buffer, offset=0, range=buf.size)],
                )
                for b, buf in self._wavefront_mlt_pass.descriptor_bindings
            ]
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)
        self._wf_mlt_pass_dims = key
        return self._wavefront_mlt_pass

    def _destroy_wavefront_mlt_pass(self):
        if self._wavefront_mlt_pass is not None:
            self._wavefront_mlt_pass.destroy()
            self._wavefront_mlt_pass = None
        self._wf_mlt_pass_dims = None

    def _mlt_iterations_per_frame(self) -> int:
        """Mutation iterations per accumulation frame: ~1 mutation/pixel/frame
        (`mpp_actual = iterations × nChains / pixels` is packed into the MLT
        uniform tail so the resolve divides by the ACTUAL budget, design D4)."""
        pixels = max(1, self.width * self.height)
        return max(1, round(pixels / max(1, int(self.mlt_num_chains))))

    def _submit_one_shot_compute(self, record_fn) -> None:
        """Allocate, record, submit, and wait one command buffer on the compute
        queue — the synchronous seam for one-shot GPU work (the MLT bootstrap +
        chain-init phases). Mirrors StorageBuffer.upload_sync's submit."""
        cmd = vk.vkAllocateCommandBuffers(
            self.ctx.device, vk.VkCommandBufferAllocateInfo(
                commandPool=self.ctx.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1))[0]
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        record_fn(cmd)
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def _run_wavefront_mlt_bootstrap(self, mlt, scene_set) -> None:
        """Synchronous MLT (re)seed at an accumulation reset (design D3):
        bootstrap dispatch → weight readback → host CDF resample (b + chain
        seeds) → seed upload → chain-init dispatch. Runs like other one-shot
        GPU work — its own submits, awaited before the frame records."""
        from skinny.mlt_bootstrap import resample_chain_seeds

        self._mlt_seed = self._next_mlt_seed()
        mlt.b = 0.0
        mlt.seeded = False
        # The bootstrap/init kernels read fc.mltSeed — re-upload before they run.
        self.uniform_buffer.upload(self._pack_uniforms())
        self._submit_one_shot_compute(lambda c: mlt.record_bootstrap(c, scene_set))
        weights = mlt.read_bootstrap_weights()
        b, seeds = resample_chain_seeds(weights, mlt.num_chains, self._mlt_seed)
        mlt.upload_chain_seeds(seeds)
        self._submit_one_shot_compute(lambda c: mlt.record_init(c, scene_set))
        mlt.b = b
        mlt.seeded = True
        # The frame's resolve reads fc.mltB — re-upload now that b is known
        # (the frame command buffer has not been submitted yet).
        self.uniform_buffer.upload(self._pack_uniforms())

    def _record_wavefront_dispatch(self, cmd, scene_set):
        """Record the active wavefront integrator's dispatch into ``cmd`` —
        shared by the windowed + headless Vulkan seams. SPPM (integrator 2) needs
        the per-frame photon count + first-frame flag; MLT (integrator 3) runs
        the synchronous bootstrap round-trip at an accumulation reset before
        recording its frame; path/bdpt take the scene set alone. SPPM and MLT
        fall back to the path tracer when their pass is unbuildable (e.g. a
        megakernel-mode session's layout, not yet wired)."""
        if self.integrator_index == 3:  # INTEGRATOR_MLT
            mlt = self._ensure_wavefront_mlt_pass()
            if mlt is not None:
                if self.accum_frame == 0 or not mlt.seeded:
                    self._run_wavefront_mlt_bootstrap(mlt, scene_set)
                mlt.record_frame(
                    cmd, scene_set, iterations=self._mlt_iterations_per_frame())
                return
        if self.integrator_index == 2:  # INTEGRATOR_SPPM
            sppm = self._ensure_wavefront_sppm_pass()
            if sppm is not None:
                sppm.record_dispatch(
                    cmd, scene_set, photons=self._sppm_photons_emitted,
                    first_frame=(self.accum_frame == 0))
                return
        if self.integrator_index == 1:
            staged = self._ensure_wavefront_bdpt_pass()
        else:
            staged = self._ensure_wavefront_path_pass()
        if staged is not None:
            staged.record_dispatch(cmd, scene_set)
        else:
            self._ensure_wavefront_env_pass().record_dispatch(cmd)

    def _ensure_wavefront_path_pass_metal(self):
        """Build (once) the Metal staged wavefront path tracer (change
        metal-wavefront-parity phase 3). Returns it, or None before the scene
        bindings exist. The pass owns its path-state/hit/queue buffers, sized
        from the reflected MSL strides; the ReSTIR DI reuse pass (phase 5)
        attaches through the same bounce-0 hook as Vulkan, so the reuse mode
        and config are part of the rebuild key. The neural pre-pass (phase 6)
        attaches through the every-bounce hook; neural is in the key because
        selecting it recompiles the shade kernels with SKINNY_METAL_NEURAL=1
        (the weight buffers are slot-cap stubs otherwise)."""
        if self._scene_bindings is None:
            return None
        has_nonflat = any(int(t) != MATERIAL_TYPE_FLAT for t in self._material_types)
        # Reuse mode is part of the key so switching none↔ReSTIR rebuilds the
        # pass (and its ReSTIR sub-pass) — the seam's pass-structural contract.
        reuse_mode = int(self._active_reuse().reuse_mode)
        _rcfg = getattr(self, "_restir_config", None)
        # Record mode is part of the key so arming/disarming the wavefront
        # record drain rebuilds the pass with SKINNY_METAL_RECORDS on/off
        # (change metal-record-drain) — mirroring the Vulkan builder's wf_record
        # key entry.
        wf_record = bool(getattr(self, "_wf_record_active", False))
        key = (self.width, self.height, has_nonflat,
               self._graph_set_signature(), reuse_mode,
               int(self.restir_regime_index) if reuse_mode == 1 else None,
               tuple(sorted(_rcfg.items())) if _rcfg else None,
               self._neural_active(), wf_record)
        if self._wavefront_path_pass is not None and self._wf_path_pass_dims == key:
            # Live ReSTIR tuning: refresh the config blob each frame so slider
            # changes (mLight/mBsdf/k/radius/mCap/biased) take effect without a
            # pass rebuild (recompile). Pass structure is unchanged.
            if self._restir_pass is not None:
                self._restir_pass.config = self._restir_build_config()
            return self._wavefront_path_pass
        self._destroy_wavefront_path_pass()
        from skinny.metal_wavefront import (
            MetalNeuralProposalPass,
            MetalRestirDiPass,
            MetalWavefrontPathPass,
        )
        num_pixels = self.width * self.height
        cap = int(getattr(self, "_wf_stream_cap", None)
                  or MetalWavefrontPathPass.STREAM_CAP)
        stream_size = max(1, min(num_pixels, cap))
        self._wavefront_path_pass = MetalWavefrontPathPass(
            self.ctx, self.shader_dir, stream_size, num_pixels,
            build_catchall=has_nonflat,
            record_capacity=(stream_size if wf_record else 0),
            graph_fragments=list(self._scene_graph_fragments),
            neural_config=self._effective_neural_config(),
            neural_active=self._neural_active(),
            records_active=wf_record,
            spectral=self._spectral,
        )
        # ReSTIR DI reuse plugin (phase 5): the pass owns the persistent
        # reservoir/G-buffer StorageBuffers and binds the path pass's
        # wfState/wfHits by name at dispatch. Constructing it only on
        # reuse_mode == RESTIR_DI IS the capability gate (the megakernel on
        # either device folds reuseMode to 0 — identity reuse).
        self._restir_pass = None
        if reuse_mode == 1:  # RESTIR_DI
            self._restir_pass = MetalRestirDiPass(
                self.ctx, self.shader_dir, stream_size,
                config=self._restir_build_config(),
            )
            self._wavefront_path_pass.set_restir(self._restir_pass)
        # Neural directional proposal (bit2, phase 6): upload the frozen
        # weights into the renderer's backend-neutral 33/34/35 buffers
        # (`set_data` on Metal — no external-memory interop, design D6) and
        # hook the forward pre-pass every bounce, mirroring the Vulkan
        # builder. Constructing it only when neural is active (always
        # wavefront here) IS the capability gate; deselection rebuilds the
        # pass (key change) and `_destroy_wavefront_path_pass` releases it.
        self._neural_pass = None
        if self._neural_active():
            self._sync_neural_weights()
            self._neural_pass = MetalNeuralProposalPass(
                self.ctx, self.shader_dir, self._wavefront_path_pass,
                stream_size, network_version=self._neural_network_version,
                neural_config=self._effective_neural_config(),
            )
            self._wavefront_path_pass.set_neural(self._neural_pass)
        self._wf_path_pass_dims = key
        # The scene-build uploads ran before any Metal reflection existed
        # (wavefront mode compiles no megakernel), so the per-graph SSBOs and
        # std-surface records were packed at scalar offsets. Re-relocate them
        # now that this pass exposes the reflected MSL layouts.
        self._upload_graph_param_buffers()
        mats = getattr(self, "_last_uploaded_materials", None)
        if mats:
            self._upload_flat_materials(mats)
        return self._wavefront_path_pass

    def _destroy_wavefront_path_pass(self) -> None:
        if getattr(self, "_neural_pass", None) is not None:
            self._neural_pass.destroy()
            self._neural_pass = None
        if getattr(self, "_restir_pass", None) is not None:
            self._restir_pass.destroy()
            self._restir_pass = None
        if self._wavefront_path_pass is not None:
            self._wavefront_path_pass.destroy()
            self._wavefront_path_pass = None
        for attr in ("_wf_path_state_buf", "_wf_path_hit_buf"):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.destroy()
                setattr(self, attr, None)
        self._wf_path_pass_dims = None

    def _ensure_wavefront_bdpt_pass(self):
        """Build (once) the staged wavefront bdpt pass for the active backend.
        Returns it, or None before the scene bindings exist. On Vulkan it
        allocates the per-lane eye/light subpath-vertex buffers + aux buffer;
        the Metal pass owns its buffers (sized from reflected MSL strides)."""
        if self.is_metal:
            return self._ensure_wavefront_bdpt_pass_metal()
        if not hasattr(self.ctx, "compute_queue"):
            return None
        if self._scene_bindings is None or self.descriptor_sets is None:
            return None
        if (self._wavefront_bdpt_pass is not None
                and self._wf_bdpt_pass_dims == (self.width, self.height)):
            return self._wavefront_bdpt_pass
        self._destroy_wavefront_bdpt_pass()
        from skinny.vk_wavefront import WavefrontBdptPass
        # Tiled streaming: subpath-vertex + aux buffers hold a fixed-size stream
        # (capped), not one entry per pixel, so VRAM doesn't scale with
        # resolution (each lane owns 2×BDPT_MAX_VERTS vertices). `_wf_stream_cap`
        # overrides the cap for tests.
        num_pixels = self.width * self.height
        cap = int(getattr(self, "_wf_stream_cap", None) or WavefrontBdptPass.STREAM_CAP)
        stream_size = max(1, min(num_pixels, cap))
        # Spectral eye/light vertices (Spectrum throughput/emission) and aux
        # (Spectrum roles + SampledWavelengths) are wider than RGB. Size against
        # the mirrored spectral stride, floored by the RGB headroom constants so
        # RGB (spectral=False) is byte-identical.
        vert_stride = WavefrontBdptPass.VERTEX_STRIDE
        aux_stride = WavefrontBdptPass.AUX_STRIDE
        if self._spectral:
            from skinny.wavefront_layout import bdpt_vertex_size, wf_bdpt_aux_size
            vert_stride = max(vert_stride, bdpt_vertex_size(spectral=True))
            aux_stride = max(aux_stride, wf_bdpt_aux_size(spectral=True))
        vert_bytes = stream_size * WavefrontBdptPass.BDPT_MAX_VERTS * vert_stride
        aux_bytes = stream_size * aux_stride
        self._wf_bdpt_eye_buf = self._gpu.StorageBuffer(self.ctx, vert_bytes)
        self._wf_bdpt_light_buf = self._gpu.StorageBuffer(self.ctx, vert_bytes)
        self._wf_bdpt_aux_buf = self._gpu.StorageBuffer(self.ctx, aux_bytes)
        self._wavefront_bdpt_pass = WavefrontBdptPass(
            self.ctx, self.shader_dir, self._scene_set0_layout,
            self._wf_bdpt_eye_buf.buffer, self._wf_bdpt_light_buf.buffer,
            self._wf_bdpt_aux_buf.buffer, vert_bytes, aux_bytes,
            stream_size, num_pixels, walk_mode=self.bdpt_walk_mode,
            spectral=self._spectral,
        )
        self._wf_bdpt_pass_dims = (self.width, self.height)
        return self._wavefront_bdpt_pass

    def _ensure_wavefront_bdpt_pass_metal(self):
        """Build (once) the Metal staged wavefront bdpt pass (change
        metal-wavefront-parity phase 4). Returns it, or None before the scene
        bindings exist. The pass owns its eye/light/aux + counting-sort
        buffers, sized from the reflected MSL strides."""
        if self._scene_bindings is None:
            return None
        heavy = self._has_heavy_nonflat()
        key = (self.width, self.height, self.bdpt_walk_mode,
               self._graph_set_signature(), heavy)
        if self._wavefront_bdpt_pass is not None and self._wf_bdpt_pass_dims == key:
            return self._wavefront_bdpt_pass
        self._destroy_wavefront_bdpt_pass()
        from skinny.metal_wavefront import MetalWavefrontBdptPass
        num_pixels = self.width * self.height
        cap = int(getattr(self, "_wf_stream_cap", None)
                  or MetalWavefrontBdptPass.STREAM_CAP)
        if heavy:  # bound the heavy per-tile eye submit (see the band constant)
            cap = min(cap, _METAL_WAVEFRONT_HEAVY_EYE_BAND_LANES)
        stream_size = max(1, min(num_pixels, cap))
        self._wavefront_bdpt_pass = MetalWavefrontBdptPass(
            self.ctx, self.shader_dir, stream_size, num_pixels,
            walk_mode=self.bdpt_walk_mode,
            graph_fragments=list(self._scene_graph_fragments),
            spectral=self._spectral,
        )
        self._wf_bdpt_pass_dims = key
        # The scene-build uploads ran before any Metal reflection existed
        # (wavefront mode compiles no megakernel), so the per-graph SSBOs and
        # std-surface records were packed at scalar offsets. Re-relocate them
        # now that this pass exposes the reflected MSL layouts.
        self._upload_graph_param_buffers()
        mats = getattr(self, "_last_uploaded_materials", None)
        if mats:
            self._upload_flat_materials(mats)
        return self._wavefront_bdpt_pass

    def _destroy_wavefront_bdpt_pass(self) -> None:
        if self._wavefront_bdpt_pass is not None:
            self._wavefront_bdpt_pass.destroy()
            self._wavefront_bdpt_pass = None
        for attr in ("_wf_bdpt_eye_buf", "_wf_bdpt_light_buf", "_wf_bdpt_aux_buf"):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.destroy()
                setattr(self, attr, None)
        self._wf_bdpt_pass_dims = None

    def build_wavefront_trace_pass(self, module: str, entry: str,
                                   include_env: bool = False,
                                   include_lights: bool = False):
        """Build a wavefront pass whose kernel calls `traceScene`. Binds the
        renderer's shared geometry/BVH/instance/material buffers at the binding
        numbers traceScene reflects (0/2/5/6/7/12/13/16; 13/16 from the
        alpha-cutout path). `include_env` adds the env map (4); `include_lights`
        adds the distant-light buffer (20) — used by the diffuse shade kernel.
        The spec must match exactly what the kernel's SPIR-V reflects. Call
        after geometry is loaded (vertex/index/BVH buffers reallocate on scene
        reload, so rebuild the pass after a reload)."""
        from skinny.vk_wavefront import BoundComputePass
        sb = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        specs = [
            {"binding": 0, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             "buffer": self.uniform_buffer.buffer, "range": self.uniform_size},
            {"binding": 2, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             "view": self.accum_image.view, "layout": vk.VK_IMAGE_LAYOUT_GENERAL},
            {"binding": 5, "type": sb, "buffer": self.vertex_buffer.buffer, "range": self.vertex_buffer.size},
            {"binding": 6, "type": sb, "buffer": self.index_buffer.buffer, "range": self.index_buffer.size},
            {"binding": 7, "type": sb, "buffer": self.bvh_buffer.buffer, "range": self.bvh_buffer.size},
            {"binding": 12, "type": sb, "buffer": self.instance_buffer.buffer, "range": self.instance_buffer.size},
            {"binding": 13, "type": sb, "buffer": self.flat_material_buffer.buffer, "range": self.flat_material_buffer.size},
            {"binding": 16, "type": sb, "buffer": self.material_types_buffer.buffer, "range": self.material_types_buffer.size},
        ]
        if include_env:
            specs.append({
                "binding": 4, "type": vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                "sampler": self.env_image.sampler, "view": self.env_image.view,
                "layout": vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            })
        if include_lights:
            specs.append({
                "binding": 20, "type": sb,
                "buffer": self.distant_lights_buffer.buffer,
                "range": self.distant_lights_buffer.size,
            })
        return BoundComputePass(
            self.ctx, self.shader_dir, module, entry, specs, self.width, self.height,
        )

    def build_wavefront_material_pass(self):
        """Build the per-material albedo wavefront pass (`wavefront_material`):
        camera ray → BVH → evalSceneGraphBaseColor → material base colour.
        Binds the traceScene set + env (4) + the combined graph-param buffer
        (single slot GRAPH_BINDING_BASE) + the bindless texture array (14, from
        the texture pool). Over-providing bindings the kernel may not reference
        is fine — the SPIR-V uses a subset of the layout."""
        from skinny.vk_compute import BINDLESS_TEXTURE_CAPACITY, GRAPH_BINDING_BASE
        from skinny.vk_wavefront import BoundComputePass
        sb = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        specs = [
            {"binding": 0, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             "buffer": self.uniform_buffer.buffer, "range": self.uniform_size},
            {"binding": 2, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             "view": self.accum_image.view, "layout": vk.VK_IMAGE_LAYOUT_GENERAL},
            {"binding": 4, "type": vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             "sampler": self.env_image.sampler, "view": self.env_image.view,
             "layout": vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {"binding": 5, "type": sb, "buffer": self.vertex_buffer.buffer, "range": self.vertex_buffer.size},
            {"binding": 6, "type": sb, "buffer": self.index_buffer.buffer, "range": self.index_buffer.size},
            {"binding": 7, "type": sb, "buffer": self.bvh_buffer.buffer, "range": self.bvh_buffer.size},
            {"binding": 12, "type": sb, "buffer": self.instance_buffer.buffer, "range": self.instance_buffer.size},
            {"binding": 13, "type": sb, "buffer": self.flat_material_buffer.buffer, "range": self.flat_material_buffer.size},
            {"binding": 16, "type": sb, "buffer": self.material_types_buffer.buffer, "range": self.material_types_buffer.size},
        ]
        if self._graph_params_combined is not None:
            specs.append({"binding": GRAPH_BINDING_BASE, "type": sb,
                          "buffer": self._graph_params_combined.buffer,
                          "range": self._graph_params_combined.size})
        slots = [(idx, s.sampler, s.view) for idx, s in self.texture_pool.filled_slots()]
        specs.append({
            "binding": 14, "type": vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            "array_count": BINDLESS_TEXTURE_CAPACITY, "slots": slots,
        })
        return BoundComputePass(
            self.ctx, self.shader_dir, "wavefront/wavefront_material",
            "wavefrontMaterial", specs, self.width, self.height,
        )

    def build_wavefront_shade_passes(self):
        """Build one per-material wavefront shade pass per scene graph fragment —
        the staged per-material-pipeline shade (the compile-win). For each
        `GraphFragment` this writes `shaders/wavefront/shade_<name>.slang` from
        `emit_wavefront_shade_module` (which imports ONLY that graph's
        `generated.<name>_graph` module, making it an independent compilation
        unit) and compiles a `BoundComputePass` for its `shadeSurface_<name>`
        entry. Each pass traces, shades only the pixels whose material maps to
        its graphId, and overwrites the base colour; together they cover every
        materialised hit — the fused `wavefront_material` pass, partitioned into
        one pipeline per material.

        Bindings: the traceScene set (0/2/5/6/7/12/13/16), this graph's param
        SSBO at GRAPH_BINDING_BASE, and the 128-slot bindless texture pool at 14
        (over-provided — graphs that sample no textures reflect no 14, which the
        superset layout tolerates). Returns a `ShadePassGroup`; plug it into the
        `_wavefront_debug_pass` seam. Rebuild after a scene/geometry reload.
        """
        from skinny.materialx_runtime import assign_graph_ids
        from skinny.vk_compute import (
            BINDLESS_TEXTURE_CAPACITY,
            GRAPH_BINDING_BASE,
            emit_wavefront_shade_module,
            graph_param_combined_stride,
        )
        from skinny.vk_wavefront import (
            BoundComputePass,
            ShadePassGroup,
            compile_shade_module_cached,
            shared_shader_hash,
        )

        sb = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        id_map = assign_graph_ids(self._scene_graph_fragments)
        # Per-slot stride of the shared combined graph-param buffer — computed
        # over ALL scene graphs so every shade pass indexes it identically.
        combined_stride = graph_param_combined_stride(self._scene_graph_fragments)
        wf_dir = self.shader_dir / "wavefront"
        gen_dir = self.shader_dir / "generated"
        shared_hash = shared_shader_hash(self.shader_dir)
        slots = [(idx, s.sampler, s.view) for idx, s in self.texture_pool.filled_slots()]
        passes = []
        compiles: list[tuple[str, bool]] = []  # (graph_name, was_cached)
        keys: dict[str, str] = {}
        for gf in self._scene_graph_fragments:
            name = gf.sanitized_name
            src = emit_wavefront_shade_module(
                gf, id_map[gf.target_name], GRAPH_BINDING_BASE, combined_stride)
            (wf_dir / f"shade_{name}.slang").write_text(src, encoding="utf-8")
            # The module imports only this graph's generated module — fold its
            # bytes into the cache key so each material caches independently.
            graph_file = gen_dir / f"{name}_graph.slang"
            dep = [graph_file.read_bytes()] if graph_file.exists() else []
            spv, cached, key = compile_shade_module_cached(
                self.shader_dir, f"wavefront/shade_{name}",
                f"shadeSurface_{name}", dep, shared_hash,
            )
            compiles.append((name, cached))
            keys[name] = key
            specs = [
                {"binding": 0, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 "buffer": self.uniform_buffer.buffer, "range": self.uniform_size},
                {"binding": 2, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                 "view": self.accum_image.view, "layout": vk.VK_IMAGE_LAYOUT_GENERAL},
                {"binding": 5, "type": sb, "buffer": self.vertex_buffer.buffer, "range": self.vertex_buffer.size},
                {"binding": 6, "type": sb, "buffer": self.index_buffer.buffer, "range": self.index_buffer.size},
                {"binding": 7, "type": sb, "buffer": self.bvh_buffer.buffer, "range": self.bvh_buffer.size},
                {"binding": 12, "type": sb, "buffer": self.instance_buffer.buffer, "range": self.instance_buffer.size},
                {"binding": 13, "type": sb, "buffer": self.flat_material_buffer.buffer, "range": self.flat_material_buffer.size},
                {"binding": 16, "type": sb, "buffer": self.material_types_buffer.buffer, "range": self.material_types_buffer.size},
            ]
            if self._graph_params_combined is not None:
                specs.append({"binding": GRAPH_BINDING_BASE, "type": sb,
                              "buffer": self._graph_params_combined.buffer,
                              "range": self._graph_params_combined.size})
            specs.append({
                "binding": 14, "type": vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                "array_count": BINDLESS_TEXTURE_CAPACITY, "slots": slots,
            })
            passes.append(BoundComputePass(
                self.ctx, self.shader_dir, f"wavefront/shade_{name}",
                f"shadeSurface_{name}", specs, self.width, self.height,
                spv_path=spv,
            ))
        group = ShadePassGroup(self.ctx, passes)
        # Compile-win bookkeeping: which materials hit the SPIR-V cache (compiled
        # nothing) vs missed (compiled one kernel). Adding a material misses only
        # its own key; resident materials are cache hits.
        from skinny.vk_wavefront import _SHADE_CACHE_DIRNAME, _build_dir
        group.shade_compiles = compiles
        group.shade_keys = keys
        group.cache_dir = _build_dir() / _SHADE_CACHE_DIRNAME
        return group

    def read_accumulation(self) -> "np.ndarray":
        """Copy the linear-HDR accumulation image to host as an (H, W, 4)
        float32 array. For A/B comparison that must not depend on tonemapping
        (see CLAUDE.md headless notes)."""
        import numpy as np

        if self.is_metal:
            # Metal storage images read back directly (drains the device);
            # no Vulkan command-buffer / layout-transition machinery.
            return np.asarray(
                self.accum_image.read_rgba(), dtype=np.float32)

        w, h = self.width, self.height
        readback = self._gpu.ReadbackBuffer(self.ctx, w, h, bytes_per_pixel=16)  # RGBA32F
        f = self.current_frame
        vk.vkWaitForFences(self.ctx.device, 1, [self.in_flight_fences[f]], vk.VK_TRUE, 2**64 - 1)
        cmd = self.command_buffers[f]
        vk.vkResetCommandBuffer(cmd, 0)
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        rng = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1)
        to_src = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image=self.accum_image.image, subresourceRange=rng)
        vk.vkCmdPipelineBarrier(cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1, [to_src])
        readback.record_copy_from(cmd, self.accum_image.image)
        to_gen = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self.accum_image.image, subresourceRange=rng)
        vk.vkCmdPipelineBarrier(cmd, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, None, 0, None, 1, [to_gen])
        vk.vkEndCommandBuffer(cmd)
        vk.vkResetFences(self.ctx.device, 1, [self.in_flight_fences[f]])
        vk.vkQueueSubmit(self.ctx.compute_queue, 1,
                         [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
                         self.in_flight_fences[f])
        vk.vkWaitForFences(self.ctx.device, 1, [self.in_flight_fences[f]], vk.VK_TRUE, 2**64 - 1)
        data = readback.read()
        readback.destroy()
        return np.frombuffer(data, dtype=np.float32).reshape(h, w, 4)

    def refresh_user_presets(self) -> None:
        """Re-scan ~/.skinny/presets/ and rebuild the preset list.

        Built-ins (first N entries) are preserved; user entries are replaced.
        Called after a save/delete from the Tk panel so the combobox list
        reflects on-disk reality.
        """
        self.presets = list(PRESETS) + load_user_presets()
        if self.preset_index >= len(self.presets):
            self.preset_index = 0

    def _furnace_environment(self) -> Environment:
        """Return a constant-white HDR environment for furnace-mode tests.

        Cached after first build so the per-frame scene rebuild stays cheap.
        """
        cached = getattr(self, "_furnace_env_cache", None)
        if cached is not None:
            return cached
        from skinny.environment import ENV_HEIGHT, ENV_WIDTH
        white = np.ones((ENV_HEIGHT, ENV_WIDTH, 4), dtype=np.float32)
        env = Environment(name="Furnace (white)", _data=white)
        self._furnace_env_cache = env
        return env

    def _build_scene_from_state(self) -> Scene:
        """Materialize the current UI state into a `Scene`.

        Today this is a thin wrapper since the renderer still owns env /
        mesh / lights individually. The Scene is the place where Phase B-2/3
        (TLAS, multi-instance) and Phase C (MaterialX-driven materials) will
        accumulate state, so call sites that consume scene-level inputs go
        through `self.scene` rather than `self.*` directly.
        """
        fallback_env: Environment | None = (
            self.environments[self.env_index]
            if 0 <= self.env_index < len(self.environments)
            else None
        )
        # Phase B-3 will populate `instances` with the active mesh + a
        # transform; for now we track environment, lights, mm_per_unit, and
        # furnace mode through the scene and leave mesh-side state on the
        # renderer.
        # Pigment overlay (today's tattoo) lives on the active material; the
        # selected Tattoo object's data is the source for the GPU upload, and
        # the slider modulates density.
        _active_tattoo = (
            self.tattoos[self.tattoo_index]
            if self.tattoos and 0 <= self.tattoo_index < len(self.tattoos)
            else None
        )
        # E-6: furnace mode replaces the env / lights with a constant-white
        # IBL and disables analytic lights at the *scene* level, so material
        # evaluators don't need their own special-case branches for the
        # energy-conservation test.
        is_furnace = (self.furnace_index != 0)
        if is_furnace:
            env_for_scene: Environment | None = self._furnace_environment()
            env_intensity_for_scene = 1.0
            direct_enabled = False
        else:
            from skinny.scene import scene_environment_for_authority
            env_for_scene = scene_environment_for_authority(
                self._usd_scene,
                fallback_env,
                uses_default_lights=self.uses_default_lights,
            )
            env_intensity_for_scene = (
                float(self.env_intensity) if self.uses_default_lights else 0.0
            )
            direct_enabled = (
                self.uses_default_lights and self.direct_light_index == 0
            )
        snapshot = build_default_scene(
            environment=env_for_scene,
            env_intensity=env_intensity_for_scene,
            mesh=None,
            light_direction=self.light_direction,
            light_radiance=self.light_radiance,
            direct_light_enabled=direct_enabled,
            mm_per_unit=float(self.mm_per_unit),
            furnace_mode=is_furnace,
        )
        # Preserve the authored DomeLight object itself so its enabled and
        # intensity state remain authoritative. build_default_scene clones an
        # Environment into a fallback LightEnvHDR and is used only for that
        # fallback/furnace path.
        if not is_furnace and not self.uses_default_lights:
            snapshot.environment = env_for_scene
        return snapshot

    def _frame_camera_to_scene(self, scene: Scene) -> None:
        """Position the orbit camera from a USD authored UsdGeom.Camera
        when present, falling back to an auto-frame around the scene's
        world AABB.

        Auto-frame: target = bounds centre, distance fits the bounding
        sphere inside the vertical FOV with a small margin. Override:
        target = position + forward·focus_distance (or auto-distance when
        focusDistance is unauthored), yaw/pitch derived from forward.
        """
        # Default off; an authored override re-asserts it from its `mirrored`
        # flag below. Resetting here (not only in _clear_model_state) means a
        # reused renderer loading an override-less scene after a mirrored one
        # never keeps a stale mirror via the set_usd_scene path.
        self._camera_mirror = False
        if scene.camera_override is not None:
            self._apply_camera_override(scene)
            return

        bounds = scene.world_bounds()
        if bounds is None:
            return
        amin, amax = bounds
        diag = amax - amin
        radius = float(np.linalg.norm(diag) * 0.5)
        if radius < 1e-6:
            return
        center = ((amin + amax) * 0.5).astype(np.float32)

        cap = _orbit_distance_cap(float(np.max(diag)))
        fov_v_rad = np.radians(self.orbit_camera.fov)
        margin = 1.4
        distance = radius / np.tan(fov_v_rad * 0.5) * margin
        # Respect OrbitCamera.zoom's clamp range so user wheel-zoom stays
        # consistent with the seeded value.
        distance = float(np.clip(distance, 0.5, cap))

        yaw, pitch = _hero_yaw_pitch()
        self.orbit_camera.max_distance = cap
        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = yaw
        self.orbit_camera.pitch = pitch

    def _apply_camera_override(self, scene: Scene) -> None:
        """Convert scene.camera_override → OrbitCamera (target, distance,
        yaw, pitch, fov). Sets target = position + forward·focus_distance;
        when focus_distance is unauthored, picks a distance that puts the
        target at the centre of the world bounds (or 5 world units when
        bounds aren't useful).
        """
        ov = scene.camera_override
        if ov is None:
            return
        self._camera_mirror = bool(ov.mirrored)
        self._override_to_orbit(self.orbit_camera, ov, scene)
        d = float(self.orbit_camera.focus_distance)
        self.free_camera.focal_length_mm = float(ov.focal_length_mm)
        self.free_camera.vertical_aperture_mm = float(ov.vertical_aperture_mm)
        self.free_camera.fstop = float(ov.fstop)
        self.free_camera.focus_distance = float(d)
        self.free_camera.lens = ov.lens
        # pbrt film exposure (change pbrt-radiometric-parity): the authored camera
        # carries ISO + exposure time; the renderer applies their imaging ratio as a
        # live output scale instead of having it baked into emitters at import.
        self.film.iso = float(getattr(ov, "iso", 100.0))
        self.film.exposure_time = float(getattr(ov, "exposure_time", 1.0))

    def _override_to_orbit(
        self, cam: "OrbitCamera", ov: "CameraOverride", scene: "Scene"
    ) -> None:
        """Set an OrbitCamera's params from a CameraOverride.

        Reproduces the authored eye position + look direction exactly
        (orbit position == ov.position) and derives vertical FOV from focal
        length + aperture. Used for both the load-time camera framing and the
        per-frame USD camera follower in `camera_mode == "usd"`.
        """
        # Distance cap scales with scene size so a large authored scene's
        # focus distance isn't clamped down to the 50-unit floor.
        bounds = scene.world_bounds()
        cap = (
            _orbit_distance_cap(float(np.max(bounds[1] - bounds[0])))
            if bounds is not None else 50.0
        )
        # Pick a focus distance: authored value if present, else aim at
        # bounds centre, else fall back to a 5-unit default.
        d = ov.focus_distance
        if d is None or d <= 1e-6:
            if bounds is not None:
                amin, amax = bounds
                center = (amin + amax) * 0.5
                d = float(np.linalg.norm(center.astype(np.float32) - ov.position))
            if not d:
                d = 5.0
        d = float(np.clip(d, 0.5, cap))

        target = (ov.position + ov.forward * d).astype(np.float32)

        # OrbitCamera.position = target + d·(cos(p)sin(y), sin(p), cos(p)cos(y))
        # We need pos - target = -forward·d, so the spherical vector is
        # -forward.
        s = -ov.forward
        pitch = float(np.arcsin(np.clip(s[1], -1.0, 1.0)))
        yaw   = float(np.arctan2(s[0], s[2]))

        # Vertical FOV from focal length + vertical aperture (mm).
        fov_v_deg = float(np.degrees(
            2.0 * np.arctan(0.5 * ov.vertical_aperture_mm /
                            max(ov.focal_length_mm, 1e-3))
        ))

        cam.max_distance = cap
        cam.target = target
        cam.distance = d
        cam.yaw = yaw
        cam.pitch = pitch
        # Honor the authored up so a non-Y-up (e.g. pbrt Z-up) camera keeps its
        # roll; the eye/yaw/pitch math above is unchanged. Defaults to +Y.
        cam.up = np.asarray(getattr(ov, "up", (0.0, 1.0, 0.0)), np.float32).reshape(3)
        cam.fov = fov_v_deg
        cam.focal_length_mm = float(ov.focal_length_mm)
        cam.vertical_aperture_mm = float(ov.vertical_aperture_mm)
        cam.fstop = float(ov.fstop)
        cam.focus_distance = float(d)
        cam.lens = ov.lens

    def _frame_camera_to_mesh(self, source: MeshSource) -> None:
        """Auto-fit orbit camera to a MeshSource's bounding box."""
        amin = source.positions.min(axis=0)
        amax = source.positions.max(axis=0)
        diag = amax - amin
        radius = float(np.linalg.norm(diag) * 0.5)
        if radius < 1e-6:
            return
        center = ((amin + amax) * 0.5).astype(np.float32)

        cap = _orbit_distance_cap(float(np.max(diag)))
        fov_v_rad = np.radians(self.orbit_camera.fov)
        margin = 1.4
        distance = radius / np.tan(fov_v_rad * 0.5) * margin
        distance = float(np.clip(distance, 0.5, cap))

        yaw, pitch = _hero_yaw_pitch()
        self.orbit_camera.max_distance = cap
        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = yaw
        self.orbit_camera.pitch = pitch

    def _clear_model_state(self) -> None:
        """Reset all model/scene state so a fresh load starts clean."""
        self.models.clear()
        self._mesh_sources.clear()
        self.model_index = -1
        self._usd_scene = None
        self._usd_stage = None
        self._usd_edit_layer = None
        self._edit_layer_default_path = None
        self._prim_to_instances = {}
        self._scene_graph = None
        self._last_projected_default_lights = None
        self._last_aux_light_authority_token = None
        self._usd_model_index = -1
        self._camera_mirror = False
        self._usd_bake_done = None
        self._usd_uploaded_count = 0
        self._baked_source_idx = -1
        self._displacement_cache.clear()
        self._normal_cache.clear()
        self._dirty_since = None
        # Reset GPU mesh to dummy so stale geometry doesn't render
        self._upload_mesh(self._dummy_mesh)
        self._upload_detail_maps(None)
        self._per_material_furnace = [False] * self.material_capacity
        self._scene_graph_fragments = []
        self._material_graph_ids.clear()
        self._material_graph_overrides.clear()
        self._mtlx_scene_materials.clear()

    def load_environment_from_path(self, path: Path) -> int:
        """Load an HDR environment from an arbitrary path and select it.

        Appends a new ``Environment`` to ``self.environments``, bumps
        ``env_index`` to point at it, and triggers an upload on the next
        frame via the standard ``_ensure_env_uploaded`` path. Returns the
        new ``env_index``.

        Supported formats: ``.hdr`` (Radiance), ``.exr``, ``.pfm``.
        """
        from skinny.environment import make_environment_from_path
        env = make_environment_from_path(path)
        self.environments.append(env)
        self.env_index = len(self.environments) - 1
        try:
            self._ensure_env_uploaded()
        except Exception as exc:  # noqa: BLE001
            print(f"[skinny] env upload failed for {path.name}: {exc}")
        return self.env_index

    def _refresh_material_python_ids(self) -> "dict[int, int]":
        """Rebuild `_material_python_ids` from the live scene + current
        `python_materials/` listing.

        Called from `_upload_material_types` (every type upload) so a
        material edit that adds/removes a Python material file flows
        through without requiring a separate `_upload_flat_materials`
        round-trip.
        """
        from skinny.megakernel_sources import python_material_ids as _ids_fn
        ids = _ids_fn()
        out: dict[int, int] = {}
        scene = self._usd_scene if self._usd_scene is not None else self.scene
        for i, mat in enumerate(scene.materials):
            mod = getattr(mat, "python_module", None)
            if mod and mod in ids:
                out[i] = ids[mod]
        self._material_python_ids = out
        return out

    def active_python_module(self) -> "str | None":
        """First `python_module` hint declared by any current scene material.

        Returns the module name (e.g. ``"python_materials.preview_surface_material"``)
        or ``None`` when no scene material is bound to a Python-authored
        slangpile material. Used by the Python Material Editor dock to
        decide which source file to load.
        """
        modules = self.scene_python_modules()
        return modules[0] if modules else None

    def scene_python_modules(self) -> "list[str]":
        """Deduped list of every `python_module` declared by scene materials,
        preserving the scene's material order. Empty when no scene material
        is bound to a Python-authored slangpile material.

        Reads from `_usd_scene` (the authored material list from
        `usd_loader._read_usd_stage`) when a USD scene is loaded. `self.scene`
        is the per-frame render snapshot rebuilt by `_build_scene_from_state`
        and only carries a placeholder material, so it's not the source of
        truth for authored material metadata.
        """
        source = self._usd_scene if self._usd_scene is not None else self.scene
        seen: set[str] = set()
        out: list[str] = []
        for mat in source.materials:
            mod = getattr(mat, "python_module", None)
            if mod and mod not in seen:
                seen.add(mod)
                out.append(mod)
        return out

    def load_model_from_path(self, path: Path) -> int:
        """Load a model file (USDA/USDC/USDZ/OBJ), replacing any previous model.

        Returns the index of the newly loaded model. Loading runs in a
        background thread; the model appears in the UI as soon as it's ready.
        """
        import threading as _threading

        self._clear_model_state()
        ext = path.suffix.lower()

        if ext in (".usda", ".usdc", ".usdz"):
            self._load_usd_model(path)
            return 0

        if ext == ".obj":
            self.models.append(f"(loading {path.name}...)")
            self.model_index = 0

            def _bg_load() -> None:
                try:
                    if path.parent.is_dir():
                        src = _load_model_dir(path.parent)
                        if src is None:
                            src = load_obj_source(path)
                    else:
                        src = load_obj_source(path)
                    self._mesh_sources.append(src)
                    self.models[0] = src.name
                    self._frame_camera_to_mesh(src)
                    self.model_index = 0
                    print(
                        f"[skinny] loaded model '{src.name}' "
                        f"({src.positions.shape[0]} verts, "
                        f"{src.tri_idx.shape[0]} tris)"
                    )
                    # OBJ loads don't traverse `_gen_scene_materials`, so
                    # the lazy pipeline build never fires from the USD
                    # poll. Build an empty-graph pipeline here so the
                    # renderer has something to dispatch. Keyed on the scene
                    # bindings (None until first build) since `self.pipeline`
                    # is always None in wavefront mode.
                    if self._scene_bindings is None:
                        self._build_pipeline_for_current_graphs()
                except Exception as exc:  # noqa: BLE001
                    print(f"[skinny] failed to load {path.name}: {exc}")
                    if self.models:
                        self.models[0] = f"(failed: {path.name})"

            _threading.Thread(
                target=_bg_load, daemon=True, name="skinny-load-model",
            ).start()
            return 0

        raise ValueError(f"Unsupported model format: {ext}")

    def _load_usd_model(self, path: Path) -> None:
        """Load a USD file as the active model, replacing any previous."""
        import threading as _threading

        self.models.append(f"USD: (loading {path.name}...)")
        self._usd_model_index = 0
        self.model_index = 0
        self._usd_bake_done = _threading.Event()

        def _bg_usd_stream() -> None:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from skinny.usd_loader import (
                _read_usd_stage,
                bake_usd_prim,
                build_animation_index,
                build_playback_clock,
            )
            scene, prim_data, stage = _read_usd_stage(
                path, use_usd_mtlx_plugin=self._use_usd_mtlx_plugin,
                keep_stage=True,
            )
            # Keep the stage around so scene-graph edits (DomeLight HDR
            # path, etc.) can mutate USD prim attrs as the source of truth.
            self._usd_stage = stage
            # Attach the non-destructive session edit layer so the runtime
            # scene-graph editing API authors there, never the original file.
            self._attach_edit_layer()
            # Build the playback clock + animated-prim index from the stage.
            # Single ref assignments, read on the main thread in update().
            if stage is not None:
                try:
                    from pxr import UsdGeom as _UsdGeom
                    from skinny.usd_loader import (
                        _up_axis_rt,
                        extract_skeletal_bindings,
                        extract_ui_controls,
                    )
                    index = build_animation_index(stage)
                    self._anim_index = index
                    self.clock = build_playback_clock(stage, index)
                    self._usd_up_axis_rt = _up_axis_rt(
                        str(_UsdGeom.GetStageUpAxis(stage))
                    )
                    # SkeletalScene retains the stage + cache so its skinning
                    # queries stay valid for per-frame ComputeSkinningTransforms.
                    self._skeletal = extract_skeletal_bindings(stage)
                    self._usd_controls = extract_ui_controls(stage)
                    self._last_eval_time_code = None
                except Exception as exc:  # noqa: BLE001
                    self._anim_index = None
                    self._skeletal = None
                    self._usd_controls = []
                    self.clock = PlaybackClock()
                    print(f"[skinny] animation index build failed: {exc}")
            # Build scene graph here in the background thread while we
            # have exclusive access to the stage — avoids GIL conflicts
            # with GLFW poll_events on the main thread.
            sg = None
            if stage is not None:
                from skinny.scene_graph import build_scene_graph
                try:
                    sg = build_scene_graph(stage, scene)
                except Exception as exc:
                    import traceback
                    print(f"[skinny] scene graph build failed: {exc}")
                    traceback.print_exc()
            self._usd_metadata_queue.put((scene, sg))
            print(
                f"[skinny] USD stage read: {len(prim_data)} meshes, "
                f"baking in background"
            )
            cache_idx = load_cache_index()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = {
                    pool.submit(
                        bake_usd_prim, src, xform, mat_id, cache_idx,
                    ): src.name
                    for src, xform, mat_id in prim_data
                }
                for fut in as_completed(futs):
                    try:
                        inst = fut.result()
                        self._usd_instance_queue.put(inst)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[skinny] USD bake failed for {futs[fut]}: {exc}")
            self._usd_bake_done.set()

        _threading.Thread(
            target=_bg_usd_stream, daemon=True, name="skinny-usd-stream",
        ).start()

    def _mtlx_skin_overrides(self) -> dict[str, object]:
        """Map the renderer's SkinParameters dataclass into MaterialX input
        name → value pairs that match what the gen-reflected
        M_skinny_skin_default UBO expects. Inputs without a SkinParameters
        equivalent (`*_pigment`, `layer_bottom_absorption/scattering`,
        `skin_surface_*`) are left out so pack_material_values uses the
        MaterialX-authored defaults.
        """
        s = self.skin
        scatter = tuple(float(c) for c in s.scattering_coefficient)
        return {
            # Top layer (epidermis): melanin, thickness, scattering, g, ior
            "layer_top_melanin":           s.melanin_fraction,
            "layer_top_thickness":         s.epidermis_thickness_mm,
            "layer_top_scattering_coeff":  scatter,
            "layer_top_anisotropy":        s.anisotropy_g,
            "layer_top_ior":               s.ior,
            # Middle layer (dermis): hemoglobin, oxygenation, thickness,
            # scattering, g, ior. pigment stays at MaterialX default
            # (zero alpha = no overlay).
            "layer_middle_hemoglobin":         s.hemoglobin_fraction,
            "layer_middle_blood_oxygenation":  s.blood_oxygenation,
            "layer_middle_thickness":          s.dermis_thickness_mm,
            "layer_middle_scattering_coeff":   scatter,
            "layer_middle_anisotropy":         s.anisotropy_g,
            "layer_middle_ior":                s.ior,
            # Bottom layer (subcut): only thickness varies in
            # SkinParameters; anisotropy + ior are shared from the
            # global, others fall back to the layer's MaterialX defaults
            # (fixed-physics fat absorption + scattering).
            "layer_bottom_thickness":   s.subcut_thickness_mm,
            "layer_bottom_anisotropy":  s.anisotropy_g,
            "layer_bottom_ior":         s.ior,
            # Surface stack: roughness, ior, pore + hair sliders.
            "skin_bsdf_roughness":     s.roughness,
            "skin_bsdf_ior":           s.ior,
            "skin_bsdf_pore_density":  s.pore_density,
            "skin_bsdf_pore_depth":    s.pore_depth,
            "skin_bsdf_hair_density":  s.hair_density,
            "skin_bsdf_hair_tilt":     s.hair_tilt,
        }

    def _pack_mtlx_skin(self) -> bytes:
        """Pack the current SkinParameters into the gen-reflected UBO bytes.

        Returns empty bytes if the runtime didn't load or hasn't generated
        the skin material yet — caller is expected to gate on the result
        size before uploading.
        """
        cm = self._mtlx_skin_material
        if cm is None or not cm.uniform_block:
            return b""
        from skinny.materialx_runtime import pack_material_values
        return pack_material_values(cm.uniform_block, self.mtlx_overrides)

    def _pack_mtlx_skin_array(self) -> bytes:
        """Pack one MtlxSkinParams record per material slot, concatenated.

        Skin-typed slots (mtlx_target_name == "M_skinny_skin_default")
        get the global mtlx_overrides merged with per-material overrides.
        All other slots are zeroed.
        """
        cm = self._mtlx_skin_material
        if cm is None or not cm.uniform_block:
            return b""
        from skinny.materialx_runtime import pack_material_values

        base = dict(self.mtlx_overrides)
        scene_mats = (
            self._usd_scene.materials if self._usd_scene is not None else []
        )

        out = bytearray()
        for slot in range(self.material_capacity):
            if slot >= len(scene_mats):
                out += b"\x00" * self.mtlx_skin_record_size
                continue
            mat = scene_mats[slot]
            if mat.mtlx_target_name == "M_skinny_skin_default":
                # Merge: per-material overrides take precedence over
                # the SkinParameters-derived base, and the user's direct
                # mtlx.* edits trump everything (already merged into base).
                merged = dict(base)
                for k, v in mat.parameter_overrides.items():
                    merged[k] = v
                out += pack_material_values(cm.uniform_block, merged)
            else:
                # Non-skin slot: zero record (shader's materialTypes
                # gating means this is never read).
                out += b"\x00" * self.mtlx_skin_record_size
        return bytes(out)

    def _pack_mtlx_skin_array_msl(self) -> bytes:
        """Metal sibling of `_pack_mtlx_skin_array`: relocate every scalar-layout
        record into the reflected MSL element layout of the `mtlxSkin`
        `StructuredBuffer` (design D3, like `_pack_uniforms_msl` does for `fc`).

        The gen-reflected skin UBO (`cm.uniform_block`, what `pack_material_values`
        emits) packs `float3` at scalar offsets (no vec3→16 promotion under
        `-fvk-use-scalar-layout`). Metal's MSL layout pads each `float3` to 16 B and
        16-aligns it, shifting every later field — so the scalar record cannot be
        float4-wrapped the way `BvhNode`/`FlatMaterialParams` are. Instead copy each
        field's data bytes from its scalar offset (`UniformField.offset`) to its
        reflected MSL offset (`pipeline.mtlx_skin_layout[name]`), record by record,
        at the MSL element stride. Offsets come from live reflection, never a
        hand-table. Falls back to the scalar packer when reflection is unavailable
        (no skin material / buffer dead-stripped) so frame 0 still has valid bytes."""
        layout = getattr(self._msl_layout_source, "mtlx_skin_layout", None)
        stride = getattr(self._msl_layout_source, "mtlx_skin_stride", 0)
        cm = self._mtlx_skin_material
        if not layout or not stride or cm is None or not cm.uniform_block:
            return self._pack_mtlx_skin_array()
        from skinny.materialx_runtime import pack_material_values

        base = dict(self.mtlx_overrides)
        scene_mats = (
            self._usd_scene.materials if self._usd_scene is not None else []
        )
        fields = cm.uniform_block

        def to_msl(scalar: bytes) -> bytes:
            rec = bytearray(stride)
            for f in fields:
                moff = layout.get(f.name)
                if moff is None:
                    continue  # field dead-stripped from the MSL struct
                rec[moff[0]:moff[0] + f.size] = scalar[f.offset:f.offset + f.size]
            return bytes(rec)

        out = bytearray()
        for slot in range(self.material_capacity):
            if slot >= len(scene_mats):
                out += b"\x00" * stride
                continue
            mat = scene_mats[slot]
            if mat.mtlx_target_name == "M_skinny_skin_default":
                merged = dict(base)
                for k, v in mat.parameter_overrides.items():
                    merged[k] = v
                out += to_msl(pack_material_values(cm.uniform_block, merged))
            else:
                out += b"\x00" * stride
        return bytes(out)

    def _init_materialx_runtime(self) -> None:
        """Bootstrap MaterialLibrary and pre-generate skin material Slang.

        Best-effort: any failure (MaterialX import error, missing impl
        files, gen exception) is logged and skipped — the renderer keeps
        running off the static skinny_skin_layered_bsdf_genslang.slang
        import. Future milestones will harden this once we actually rely
        on the runtime output for shading.
        """
        try:
            from skinny.materialx_runtime import MaterialLibrary
        except ImportError as e:
            print(f"[skinny] materialx_runtime unavailable: {e}")
            return

        try:
            lib = MaterialLibrary.from_install()
            lib.load()
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[skinny] MaterialLibrary load failed: {e}")
            return

        try:
            cm = lib.generate("M_skinny_skin_default", compile_check=False)
        except (KeyError, RuntimeError) as e:
            print(f"[skinny] MaterialX generate failed: {e}")
            return

        self._mtlx_library = lib
        self._mtlx_skin_material = cm
        n_nodedefs = len(lib.list_skinny_nodedefs())
        n_uniforms = len(cm.uniform_block)
        n_funcs = len(cm.functions_emitted)
        size_kb = len(cm.pixel_source) / 1024.0
        print(
            f"[skinny] MaterialX runtime ready: {n_nodedefs} skinny nodedefs, "
            f"target={cm.target_name!r}, "
            f"slang={size_kb:.1f}KB, "
            f"functions_emitted={n_funcs}, uniform_fields={n_uniforms}"
        )

        # Round-trip check: pack the current SkinParameters through the
        # gen-reflected layout and report the byte size + how many fields
        # got an explicit override (vs. falling back to MaterialX defaults).
        # Milestone 3 will upload these bytes to a GPU buffer at binding 15.
        packed = self._pack_mtlx_skin()
        self.mtlx_skin_record_size = len(packed)
        n_overrides = len(self.mtlx_overrides)
        print(
            f"[skinny] MaterialX skin override pack: {len(packed)} bytes "
            f"({n_overrides}/{n_uniforms} fields driven by SkinParameters)"
        )

        self._mtlx_scene_materials: dict[int, object] = {}
        self._gen_scene_materials()

    def _gen_scene_materials(self) -> None:
        """Run MaterialX gen for each non-skin scene material.

        Populates `_mtlx_scene_materials`, `_scene_graph_fragments`,
        `_material_graph_ids`, and `_material_graph_overrides`. Called at
        init and again whenever `_usd_scene` changes so dynamically loaded
        models get their MaterialX-driven materials (marble, wood, …).

        Materials whose MaterialX target wraps a nodegraph driving
        `base_color` (`generate_for_compute` returns a GraphFragment) get
        a graphId ≥ 2; the renderer encodes that id in materialTypes and
        the compute shader's evalSceneGraph dispatches to the gen-emitted
        evaluator. Pure constant-input materials (Glass, Brass, …) fall
        through to the existing flat / std_surface SSBO path with
        graphId == 0.
        """
        lib = self._mtlx_library
        scene = self._usd_scene
        if scene is None or not scene.materials:
            return

        # MaterialX runtime missing → skip graph generation but still
        # rebuild the pipeline so the FLAT-material path can render USD
        # scenes without the gen-slang dependency.
        if lib is None:
            if (
                self._scene_bindings is None
                or self._graph_set_signature() != self._pipeline_built_for_targets
            ):
                self._build_pipeline_for_current_graphs()
            return

        from skinny.materialx_runtime import (
            assign_graph_ids,
        )
        self._mtlx_scene_materials.clear()
        self._material_graph_ids.clear()
        self._material_graph_overrides.clear()
        ok_std = 0
        ok_mtlx = 0
        fail = 0
        total_kb = 0.0

        # First pass: gen each material, collect distinct GraphFragments.
        # Fragment identity = target_name; multiple materials sharing a
        # target reuse the same fragment + SSBO struct (the slot index
        # carries per-material overrides).
        fragments_by_target: dict[str, object] = {}
        per_mat_target: dict[int, str] = {}
        for i, mat in enumerate(scene.materials):
            if i == 0:
                continue
            try:
                if mat.mtlx_target_name:
                    if mat.mtlx_document is not None:
                        lib.import_document(mat.mtlx_document)
                    cm = lib.generate(
                        mat.mtlx_target_name, compile_check=False
                    )
                else:
                    cm = lib.compile_for_scene_material(mat)
            except Exception as e:  # noqa: BLE001
                print(f"[skinny] mat[{i}] {mat.name!r}: gen FAIL  "
                      f"{type(e).__name__}: {e}")
                fail += 1
                continue
            self._mtlx_scene_materials[i] = cm
            target = getattr(cm, "target_name", None) or mat.mtlx_target_name
            if target and target not in fragments_by_target:
                try:
                    # Reuse the CompiledMaterial we already built above —
                    # generate_for_compute would otherwise re-run gen for
                    # every graph-bound material.
                    gf = lib.generate_for_compute(
                        target, write_to_disk=False, compiled=cm,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[skinny] mat[{i}] {mat.name!r}: graph-extract FAIL  "
                          f"{type(e).__name__}: {e}")
                    gf = None
                if gf is not None:
                    fragments_by_target[target] = gf
            if target in fragments_by_target:
                per_mat_target[i] = target

            if mat.mtlx_target_name:
                ok_mtlx += 1
            else:
                ok_std += 1
            total_kb += len(cm.pixel_source) / 1024.0

        self._scene_graph_fragments = list(fragments_by_target.values())
        id_map = assign_graph_ids(self._scene_graph_fragments)
        for mat_idx, target in per_mat_target.items():
            gid = id_map.get(target)
            if gid is None:
                continue
            self._material_graph_ids[mat_idx] = gid
            mat = scene.materials[mat_idx]
            self._material_graph_overrides[mat_idx] = dict(
                getattr(mat, "parameter_overrides", {}) or {}
            )

        if ok_std or ok_mtlx or fail:
            n_graphs = len(self._scene_graph_fragments)
            n_graph_mats = len(self._material_graph_ids)
            print(
                f"[skinny] MaterialX per-scene-material gen: "
                f"{ok_std} std_surface, {ok_mtlx} mtlx-targeted, "
                f"{fail} fail, {n_graphs} graphs / {n_graph_mats} graph-bound mats, "
                f"total {total_kb:.1f}KB slang"
            )

        # Build pipeline on first call (lazy — `_init_gpu` left it None to
        # avoid a wasted compile against an empty fragment list at startup),
        # or rebuild when the scene's MaterialX nodegraph set differs from
        # what the live pipeline was compiled against. The signature
        # `_graph_set_signature()` pairs each target name with a stable
        # hash of the emitted Slang, so two scenes that use the same
        # target_name from different `.mtlx` documents (different node
        # wiring, different texture paths) still trigger a rebuild.
        if (
            self._scene_bindings is None
            or self._graph_set_signature() != self._pipeline_built_for_targets
        ):
            self._build_pipeline_for_current_graphs()

    def _init_default_light_stage(self) -> None:
        """Create an anonymous in-memory stage with /Skinny/DefaultLight as a
        UsdLuxDistantLight and /Skinny/DefaultDome as a UsdLuxDomeLight.
        Both prims mirror the renderer's current state so the scene graph
        editor treats them identically to imported USD lights.
        """
        try:
            from pxr import Usd, UsdGeom, UsdLux
        except Exception:
            return
        try:
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            xf = UsdGeom.Xform.Define(stage, "/Skinny")
            light = UsdLux.DistantLight.Define(stage, "/Skinny/DefaultLight")
            dome = UsdLux.DomeLight.Define(stage, "/Skinny/DefaultDome")
            stage.SetDefaultPrim(xf.GetPrim())
            self._default_light_stage = stage
            self._default_light_prim = light.GetPrim()
            self._default_dome_prim = dome.GetPrim()
        except Exception:
            self._default_light_stage = None
            self._default_light_prim = None
            self._default_dome_prim = None

    def _sync_default_dome_prim(self) -> None:
        """Push current env state onto /Skinny/DefaultDome.

        Writes the resolved HDR path of ``self.environments[self.env_index]``
        into ``inputs:texture:file`` (empty asset for procedural envs) and
        ``self.env_intensity`` into ``inputs:intensity``. Keeps the USD
        prim as the source of truth surfaced in the scene graph.
        """
        prim = self._default_dome_prim
        if prim is None:
            return
        try:
            from pxr import Sdf, UsdLux
            dome = UsdLux.DomeLight(prim)
            env = (
                self.environments[self.env_index]
                if 0 <= self.env_index < len(self.environments)
                else None
            )
            env_path = getattr(env, "path", None) if env is not None else None
            asset_path = str(env_path) if env_path is not None else ""
            dome.CreateTextureFileAttr().Set(Sdf.AssetPath(asset_path))
            dome.CreateTextureFormatAttr().Set("latlong")
            dome.CreateIntensityAttr().Set(float(self.env_intensity))
        except Exception:
            pass

    def _sync_default_light_prim(self) -> None:
        """Push current scalar light state onto the in-memory prim attrs."""
        prim = self._default_light_prim
        if prim is None:
            return
        try:
            from pxr import Gf, UsdLux, UsdGeom
            light = UsdLux.DistantLight(prim)
            light.CreateColorAttr().Set(Gf.Vec3f(
                float(self.light_color_r),
                float(self.light_color_g),
                float(self.light_color_b),
            ))
            light.CreateIntensityAttr().Set(float(self.light_intensity))
            light.CreateExposureAttr().Set(0.0)
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            rot_op = next(
                (op for op in ops
                 if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ),
                None,
            )
            if rot_op is None:
                rot_op = xformable.AddRotateXYZOp()
            # USD distant light shines down its local -Z. Map elevation/
            # azimuth (renderer convention) to a rotation that aims -Z at
            # the light direction.
            rot_op.Set(Gf.Vec3f(
                float(-self.light_elevation),
                float(self.light_azimuth),
                0.0,
            ))
        except Exception:
            pass

    def _update_light(self) -> None:
        az = np.radians(self.light_azimuth)
        el = np.radians(self.light_elevation)
        d = np.array([
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ], dtype=np.float32)
        self.light_direction = d / np.linalg.norm(d)
        color = np.array(
            [self.light_color_r, self.light_color_g, self.light_color_b],
            dtype=np.float32,
        )
        self.light_radiance = color * self.light_intensity
        self._sync_default_light_prim()

    def _graph_set_signature(self) -> tuple:
        """Hashable identity of the active MaterialX graph set.

        Pairs each fragment's `target_name` with a content hash of its
        emitted Slang so the rebuild gate distinguishes scenes that
        share a target name but came from different `.mtlx` documents.
        """
        import hashlib
        return tuple(
            (gf.target_name,
             hashlib.blake2b(gf.slang_source.encode("utf-8"),
                             digest_size=8).hexdigest())
            for gf in self._scene_graph_fragments
        )

    def _build_pipeline_for_current_graphs(self) -> None:
        """Build (or rebuild) the scene bindings + descriptor pool/sets — and,
        in megakernel mode only, the megakernel compute pipeline — against the
        current `_scene_graph_fragments`.

        Handles both first-build (scene bindings are None at startup; built
        lazily once `_gen_scene_materials` has populated the fragment list from
        a loaded scene) and rebuild (scene graph set changed mid-session).

        The set-0 descriptor-set layout includes one storage buffer per
        MaterialX nodegraph fragment at GRAPH_BINDING_BASE+idx, and the
        aggregator's `evalSceneGraph` switch hard-codes the fragment list —
        both are re-emitted (+ the megakernel recompiled, in megakernel mode)
        whenever the fragment set changes.

        Execution-mode gating: the megakernel `main_pass` pipeline is compiled
        ONLY in megakernel mode. In wavefront mode the scene bindings are built
        standalone (`scene_bindings_only` — no main_pass slangc, no driver
        pipeline; `self.pipeline` stays None) and the wavefront stage pipelines
        are built lazily by `_ensure_wavefront_*` on the first wavefront frame.

        If slangc fails in megakernel mode (malformed extracted fragment,
        extractor regression, …) we fall back to an empty-graph build so the
        rest of the scene still renders; affected materials show magenta via
        evalSceneGraph's `default` case.
        """
        megakernel = self.execution_mode_index == EXECUTION_MEGAKERNEL
        is_rebuild = self._scene_bindings is not None
        if is_rebuild:
            # Backend-neutral device drain. Metal routes through slang-rhi's
            # ``Device.wait_for_idle``; Vulkan calls ``vkDeviceWaitIdle``.
            self.ctx.wait_idle()
            if self.descriptor_pool is not None:
                # Vulkan-only: descriptor pools are a VK concept (Metal uses
                # argument buffers and never populates ``descriptor_pool``).
                # Pool destroy frees descriptor sets implicitly.
                vk.vkDestroyDescriptorPool(
                    self.ctx.device, self.descriptor_pool, None,
                )
                self.descriptor_pool = None
                self.descriptor_sets = None
            # The wavefront stage passes reuse the set-0 layout owned by the
            # scene bindings, so a rebuild of the layout invalidates them too.
            self._destroy_wavefront_path_pass()
            self._destroy_wavefront_bdpt_pass()
            # MLT also holds descriptor writes in the (freed) scene sets — a
            # rebuild must force the ensure/rebind path (change mlt-integrator).
            self._destroy_wavefront_mlt_pass()
            # `self.pipeline` (megakernel) shares the scene-bindings object in
            # megakernel mode; destroy the scene bindings once, then clear the
            # alias. In wavefront mode `self.pipeline` is already None.
            self.pipeline = None
            # Preview pipeline shares the set-0 descriptor-set layout, so a
            # rebuild invalidates its reference. Drop it; render_material_preview
            # re-creates it lazily on next call.
            if self._preview_pipeline is not None:
                self._preview_pipeline.destroy()
                self._preview_pipeline = None
            self._scene_bindings.destroy()
            self._scene_bindings = None

        # Snapshot the attempted signature BEFORE the build — if slangc
        # fails we still want to record what we tried so the gate in
        # `_gen_scene_materials` doesn't trigger an infinite retry loop
        # for the same broken fragment set on every subsequent scene
        # poll.
        attempted_sig = self._graph_set_signature()
        if megakernel:
            try:
                self._scene_bindings = self._gpu.ComputePipeline(
                    self.ctx,
                    self.shader_dir,
                    entry_module="main_pass",
                    entry_point="mainImage",
                    graph_fragments=list(self._scene_graph_fragments),
                    spectral=self._spectral,
                )
            except RuntimeError as e:
                action = "rebuild" if is_rebuild else "build"
                print(
                    f"[skinny] WARNING: pipeline {action} with "
                    f"{len(self._scene_graph_fragments)} MaterialX graph(s) "
                    f"failed:\n  {e}\n"
                    f"[skinny]   → falling back to empty-graph pipeline. "
                    f"Affected materials will render magenta."
                )
                self._scene_graph_fragments = []
                self._material_graph_ids.clear()
                self._scene_bindings = self._gpu.ComputePipeline(
                    self.ctx,
                    self.shader_dir,
                    entry_module="main_pass",
                    entry_point="mainImage",
                    graph_fragments=[],
                    spectral=self._spectral,
                )
            # In megakernel mode the scene bindings ARE the compiled pipeline.
            self.pipeline = self._scene_bindings
        else:
            # Wavefront mode: build the scene plumbing (set-0 layout + material
            # emission + graph bindings) WITHOUT compiling main_pass. The
            # wavefront stage pipelines are built lazily in the render gate.
            self._scene_bindings = self._gpu.ComputePipeline.scene_bindings_only(
                self.ctx,
                self.shader_dir,
                graph_fragments=list(self._scene_graph_fragments),
                spectral=self._spectral,
            )
            self.pipeline = None
        # `built_sig` reflects what we *attempted*, not the post-fallback
        # state — keeps the rebuild gate idempotent.
        built_sig = attempted_sig

        if self.is_metal:
            # Metal binds resources at dispatch (no Vulkan descriptor pool/sets);
            # only the per-graph SSBO data + material-type codes need uploading.
            # The texture-pool textures are bound directly from the pool at
            # dispatch, so `_update_texture_pool_descriptors` (a Vulkan
            # descriptor-write) is skipped.
            self.descriptor_sets = None
            self._upload_graph_param_buffers()
            self._upload_material_types()
        else:
            # Allocate descriptor pool + sets sized for the new fragment count;
            # then push graph SSBOs, texture-pool slots, and per-material type
            # codes against the freshly-allocated descriptor sets.
            self._create_descriptors()
            self._upload_graph_param_buffers()
            self._update_texture_pool_descriptors()
            self._upload_material_types()
        self._pipeline_built_for_targets = built_sig
        if is_rebuild:
            print(
                f"[skinny] pipeline rebuilt for "
                f"{len(self._scene_graph_fragments)} MaterialX graph(s)"
            )

    def _rebuild_pipeline_for_graphs(self) -> None:
        """Back-compat shim; the unified builder handles both paths."""
        self._build_pipeline_for_current_graphs()

    def _init_gpu(self) -> None:
        # Pipeline + descriptor pool/sets are built lazily by
        # `_build_pipeline_for_current_graphs`, triggered from
        # `_gen_scene_materials` once a scene's MaterialX fragment set is
        # known (USD metadata arrival via `_poll_usd_streaming`) or from
        # the OBJ-load path with an empty fragment set. This avoids a
        # wasted ~9 s slangc compile at startup against an empty fragment
        # list that's immediately discarded when the scene loads.
        self.pipeline = None
        # Backend-independent scene plumbing (set-0 layout + material/dispatcher
        # emission + graph-binding map). Built in BOTH modes; in megakernel mode
        # it IS `self.pipeline` (a full ComputePipeline), in wavefront mode it is
        # a `scene_bindings_only` build with `.pipeline is None`.
        self._scene_bindings = None
        self.descriptor_pool = None
        self.descriptor_sets = None

        # Uniform buffer — FrameConstants + SkinParams + light. Sized with
        # headroom over the scalar blob (see _VK_UNIFORM_BUFFER_BYTES): the
        # upload path truncates silently, so this must stay ≥ len(_pack_uniforms()).
        self.uniform_size = _VK_UNIFORM_BUFFER_BYTES
        self.uniform_buffer = self._gpu.UniformBuffer(self.ctx, self.uniform_size)

        # Per-material skin UBO array (binding 15). StructuredBuffer of
        # MtlxSkinParams, one per material slot — only skin-typed slots
        # (mtlx_target_name == "M_skinny_skin_default") carry data; other
        # slots are zeroed. Filled per-frame via _pack_mtlx_skin_array.
        # Each record is 164 scalar-layout bytes (27 fields, no vec3 padding).
        # _init_materialx_runtime may have set this already from reflection.
        if not hasattr(self, 'mtlx_skin_record_size') or self.mtlx_skin_record_size == 0:
            self.mtlx_skin_record_size = 164
        # On Metal the per-element stride grows (each `float3` pads to 16 B in MSL),
        # and the pipeline reflection that yields the exact stride is built lazily
        # after this point — so size each slot at a safe 256 B ceiling (≥ any MSL
        # stride for the 164 B / 27-field record). The MSL repack writes records at
        # the reflected stride; the buffer only needs to be large enough.
        slot_bytes = 256 if self.is_metal else self.mtlx_skin_record_size
        self.mtlx_skin_buffer = self._gpu.StorageBuffer(
            self.ctx,
            self.material_capacity * slot_bytes + 256,
        )
        # Seed with current SkinParameters → MaterialX defaults so the
        # buffer is valid on frame 0.
        seed = self._pack_mtlx_skin_array()
        if seed:
            self.mtlx_skin_buffer.upload_sync(seed)

        # Persistent HDR accumulation image (progressive convergence).
        # transfer_src=True so screenshot path can copy raw float radiance
        # to a host-visible staging buffer for EXR/HDR export.
        self.accum_image = self._gpu.StorageImage(
            self.ctx, self.width, self.height, transfer_src=True,
        )

        # Per-frame HUD overlay (R8 alpha mask rasterised by Pillow).
        # Pre-zero the staging buffer so the GPU image starts clean even if
        # render() never gets to upload before render_headless() / a
        # screenshot dispatch reads it.
        self.hud_overlay = self._gpu.HudOverlay(self.ctx, self.width, self.height)
        self.hud_overlay.upload(bytes(self.width * self.height))

        # HDR environment texture (RGBA32F, equirectangular).
        from skinny.environment import ENV_HEIGHT, ENV_WIDTH
        self.env_image = self._gpu.SampledImage(self.ctx, ENV_WIDTH, ENV_HEIGHT)

        # Heterogeneous-medium density grid (binding 26, `volumeDensity`,
        # nanovdb-volume-rendering). ALWAYS bound — PARTIALLY_BOUND-style gating
        # is not available for a single binding, so a 1×1×1 zero texture stands
        # in until a scene with a UsdVol.Volume grid loads (same always-bound
        # pattern as the env/tattoo maps); densityAt then reads 0 everywhere.
        # Replaced (destroy + re-create + rebind) per scene by _sync_volume_grid.
        self.volume_density_image = self._gpu.SampledImage3D(self.ctx, 1, 1, 1)
        self.volume_density_image.upload_sync(np.zeros((1, 1, 1), np.float16))
        # Environment importance-sampling CDFs — ONE combined buffer (binding 31,
        # `envDistCdf`): the marginal CDF ([ENV_HEIGHT+1] floats) followed by the
        # conditional CDF ([ENV_HEIGHT*(ENV_WIDTH+1)] floats) at element offset
        # ENV_HEIGHT+1 (change combine-graph-param-buffers — frees a Metal buffer
        # slot). Uploaded by _ensure_env_uploaded; drives env NEE + MIS.
        self.env_dist_buffer = self._gpu.StorageBuffer(
            self.ctx, ((ENV_HEIGHT + 1) + ENV_HEIGHT * (ENV_WIDTH + 1)) * 4)
        # Env sphere luminance integral ∫L dω (Φ_env = πR²·envIntensity·∫L dω)
        # — an SPPM photon-group power input, cached alongside the importance
        # CDF. MUST default before the _ensure_env_uploaded() call below (which
        # computes it); a later re-init would clobber the computed value and
        # silently zero the env photon group.
        self._env_lum_integral: float = 0.0
        self._ensure_env_uploaded()

        # Hero-wavelength spectral upsample tables (bindings 45/46/47), created
        # ONCE and only for the spectral megakernel variant — the RGB path
        # allocates nothing here. Static pbrt data: the sRGB→spectrum sigmoid
        # table (scale grid + [3,res,res,res,3] cube, res==64) and the CIE D65
        # SPD (95 samples on the 360-830/5 nm grid). Uploaded C-order float32.
        self._spectral_scale_buffer = None
        self._spectral_data_buffer = None
        self._spectral_d65_buffer = None
        self._spectral_metals_buffer = None
        # Per-emissive-triangle blackbody metadata (binding 49, Group 6.1): a
        # float2 (temperature_K, scale) parallel-indexed to emissive_tri_buffer
        # (binding 18). Allocated below alongside the triangle buffer, spectral
        # variant only. scale = spectral.blackbody_scale(T, emissiveColor).
        self._spectral_emitters_buffer = None
        if self._spectral:
            from skinny.pbrt import spectral as _spectral_mirror
            from skinny.pbrt.data import spectral_tables
            res, scale, data = spectral_tables.load_srgb_upsample_table()
            # Upload the UNIT-LUMINANCE-normalized D65 (pbrt whitepoint), matching
            # spectral.upsample_illuminant so the GPU upsampleIlluminant ≡ the numpy
            # mirror. Raw D65 (~100× luminance) would make every emitter blow out.
            d65 = _spectral_mirror.d65_normalized()
            if res != 64:
                raise ValueError(
                    f"spectral upsample table res must be 64, got {res}")
            if d65.size != 95:
                raise ValueError(
                    f"spectral D65 SPD must have 95 samples, got {d65.size}")
            scale_arr = np.ascontiguousarray(scale, dtype=np.float32).ravel()
            data_arr = np.ascontiguousarray(data, dtype=np.float32).ravel(order="C")
            d65_arr = np.ascontiguousarray(d65, dtype=np.float32).ravel()
            self._spectral_scale_buffer = self._gpu.StorageBuffer(
                self.ctx, scale_arr.size * 4)
            self._spectral_scale_buffer.upload_sync(scale_arr.tobytes())
            self._spectral_data_buffer = self._gpu.StorageBuffer(
                self.ctx, data_arr.size * 4)
            self._spectral_data_buffer.upload_sync(data_arr.tobytes())
            self._spectral_d65_buffer = self._gpu.StorageBuffer(
                self.ctx, d65_arr.size * 4)
            self._spectral_d65_buffer.upload_sync(d65_arr.tobytes())
            # Named-conductor eta/k (Group 6.2, binding 48): every metal in
            # _SPECTRAL_METAL_ORDER (shader ids 1..N), each [eta(95) | k(95)] on the
            # 360-830/5nm grid, concatenated → N·190 floats. Order MUST match
            # namedMetalEtaK's (metalId-1)*190 indexing in bindings.slang, and N MUST
            # match SPECTRAL_METAL_COUNT there — a shader bound below N silently drops
            # the extra metals to RGB Schlick.
            metal_blocks = []
            for name in _SPECTRAL_METAL_ORDER:
                ek = spectral_tables.named_metal_spectrum(name)
                if ek is None:
                    raise ValueError(f"spectral: missing named-metal curve '{name}'")
                eta, k = ek
                if eta.size != 95 or k.size != 95:
                    raise ValueError(
                        f"spectral metal '{name}' eta/k must be 95 samples, "
                        f"got {eta.size}/{k.size}")
                metal_blocks.append(np.asarray(eta, dtype=np.float32).ravel())
                metal_blocks.append(np.asarray(k, dtype=np.float32).ravel())
            metals_arr = np.ascontiguousarray(
                np.concatenate(metal_blocks), dtype=np.float32)
            self._spectral_metals_buffer = self._gpu.StorageBuffer(
                self.ctx, metals_arr.size * 4)
            self._spectral_metals_buffer.upload_sync(metals_arr.tobytes())

        # Neural-proposal frozen weights (bindings 33/34/35). Sized for the fixed
        # flow architecture and seeded with a dummy (zero) net so the inline flow
        # inverse in proposal.slang always has valid descriptors, even with the
        # neural proposal inactive. Real per-scene weights overwrite them on
        # activation (_sync_neural_weights). The same dummy is the 1a bring-up net.
        from skinny.sampling.neural_weights import make_dummy_weights
        _ncfg = self._effective_neural_config()
        _nw = make_dummy_weights(_ncfg)
        _nwb = _nw.weight_bytes_for(_ncfg.precision)   # half bytes in the fp16 modes
        _nbb = _nw.bias_bytes_for(_ncfg.precision)
        # Under `--neural-handoff interop` the weight buffers are GPU-shareable,
        # per backend (change metal-neural-interop, design D4): on Vulkan they
        # are CUDA-exportable (VK_KHR_external_memory, task 5.1; a guarded no-op
        # on devices without the extension); on Metal the weights+biases live in
        # UMA shared storage the interop publisher writes in place at the
        # frame-boundary swap. Binding 35 (layer headers) is immutable after
        # build and stays device-local on Metal. The file handoff uses plain
        # device-local buffers everywhere.
        _interop_neural = (self._neural_handoff_kind == "interop")
        _is_metal = bool(getattr(self.ctx, "is_metal", False))
        _ext_neural = _interop_neural and not _is_metal
        _shared_neural = (_interop_neural and _is_metal
                          and getattr(self.ctx, "supports_shared_memory", False))
        self.neural_weights_buffer = self._gpu.StorageBuffer(
            self.ctx, max(len(_nwb), 4), external=_ext_neural, shared=_shared_neural)
        self.neural_biases_buffer = self._gpu.StorageBuffer(
            self.ctx, max(len(_nbb), 4), external=_ext_neural, shared=_shared_neural)
        self.neural_layers_buffer = self._gpu.StorageBuffer(self.ctx, max(len(_nw.header_bytes), 4), external=_ext_neural)
        self.neural_weights_buffer.upload_sync(_nwb)
        self.neural_biases_buffer.upload_sync(_nbb)
        self.neural_layers_buffer.upload_sync(_nw.header_bytes)
        self._neural_weights_loaded = None   # path of the net currently uploaded (None = dummy)
        # Interop handoff (task 5.2): an exportable timeline semaphore orders the
        # CUDA weight-write vs the Vulkan read. Allocated only under
        # `--neural-handoff interop` on Vulkan; a guarded no-op
        # (export_handle()→None) on devices without external-semaphore support.
        # Vulkan-only interop primitive — imported lazily so the Metal path
        # (`_ext_neural` False there; its sync is the frame-boundary in-place
        # write, no semaphore) never pulls in `vulkan` (task 2.3).
        if _ext_neural:
            from skinny.vk_compute import ExternalTimelineSemaphore
            self.neural_timeline_semaphore = ExternalTimelineSemaphore(self.ctx)
        else:
            self.neural_timeline_semaphore = None

        # Neural training-record dump (bindings 36/37, task 5.1). 1-element dummy
        # append buffer + counter so the descriptors are always valid; the
        # `mainImageRecord` entry never runs except inside dump_path_records,
        # which reallocates `record_buffer` to the per-frame capacity, binds it,
        # and reads it back. `mainImage` never touches these (dead-stripped).
        from skinny.sampling.path_records import RECORD_STRIDE
        self.record_buffer = self._gpu.StorageBuffer(self.ctx, RECORD_STRIDE)   # 1 dummy record
        self.record_counter = self._gpu.StorageBuffer(self.ctx, 8)             # [count, capacity]
        self._record_pipeline = None   # lazily-built mainImageRecord ComputePipeline
        self._drain_buffer = None      # persistent live-drain target (task 1.2)

        # Tattoo texture (RGBA32F, spherical UV). Seeded with a blank so the
        # descriptor is valid even before the user flips off "None".
        self.tattoo_image = self._gpu.SampledImage(self.ctx, TATTOO_WIDTH, TATTOO_HEIGHT)
        self.tattoo_image.upload_sync(blank_tattoo_data())
        self._ensure_tattoo_uploaded()

        # Per-model detail maps — RGBA8, 2K square. Three images cover
        # normal / roughness / displacement respectively. Seeded with
        # blanks so the descriptors are valid on frame 1.
        self.normal_image = self._gpu.SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format="rgba8_unorm", bytes_per_pixel=4,
        )
        self.roughness_image = self._gpu.SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format="rgba8_unorm", bytes_per_pixel=4,
        )
        self.displacement_image = self._gpu.SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format="rgba8_unorm", bytes_per_pixel=4,
        )
        self.normal_image.upload_sync(blank_normal_bytes())
        self.roughness_image.upload_sync(blank_roughness_bytes())
        self.displacement_image.upload_sync(blank_displacement_bytes())

        # Mesh storage buffers — always bound even when the SDF path is
        # active, so the shader's StructuredBuffer bindings are valid.
        # Sized for the largest source mesh (displacement doesn't change
        # vertex/triangle counts).
        self._dummy_mesh = dummy_mesh()
        max_v = max(
            (src.positions.shape[0] for src in self._mesh_sources),
            default=self._dummy_mesh.num_vertices,
        )
        max_t = max(
            (src.tri_idx.shape[0] for src in self._mesh_sources),
            default=self._dummy_mesh.num_triangles,
        )
        # BVH node count is <= 2·tri_count with our leaf size of 4, but
        # we over-size to keep headroom — cheaper than reallocation on rebake.
        v_size = max_v * 32 + 256
        i_size = max_t * 12 + 256
        b_size = max(max_t * 32, self._dummy_mesh.num_nodes * 32) + 256

        # USD-side budget: when a USD scene is supplied, the buffers must
        # hold every loaded mesh concatenated back-to-back. Take the max
        # of the legacy OBJ rebake budget and the USD concat total so
        # toggling between USD and OBJ slots never overflows.
        if self._usd_scene is not None and self._usd_scene.instances:
            usd_v_bytes = sum(
                inst.mesh.num_vertices * 32 for inst in self._usd_scene.instances
            )
            usd_i_bytes = sum(
                inst.mesh.num_triangles * 12 for inst in self._usd_scene.instances
            )
            usd_b_bytes = sum(
                inst.mesh.num_nodes * 32 for inst in self._usd_scene.instances
            )
            v_size = max(v_size, usd_v_bytes + 256)
            i_size = max(i_size, usd_i_bytes + 256)
            b_size = max(b_size, usd_b_bytes + 256)
        self.vertex_buffer = self._gpu.StorageBuffer(self.ctx, v_size)
        self.index_buffer = self._gpu.StorageBuffer(self.ctx, i_size)
        self.bvh_buffer = self._gpu.StorageBuffer(self.ctx, b_size)
        # Upload the dummy mesh so the buffers are valid on first frame
        # even before the user picks a real mesh (or if none are present).
        self._upload_mesh(self._dummy_mesh)

        # TLAS instance buffer — one record per renderable mesh instance.
        # Phase B always carries exactly one identity-transform instance, so
        # the GPU's broad-phase loop is mathematically a no-op pass-through
        # to the BLAS traversal. Sized for INSTANCE_CAPACITY entries up front
        # so the upload path can grow into multi-mesh scenes (Phase D)
        # without reallocation.
        self.instance_capacity = 16
        self.instance_buffer = self._gpu.StorageBuffer(
            self.ctx, self.instance_capacity * INSTANCE_STRIDE + 256
        )
        self._upload_instances([np.eye(4, dtype=np.float32)], material_ids=[0])
        self._num_instances = 1

        # Flat-material parameter buffer — one record per scene material.
        # Sized for FLAT_MATERIAL_CAPACITY entries up front.
        self.flat_material_buffer = self._gpu.StorageBuffer(
            self.ctx, self.material_capacity * FLAT_MATERIAL_STRIDE + 256
        )
        # Initialize with one zeroed record so the buffer is valid even
        # before any USD scene is loaded.
        self.flat_material_buffer.upload_sync(b"\x00" * FLAT_MATERIAL_STRIDE)
        self._num_flat_materials = 0
        # Per-material blackbody emission (binding 51, Group 6.1 follow-up),
        # spectral variant only: float2 (temperature_K, scale) per flat material,
        # indexed by materialId. Lets a camera-visible / BSDF-hit blackbody emitter
        # use the exact Planck SPD (matching NEE) instead of the RGB upsample.
        # Sized/grown parallel to flat_material_buffer; zeros ⇒ RGB upsample.
        self._spectral_mat_emission_buffer = None
        if self._spectral:
            self._spectral_mat_emission_buffer = self._gpu.StorageBuffer(
                self.ctx, self.material_capacity * SPECTRAL_EMITTER_STRIDE + 16
            )
            self._spectral_mat_emission_buffer.upload_sync(
                b"\x00" * (self.material_capacity * SPECTRAL_EMITTER_STRIDE)
            )

        # Bindless texture array (binding 14). Slots are populated lazily by
        # `_upload_flat_materials` from each Material.texture_paths entry.
        self.texture_pool = TexturePool(self.ctx, self._gpu)

        # Per-material type-code buffer (binding 16). One uint per slot,
        # written each time _upload_flat_materials runs.
        self.material_types_buffer = self._gpu.StorageBuffer(
            self.ctx, self.material_capacity * 4 + 16
        )
        self._material_types: list[int] = [MATERIAL_TYPE_FLAT]
        # Seed with MATERIAL_TYPE_FLAT so no slot defaults to skin.
        init_types = bytearray()
        for _ in range(self.material_capacity):
            init_types += struct.pack("I", MATERIAL_TYPE_FLAT)
        self.material_types_buffer.upload_sync(bytes(init_types))

        # Sphere-light buffer (binding 17). Filled from
        # scene.lights_sphere; fc.numSphereLights bounds the active range.
        self.sphere_lights_buffer = self._gpu.StorageBuffer(
            self.ctx, SPHERE_LIGHT_CAPACITY * SPHERE_LIGHT_STRIDE + 16
        )
        self.sphere_lights_buffer.upload_sync(
            b"\x00" * (SPHERE_LIGHT_CAPACITY * SPHERE_LIGHT_STRIDE)
        )
        self._num_sphere_lights: int = 0
        # Σ(lum·r²) over the packed sphere lights — feeds the SPPM photon-group
        # power distribution (Φ_S = 4π²·Σ(lum·r²)); refreshed by every
        # _upload_sphere_lights call so live light edits stay consistent.
        self._sphere_power_sum: float = 0.0

        # Distant-light buffer (binding 20). Filled from scene.lights_dir;
        # fc.numDistantLights bounds the active range. Replaces the legacy
        # single lightDirection/lightRadiance uniforms so the integrators
        # can iterate every authored distant light via DirectionalLightImpl
        # (ILight).
        self.distant_lights_buffer = self._gpu.StorageBuffer(
            self.ctx, DISTANT_LIGHT_CAPACITY * DISTANT_LIGHT_STRIDE + 16
        )
        self.distant_lights_buffer.upload_sync(
            b"\x00" * (DISTANT_LIGHT_CAPACITY * DISTANT_LIGHT_STRIDE)
        )
        self._num_distant_lights: int = 0
        # Σlum over the packed distant lights (Φ_D = πR²·Σlum) — an SPPM
        # photon-group power input (see _sppm_photon_group_pmf).
        # _sppm_group_pmf_override (a 4-tuple) bypasses the power distribution
        # and packs verbatim — the forced-group flux-normalization probe hook
        # ([0,0,0,1] = all-env). NOTE: the companion _env_lum_integral is
        # initialized *before* the env buffer construction above —
        # _ensure_env_uploaded() already ran at construction and a default here
        # would clobber the computed integral (env pmf silently 0).
        self._distant_lum_sum: float = 0.0
        self._sppm_group_pmf_override: tuple | None = None
        # Per-distant-light authored illuminant SPD (binding 50, Group 6.3),
        # spectral variant only — fixed capacity (distant lights never grow past
        # DISTANT_LIGHT_CAPACITY), so no rebind path. Filled in
        # _upload_distant_lights; zeros when no light carries an SPD.
        self._spectral_light_spd_buffer = None
        if self._spectral:
            self._spectral_light_spd_buffer = self._gpu.StorageBuffer(
                self.ctx, DISTANT_LIGHT_CAPACITY * SPECTRAL_LIGHT_SPD_STRIDE + 16
            )
            self._spectral_light_spd_buffer.upload_sync(
                b"\x00" * (DISTANT_LIGHT_CAPACITY * SPECTRAL_LIGHT_SPD_STRIDE)
            )

        # Emissive-triangle buffer (binding 18). Built from scene instances
        # whose material has non-zero emissiveColor. The shader samples one
        # triangle per pixel per frame for next-event estimation. EMISSIVE_TRI_CAPACITY
        # is only the initial capacity — _upload_emissive_triangles grows it to the
        # actual emissive-triangle count (no silent 256-cap; change emissive-mesh-nee).
        self.emissive_tri_capacity: int = EMISSIVE_TRI_CAPACITY
        self.emissive_tri_buffer = self._gpu.StorageBuffer(
            self.ctx, self.emissive_tri_capacity * EMISSIVE_TRI_STRIDE + 16
        )
        self.emissive_tri_buffer.upload_sync(
            b"\x00" * (self.emissive_tri_capacity * EMISSIVE_TRI_STRIDE)
        )
        self._num_emissive_tris: int = 0
        # Spectral emitter metadata (binding 49): float2 (T, scale) per triangle,
        # sized/grown parallel to emissive_tri_buffer. Spectral variant only, so
        # the RGB descriptor layout stays byte-identical.
        if self._spectral:
            self._spectral_emitters_buffer = self._gpu.StorageBuffer(
                self.ctx, self.emissive_tri_capacity * SPECTRAL_EMITTER_STRIDE + 16
            )
            self._spectral_emitters_buffer.upload_sync(
                b"\x00" * (self.emissive_tri_capacity * SPECTRAL_EMITTER_STRIDE)
            )
        # Σ(area·Rec709-lum) over emissive triangles → FrameConstants.emissiveTotalPower
        # (the path tracer's BSDF-hit MIS weight). Set in _upload_emissive_triangles.
        self._emissive_total_power: float = 0.0
        # Test hook (change emissive-mesh-nee): force uniform-by-index emissive
        # selection (build the inline CDF uniform) for the power-vs-uniform A/B.
        self._emissive_uniform_selection: bool = False

        # StdSurfaceParams buffer (binding 19). One 256-byte record per
        # material slot, carrying the full MaterialX standard_surface input
        # set for evalStdSurfaceBSDF in mtlx_std_surface.slang.
        self.std_surface_buffer = self._gpu.StorageBuffer(
            self.ctx, self.material_capacity * STD_SURFACE_STRIDE + 16
        )
        self.std_surface_buffer.upload_sync(
            b"\x00" * (self.material_capacity * STD_SURFACE_STRIDE)
        )

        # BDPT light-tracer splat buffer (binding 21). 3 × uint32 per pixel
        # (Q22.10 fixed-point, atomic-add target). Cleared via fill_zero_sync
        # whenever the accumulation resets so the running mean stays correct.
        self.light_splat_buffer = self._gpu.StorageBuffer(
            self.ctx, self.width * self.height * 3 * 4
        )
        self.light_splat_buffer.fill_zero_sync()

        # Gizmo segment buffer (binding 22). Holds at most
        # GIZMO_SEGMENT_CAPACITY 32-byte records (2 float2 endpoints, float3
        # colour, float half-width). Repacked every frame from
        # ``self.gizmo`` when the user has selected an instance.
        from skinny.gizmo import (
            GIZMO_SEGMENT_CAPACITY, GIZMO_SEGMENT_STRIDE, TransformGizmo,
        )
        self.gizmo_segment_capacity = GIZMO_SEGMENT_CAPACITY
        self.gizmo_segment_stride = GIZMO_SEGMENT_STRIDE
        self.gizmo_segments_buffer = self._gpu.StorageBuffer(
            self.ctx,
            self.gizmo_segment_capacity * self.gizmo_segment_stride + 16,
        )
        self.gizmo_segments_buffer.upload_sync(
            b"\x00" * (self.gizmo_segment_capacity * self.gizmo_segment_stride)
        )
        self.gizmo = TransformGizmo()
        self._num_gizmo_segments: int = 0
        self.show_focus_overlay: bool = False
        self.lens_vignette_debug: bool = False

        # Viewport zoom-rect: a sub-rectangle of the output (in
        # normalised pixel coords) that gets stretched to fill the
        # window. (0,0)→(1,1) means no zoom; tighter bounds magnify a
        # selected region without moving the camera.
        self.zoom_rect: list[float] = [0.0, 0.0, 1.0, 1.0]
        # Live drag rectangle (pixel coords) — drawn as a yellow outline
        # via the gizmo segment list while the user picks a sub-region.
        self._zoom_drag_overlay: Optional[tuple[float, float, float, float]] = None

        # Thick-lens element buffer (binding 23). Each element is a
        # 16-byte float4: (radius_world, thickness_world, ior, half_aperture_world).
        # Capped at MAX_LENS_ELEMENTS so the SSBO size is fixed at startup.
        # Repacked from the active camera's `LensSystem` whenever the lens
        # signature changes; otherwise reused frame to frame.
        self.lens_element_capacity = MAX_LENS_ELEMENTS
        self.lens_element_stride = 16   # float4
        self.lens_elements_buffer = self._gpu.StorageBuffer(
            self.ctx,
            self.lens_element_capacity * self.lens_element_stride + 16,
        )
        self.lens_elements_buffer.upload_sync(
            b"\x00" * (self.lens_element_capacity * self.lens_element_stride)
        )
        self._packed_lens_signature: object = None
        self._lens_film_distance_world: float = 0.0
        self._lens_rear_z_world: float = 0.0
        self._lens_rear_aperture_world: float = 0.0
        self._lens_front_z_world: float = 0.0
        self._lens_iris_z_world: float = 0.0
        self._lens_active_count: int = 0
        self._lens_film_diag_world: float = 0.0
        self._lens_num_pupil_bounds: int = 0
        # Exit-pupil bounds buffer (binding 24): 64 × float4
        # (xMin, xMax, yMin, yMax) per film-radius bin. PBRT's
        # `BoundExitPupil`. Lets the shader sample only the rear-disk
        # subregion that produces non-vignetted rays for each pixel,
        # keeping the rendered area full-screen even at small fstops.
        self.lens_pupil_capacity = 64
        self.lens_pupil_stride = 16
        self.lens_pupil_buffer = self._gpu.StorageBuffer(
            self.ctx,
            self.lens_pupil_capacity * self.lens_pupil_stride + 16,
        )
        self.lens_pupil_buffer.upload_sync(
            b"\x00" * (self.lens_pupil_capacity * self.lens_pupil_stride)
        )

        # Bumped any time apply_material_override mutates a scene material's
        # parameter_overrides. Hashed into _current_state_hash so the
        # progressive accumulation resets on a slider drag in the
        # per-material panel.
        self._material_version: int = 0

        # Offscreen output image + readback buffer. Always created — used by
        # render_headless() (web path) and by save_screenshot() in both
        # windowed and headless modes. In windowed mode render() rebinds
        # binding 1 to the swapchain image per frame, so this offscreen
        # only sees writes during the screenshot path.
        # Must be created before _create_descriptors which writes binding 1.
        self._offscreen_output = self._gpu.StorageImage(
            self.ctx, self.width, self.height,
            format="rgba8_unorm",
            transfer_src=True,
        )
        self._readback = self._gpu.ReadbackBuffer(self.ctx, self.width, self.height)

        # BXDF visualizer output (binding 30). Host-visible SSBO holding
        # the picked HitInfo and (future) BXDF eval grid. Sized for a
        # 128 × 64 float4 lobe grid + 32-slot header, plus headroom.
        self.tool_buffer = self._gpu.HostStorageBuffer(self.ctx, 128 * 64 * 16 + 4096)
        self._pick_armed: bool = False
        self._pick_pixel: tuple[int, int] = (0, 0)
        self._pick_frame_count: int = 0
        self._pending_pick_callbacks: list = []

        # Descriptor pool + sets are created lazily inside
        # `_build_pipeline_for_current_graphs` because pool sizing depends
        # on `_scene_graph_fragments`, which is empty here at startup.

        # Command buffers
        self.command_buffers = self.ctx.allocate_command_buffers(MAX_FRAMES_IN_FLIGHT)

        # Synchronisation. The Metal megakernel dispatch is synchronous
        # (submit + wait_for_idle each frame), so it needs no Vulkan
        # fences/semaphores — the present-smoke fence is owned by the surface.
        if self.is_metal:
            self.image_available = []
            self.render_finished = []
            self.in_flight_fences = []
        elif self.ctx.swapchain_info is not None:
            swapchain_image_count = len(self.ctx.swapchain_info.images)
            self.image_available = [
                vk.vkCreateSemaphore(
                    self.ctx.device, vk.VkSemaphoreCreateInfo(), None
                )
                for _ in range(MAX_FRAMES_IN_FLIGHT)
            ]
            self.render_finished = [
                vk.vkCreateSemaphore(
                    self.ctx.device, vk.VkSemaphoreCreateInfo(), None
                )
                for _ in range(swapchain_image_count)
            ]
        else:
            self.image_available = []
            self.render_finished = []

        if not self.is_metal:
            self.in_flight_fences = [
                vk.vkCreateFence(
                    self.ctx.device,
                    vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT),
                    None,
                )
                for _ in range(MAX_FRAMES_IN_FLIGHT)
            ]

        self.current_frame = 0

    def _create_descriptors(self) -> None:
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                # One UBO per descriptor set (FrameConstants at binding 0).
                descriptorCount=MAX_FRAMES_IN_FLIGHT,
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=MAX_FRAMES_IN_FLIGHT,
            ),
        ]
        # Storage-image descriptors per frame: swapchain + accumulation + HUD.
        pool_sizes[1] = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=MAX_FRAMES_IN_FLIGHT * 3,
        )
        pool_sizes.append(
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                # env + tattoo + n/r/d + volume density grid (6)
                # + bindless flat-material array.
                descriptorCount=MAX_FRAMES_IN_FLIGHT * (6 + self._gpu.BINDLESS_TEXTURE_CAPACITY),
            )
        )
        # Storage buffers per frame: vertices, indices, BVH nodes, TLAS
        # instances, flat-material params, material type codes,
        # per-material skin UBO array, sphere lights, emissive triangles,
        # StdSurfaceParams, plus the ONE combined MaterialX graph-param buffer
        # (binding GRAPH_BINDING_BASE) when the scene carries any graph.
        graph_slot = 1 if (getattr(self, "_scene_graph_fragments", []) or []) else 0
        pool_sizes.append(
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                # 17 fixed = vertices+indices+bvh+instances+flatMaterials+
                #      materialTypes+mtlxSkin+sphereLights+emissiveTris+
                #      stdSurface+lightSplat+gizmoSegments+
                #      lensElements+lensPupilBounds+distantLights+toolBuffer+
                #      envDistCdf (one combined env CDF buffer).
                # +3 neural weights (33/34/35) +2 record dump (36/37) = 22.
                # +7 spectral buffers (45/46/47 upsample + 48 conductor eta/k +
                # 49 emissive blackbody + 50 distant-light SPD + 51 per-material
                # blackbody) only for the spectral megakernel variant.
                # +6 MLT chain buffers (52–57, change mlt-integrator/spectral-mlt) only on
                # the wavefront (`scene_bindings_only`) layout.
                descriptorCount=MAX_FRAMES_IN_FLIGHT
                * (22 + graph_slot + (7 if self._spectral else 0)
                   + (6 if getattr(self._scene_bindings, "mlt_bindings", False) else 0)),
            )
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            flags=0x00000002,  # VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT
            maxSets=MAX_FRAMES_IN_FLIGHT,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.ctx.device, pool_info, None
        )

        layouts = [self._scene_set0_layout] * MAX_FRAMES_IN_FLIGHT
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=MAX_FRAMES_IN_FLIGHT,
            pSetLayouts=layouts,
        )
        self.descriptor_sets = vk.vkAllocateDescriptorSets(
            self.ctx.device, alloc_info
        )

        # Write descriptors (UBO at binding 0, accumulation image at binding 2).
        # Binding 1 (swapchain image) is updated per-frame in render() because
        # the acquired image index changes. In headless mode, binding 1 points
        # to the persistent offscreen output image and is written here once.
        for ds in self.descriptor_sets:
            buf_info = vk.VkDescriptorBufferInfo(
                buffer=self.uniform_buffer.buffer,
                offset=0,
                range=self.uniform_size,
            )
            accum_info = vk.VkDescriptorImageInfo(
                imageView=self.accum_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            hud_info = vk.VkDescriptorImageInfo(
                imageView=self.hud_overlay.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            env_info = vk.VkDescriptorImageInfo(
                sampler=self.env_image.sampler,
                imageView=self.env_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            tattoo_info = vk.VkDescriptorImageInfo(
                sampler=self.tattoo_image.sampler,
                imageView=self.tattoo_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            normal_info = vk.VkDescriptorImageInfo(
                sampler=self.normal_image.sampler,
                imageView=self.normal_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            rough_info = vk.VkDescriptorImageInfo(
                sampler=self.roughness_image.sampler,
                imageView=self.roughness_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            disp_info = vk.VkDescriptorImageInfo(
                sampler=self.displacement_image.sampler,
                imageView=self.displacement_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            # Heterogeneous-medium density grid (binding 26, always bound —
            # the 1×1×1 zero fallback until a scene grid uploads; per-scene
            # swaps re-write via _rebind_volume_descriptor).
            volume_info = vk.VkDescriptorImageInfo(
                sampler=self.volume_density_image.sampler,
                imageView=self.volume_density_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            vtx_info = vk.VkDescriptorBufferInfo(
                buffer=self.vertex_buffer.buffer, offset=0, range=self.vertex_buffer.size,
            )
            idx_info = vk.VkDescriptorBufferInfo(
                buffer=self.index_buffer.buffer, offset=0, range=self.index_buffer.size,
            )
            bvh_info = vk.VkDescriptorBufferInfo(
                buffer=self.bvh_buffer.buffer, offset=0, range=self.bvh_buffer.size,
            )
            inst_info = vk.VkDescriptorBufferInfo(
                buffer=self.instance_buffer.buffer, offset=0, range=self.instance_buffer.size,
            )
            mat_info = vk.VkDescriptorBufferInfo(
                buffer=self.flat_material_buffer.buffer,
                offset=0,
                range=self.flat_material_buffer.size,
            )
            mtlx_skin_info = vk.VkDescriptorBufferInfo(
                buffer=self.mtlx_skin_buffer.buffer,
                offset=0,
                range=self.mtlx_skin_buffer.size,
            )
            mat_types_info = vk.VkDescriptorBufferInfo(
                buffer=self.material_types_buffer.buffer,
                offset=0,
                range=self.material_types_buffer.size,
            )
            sphere_lights_info = vk.VkDescriptorBufferInfo(
                buffer=self.sphere_lights_buffer.buffer,
                offset=0,
                range=self.sphere_lights_buffer.size,
            )
            emissive_tri_info = vk.VkDescriptorBufferInfo(
                buffer=self.emissive_tri_buffer.buffer,
                offset=0,
                range=self.emissive_tri_buffer.size,
            )
            std_surface_info = vk.VkDescriptorBufferInfo(
                buffer=self.std_surface_buffer.buffer,
                offset=0,
                range=self.std_surface_buffer.size,
            )
            light_splat_info = vk.VkDescriptorBufferInfo(
                buffer=self.light_splat_buffer.buffer,
                offset=0,
                range=self.light_splat_buffer.size,
            )
            gizmo_info = vk.VkDescriptorBufferInfo(
                buffer=self.gizmo_segments_buffer.buffer,
                offset=0,
                range=self.gizmo_segments_buffer.size,
            )
            lens_info = vk.VkDescriptorBufferInfo(
                buffer=self.lens_elements_buffer.buffer,
                offset=0,
                range=self.lens_elements_buffer.size,
            )
            lens_pupil_info = vk.VkDescriptorBufferInfo(
                buffer=self.lens_pupil_buffer.buffer,
                offset=0,
                range=self.lens_pupil_buffer.size,
            )
            distant_lights_info = vk.VkDescriptorBufferInfo(
                buffer=self.distant_lights_buffer.buffer,
                offset=0,
                range=self.distant_lights_buffer.size,
            )
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=0,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    pBufferInfo=[buf_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=2,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[accum_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=3,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[hud_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=4,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[env_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=5,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vtx_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=6,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[idx_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=7,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[bvh_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=8,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[tattoo_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=9,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[normal_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=10,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[rough_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=11,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[disp_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=26,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[volume_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=12,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[inst_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=13,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[mat_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=15,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[mtlx_skin_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=16,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[mat_types_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=17,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[sphere_lights_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=18,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[emissive_tri_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=19,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[std_surface_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=21,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[light_splat_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=22,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[gizmo_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=23,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[lens_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=24,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[lens_pupil_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=20,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[distant_lights_info],
                ),
            ]
            output_info = vk.VkDescriptorImageInfo(
                imageView=self._offscreen_output.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=ds,
                dstBinding=1,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[output_info],
            ))
            tool_info = vk.VkDescriptorBufferInfo(
                buffer=self.tool_buffer.buffer, offset=0, range=self.tool_buffer.size,
            )
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=ds,
                dstBinding=30,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[tool_info],
            ))
            env_dist_info = vk.VkDescriptorBufferInfo(
                buffer=self.env_dist_buffer.buffer, offset=0,
                range=self.env_dist_buffer.size,
            )
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=ds,
                dstBinding=31,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[env_dist_info],
            ))
            # Neural proposal weights (33/34/35). Always bound (dummy net when
            # inactive) so the inline flow inverse in proposal.slang has valid
            # descriptors on every pipeline that uses this layout.
            for _b, _buf in ((33, self.neural_weights_buffer),
                             (34, self.neural_biases_buffer),
                             (35, self.neural_layers_buffer),
                             # Training-record dump (36 = PathRecord append, 37 =
                             # counter); dummies until dump_path_records rebinds.
                             (36, self.record_buffer),
                             (37, self.record_counter)):
                writes.append(vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=_b, dstArrayElement=0, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vk.VkDescriptorBufferInfo(
                        buffer=_buf.buffer, offset=0, range=_buf.size)],
                ))
            # MLT chain buffers (52–57, change mlt-integrator/spectral-mlt) — only the
            # wavefront (`scene_bindings_only`) layout declares them; dummies
            # until `_ensure_wavefront_mlt_pass` rebinds the real chain
            # buffers (the 36/37 record-dump precedent).
            if getattr(self._scene_bindings, "mlt_bindings", False):
                for _b in (52, 53, 54, 55, 56, 57):
                    writes.append(vk.VkWriteDescriptorSet(
                        dstSet=ds, dstBinding=_b, dstArrayElement=0, descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[vk.VkDescriptorBufferInfo(
                            buffer=self.record_counter.buffer, offset=0,
                            range=self.record_counter.size)],
                    ))
            # Spectral upsample tables (45/46/47) — only the spectral variant's
            # layout declares these bindings; the RGB path writes nothing.
            if self._spectral:
                for _b, _buf in ((45, self._spectral_scale_buffer),
                                 (46, self._spectral_data_buffer),
                                 (47, self._spectral_d65_buffer),
                                 (48, self._spectral_metals_buffer),
                                 (49, self._spectral_emitters_buffer),
                                 (50, self._spectral_light_spd_buffer),
                                 (51, self._spectral_mat_emission_buffer)):
                    writes.append(vk.VkWriteDescriptorSet(
                        dstSet=ds, dstBinding=_b, dstArrayElement=0,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[vk.VkDescriptorBufferInfo(
                            buffer=_buf.buffer, offset=0, range=_buf.size)],
                    ))
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _rebind_scene_descriptors(self) -> None:
        """Re-write descriptor bindings 12, 13, 15, 16 after buffer reallocation."""
        # Metal has no Vulkan descriptor sets; `_build_metal_binds` re-reads each
        # buffer reference fresh at every dispatch, so a realloc is picked up
        # automatically. Without this guard the Vulkan `VkDescriptorBufferInfo`
        # call below fails on a slangpy buffer with
        # `TypeError: an integer is required` (e.g. a 20+-instance scene grows the
        # instance buffer and trips the rebind mid-stream).
        if self.is_metal:
            return
        inst_info = vk.VkDescriptorBufferInfo(
            buffer=self.instance_buffer.buffer, offset=0,
            range=self.instance_buffer.size,
        )
        mat_info = vk.VkDescriptorBufferInfo(
            buffer=self.flat_material_buffer.buffer, offset=0,
            range=self.flat_material_buffer.size,
        )
        mtlx_skin_info = vk.VkDescriptorBufferInfo(
            buffer=self.mtlx_skin_buffer.buffer, offset=0,
            range=self.mtlx_skin_buffer.size,
        )
        mat_types_info = vk.VkDescriptorBufferInfo(
            buffer=self.material_types_buffer.buffer, offset=0,
            range=self.material_types_buffer.size,
        )
        for ds in self.descriptor_sets:
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=12, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[inst_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=13, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[mat_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=15, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[mtlx_skin_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=16, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[mat_types_info],
                ),
            ]
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _rebind_aux_material_descriptors(self) -> None:
        """Re-write descriptor binding 19 after buffer reallocation."""
        # Vulkan-only (see `_rebind_scene_descriptors`): Metal rebinds the live
        # std_surface buffer at dispatch, so this is a no-op there.
        if self.is_metal:
            return
        ss_info = vk.VkDescriptorBufferInfo(
            buffer=self.std_surface_buffer.buffer, offset=0,
            range=self.std_surface_buffer.size,
        )
        for ds in self.descriptor_sets:
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=19, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[ss_info],
                ),
            ]
            # Spectral (Group 6.1 follow-up): binding 51 grows with the material
            # buffer, so rewrite it here too or it points at the freed buffer.
            if self._spectral and self._spectral_mat_emission_buffer is not None:
                writes.append(vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=51, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vk.VkDescriptorBufferInfo(
                        buffer=self._spectral_mat_emission_buffer.buffer, offset=0,
                        range=self._spectral_mat_emission_buffer.size)],
                ))
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _rebind_emissive_descriptors(self) -> None:
        """Re-write descriptor binding 18 after the emissive-triangle buffer grows
        (change emissive-mesh-nee)."""
        # Vulkan-only (see `_rebind_scene_descriptors`): native Metal re-reads the
        # live emissive_tri_buffer reference at dispatch, so this is a no-op there.
        if self.is_metal:
            return
        emissive_tri_info = vk.VkDescriptorBufferInfo(
            buffer=self.emissive_tri_buffer.buffer, offset=0,
            range=self.emissive_tri_buffer.size,
        )
        for ds in self.descriptor_sets:
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=18, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[emissive_tri_info],
                ),
            ]
            # Spectral (Group 6.1): the parallel-indexed spectralEmitters buffer
            # (binding 49) is destroyed+recreated in the SAME growth step, so its
            # descriptor must be rewritten here too — otherwise binding 49 keeps
            # pointing at the freed buffer and emitterBlackbody() reads stale memory.
            if self._spectral and self._spectral_emitters_buffer is not None:
                writes.append(vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=49, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vk.VkDescriptorBufferInfo(
                        buffer=self._spectral_emitters_buffer.buffer, offset=0,
                        range=self._spectral_emitters_buffer.size)],
                ))
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _rebind_mesh_descriptors(self) -> None:
        """Re-write descriptor bindings 5, 6, 7 after buffer reallocation."""
        # Vulkan-only (see `_rebind_scene_descriptors`): Metal rebinds the live
        # vertex/index/BVH buffers at dispatch, so this is a no-op there. The mesh-
        # grow path ("growing mesh buffers …") on a 20+-instance scene hits this
        # first, otherwise crashing with `TypeError: an integer is required`.
        if self.is_metal:
            return
        vtx_info = vk.VkDescriptorBufferInfo(
            buffer=self.vertex_buffer.buffer, offset=0, range=self.vertex_buffer.size,
        )
        idx_info = vk.VkDescriptorBufferInfo(
            buffer=self.index_buffer.buffer, offset=0, range=self.index_buffer.size,
        )
        bvh_info = vk.VkDescriptorBufferInfo(
            buffer=self.bvh_buffer.buffer, offset=0, range=self.bvh_buffer.size,
        )
        for ds in self.descriptor_sets:
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=5, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vtx_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=6, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[idx_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=7, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[bvh_info],
                ),
            ]
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _ensure_mesh_buffer_capacity(
        self, num_vertices: int, num_triangles: int, num_nodes: int,
    ) -> None:
        """Grow mesh storage buffers if needed, rebinding descriptors."""
        v_need = num_vertices * 32 + 256
        i_need = num_triangles * 12 + 256
        b_need = num_nodes * 32 + 256

        if (v_need <= self.vertex_buffer.size
                and i_need <= self.index_buffer.size
                and b_need <= self.bvh_buffer.size):
            return

        # Backend-neutral drain before freeing in-flight mesh buffers.
        self.ctx.wait_idle()

        v_new = max(self.vertex_buffer.size * 2, v_need)
        i_new = max(self.index_buffer.size * 2, i_need)
        b_new = max(self.bvh_buffer.size * 2, b_need)

        print(
            f"[skinny] growing mesh buffers: "
            f"vtx {self.vertex_buffer.size}→{v_new}, "
            f"idx {self.index_buffer.size}→{i_new}, "
            f"bvh {self.bvh_buffer.size}→{b_new}"
        )

        self.vertex_buffer.destroy()
        self.index_buffer.destroy()
        self.bvh_buffer.destroy()

        self.vertex_buffer = self._gpu.StorageBuffer(self.ctx, v_new)
        self.index_buffer = self._gpu.StorageBuffer(self.ctx, i_new)
        self.bvh_buffer = self._gpu.StorageBuffer(self.ctx, b_new)

        # Metal binds mesh buffers by reference at dispatch (no Vulkan descriptor
        # sets — `descriptor_sets is None`), so the next dispatch picks up the
        # freshly-allocated buffers automatically; the descriptor rewrite is a
        # Vulkan-only step. It is also skipped when the descriptor sets don't
        # exist yet (a large mesh can grow the buffers before the pipeline build
        # allocates them — e.g. loading a big OBJ before the lazy build runs); the
        # subsequent `_build_pipeline_for_current_graphs` writes the descriptors
        # against the current buffers.
        if not self.is_metal and self.descriptor_sets is not None:
            self._rebind_mesh_descriptors()
        self.vertex_buffer.upload_sync(self._dummy_mesh.vertex_bytes)
        self.index_buffer.upload_sync(self._dummy_mesh.index_bytes)
        self.bvh_buffer.upload_sync(self._dummy_mesh.bvh_bytes)

    def _poll_usd_streaming(self) -> None:
        """Poll USD background thread: metadata first, then instances."""
        if self._usd_bake_done is None:
            return
        import queue as _queue

        # Phase 1: metadata (lights, camera, materials, mm_per_unit)
        if self._usd_scene is None:
            try:
                scene, sg = self._usd_metadata_queue.get_nowait()
            except _queue.Empty:
                return
            self._usd_scene = scene
            self._scene_graph = sg
            self._gen_scene_materials()
            self._frame_camera_to_scene(scene)
            # Seed the USD camera follower from the default-time camera so the
            # user can switch to usd mode before pressing play.
            if scene.camera_override is not None:
                self._override_to_orbit(self.usd_camera, scene.camera_override, scene)
            # Apply authored skinny:ui:default values now that the scene +
            # materials exist (material/usd targets need them).
            if self._usd_controls:
                self._apply_control_defaults()
            # Inject /Skinny/MainCamera *after* _apply_camera_override so the
            # node snapshot captures any authored thick lens.
            self._refresh_camera_node()
            # Layer synthetic /Skinny/DefaultLight + /Skinny/DefaultDome onto
            # the loaded graph so the user can still see the renderer-owned
            # direct light + IBL when the USD asset omits them.
            self._inject_default_lights_into_scene_graph()
            if scene.mm_per_unit != 120.0:
                self.mm_per_unit = scene.mm_per_unit
            # Film per-sample radiance clamp from the imported pbrt film
            # (change film-maxcomponent-clamp); 0 = disabled (no-op).
            self.film_max_component = float(getattr(scene, "film_max_component", 0.0) or 0.0)
            # Heterogeneous-medium density grid (nanovdb-volume-rendering):
            # upload once per scene, after mm_per_unit is final and before the
            # material upload below packs the volume σ folds.
            self._sync_volume_grid(scene)
            if scene.instances:
                self._upload_usd_scene()
                self._usd_uploaded_count = len(scene.instances)
            print(
                f"[skinny] USD metadata applied — "
                f"{len(scene.materials)} materials, "
                f"{len(scene.lights_dir)} dir lights"
            )

        # Phase 2: baked mesh instances
        added = 0
        first_name = None
        while True:
            try:
                inst = self._usd_instance_queue.get_nowait()
            except _queue.Empty:
                break
            self._usd_scene.instances.append(inst)
            added += 1
            if first_name is None:
                first_name = inst.name
            print(
                f"[skinny] USD streamed '{inst.name}' — "
                f"{len(self._usd_scene.instances)} instance(s)"
            )
        if first_name is not None and self.models[self._usd_model_index].endswith("(loading...)"):
            self.models[self._usd_model_index] = f"USD: {first_name}"

        if added > 0 and self._is_usd_active():
            self._upload_usd_scene()
            self._usd_uploaded_count = len(self._usd_scene.instances)

        if added > 0 and self._scene_graph is not None:
            from skinny.scene_graph import populate_instance_refs
            updated = populate_instance_refs(self._scene_graph, self._usd_scene)
            if updated:
                # In-place ref/metadata update — bump the version so the dock's
                # open property panel doesn't go stale (same tree object).
                self._scene_graph_version = getattr(
                    self, "_scene_graph_version", 0) + 1
                print(
                    f"[skinny] scene graph: attached {updated} instance ref(s)"
                )

        if self._usd_bake_done.is_set() and self._usd_instance_queue.empty():
            self._usd_bake_done = None
            # Re-frame now that all geometry exists. The metadata-phase frame
            # ran before any instance streamed in, so world_bounds() was None
            # and it early-returned, leaving the camera at its defaults.
            if self._is_usd_active():
                self._frame_camera_to_scene(self._usd_scene)
                self._refresh_camera_node()
            # Build GPU skinning passes now that all skinned BLASes are baked
            # + uploaded (so their rest bytes + buffer offsets are final).
            _idx = getattr(self, "_anim_index", None)
            if _idx is not None and _idx.skinned_mesh_paths:
                self._build_skinning_passes()
            print(
                f"[skinny] USD streaming complete — "
                f"{len(self._usd_scene.instances)} instance(s)"
            )


    @property
    def scene_graph(self):
        return self._scene_graph

    def _is_usd_active(self) -> bool:
        return (
            self._usd_model_index >= 0
            and self.model_index == self._usd_model_index
        )

    def _upload_mesh(self, mesh: Mesh) -> None:
        # Legacy single-mesh OBJ path: writes one BLAS at offset 0, clobbering
        # any suballocated slab layout. Reset the slab allocator so a later USD
        # resync rebuilds from scratch instead of trusting stale offsets.
        from skinny.slab_allocator import SlabAllocator
        self._slab_alloc = SlabAllocator()
        self._slab_content_fp = {}
        self._ensure_mesh_buffer_capacity(
            mesh.num_vertices, mesh.num_triangles, mesh.num_nodes,
        )
        self.vertex_buffer.upload_sync(mesh.vertex_bytes)
        self.index_buffer.upload_sync(mesh.index_bytes)
        self.bvh_buffer.upload_sync(mesh.bvh_bytes)

    def _upload_usd_scene(self) -> None:
        """Upload every instance from `self._usd_scene` to the GPU.

        Concats all instance meshes into the unified buffers and writes
        one Instance record per instance with the correct BLAS offsets +
        world transform + material_id. Called from __init__ and from
        `_rebake_if_needed` when the user toggles back to the USD slot
        from an OBJ entry that overwrote the buffers.
        """
        scene = self._usd_scene
        if scene is None:
            self._prim_to_instances = {}
            return
        if not scene.instances:
            self._prim_to_instances = {}
            # Reset the TLAS to zero records — otherwise _num_instances keeps its
            # prior value (the analytic head's 1 on a fresh renderer, or N from a
            # replaced scene), leaving ghost geometry walked by rays. Covers a
            # freshly created empty scene AND removing a scene's last instance.
            self._upload_instances([])
            if self.uses_default_lights:
                self._upload_distant_lights([])
            else:
                self._upload_distant_lights(scene.lights_dir)
            self._sync_auxiliary_light_authority(force=True)
            return
        # Rebuild the prim-path → instance-index map for the editing API. One
        # prim may expand to several instances (e.g. multi-material meshes).
        prim_index: dict[str, list[int]] = {}
        for i, inst in enumerate(scene.instances):
            if inst.prim_path:
                prim_index.setdefault(inst.prim_path, []).append(i)
        self._prim_to_instances = prim_index
        offsets = self._upload_meshes_suballocated(scene.instances)
        # TLAS records only for enabled instances. Disabled instances keep
        # their BLAS data resident (cheap) but never get walked by rays.
        enabled_idx = [
            i for i, inst in enumerate(scene.instances) if inst.enabled
        ]
        transforms = [scene.instances[i].transform for i in enabled_idx]
        material_ids = [scene.instances[i].material_id for i in enabled_idx]
        enabled_offsets = [offsets[i] for i in enabled_idx]
        self._upload_instances(
            transforms,
            material_ids=material_ids,
            blas_offsets=enabled_offsets,
        )
        # Cache the TLAS layout so the animation path can re-upload just the
        # instance transforms (cheap) without re-concatenating geometry.
        self._usd_instance_layout = (enabled_idx, enabled_offsets, material_ids)
        self._upload_flat_materials(scene.materials)
        if self.uses_default_lights:
            self._upload_distant_lights([])
        else:
            self._upload_distant_lights(scene.lights_dir)
        self._sync_auxiliary_light_authority(force=True)

    def _reupload_instance_transforms(self) -> None:
        """Re-upload only the TLAS instance records using the cached layout.

        Cheap: rebuilds the small instance buffer from current
        `_usd_scene.instances[*].transform` without re-concatenating geometry
        or rebuilding any BVH. Falls back to a full upload if no layout was
        cached yet (e.g. instance set changed).
        """
        layout = self._usd_instance_layout
        scene = self._usd_scene
        if layout is None or scene is None or not scene.instances:
            return
        enabled_idx, enabled_offsets, material_ids = layout
        try:
            transforms = [scene.instances[i].transform for i in enabled_idx]
        except IndexError:
            # Instance set changed under us; full upload rebuilds the layout.
            self._upload_usd_scene()
            return
        self._upload_instances(
            transforms, material_ids=material_ids, blas_offsets=enabled_offsets,
        )

    def _reextract_animated_lights(self, stage, time, rt) -> None:
        """Rebuild distant + sphere lights from the stage at `time`.

        Re-extracts every distant/sphere light (cheap; a handful of prims) so
        animated intensity/colour/direction track playback. The stage is the
        source of truth — scene-graph editor edits are written back to it — so
        a full re-extract stays consistent with manual edits. Dome/env and
        emissive rect/disk animation are out of scope for this change.
        """
        from pxr import UsdLux
        from skinny.usd_loader import _extract_distant_light, _extract_sphere_light
        scene = self._usd_scene
        if scene is None:
            return
        new_dir = []
        new_sphere = []
        for prim in stage.Traverse():
            if not prim.IsActive() or prim.IsAbstract():
                continue
            if prim.IsA(UsdLux.DistantLight):
                ld = _extract_distant_light(prim, time)
                if ld is not None:
                    if rt is not None:
                        ld.direction = (ld.direction @ rt).astype(np.float32)
                    new_dir.append(ld)
            elif prim.IsA(UsdLux.SphereLight):
                ls = _extract_sphere_light(prim, time)
                if ls is not None:
                    if rt is not None:
                        ls.position = (ls.position @ rt).astype(np.float32)
                    new_sphere.append(ls)
        scene.lights_dir = new_dir
        scene.lights_sphere = new_sphere
        # Distant SSBO re-uploads every frame in update(); sphere does not.
        self._upload_sphere_lights(new_sphere)

    def _apply_animation_frame(self) -> None:
        """Re-evaluate animated USD prims at the clock's current time code.

        Cheap path: instance transforms (TLAS re-upload), distant/sphere
        lights, and the USD camera follower. No mesh rebake or BVH rebuild.
        No-op unless a USD scene with authored animation is loaded and the
        time code advanced since the last evaluation.
        """
        stage = self._usd_stage
        index = self._anim_index
        scene = self._usd_scene
        if stage is None or index is None or scene is None:
            return
        if not self.clock.has_animation:
            return
        tc = float(self.clock.current_time_code)
        if self._last_eval_time_code is not None and tc == self._last_eval_time_code:
            return

        from pxr import Usd
        from skinny.usd_loader import _extract_camera, _world_transform
        time = Usd.TimeCode(tc)
        rt = self._usd_up_axis_rt
        rt4 = None
        if rt is not None:
            rt4 = np.eye(4, dtype=np.float32)
            rt4[:3, :3] = rt

        # 1. Animated transforms → re-upload instance records.
        if index.xform_paths and scene.instances:
            xset = set(index.xform_paths)
            moved = False
            for inst in scene.instances:
                if inst.name in xset:
                    prim = stage.GetPrimAtPath(inst.name)
                    if prim and prim.IsValid():
                        m = _world_transform(prim, time)
                        inst.transform = (
                            (m @ rt4).astype(np.float32) if rt4 is not None else m
                        )
                        moved = True
            if moved:
                self._reupload_instance_transforms()

        # 2. Animated lights → re-extract distant + sphere.
        if index.light_paths:
            self._reextract_animated_lights(stage, time, rt)

        # 3. USD camera follower. Keep usd_camera tracking the latest evaluated
        # time so switching into usd mode is instant; the camera property only
        # returns it when camera_mode == "usd".
        if index.camera_animated:
            ov = _extract_camera(stage, time)
            if ov is not None and rt is not None:
                ov.position = (ov.position @ rt).astype(np.float32)
                ov.forward = (ov.forward @ rt).astype(np.float32)
            self._usd_camera_override = ov
            if ov is not None:
                self._override_to_orbit(self.usd_camera, ov, scene)

        # 4. Skeletal (UsdSkel) skinning. Interim CPU path: deform points on the
        # CPU, rebuild each skinned BLAS, re-upload. GPU skinning + BVH refit
        # replace this in a later step.
        if index.skinned_mesh_paths:
            self._apply_skeletal_frame(tc)

        self._last_eval_time_code = tc

    def _apply_skeletal_frame(self, time: float) -> None:
        """CPU linear-blend-skin every skinned instance at `time` and re-upload.

        Deformed points are produced in the same space as the authored rest
        points (geomBind-relative), so the instance's existing TLAS transform
        — already carrying the prim placement + up-axis correction — positions
        them correctly without any per-frame transform change.
        """
        skel = self._skeletal
        scene = self._usd_scene
        if skel is None or not skel.has_skinning or scene is None or not scene.instances:
            return
        from skinny.usd_loader import compute_joint_matrices

        # GPU path: upload per-mesh joint matrices, dispatch skin + BVH refit.
        if self._skinning_passes is not None:
            for mg in self._skinning_passes.meshes:
                mg.upload_joint_matrices(compute_joint_matrices(mg.binding, time))
            self._skinning_passes.dispatch(do_refit=True)
            return

        # CPU fallback (Metal / no GPU passes): deform on the CPU, rebuild each
        # skinned BLAS, re-upload.
        from dataclasses import replace as _replace
        from skinny.mesh import bake_mesh
        from skinny.usd_loader import _smooth_normals, lbs_points

        by_path = {b.prim_path: b for b in skel.meshes}
        changed = False
        for inst in scene.instances:
            binding = by_path.get(inst.name)
            if binding is None or inst.source is None:
                continue
            mats = compute_joint_matrices(binding, time)
            deformed = lbs_points(
                binding.rest_points, binding.joint_indices,
                binding.joint_weights, mats,
            )
            src = inst.source
            deformed_src = _replace(
                src,
                positions=deformed.astype(np.float32),
                normals=_smooth_normals(deformed, src.tri_idx),
            )
            inst.mesh = bake_mesh(
                deformed_src, displacement_bytes=None,
                displacement_res=0, displacement_scale_world=0.0,
            )
            changed = True
        if changed:
            self._upload_usd_scene()

    def _build_skinning_passes(self) -> None:
        """Build GPU skinning + refit passes for the loaded skinned scene.

        Vulkan only (uses the compute queue directly); on other backends this
        is a no-op and the CPU skinning fallback runs instead. Captures each
        skinned instance's rest-pose BLAS bytes + its offsets in the shared
        concatenated buffers (from `_usd_instance_layout`).
        """
        if not hasattr(self.ctx, "compute_queue"):
            return  # non-Vulkan backend → CPU fallback
        skel = self._skeletal
        scene = self._usd_scene
        layout = self._usd_instance_layout
        if skel is None or not skel.has_skinning or scene is None or layout is None:
            return
        enabled_idx, enabled_offsets, _mids = layout
        off_by_inst = {ii: enabled_offsets[k] for k, ii in enumerate(enabled_idx)}
        by_path = {b.prim_path: b for b in skel.meshes}
        try:
            from skinny.vk_skinning import SkinningPasses, _SkinnedMeshGPU
            mesh_gpus = []
            for i, inst in enumerate(scene.instances):
                binding = by_path.get(inst.name)
                if binding is None or i not in off_by_inst:
                    continue
                node_off, tri_off, vert_off = off_by_inst[i]
                mesh_gpus.append(_SkinnedMeshGPU(
                    self.ctx, binding,
                    rest_vertex_bytes=inst.mesh.vertex_bytes,
                    vertex_offset=vert_off, node_offset=node_off,
                    node_count=inst.mesh.num_nodes, index_offset=tri_off * 3,
                ))
            if not mesh_gpus:
                return
            if self._skinning_passes is None:
                self._skinning_passes = SkinningPasses(
                    self.ctx, self.shader_dir,
                    self.vertex_buffer, self.index_buffer, self.bvh_buffer,
                )
            self._skinning_passes.prepare(mesh_gpus)
            print(f"[skinny] GPU skinning ready — {len(mesh_gpus)} skinned mesh(es)")
        except Exception as exc:  # noqa: BLE001
            self._skinning_passes = None
            print(f"[skinny] GPU skinning unavailable ({exc}); using CPU skinning")

    def _apply_control_defaults(self) -> None:
        """Apply authored `skinny:ui:default` values to their targets at load."""
        from skinny.usd_loader import resolve_control_binding
        for spec in self._usd_controls:
            if spec.default is None:
                continue
            try:
                _get, setter = resolve_control_binding(self, spec)
                setter(spec.default)
            except Exception as exc:  # noqa: BLE001
                print(f"[skinny] control default failed for {spec.target}: {exc}")

    def _refresh_usd_live_state(self) -> None:
        """Re-read live-applicable scene state from the stage after a `usd:`
        control edited an attribute. Covers lights, instance transforms, and the
        camera (the cheap set); other attributes apply on reload.
        """
        stage = self._usd_stage
        scene = self._usd_scene
        if stage is None or scene is None:
            return
        from pxr import Usd
        from skinny.usd_loader import _extract_camera, _world_transform
        t = Usd.TimeCode.Default()
        rt = self._usd_up_axis_rt
        rt4 = None
        if rt is not None:
            rt4 = np.eye(4, dtype=np.float32)
            rt4[:3, :3] = rt

        self._reextract_animated_lights(stage, t, rt)

        if scene.instances:
            for inst in scene.instances:
                prim = stage.GetPrimAtPath(inst.name)
                if prim and prim.IsValid():
                    m = _world_transform(prim, t)
                    inst.transform = (
                        (m @ rt4).astype(np.float32) if rt4 is not None else m
                    )
            self._reupload_instance_transforms()

        ov = _extract_camera(stage, t)
        if ov is not None:
            if rt is not None:
                ov.position = (ov.position @ rt).astype(np.float32)
                ov.forward = (ov.forward @ rt).astype(np.float32)
            self._usd_camera_override = ov

        # Force an accumulation reset (raw USD edits aren't in the state hash).
        self._material_version += 1

    def _default_edit_path(self, root_layer) -> "str | None":
        """Default on-disk save target for the edit layer: sibling
        ``<scene>.edits.usda`` next to the stage's root file, or None for an
        anonymous/in-memory root that has no real path."""
        rp = getattr(root_layer, "realPath", "") or ""
        if not rp:
            return None
        from pathlib import Path
        p = Path(rp)
        return str(p.parent / f"{p.stem}.edits.usda")

    def _attach_edit_layer(self) -> None:
        """Route non-destructive edits to ``self._usd_stage``'s session layer.

        Makes the stage's session layer the edit target. The session layer sits
        ABOVE the entire root layer stack in strength, so an edit overrides any
        opinion authored in the root/file layer -- unlike a root SUBLAYER, which
        is WEAKER than the root layer and therefore cannot override a
        file-authored attribute (a `set_transform` on a prim whose
        `xformOp:transform` lives in the file would be silently ignored, and the
        clear+add author path would raise a duplicate-op error). The session
        layer is in-memory and is never written to disk by an edit; the original
        file is untouched. ``save_edits()`` persists these overrides on request.
        No-op if no stage is loaded or an edit layer is already attached.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is not None:
            return
        from pxr import Usd
        edit_layer = stage.GetSessionLayer()
        stage.SetEditTarget(Usd.EditTarget(edit_layer))
        self._usd_edit_layer = edit_layer
        self._edit_layer_default_path = self._default_edit_path(stage.GetRootLayer())

    def set_usd_scene(self, scene: "Scene", stage=None) -> None:
        """Make `scene` the active USD scene synchronously and upload it.

        Composes the same finalize steps the async streaming path runs, but
        blocking and re-callable. Intended for the headless render API: load
        a scene (e.g. from a caller-mutated Usd.Stage at time T), call this,
        then `update()` + `render_headless()`.

        Per call, geometry, flat materials, and lights are re-uploaded, so a
        mutated stage's moved transforms / deforming meshes / animated lights
        update. Fallback light and environment controls retain their values
        across authored scenes. An authored `camera_override` is re-applied
        every call (animated cameras track); with no authored camera the orbit
        framing is set once and does not follow a moving scene.

        Limitations (headless / non-interactive use): the scene-graph tree
        (`self.scene_graph`) is not built here, so scene-graph inspection APIs
        are unavailable after this call. Call `update()` before
        `render_headless()` so the accumulation buffer resets for the new
        scene.

        Not part of the live UI path.
        """
        first = self._usd_scene is None

        # Enter the USD-active state so update()/render treat this scene as
        # the subject instead of the default analytic head.
        if self._usd_model_index < 0:
            self.models.append("USD: (headless)")
            self._usd_model_index = len(self.models) - 1
        self.model_index = self._usd_model_index

        self._usd_scene = scene
        # Film per-sample radiance clamp from the imported pbrt film (change
        # film-maxcomponent-clamp); 0 = disabled (no-op). Applied on the
        # synchronous scene-swap path (headless / parity harness) — the streaming
        # interactive path sets it in _poll_usd_streaming.
        self.film_max_component = float(getattr(scene, "film_max_component", 0.0) or 0.0)
        # When the caller owns a stage (headless editing entry, design D9), take
        # ownership and attach the non-destructive edit layer so the runtime
        # scene-graph editing API (add_model/remove_node/set_transform) works.
        if stage is not None:
            self._usd_stage = stage
            self._attach_edit_layer()
        # Density grid before the material upload (the volume σ folds read the
        # grid state + the live self.mm_per_unit — the same source the walk's
        # fc.mmPerUnit is packed from, so fold and walk always agree).
        self._sync_volume_grid(scene)
        self._gen_scene_materials()           # guarded: rebuilds pipeline only on graph-set change
        if first:
            self._frame_camera_to_scene(scene)
        elif scene.camera_override is not None:
            self._frame_camera_to_scene(scene)  # animated authored camera
        self._upload_usd_scene()              # every call: geometry + materials + lights

    def create_empty_scene(self) -> None:
        """Synthesize a bare editable in-memory USD stage and make it active.

        Gives the renderer a fresh editable stage — a single ``/World`` Xform
        default prim (Y-up, 1 m/unit), no authored geometry/lights/camera — so
        the runtime scene-graph editing API (`add_model`/`add_primitive`/
        `add_light`/`save_edits`) works with no scene ever loaded from disk. The
        MCP `scene_create` tool routes here.

        `/World` is load-bearing: the add helpers parent new prims under it.
        Lights/dome/camera are intentionally omitted — `_resync_geometry_from_stage`
        re-injects the synthetic `/Skinny/DefaultLight`+`DefaultDome`+`MainCamera`,
        so authoring real ones would only create duplicates. Any previously
        loaded stage + edit layer are replaced (caller enforces the refuse/force
        policy).
        """
        from pxr import Usd, UsdGeom

        from skinny.usd_loader import load_scene_from_stage

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        # Reset stale per-scene state carried by the load path so force-replacing
        # a Z-up / animated scene doesn't inherit its rotation or clock. A fresh
        # Y-up empty stage wants exactly these defaults.
        self.clock = PlaybackClock()
        self._anim_index = None
        self._skeletal = None
        self._usd_controls = []
        self._usd_up_axis_rt = None
        self._usd_bake_done = None
        self.film_max_component = 0.0
        # The new stage declares no authored camera; a replaced scene may have
        # left the renderer horizontally mirrored or in USD-camera mode, where
        # input dispatch would call look() on the OrbitCamera and raise. Return
        # to a valid free-look/orbit state.
        self._camera_mirror = False
        if self.camera_mode == "usd":
            self.camera_mode = "orbit"

        # An empty stage has no mesh/gprim geometry — allow_empty returns a
        # well-formed empty Scene rather than raising.
        scene = load_scene_from_stage(
            stage, use_usd_mtlx_plugin=self._use_usd_mtlx_plugin, allow_empty=True,
        )
        # Adopt the new stage's physical scale (metersPerUnit 1 -> 1000 mm/unit)
        # so a force-replace doesn't render later skin/volume content at the
        # previous scene's scale (self.mm_per_unit is a separate renderer field).
        self.mm_per_unit = float(scene.mm_per_unit)

        # Enter the USD-active state (mirrors set_usd_scene): the label append is
        # required, not just the index — sites index self.models[_usd_model_index].
        if self._usd_model_index < 0:
            self.models.append("USD: (empty)")
            self._usd_model_index = len(self.models) - 1
        self.model_index = self._usd_model_index

        self._usd_scene = scene
        self._usd_stage = stage
        self._usd_edit_layer = None
        self._attach_edit_layer()
        # Builds scene_graph, re-injects synthetic default light/dome/camera,
        # uploads (empty TLAS via the zero-instance branch), bumps versions.
        self._resync_geometry_from_stage()

    # ── Runtime scene-graph editing (usd-scene-editing) ─────────────────

    def _resync_geometry_from_stage(self) -> None:
        """Re-read the authoritative stage into the flat scene, GPU buffers, and
        derived scene graph.

        Used after add/remove edits. Re-reads instances + materials + lights +
        camera via ``load_scene_from_stage``; meshes are cached by content hash,
        so unchanged prims are not re-baked. Runtime ``enabled`` flags (not
        authored to the stage) are carried across by prim path for both
        instances and lights, so an unrelated edit does not lose a user toggle.
        Authored environment state is replaced or cleared from the re-read
        stage. Finally rebuilds the derived scene graph (with synthesized
        default lights re-injected) and bumps the
        version so the UI panels repaint, and so a deleted light/camera prim
        drops out of the render.
        """
        stage = self._usd_stage
        scene = self._usd_scene
        if stage is None or scene is None:
            return
        from skinny.usd_loader import load_scene_from_stage
        prev_inst_enabled = {
            inst.prim_path: inst.enabled
            for inst in scene.instances if inst.prim_path
        }
        prev_light_enabled = {
            lt.prim_path: lt.enabled
            for lt in (*scene.lights_dir, *scene.lights_sphere) if lt.prim_path
        }
        new_scene = load_scene_from_stage(
            stage, use_usd_mtlx_plugin=self._use_usd_mtlx_plugin,
            allow_empty=True,
        )
        for inst in new_scene.instances:
            if inst.prim_path in prev_inst_enabled:
                inst.enabled = prev_inst_enabled[inst.prim_path]
        for lt in (*new_scene.lights_dir, *new_scene.lights_sphere):
            if lt.prim_path in prev_light_enabled:
                lt.enabled = prev_light_enabled[lt.prim_path]
        # Swap instances + materials together so material_ids stay consistent;
        # take the re-read lights + environment + camera too so deleting one
        # drops it.
        scene.instances = new_scene.instances
        scene.materials = new_scene.materials
        scene.lights_dir = new_scene.lights_dir
        scene.lights_sphere = new_scene.lights_sphere
        scene.environment = new_scene.environment
        scene.has_authored_lighting = new_scene.has_authored_lighting
        scene.camera_override = new_scene.camera_override
        scene.volume_grid = getattr(new_scene, "volume_grid", None)
        self._sync_volume_grid(scene)
        self._gen_scene_materials()
        self._upload_usd_scene()  # also uploads distant + sphere lights
        self._material_version += 1
        # Rebuild the derived scene graph so the UI panels (which poll a version
        # counter / object id) repaint to reflect the edit.
        try:
            from skinny.scene_graph import build_scene_graph
            self._scene_graph = build_scene_graph(stage, scene)
            self._inject_default_lights_into_scene_graph()
            self._refresh_camera_node()
        except Exception as exc:  # noqa: BLE001
            print(f"[skinny] scene graph rebuild after edit failed: {exc}")
        # Bump explicitly: _inject_default_lights_into_scene_graph early-returns
        # (without bumping) when there is no default-light stage, e.g. headless.
        self._scene_graph_version = getattr(self, "_scene_graph_version", 0) + 1

    def _instance_indices_under_path(self, prim_path: str) -> list[int]:
        """Indices of every instance at or below ``prim_path`` (subtree match).

        Authoring a transform on an ancestor Xform changes the world transform
        of its descendant meshes, so transform resync must match the subtree.
        """
        scene = self._usd_scene
        if scene is None:
            return []
        pref = prim_path.rstrip("/")
        out: list[int] = []
        for i, inst in enumerate(scene.instances):
            pp = inst.prim_path
            if pp and (pp == pref or pp.startswith(pref + "/")):
                out.append(i)
        return out

    def _resync_instance_transforms(self, prim_path: str) -> None:
        """Recompute world transforms for instances under ``prim_path`` from the
        stage and re-upload only the TLAS records (no geometry re-bake)."""
        scene = self._usd_scene
        stage = self._usd_stage
        if scene is None or stage is None:
            return
        indices = self._instance_indices_under_path(prim_path)
        if not indices:
            return
        from pxr import Usd
        from skinny.usd_loader import _world_transform
        rt4 = None
        rt = self._usd_up_axis_rt
        if rt is not None:
            rt4 = np.eye(4, dtype=np.float32)
            rt4[:3, :3] = rt
        for i in indices:
            iprim = stage.GetPrimAtPath(scene.instances[i].prim_path)
            if not iprim or not iprim.IsValid():
                continue
            m = _world_transform(iprim, Usd.TimeCode.Default())
            scene.instances[i].transform = (
                (m @ rt4).astype(np.float32) if rt4 is not None else m
            )
        self._reupload_instance_transforms()

    def _author_local_transform(self, xformable, matrix) -> None:
        """Author ``matrix`` as the prim's single ``xformOp:transform`` in the
        active edit target. Pure logic in `skinny.usd_edit` so it is unit-tested
        without a GPU renderer."""
        from skinny.usd_edit import author_local_transform
        author_local_transform(xformable, matrix)

    def _unique_prim_path(self, base: str) -> str:
        """``base`` if free, else ``base_1``, ``base_2``, … (avoids prim clash)."""
        stage = self._usd_stage
        candidate = base
        i = 1
        while stage.GetPrimAtPath(candidate).IsValid():
            candidate = f"{base}_{i}"
            i += 1
        return candidate

    def add_model(
        self,
        usd_path,
        parent_prim_path: str = "/World",
        name: "str | None" = None,
        transform=None,
        validate=None,
    ) -> str:
        """Add a USD model to the live stage by reference and return its prim path.

        Defines an ``Xform`` under ``parent_prim_path`` in the edit layer, adds a
        reference to ``usd_path``, authors ``transform`` (identity when omitted),
        then re-reads the stage so the new geometry uploads. Accepts USD only.

        ``validate``, when given, is called as ``validate(stage, added_prim)``
        after the reference has recomposed but before the geometry re-sync, so a
        policy layer (the MCP path allowlist) can inspect the composed result —
        newly introduced layers and asset-valued attributes — and veto the add.
        A raised exception from ``validate`` triggers the same rollback as an
        authoring failure and propagates to the caller.

        Raises ``ValueError`` for a missing/invalid path (no stage mutation) and
        rolls back the authored prim -- and any parent prims this call created --
        if the re-read fails or ``validate`` rejects the result.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is None:
            raise RuntimeError("add_model requires a loaded USD stage")
        from pathlib import Path as _Path
        p = _Path(str(usd_path))
        if p.suffix.lower() == ".obj":
            raise ValueError(
                "add_model accepts USD references only; OBJ adds are deferred"
            )
        if not p.exists():
            raise ValueError(f"add_model: file not found: {usd_path}")
        from pxr import Sdf, Tf, Usd, UsdGeom
        if Sdf.Layer.FindOrOpen(str(p)) is None:
            raise ValueError(f"add_model: not a valid USD file: {usd_path}")
        leaf = name or Tf.MakeValidIdentifier(p.stem)
        parent = parent_prim_path.rstrip("/") or "/"
        prim_path = self._unique_prim_path(
            f"/{leaf}" if parent == "/" else f"{parent}/{leaf}"
        )
        parent_parts = [part for part in parent.split("/") if part]
        parent_paths = [
            "/" + "/".join(parent_parts[:i])
            for i in range(1, len(parent_parts) + 1)
        ]
        missing_parent_paths = [
            path for path in parent_paths
            if not stage.GetPrimAtPath(path).IsValid()
        ]
        resync_started = False
        try:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                if parent != "/" and not stage.GetPrimAtPath(parent).IsValid():
                    UsdGeom.Xform.Define(stage, parent)
                xform = UsdGeom.Xform.Define(stage, prim_path)
                xform.GetPrim().GetReferences().AddReference(str(p))
                if transform is not None:
                    self._author_local_transform(xform, transform)
            if validate is not None:
                validate(stage, xform.GetPrim())
            resync_started = True
            self._resync_geometry_from_stage()
        except Exception:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                if stage.GetPrimAtPath(prim_path).IsValid():
                    stage.RemovePrim(prim_path)
                for created_parent_path in reversed(missing_parent_paths):
                    if stage.GetPrimAtPath(created_parent_path).IsValid():
                        stage.RemovePrim(created_parent_path)
            if resync_started:
                try:
                    self._resync_geometry_from_stage()
                except Exception:  # preserve the original creation failure
                    pass
            raise
        return prim_path

    def add_light(
        self,
        light_type: str,
        parent_prim_path: str = "/World",
        name: "str | None" = None,
        transform=None,
        intensity: "float | None" = None,
        color=None,
    ) -> str:
        """Author a supported ``UsdLux`` light and return its unique prim path.

        The light is defined in the active non-destructive edit layer, receives
        explicit property-editor-friendly defaults, and triggers the same full
        stage resync as add/remove edits. ``light_type`` must be one of
        DistantLight, SphereLight, DomeLight, RectLight, or DiskLight.

        ``intensity`` and ``color`` (a 3-tuple), when given, are authored at
        define time instead of the fixed defaults -- a post-creation property
        write would not persist to a saved edit layer.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is None:
            raise RuntimeError("add_light requires a loaded USD stage")

        from pxr import Gf, Tf, Usd, UsdGeom, UsdLux

        schemas = {
            "DistantLight": UsdLux.DistantLight,
            "SphereLight": UsdLux.SphereLight,
            "DomeLight": UsdLux.DomeLight,
            "RectLight": UsdLux.RectLight,
            "DiskLight": UsdLux.DiskLight,
        }
        schema = schemas.get(str(light_type))
        if schema is None:
            supported = ", ".join(schemas)
            raise ValueError(
                f"add_light: unsupported light type {light_type!r}; "
                f"expected one of {supported}"
            )

        parent = str(parent_prim_path or "/World").rstrip("/") or "/"
        leaf = Tf.MakeValidIdentifier(name or str(light_type))
        prim_path = self._unique_prim_path(
            f"/{leaf}" if parent == "/" else f"{parent}/{leaf}"
        )
        parent_parts = [part for part in parent.split("/") if part]
        parent_paths = [
            "/" + "/".join(parent_parts[:i])
            for i in range(1, len(parent_parts) + 1)
        ]
        missing_parent_paths = [
            path for path in parent_paths
            if not stage.GetPrimAtPath(path).IsValid()
        ]
        resync_started = False
        try:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                if parent != "/" and not stage.GetPrimAtPath(parent).IsValid():
                    UsdGeom.Xform.Define(stage, parent)
                light = schema.Define(stage, prim_path)
                r, g, b = color if color is not None else (1.0, 1.0, 1.0)
                light.CreateColorAttr().Set(Gf.Vec3f(float(r), float(g), float(b)))
                light.CreateIntensityAttr().Set(
                    float(intensity) if intensity is not None else 1.0
                )
                light.CreateExposureAttr().Set(0.0)
                if light_type == "DistantLight":
                    light.CreateAngleAttr().Set(0.53)
                elif light_type == "SphereLight":
                    light.CreateRadiusAttr().Set(0.5)
                elif light_type == "RectLight":
                    light.CreateWidthAttr().Set(1.0)
                    light.CreateHeightAttr().Set(1.0)
                elif light_type == "DiskLight":
                    light.CreateRadiusAttr().Set(0.5)
                if transform is not None:
                    self._author_local_transform(
                        UsdGeom.Xformable(light.GetPrim()), transform,
                    )
            resync_started = True
            self._resync_geometry_from_stage()
        except Exception:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                if stage.GetPrimAtPath(prim_path).IsValid():
                    stage.RemovePrim(prim_path)
                for created_parent_path in reversed(missing_parent_paths):
                    if stage.GetPrimAtPath(created_parent_path).IsValid():
                        stage.RemovePrim(created_parent_path)
            if resync_started:
                try:
                    self._resync_geometry_from_stage()
                except Exception:  # preserve the original creation failure
                    pass
            raise
        return prim_path

    def add_primitive(
        self,
        prim_type: str,
        parent_prim_path: str = "/World",
        name: "str | None" = None,
        transform=None,
        color=None,
        roughness: "float | None" = None,
        metallic: "float | None" = None,
        skip_inline_material: bool = False,
    ) -> str:
        """Author an analytic gprim with its own bound preview-surface material.

        Defines one of Sphere/Cube/Cylinder/Cone/Capsule/Plane -- the types
        ``usd_gprims.tessellate_gprim`` meshes -- in the active edit layer,
        together with a dedicated ``UsdShade`` material carrying a
        ``UsdPreviewSurface`` shader, and binds it. A primitive is never
        authored bare: an unbound prim resolves to the protected fallback
        material slot (index 0) and could never be re-colored afterwards.

        ``skip_inline_material`` (mcp-material-authoring, design D6) omits
        the inline material entirely -- the caller (``scene_add_primitive``
        with a ``material`` argument) binds an existing or freshly-created
        ``/Materials`` holder instead, and authoring the usual inline one
        first would leave it an orphaned, unbound sibling prim.
        ``color``/``roughness``/``metallic`` are ignored in that case.

        Returns the gprim's unique prim path.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is None:
            raise RuntimeError("add_primitive requires a loaded USD stage")

        from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade

        schemas = {
            "Sphere": UsdGeom.Sphere,
            "Cube": UsdGeom.Cube,
            "Cylinder": UsdGeom.Cylinder,
            "Cone": UsdGeom.Cone,
            "Capsule": UsdGeom.Capsule,
            "Plane": UsdGeom.Plane,
        }
        schema = schemas.get(str(prim_type))
        if schema is None:
            supported = ", ".join(schemas)
            raise ValueError(
                f"add_primitive: unsupported primitive type {prim_type!r}; "
                f"expected one of {supported}"
            )

        parent = str(parent_prim_path or "/World").rstrip("/") or "/"
        leaf = Tf.MakeValidIdentifier(name or str(prim_type))
        prim_path = self._unique_prim_path(
            f"/{leaf}" if parent == "/" else f"{parent}/{leaf}"
        )
        material_path = (
            None if skip_inline_material
            else self._unique_prim_path(f"{prim_path}_material")
        )
        parent_parts = [part for part in parent.split("/") if part]
        parent_paths = [
            "/" + "/".join(parent_parts[:i])
            for i in range(1, len(parent_parts) + 1)
        ]
        missing_parent_paths = [
            path for path in parent_paths
            if not stage.GetPrimAtPath(path).IsValid()
        ]
        resync_started = False
        try:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                if parent != "/" and not stage.GetPrimAtPath(parent).IsValid():
                    UsdGeom.Xform.Define(stage, parent)
                gprim = schema.Define(stage, prim_path)
                if transform is not None:
                    self._author_local_transform(
                        UsdGeom.Xformable(gprim.GetPrim()), transform,
                    )

                if material_path is not None:
                    material = UsdShade.Material.Define(stage, material_path)
                    shader = UsdShade.Shader.Define(
                        stage, f"{material_path}/PreviewSurface"
                    )
                    shader.CreateIdAttr("UsdPreviewSurface")
                    r, g, b = color if color is not None else (0.8, 0.8, 0.8)
                    shader.CreateInput(
                        "diffuseColor", Sdf.ValueTypeNames.Color3f,
                    ).Set(Gf.Vec3f(float(r), float(g), float(b)))
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                        float(roughness) if roughness is not None else 0.5
                    )
                    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                        float(metallic) if metallic is not None else 0.0
                    )
                    material.CreateSurfaceOutput().ConnectToSource(
                        shader.ConnectableAPI(), "surface",
                    )
                    UsdShade.MaterialBindingAPI.Apply(gprim.GetPrim())
                    UsdShade.MaterialBindingAPI(gprim.GetPrim()).Bind(material)
            resync_started = True
            self._resync_geometry_from_stage()
        except Exception:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                if material_path is not None and stage.GetPrimAtPath(material_path).IsValid():
                    stage.RemovePrim(material_path)
                if stage.GetPrimAtPath(prim_path).IsValid():
                    stage.RemovePrim(prim_path)
                for created_parent_path in reversed(missing_parent_paths):
                    if stage.GetPrimAtPath(created_parent_path).IsValid():
                        stage.RemovePrim(created_parent_path)
            if resync_started:
                try:
                    self._resync_geometry_from_stage()
                except Exception:  # preserve the original creation failure
                    pass
            raise
        return prim_path

    def add_material(
        self,
        name: str,
        *,
        mtlx_path: "str | None" = None,
        preview_params: "dict | None" = None,
        session_dir: "str | None" = None,
        on_rollback=None,
    ) -> str:
        """Author a material holder under ``/Materials`` in the session edit layer.

        Two forms (mcp-material-authoring, design D2):
        - ``mtlx_path`` — a typed ``UsdShade.Material`` holder carrying an
          absolute ``.mtlx`` reference (curated preset or session-synthesized
          document). The holder prim name is ``name`` **exactly** (the naming
          contract the loader's binding resolution requires); a clash raises so
          the caller dedups presets / salts synthesized names beforehand.
        - ``preview_params`` — an inline ``UsdPreviewSurface`` material with the
          full editable input set; the holder name is uniquified.

        ``session_dir`` marks a synthesized document's store directory so
        ``save_edits`` classifies it correctly. ``on_rollback`` (e.g. delete the
        session ``.mtlx``) runs if authoring fails — keeping the renderer
        decoupled from the synthesis module. Rollback removes the holder and an
        auto-created ``/Materials`` scope. The material is created but **not live**
        until a geometry prim binds it (design D8). Returns the holder prim path.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is None:
            raise RuntimeError("add_material requires a loaded USD stage")
        if (mtlx_path is None) == (preview_params is None):
            raise ValueError(
                "add_material requires exactly one of mtlx_path or preview_params"
            )
        from pxr import Tf, Usd
        from skinny import usd_material_edit as ume

        leaf = Tf.MakeValidIdentifier(name)
        if mtlx_path is not None:
            holder_path = f"/Materials/{leaf}"
            if stage.GetPrimAtPath(holder_path).IsValid():
                raise ValueError(
                    f"add_material: /Materials/{leaf} already exists; "
                    f"dedup presets and salt synthesized names before calling"
                )
        else:
            holder_path = self._unique_prim_path(f"/Materials/{leaf}")

        if session_dir is not None:
            self._material_session_dir = session_dir

        created_scope = False
        resync_started = False
        try:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                created_scope = ume.ensure_materials_scope(stage)
                if mtlx_path is not None:
                    ume.author_material_holder(stage, holder_path, mtlx_path)
                else:
                    ume.author_preview_material(stage, holder_path, preview_params or {})
            resync_started = True
            self._resync_geometry_from_stage()
        except Exception:
            with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
                if stage.GetPrimAtPath(holder_path).IsValid():
                    stage.RemovePrim(holder_path)
                if created_scope and stage.GetPrimAtPath("/Materials").IsValid():
                    stage.RemovePrim("/Materials")
            if on_rollback is not None:
                try:
                    on_rollback()
                except Exception:  # rollback is best-effort
                    pass
            if resync_started:
                try:
                    self._resync_geometry_from_stage()
                except Exception:  # preserve the original creation failure
                    pass
            raise
        return holder_path

    def bind_material(self, prim_path: str, material_path: str) -> None:
        """Bind ``material_path`` to ``prim_path`` in the session edit layer.

        Validates both paths (mcp-material-authoring, design D6/M4): the geometry
        prim must exist and be a bindable ``Gprim`` (Mesh / analytic gprim); the
        material must exist and be either ``Material``-typed or carry a ``.mtlx``
        reference. Authors explicit binding targets so the session binding
        *replaces* any file-authored binding (LIVRPS), then resyncs — which loads
        the newly bound material and restarts accumulation.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is None:
            raise RuntimeError("bind_material requires a loaded USD stage")
        from pxr import Usd, UsdGeom, UsdShade
        from skinny import usd_material_edit as ume
        from skinny.usd_loader import _prim_has_mtlx_reference

        prim = stage.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            raise ValueError(f"bind_material: prim not found: {prim_path}")
        if not prim.IsA(UsdGeom.Gprim):
            raise ValueError(
                f"bind_material: {prim_path} is not bindable geometry (Gprim)"
            )
        mat_prim = stage.GetPrimAtPath(material_path)
        if not (mat_prim and mat_prim.IsValid()):
            raise ValueError(f"bind_material: material not found: {material_path}")
        is_material = bool(UsdShade.Material(mat_prim))
        if not is_material and not _prim_has_mtlx_reference(stage, mat_prim.GetPath()):
            raise ValueError(
                f"bind_material: {material_path} is neither Material-typed nor "
                f"carries a .mtlx reference"
            )
        with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
            ume.author_binding(stage, prim_path, material_path)
        self._resync_geometry_from_stage()

    def remove_node(self, prim_path: str) -> None:
        """Remove a node by deactivating its prim (non-destructive) and re-reading.

        Raises ``ValueError`` if ``prim_path`` is not on the stage.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is None:
            raise RuntimeError("remove_node requires a loaded USD stage")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"remove_node: prim not found: {prim_path}")
        from pxr import Usd
        with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
            prim.SetActive(False)
        self._resync_geometry_from_stage()

    def set_transform(self, prim_path: str, matrix) -> None:
        """Author a prim's local transform and fast-resync its instances.

        ``matrix`` is a 4x4 (numpy/array-like) in USD row-major convention.
        Raises ``ValueError`` if ``prim_path`` is not on the stage.
        """
        stage = self._usd_stage
        if stage is None or self._usd_edit_layer is None:
            raise RuntimeError("set_transform requires a loaded USD stage")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"set_transform: prim not found: {prim_path}")
        from pxr import Usd, UsdGeom, UsdLux
        with Usd.EditContext(stage, Usd.EditTarget(self._usd_edit_layer)):
            self._author_local_transform(UsdGeom.Xformable(prim), matrix)
        light_types = (
            UsdLux.DistantLight,
            UsdLux.SphereLight,
            UsdLux.DomeLight,
            UsdLux.RectLight,
            UsdLux.DiskLight,
        )
        if any(prim.IsA(light_type) for light_type in light_types):
            # Analytic light transforms live outside the instance TLAS; re-read
            # them from USD so positions/directions and the scene graph update.
            self._resync_geometry_from_stage()
            return
        self._resync_instance_transforms(prim_path)
        self._material_version += 1

    def save_edits(self, path: "str | None" = None) -> str:
        """Persist the edit layer to disk and return the written path.

        Defaults to the sibling ``<scene>.edits.usda`` next to the loaded file.
        """
        if self._usd_edit_layer is None:
            raise RuntimeError("save_edits: no edit layer attached")
        target = path or self._edit_layer_default_path
        if target is None:
            raise ValueError(
                "save_edits: no path given and no default (in-memory stage)"
            )
        stage = self._usd_stage
        root = stage.GetRootLayer() if stage is not None else None
        # Material holders and their referenced `.mtlx` documents, read off the
        # LIVE stage before an export flattens the reference arcs away
        # (mcp-material-authoring, design D7).
        from skinny import usd_material_edit as ume
        holders = ume.collect_material_holders(stage) if stage is not None else {}
        save_dir = str(Path(str(target)).resolve().parent)

        if root is not None and root.anonymous:
            # A synthesized stage (scene_create) has no backing file: its /World
            # prim and stage metadata (defaultPrim / upAxis / metersPerUnit) live
            # on the anonymous root, not the session edit layer. Exporting only
            # the edit layer would drop them, so the saved file would reopen at
            # USD's 0.01 m/unit default. Export the composed stage instead —
            # complete and self-contained (session overrides fold in too).
            stage.Export(str(target))
            flattened = True
        else:
            self._usd_edit_layer.Export(str(target))
            flattened = False

        # Post-process the saved layer: re-anchor `.mtlx` references, copy
        # session-synthesized documents into a `materials/` bundle, strip the
        # flatten residue (anonymous branch only). Curated presets keep absolute
        # references (texture carve-out). No-op when no holders were authored.
        if holders:
            from pxr import Sdf
            saved_layer = Sdf.Layer.FindOrOpen(str(target))
            if saved_layer is not None:
                ume.finalize_saved_materials(
                    saved_layer, holders, save_dir,
                    self._material_session_dir, flattened=flattened,
                )
                saved_layer.Save()
        return str(target)

    def list_nodes(self) -> "list[dict]":
        """Editable prims of the stage as ``{path, type, active}`` dicts.

        Includes inactive prims (e.g. nodes removed via ``remove_node``) so the
        full editable graph is visible to scripts and a future scene-graph UI.
        """
        stage = self._usd_stage
        if stage is None:
            return []
        out: list[dict] = []
        for prim in stage.TraverseAll():
            if prim.IsPseudoRoot():
                continue
            out.append({
                "path": str(prim.GetPath()),
                "type": str(prim.GetTypeName()),
                "active": bool(prim.IsActive()),
            })
        return out

    def _upload_emissive_triangles(self, scene: Scene) -> None:
        """Build the emissive triangle buffer (binding 18) from scene instances.

        Walks every instance whose material has non-zero emissiveColor,
        world-transforms its source triangles, and packs each into a 64-byte
        record. The shader samples one triangle per pixel per frame for NEE.
        """
        records: list[tuple] = []
        for inst in scene.instances:
            if not inst.enabled:
                continue
            if inst.source is None:
                continue
            mat_id = inst.material_id
            if mat_id >= len(scene.materials):
                continue
            mat = scene.materials[mat_id]
            emissive = _override_color3(
                mat.parameter_overrides, "emissiveColor", (0.0, 0.0, 0.0)
            )
            if emissive[0] <= 0 and emissive[1] <= 0 and emissive[2] <= 0:
                continue
            # Spectral (Group 6.1): a blackbody area light preserves its
            # temperature on parameter_overrides["emissive_spectral"]. Evaluate the
            # exact-Planck luminance-matching scale once per instance (shared by all
            # its triangles); (0, 0) means "not a blackbody, use the RGB upsample".
            bb_temp, bb_scale = 0.0, 0.0
            if self._spectral:
                payload = mat.parameter_overrides.get("emissive_spectral")
                # payload round-trips USD as a pxr.Vt.Dictionary (NOT a py dict),
                # so duck-type on `.get` rather than isinstance(dict).
                if (payload is not None and hasattr(payload, "get")
                        and payload.get("kind") == "blackbody"):
                    from skinny.pbrt import spectral as _spectral_mirror
                    bb_temp = float(payload.get("temperature", 0.0) or 0.0)
                    if bb_temp > 0.0:
                        bb_scale = float(
                            _spectral_mirror.blackbody_scale(bb_temp, emissive)
                        )
            src = inst.source
            xform = inst.transform
            for tri in range(len(src.tri_idx)):
                i0, i1, i2 = int(src.tri_idx[tri][0]), int(src.tri_idx[tri][1]), int(src.tri_idx[tri][2])
                p0 = np.append(src.positions[i0], 1.0).astype(np.float32) @ xform
                p1 = np.append(src.positions[i1], 1.0).astype(np.float32) @ xform
                p2 = np.append(src.positions[i2], 1.0).astype(np.float32) @ xform
                p0, p1, p2 = p0[:3], p1[:3], p2[:3]
                e1 = p1 - p0
                e2 = p2 - p0
                area = 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
                if area < 1e-8:
                    continue
                records.append((p0, p1, p2, emissive, area, bb_temp, bb_scale))

        # Power-weighted importance sampling (change emissive-mesh-nee). Build a
        # normalized cumulative-power CDF over every emissive triangle (no 256
        # cap), weight w_i = area_i × Rec.709-luminance(emission_i). The CDF is
        # packed *inline* into the record's spare .w lanes (cw = inclusive
        # cumulative Σ_{j≤i} w_j / Σw in _v0.w; pSel = w_i / Σw in _v1.w) so it
        # rides binding 18 — no separate buffer, no extra Metal slot. The shader
        # binary-searches `cw` to select a triangle and reads `pSel` as the
        # selection pdf. See lights/emissive_triangle_light.slang + scene_lights.slang.
        n = len(records)
        weights = [
            area * (0.2126 * float(em[0]) + 0.7152 * float(em[1]) + 0.0722 * float(em[2]))
            for (_p0, _p1, _p2, em, area, _t, _s) in records
        ]
        total_w = float(sum(weights))
        # Σ(area·Rec709-lum) → FrameConstants.emissiveTotalPower; the path tracer's
        # BSDF-hit MIS weight reconstructs the NEE solid-angle pdf from it.
        self._emissive_total_power = total_w
        if n == 0 or total_w <= 0.0:
            # Degenerate (no emissive geometry, or zero total power): keep the
            # numEmissiveTriangles == 0 early-out so the shader skips the path and
            # the inline CDF is never read.
            self._num_emissive_tris = 0
            self._emissive_total_power = 0.0
            zeros = b"\x00" * (self.emissive_tri_capacity * EMISSIVE_TRI_STRIDE)
            self.emissive_tri_buffer.upload_sync(zeros)
            if self._spectral and self._spectral_emitters_buffer is not None:
                self._spectral_emitters_buffer.upload_sync(
                    b"\x00" * (self.emissive_tri_capacity * SPECTRAL_EMITTER_STRIDE)
                )
            return

        # Grow (and rebind) the buffer to hold every emissive triangle, doubling
        # capacity like material_capacity. Vulkan re-writes binding 18; native
        # Metal re-reads the buffer reference fresh at dispatch (no-op rebind).
        if n > self.emissive_tri_capacity:
            self.emissive_tri_capacity = max(n, self.emissive_tri_capacity * 2)
            self.emissive_tri_buffer.destroy()
            self.emissive_tri_buffer = self._gpu.StorageBuffer(
                self.ctx, self.emissive_tri_capacity * EMISSIVE_TRI_STRIDE + 16
            )
            if self._spectral and self._spectral_emitters_buffer is not None:
                self._spectral_emitters_buffer.destroy()
                self._spectral_emitters_buffer = self._gpu.StorageBuffer(
                    self.ctx, self.emissive_tri_capacity * SPECTRAL_EMITTER_STRIDE + 16
                )
            self._rebind_emissive_descriptors()

        uniform = bool(getattr(self, "_emissive_uniform_selection", False))
        data = bytearray()
        spectral_data = bytearray()   # binding 49: (T, scale) parallel to `data`
        cum = 0.0
        for i in range(n):
            p0, p1, p2, em, area, bb_temp, bb_scale = records[i]
            if self._spectral:
                spectral_data += struct.pack("ff", float(bb_temp), float(bb_scale))
            if uniform:
                # Test A/B: uniform-by-index — the same shader path then
                # reproduces exact 1/N selection.
                p_sel = 1.0 / float(n)
                cw = float(i + 1) / float(n)
            else:
                p_sel = weights[i] / total_w
                cum += weights[i]
                cw = cum / total_w
            data += struct.pack(
                "fff f fff f fff f fff f",
                float(p0[0]), float(p0[1]), float(p0[2]), float(cw),
                float(p1[0]), float(p1[1]), float(p1[2]), float(p_sel),
                float(p2[0]), float(p2[1]), float(p2[2]), 0.0,
                float(em[0]), float(em[1]), float(em[2]), float(area),
            )
        # Pin the final CDF entry to exactly 1.0 (guards float round-off so the
        # binary search always resolves a valid index for u → 1⁻).
        if not uniform and n > 0:
            struct.pack_into("f", data, (n - 1) * EMISSIVE_TRI_STRIDE + 12, 1.0)
        cap_bytes = self.emissive_tri_capacity * EMISSIVE_TRI_STRIDE
        while len(data) < cap_bytes:
            data += b"\x00" * EMISSIVE_TRI_STRIDE
        self.emissive_tri_buffer.upload_sync(bytes(data[:cap_bytes]))
        if self._spectral and self._spectral_emitters_buffer is not None:
            sp_cap = self.emissive_tri_capacity * SPECTRAL_EMITTER_STRIDE
            while len(spectral_data) < sp_cap:
                spectral_data += b"\x00" * SPECTRAL_EMITTER_STRIDE
            self._spectral_emitters_buffer.upload_sync(bytes(spectral_data[:sp_cap]))
        self._num_emissive_tris = n
        print(f"[skinny] emissive triangles: {n}"
              + (" (uniform selection)" if uniform else " (power-weighted NEE)"))

    def _upload_sphere_lights(self, lights: list) -> None:
        """Pack each LightSphere into binding 17. Active count goes to
        FrameConstants.numSphereLights for the shader to bound its loop.

        Zero-intensity lights are dropped — keeping them in the buffer
        would burn an NEE / light-walk sample on a contribution that is
        guaranteed to be black.
        """
        def _has_power(lt) -> bool:
            if getattr(lt, "intensity", None) is not None and float(lt.intensity) == 0.0:
                return False
            rad = np.asarray(getattr(lt, "radiance", (0.0, 0.0, 0.0)), np.float32)
            return bool(np.any(rad > 0.0))

        enabled = [
            lt for lt in lights
            if getattr(lt, "enabled", True) and _has_power(lt)
        ]
        n = min(len(enabled), SPHERE_LIGHT_CAPACITY)
        data = bytearray()
        for i in range(SPHERE_LIGHT_CAPACITY):
            if i < n:
                light = enabled[i]
                pos = light.position
                rad = light.radiance
                data += struct.pack(
                    "fff f fff f",
                    float(pos[0]), float(pos[1]), float(pos[2]),
                    float(light.radius),
                    float(rad[0]), float(rad[1]), float(rad[2]),
                    0.0,
                )
            else:
                data += b"\x00" * SPHERE_LIGHT_STRIDE
        self.sphere_lights_buffer.upload_sync(bytes(data))
        self._num_sphere_lights = n
        # Σ(lum·r²) over the packed set → SPPM photon-group power Φ_S = 4π²·Σ(lum·r²)
        self._sphere_power_sum = float(sum(
            (0.2126 * float(lt.radiance[0]) + 0.7152 * float(lt.radiance[1])
             + 0.0722 * float(lt.radiance[2])) * float(lt.radius) ** 2
            for lt in enabled[:n]
        ))

    @property
    def uses_default_lights(self) -> bool:
        """Whether the active scene uses Skinny's DistantLight + IBL pair."""
        from skinny.scene import scene_uses_default_lights
        return scene_uses_default_lights(
            self._usd_scene,
            usd_active=self._is_usd_active(),
        )

    def _sync_auxiliary_light_authority(self, *, force: bool = False) -> None:
        """Mirror sphere/emissive sources owned by the active scene.

        USD buffers remain resident across model switches, so counts and
        backing data must be cleared when fallback authority becomes active
        and restored when the authored USD scene becomes active again.
        """
        token = (
            self._is_usd_active(),
            id(self._usd_scene) if self._usd_scene is not None else 0,
            self.uses_default_lights,
        )
        if not force and token == self._last_aux_light_authority_token:
            return

        from skinny.scene import scene_auxiliary_lights_for_authority
        sphere_lights, emissive_scene = scene_auxiliary_lights_for_authority(
            self._usd_scene,
            uses_default_lights=self.uses_default_lights,
        )
        self._upload_sphere_lights(sphere_lights)
        self._upload_emissive_triangles(
            emissive_scene if emissive_scene is not None else Scene(),
        )
        self._last_aux_light_authority_token = token

    def _upload_distant_lights(
        self,
        lights: list,
        *,
        fallback_controls: bool = False,
    ) -> None:
        """Pack each LightDir into binding 20. Active count goes to
        FrameConstants.numDistantLights for the shader to bound its loop.

        Honoured by every NEE path (path.allLightsNEE, bdpt.connectT1,
        skin_direct.skinAllLightsEstimator) and the BDPT s≥1 light-walk
        seed (bdpt.sampleLightOrigin). ``direct_light_index`` applies only
        to Skinny's fallback light; authored USD lights keep their own
        enabled/intensity state.
        """
        from skinny.scene import select_powered_distant_lights

        authority_enabled = not (
            fallback_controls and self.direct_light_index != 0
        )
        enabled = select_powered_distant_lights(
            lights,
            authority_enabled=authority_enabled,
        )
        n = min(len(enabled), DISTANT_LIGHT_CAPACITY)
        data = bytearray()
        # Spectral (Group 6.3): a companion SPD buffer (binding 50) carries each
        # light's authored illuminant spectrum, luminance-matched to its RGB
        # radiance so switching to the SPD only changes chromaticity, not
        # brightness. `_direction.w` holds the SPD slot (-1 = upsample the RGB).
        spd_data = bytearray()
        for i in range(DISTANT_LIGHT_CAPACITY):
            if i < n:
                light = enabled[i]
                d = np.asarray(light.direction, np.float32)
                r = np.asarray(light.radiance, np.float32)
                # RGB build packs 0.0 in this lane (byte-identical to the pre-6.3
                # layout); only the spectral build uses it as an SPD slot (-1 = none).
                spd_index = -1.0 if self._spectral else 0.0
                if self._spectral and getattr(light, "spectral_spd", None) is not None:
                    scaled = self._spectral_light_spd_scaled(light.spectral_spd, r)
                    if scaled is not None:
                        spd_index = float(i)
                        spd_data += scaled.astype("<f4").tobytes()
                if len(spd_data) < (i + 1) * SPECTRAL_LIGHT_SPD_STRIDE:
                    spd_data += b"\x00" * ((i + 1) * SPECTRAL_LIGHT_SPD_STRIDE - len(spd_data))
                data += struct.pack(
                    "fff f fff f",
                    float(d[0]), float(d[1]), float(d[2]), spd_index,
                    float(r[0]), float(r[1]), float(r[2]), 0.0,
                )
            else:
                data += b"\x00" * DISTANT_LIGHT_STRIDE
                spd_data += b"\x00" * SPECTRAL_LIGHT_SPD_STRIDE
        self.distant_lights_buffer.upload_sync(bytes(data))
        if self._spectral and self._spectral_light_spd_buffer is not None:
            self._spectral_light_spd_buffer.upload_sync(bytes(spd_data))
        self._num_distant_lights = n
        # Σlum over the packed set → SPPM photon-group power Φ_D = πR²·Σlum
        self._distant_lum_sum = float(sum(
            0.2126 * float(lt.radiance[0]) + 0.7152 * float(lt.radiance[1])
            + 0.0722 * float(lt.radiance[2])
            for lt in enabled[:n]
        ))

    def _spectral_light_spd_scaled(self, spd, radiance_rgb):
        """Return the authored SPD (95 samples) scaled so its hero-λ film resolve
        reproduces the RGB radiance's luminance — so enabling the SPD shifts only
        chromaticity. Returns None for a degenerate SPD (falls back to RGB)."""
        import numpy as _np

        from skinny.pbrt import spectra as _spectra
        from skinny.pbrt.spectral import _REC709_Y, _Y_INTEGRAL

        spd = _np.asarray(spd, dtype=_np.float64).reshape(-1)
        if spd.size != SPECTRAL_LIGHT_SPD_SAMPLES or not _np.any(spd > 0.0):
            return None
        y_spd = float(_spectra.spd_to_xyz(spd)[1]) / _Y_INTEGRAL
        if y_spd <= 0.0:
            return None
        y_target = float(_np.dot(_np.asarray(radiance_rgb, dtype=_np.float64), _REC709_Y))
        return (spd * (y_target / y_spd)).astype(_np.float32)

    def _upload_flat_materials(self, materials: list) -> None:
        """Pack each scene Material into the FLAT_MATERIAL_STRIDE record format
        and upload.

        Skin-typed materials (mtlx_target_name == "M_skinny_skin_default")
        get a zeroed record (the shader dispatches to the skin path before
        reading the flat-material buffer). All other materials — including
        the fallback slot 0 — are packed as flat materials.

        Also walks each Material.texture_paths.diffuseColor (when present)
        through the bindless TexturePool so the GPU has the actual image and
        the packed record carries the resolved array slot.
        """
        # Cached so the Metal wavefront pass build can re-run this upload once
        # its reflected MSL layouts exist (the wavefront-mode scene build is
        # `scene_bindings_only` — no megakernel reflection at upload time).
        self._last_uploaded_materials = list(materials)
        if len(materials) > self.material_capacity:
            new_cap = max(len(materials), self.material_capacity * 2)
            self.material_capacity = new_cap
            self._per_material_furnace = [False] * new_cap
            self.flat_material_buffer.destroy()
            self.flat_material_buffer = self._gpu.StorageBuffer(
                self.ctx, new_cap * FLAT_MATERIAL_STRIDE + 256
            )
            self.material_types_buffer.destroy()
            self.material_types_buffer = self._gpu.StorageBuffer(
                self.ctx, new_cap * 4 + 16
            )
            self.mtlx_skin_buffer.destroy()
            mtlx_slot_bytes = 256 if self.is_metal else self.mtlx_skin_record_size
            self.mtlx_skin_buffer = self._gpu.StorageBuffer(
                self.ctx, new_cap * mtlx_slot_bytes + 256
            )
            self.std_surface_buffer.destroy()
            self.std_surface_buffer = self._gpu.StorageBuffer(
                self.ctx, new_cap * STD_SURFACE_STRIDE + 16
            )
            if self._spectral and self._spectral_mat_emission_buffer is not None:
                self._spectral_mat_emission_buffer.destroy()
                self._spectral_mat_emission_buffer = self._gpu.StorageBuffer(
                    self.ctx, new_cap * SPECTRAL_EMITTER_STRIDE + 16
                )
            self._rebind_scene_descriptors()
            self._rebind_aux_material_descriptors()
        # Python-material slots route through MATERIAL_TYPE_PYTHON, but
        # `flatMaterials[matId]` still holds the UsdPreviewSurface inputs
        # that the generated `_pyMatInputs_<id>` adapter reads. Pack flat
        # data either way; only the type tag (and packed python_id) changes.
        from skinny.megakernel_sources import python_material_ids as _py_mat_ids_fn
        py_ids = _py_mat_ids_fn()
        self._material_python_ids: dict[int, int] = {}

        data = bytearray()
        types: list[int] = []
        spectral_emis = bytearray()   # binding 51: (T, scale) parallel to `data`
        for i, mat in enumerate(materials):
            # Spectral (Group 6.1 follow-up): per-material blackbody (T, scale) for
            # the exact-Planck visible/BSDF-hit emission path. Appended for EVERY
            # material (before any `continue`) to stay parallel-indexed to materialId.
            if self._spectral:
                bb_t, bb_s = 0.0, 0.0
                _bp = mat.parameter_overrides.get("emissive_spectral")
                if (_bp is not None and hasattr(_bp, "get")
                        and _bp.get("kind") == "blackbody"):
                    bb_t = float(_bp.get("temperature", 0.0) or 0.0)
                    if bb_t > 0.0:
                        from skinny.pbrt import spectral as _sp
                        _em = _override_color3(
                            mat.parameter_overrides, "emissiveColor", (0.0, 0.0, 0.0))
                        bb_s = float(_sp.blackbody_scale(bb_t, _em))
                spectral_emis += struct.pack("ff", bb_t, bb_s)
            if mat.mtlx_target_name == "M_skinny_skin_default":
                types.append(MATERIAL_TYPE_SKIN)
                data += b"\x00" * FLAT_MATERIAL_STRIDE
                continue
            mod = getattr(mat, "python_module", None)
            if mod and mod in py_ids:
                types.append(MATERIAL_TYPE_PYTHON)
                self._material_python_ids[i] = py_ids[mod]
            elif _material_is_volume(mat):
                # Free-standing medium boundary (nanovdb-volume-rendering):
                # pbrt `Material "interface"` with a MediumInterface. Packed as
                # flat data (medium fields at 160..240); the type tag routes the
                # GPU to the index-matched pass-through medium walk.
                types.append(MATERIAL_TYPE_VOLUME)
            elif _material_is_subsurface(mat):
                # pbrt subsurface: a dielectric boundary + an inline homogeneous
                # interior medium. Still packed as flat data (the medium fields
                # ride in FlatMaterialParams 160..192); the type tag routes the
                # GPU to the interior random walk instead of opacity=0 glass.
                types.append(MATERIAL_TYPE_SUBSURFACE)
            else:
                types.append(MATERIAL_TYPE_FLAT)
            indices: dict[str, int] = {
                "diffuseColor":  TexturePool.SENTINEL,
                "roughness":     TexturePool.SENTINEL,
                "metallic":      TexturePool.SENTINEL,
                "normal":        TexturePool.SENTINEL,
                "emissiveColor": TexturePool.SENTINEL,
                "opacity":       TexturePool.SENTINEL,
            }
            _LINEAR_INPUTS = {"roughness", "metallic", "normal"}
            # Default channel selectors mirror the UsdPreviewSurface fetch
            # convention: scalars from `.r`, alpha from `.a`, colour from
            # rgb. A USD-authored `outputs:<chan>` overrides this per-slot.
            _DEFAULT_CHANNELS = {
                "diffuseColor":  "rgb",
                "roughness":     "r",
                "metallic":      "r",
                "opacity":       "a",
                "emissiveColor": "rgb",
            }
            tex_bindings = getattr(mat, "texture_bindings", None) or {}
            channels: dict[str, str] = {}
            for input_name in indices:
                tex_path = mat.texture_paths.get(input_name)
                if tex_path is None:
                    continue
                binding = tex_bindings.get(input_name)
                wrap_s = binding.wrap_s if binding is not None else "repeat"
                wrap_t = binding.wrap_t if binding is not None else "repeat"
                indices[input_name] = self.texture_pool.add_or_get(
                    tex_path,
                    linear=input_name in _LINEAR_INPUTS,
                    wrap_s=wrap_s, wrap_t=wrap_t,
                )
                if input_name in _DEFAULT_CHANNELS:
                    if binding is not None:
                        channels[input_name] = binding.channel
                    else:
                        channels[input_name] = _DEFAULT_CHANNELS[input_name]

            # Normal-map scale/bias. UsdPreviewSurface authors per-channel
            # remap on the UsdUVTexture node; the default OpenGL convention
            # is scale=(2,2,2,_), bias=(-1,-1,-1,_) which maps unorm [0,1]
            # back to signed [-1,1]. DirectX-style normal maps author
            # scale.y=-2/bias.y=+1 to flip Y. When no binding is present
            # (MaterialX fallback path or non-USD materials) we apply the
            # OpenGL default so the shader's unchanged behaviour matches
            # the previous hardcoded `*2-1`.
            normal_binding = tex_bindings.get("normal")
            if normal_binding is not None:
                n_scale = (
                    float(normal_binding.scale[0]),
                    float(normal_binding.scale[1]),
                    float(normal_binding.scale[2]),
                )
                n_bias = (
                    float(normal_binding.bias[0]),
                    float(normal_binding.bias[1]),
                    float(normal_binding.bias[2]),
                )
            else:
                n_scale = (2.0, 2.0, 2.0)
                n_bias = (-1.0, -1.0, -1.0)

            channel_mask = _encode_channel_mask(channels)
            data += pack_flat_material(
                mat,
                diffuse_texture_idx=indices["diffuseColor"],
                roughness_texture_idx=indices["roughness"],
                metallic_texture_idx=indices["metallic"],
                normal_texture_idx=indices["normal"],
                emissive_texture_idx=indices["emissiveColor"],
                opacity_texture_idx=indices["opacity"],
                normal_scale=n_scale,
                normal_bias=n_bias,
                channel_mask=channel_mask,
                # Volume-material folds (nanovdb-volume-rendering): the loaded
                # grid's world→uvw + value max (`_sync_volume_grid`) and the
                # live scene scale — inert (identity/1.0) for other materials.
                volume_world_to_uvw=self._volume_world_to_uvw,
                volume_value_max=self._volume_value_max,
                mm_per_unit=float(self.mm_per_unit),
                # Group 6.4: only substitute the named-glass Cauchy A into the
                # scalar `ior` under --spectral; the RGB pack stays byte-identical.
                spectral=self._spectral,
            )
        if not data:
            data += b"\x00" * FLAT_MATERIAL_STRIDE
            types.append(MATERIAL_TYPE_FLAT)
        self.flat_material_buffer.upload_sync(bytes(data))
        if self._spectral and self._spectral_mat_emission_buffer is not None:
            sp_cap = self.material_capacity * SPECTRAL_EMITTER_STRIDE
            if not spectral_emis:
                spectral_emis += struct.pack("ff", 0.0, 0.0)
            while len(spectral_emis) < sp_cap:
                spectral_emis += b"\x00" * SPECTRAL_EMITTER_STRIDE
            self._spectral_mat_emission_buffer.upload_sync(bytes(spectral_emis[:sp_cap]))
        self._num_flat_materials = len(materials)
        self._material_types = types
        # Spectral v1 is FLAT-only: the megakernel spectral integrator
        # (path_spectral.slang) shades MATERIAL_TYPE_FLAT and terminates the path
        # on anything else. The CLI flag-level guard (reject_spectral_unsupported)
        # can't see scene contents, so this is the scene-level refusal the design
        # deferred to renderer setup — refuse here rather than silently render
        # non-flat pixels black.
        if self._spectral:
            nonflat = sorted({int(t) for t in types if int(t) != MATERIAL_TYPE_FLAT})
            if nonflat:
                raise SystemExit(
                    "skinny: --spectral supports only flat materials in v1 "
                    "(UsdPreviewSurface / standard_surface / OpenPBR), but this scene "
                    f"has non-flat material type code(s) {nonflat} "
                    "(skin=0 / debug=2 / python=3 / subsurface=4 / volume=5). Spectral "
                    "skin/subsurface/volume are follow-ups — render without --spectral."
                )
        self._upload_material_types()
        # Pack StdSurfaceParams for every material slot into binding 19.
        # Skin-typed slots get zeroed records (the shader dispatches to the
        # skin path before reading stdSurfaceParams); flat-typed slots get
        # the full standard_surface parameter set.
        #
        # Metal reads `StructuredBuffer<StdSurfaceParams>` MSL-padded (float3 →
        # 16 B, element stride ≈400 B), so when a Metal pipeline references
        # binding 19 the scalar `pack_std_surface_params` record (256 B) must be
        # relocated into the reflected MSL layout and the buffer sized at the MSL
        # stride — the same repack the skin params (`_pack_mtlx_skin_array_msl`)
        # and per-graph SSBOs get (see `pack_std_surface_params_msl`). This is
        # inert today: the only consumer of binding 19 is the Vulkan-only
        # `preview_pass` (scalar-correct on Vulkan) and the Metal megakernel
        # dead-strips it (`loadStdSurfaceParams` uncalled), so `std_surface_layout`
        # is empty there and this falls through to the scalar path. It hardens the
        # forthcoming Metal preview/raster port. Vulkan always reads scalar
        # (stride 256, no relocation).
        ss_layout = (getattr(self._msl_layout_source, "std_surface_layout", None)
                     if self.is_metal else None) or None
        ss_stride = (getattr(self._msl_layout_source, "std_surface_stride", 0)
                     if ss_layout else 0) or STD_SURFACE_STRIDE
        ss_data = bytearray()
        for i, mat in enumerate(materials):
            if i < len(types) and types[i] == MATERIAL_TYPE_FLAT:
                rec = pack_std_surface_params(mat)
                if ss_layout:
                    rec = pack_std_surface_params_msl(rec, ss_layout, ss_stride)
                ss_data += rec
            else:
                ss_data += b"\x00" * ss_stride
        while len(ss_data) < self.material_capacity * ss_stride:
            ss_data += b"\x00" * ss_stride
        # On Metal the MSL stride outgrows the scalar-sized buffer allocated at
        # init/realloc; grow it here. The Metal bind reads `.buffer` fresh at
        # dispatch (Vulkan rebind helpers are no-ops on Metal), so the realloc is
        # picked up without a descriptor rewrite. Vulkan keeps the 256 B stride,
        # so this never grows there.
        needed = self.material_capacity * ss_stride + 16
        if self.std_surface_buffer.size < needed:
            self.std_surface_buffer.destroy()
            self.std_surface_buffer = self._gpu.StorageBuffer(self.ctx, needed)
        self.std_surface_buffer.upload_sync(
            bytes(ss_data[: self.material_capacity * ss_stride])
        )
        # Per-graph MaterialX SSBOs (bindings GRAPH_BINDING_BASE+i). Each
        # distinct GraphFragment in `_scene_graph_fragments` owns a
        # StructuredBuffer<GraphParams_X> of length material_capacity, packed
        # with that graph's overrides at slots whose material is bound to
        # this graph and zero elsewhere.
        self._upload_graph_param_buffers()
        # Refresh binding-14 descriptor writes so newly populated slots
        # are visible to the shader.
        self._update_texture_pool_descriptors()

    def _upload_graph_param_buffers(self) -> None:
        """Pack + upload the single combined graph-param buffer.

        ALL scene MaterialX graphs share one matId-major byte buffer
        (`self._graph_params_combined`) bound at `GRAPH_BINDING_BASE`
        (change combine-graph-param-buffers). Slot `matId` holds that
        material's params in scalar (std430) layout, read in-shader by
        `graphParamsCombined.Load<GraphParams_X>(matId * GRAPH_PARAM_STRIDE)`.
        `Load<T>` uses the same scalar layout on the SPIR-V and Metal targets,
        so one scalar blob feeds both backends — no per-backend MSL repack and
        no per-graph buffer. A material maps to exactly one graph, so a single
        stride (`graph_param_combined_stride`, max over the scene's graphs)
        covers every slot. The buffer is allocated on first call and grown with
        `material_capacity`.
        """
        from skinny.materialx_runtime import pack_uniform_block
        from skinny.megakernel_sources import (
            GRAPH_BINDING_BASE,
            graph_param_combined_stride,
        )

        fragments = self._scene_graph_fragments
        if not fragments:
            # No graphs: drop any stale combined buffer; nothing to bind.
            if getattr(self, "_graph_params_combined", None) is not None:
                self._graph_params_combined.destroy()
                self._graph_params_combined = None
            return

        stride = graph_param_combined_stride(fragments)
        buf_size = self.material_capacity * stride + 16
        existing = getattr(self, "_graph_params_combined", None)
        if existing is None or existing.size < buf_size:
            if existing is not None:
                existing.destroy()
            self._graph_params_combined = self._gpu.StorageBuffer(self.ctx, buf_size)

        # Pack every graph's materials into the one matId-major blob.
        data = bytearray(self.material_capacity * stride)
        frag_by_gid = {idx + 2: gf for idx, gf in enumerate(fragments)}
        for mat_idx, gid in self._material_graph_ids.items():
            gf = frag_by_gid.get(gid)  # GRAPH_ID_FIRST = 2 (assign_graph_ids)
            if gf is None or mat_idx >= self.material_capacity:
                continue
            overrides = dict(self._material_graph_overrides.get(mat_idx, {}))
            mat = (self._usd_scene.materials[mat_idx]
                   if self._usd_scene is not None else None)
            # Resolve each filename input → bindless slot. Source precedence:
            #   1. mat.parameter_overrides[name] (slider override).
            #   2. mat.texture_paths[name] (USD loader-decoded path).
            #   3. UniformField.default (the .mtlx-authored value, a path str).
            # Relative paths resolve against the .mtlx document's source URI so
            # `textures/foo.jpg` references work regardless of the CWD.
            mtlx_dir: Optional[Path] = None
            if mat is not None and mat.mtlx_document is not None:
                src_uri = mat.mtlx_document.getSourceUri()
                if src_uri:
                    mtlx_dir = Path(src_uri).resolve().parent
            for f in (u for u in gf.uniform_block if u.type_name == "filename"):
                raw = overrides.get(f.name)
                if raw is None and mat is not None:
                    raw = mat.texture_paths.get(f.name)
                if raw is None:
                    raw = f.default
                if raw is None or raw == "":
                    continue
                p = Path(str(raw))
                if not p.is_absolute() and mtlx_dir is not None:
                    p = (mtlx_dir / p).resolve()
                try:
                    slot = self.texture_pool.add_or_get(p, linear=False)
                except Exception as e:  # noqa: BLE001
                    print(f"[skinny] graph[{gf.target_name}] mat[{mat_idx}] "
                          f"'{f.name}' texture load fail ({p}): {e}")
                    slot = 0
                overrides[f.name] = slot
            packed = pack_uniform_block(gf.uniform_block, overrides)
            data[mat_idx * stride:mat_idx * stride + len(packed)] = packed
        self._graph_params_combined.upload_sync(bytes(data))

        # Metal binds the combined buffer at dispatch by name (no Vulkan
        # descriptor set); the upload above is all that's needed there.
        if self.is_metal:
            return

        # Defensive: a slangc empty-graph fallback may leave the binding absent.
        if not self._scene_graph_bindings:
            return
        info = vk.VkDescriptorBufferInfo(
            buffer=self._graph_params_combined.buffer,
            offset=0,
            range=self._graph_params_combined.size,
        )
        for ds in self.descriptor_sets:
            write = vk.VkWriteDescriptorSet(
                dstSet=ds,
                dstBinding=GRAPH_BINDING_BASE,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[info],
            )
            vk.vkUpdateDescriptorSets(self.ctx.device, 1, [write], 0, None)

    def _upload_material_types(self) -> None:
        """Pack per-material type+flags into binding 16.

        Encoding per slot (uint):
          bits  0-7 : material type code (skin=0, flat=1, python=3)
          bits  8-9 : scatter mode for skin slots (bit0=BSSRDF, bit1=volume)
          bit  10   : per-material furnace mode
          bits 11-15: reserved.
          bits 16-23: MaterialX graphId (0 = no graph; 2+ = index into
                      `_scene_graph_fragments` + GRAPH_ID_FIRST). Read by
                      shaders/bindings.slang::materialGraphId().
          bits 24-31: Python material id when type == PYTHON. Read by
                      shaders/bindings.slang::pythonMaterialId().

        Re-uploaded whenever scatter mode, per-material furnace, or the
        scene's graph binding changes.
        """
        scatter_idx = int(np.clip(self.scatter_index, 0, len(self._scatter_mode_bits) - 1))
        scatter_bits = int(self._scatter_mode_bits[scatter_idx]) & 0x3
        types = self._material_types
        py_ids = self._refresh_material_python_ids()
        type_bytes = bytearray()
        for i in range(self.material_capacity):
            t = int(types[i]) if i < len(types) else MATERIAL_TYPE_FLAT
            packed = (t & 0xFF)
            if t == MATERIAL_TYPE_SKIN:
                packed |= (scatter_bits & 0x3) << 8
            if self._per_material_furnace[i]:
                packed |= 1 << 10
            gid = self._material_graph_ids.get(i, 0)
            packed |= (gid & 0xFF) << 16
            if t == MATERIAL_TYPE_PYTHON:
                packed |= (py_ids.get(i, 0) & 0xFF) << 24
            type_bytes += struct.pack("I", packed)
        self.material_types_buffer.upload_sync(bytes(type_bytes))
        self._last_scatter_index = scatter_idx

    def iter_graph_uniforms(self, material_id: int) -> list:
        """Return the MaterialX graph uniforms driving `material_id`.

        Each entry is a `materialx_runtime.UniformField` with `name`,
        `type_name`, and `default` populated from the gen-reflected
        graph fragment. Filtered to widget-friendly scalar / vector /
        color types — filename + string uniforms (resolved through the
        texture pool / framerange tokens) are skipped because the panel
        UIs don't author them through ordinary slider controls.

        Returns `[]` when the material has no graph (constant-input
        Glass / Jade / etc., or any material with `_material_graph_ids[i]
        < GRAPH_ID_FIRST`).
        """
        from skinny.materialx_runtime import GRAPH_ID_FIRST
        gid = self._material_graph_ids.get(material_id, 0)
        if gid < GRAPH_ID_FIRST:
            return []
        idx = gid - GRAPH_ID_FIRST
        if idx < 0 or idx >= len(self._scene_graph_fragments):
            return []
        gf = self._scene_graph_fragments[idx]
        # `filename` resolves to a bindless slot via TexturePool — not
        # something the user authors through a generic widget. `string`
        # is reserved for framerange tokens. Everything else (float /
        # integer / boolean / vector2-4 / color3-4) maps onto a regular
        # slider or color picker.
        skip = {"filename", "string"}
        return [u for u in gf.uniform_block if u.type_name not in skip]

    def apply_material_override(
        self, material_id: int, key: str, value: object
    ) -> None:
        """Mutate a scene material's parameter_overrides and re-upload.

        Used by the control panel when the user drags a per-material
        slider. Bumps `_material_version` so the next frame's state-hash
        check resets the accumulation buffer.
        """
        if self._usd_scene is None:
            return
        mats = self._usd_scene.materials
        if material_id <= 0 or material_id >= len(mats):
            return
        mats[material_id].parameter_overrides[key] = value
        # Mirror into the graph-overrides cache so per-graph SSBO packs
        # see UI slider drags on MaterialX-graph materials. Without this,
        # the cache stays seeded from scene-load time and slider edits
        # silently don't take effect for graph-bound prims.
        if material_id in self._material_graph_ids:
            self._material_graph_overrides.setdefault(
                material_id, {}
            )[key] = value
        self._upload_flat_materials(mats)
        self._material_version += 1

    def apply_material_overrides(
        self, material_id: int, values: "dict[str, object]"
    ) -> None:
        """Apply several parameter overrides to one material in a single pass.

        The fan-out write path for a synthesized MaterialX material
        (mcp-material-authoring, design D5): one logical `scene_set` maps to N
        generated uniforms, and all of them must land in the same edit so the
        re-upload and accumulation reset happen exactly once. Mirrors
        `apply_material_override` (including the graph-overrides cache mirror)
        but uploads and bumps `_material_version` a single time for the batch.
        """
        if self._usd_scene is None or not values:
            return
        mats = self._usd_scene.materials
        if material_id <= 0 or material_id >= len(mats):
            return
        graph_cache = (
            self._material_graph_overrides.setdefault(material_id, {})
            if material_id in self._material_graph_ids
            else None
        )
        for key, value in values.items():
            mats[material_id].parameter_overrides[key] = value
            if graph_cache is not None:
                graph_cache[key] = value
        self._upload_flat_materials(mats)
        self._material_version += 1

    def toggle_material_furnace(self, material_id: int, enabled: bool) -> None:
        if material_id < 0 or material_id >= self.material_capacity:
            return
        if enabled:
            for i in range(self.material_capacity):
                self._per_material_furnace[i] = (i == material_id)
        else:
            self._per_material_furnace[material_id] = False
        self._upload_material_types()
        self._material_version += 1

    # ── Material preview (Material Graph Editor) ────────────────────

    def _ensure_preview_resources(self, size: int) -> bool:
        """Lazy-create preview image / readback / pipeline. False = unavailable."""
        # The preview pipeline only needs the set-0 layout + the emitted
        # material modules — both live on the scene bindings, so it works in
        # wavefront mode too (where `self.pipeline` is None).
        if self._scene_bindings is None:
            return False
        if self._preview_size != size:
            # Size changed (or first init) — tear down old image + readback
            # and the pipeline (the pipeline holds a write to the image view).
            if self._preview_pipeline is not None:
                self._preview_pipeline.destroy()
                self._preview_pipeline = None
            if self._preview_readback is not None:
                self._preview_readback.destroy()
                self._preview_readback = None
            if self._preview_image is not None:
                self._preview_image.destroy()
                self._preview_image = None
            self._preview_size = size

        if self._preview_image is None:
            self._preview_image = self._gpu.StorageImage(
                self.ctx, size, size, transfer_src=True,
            )
        if self._preview_readback is None:
            # RGBA32F = 16 bytes per pixel.
            self._preview_readback = self._gpu.ReadbackBuffer(
                self.ctx, size, size, bytes_per_pixel=16,
            )
        if self._preview_pipeline is None:
            try:
                if self.is_metal:
                    # Native-Metal preview: compile preview_pass.slang to MSL,
                    # dispatch by binding resources by name (no descriptor sets,
                    # no output image view). Change metal-tool-dock-render P1.
                    from skinny.metal_compute import PreviewPipelineMetal
                    self._preview_pipeline = PreviewPipelineMetal(
                        self.ctx, self.shader_dir,
                        graph_fragments=list(self._scene_graph_fragments),
                    )
                else:
                    from skinny.vk_compute import PreviewPipeline
                    self._preview_pipeline = PreviewPipeline(
                        self.ctx, self.shader_dir,
                        self._scene_set0_layout,
                        self._preview_image.view,
                    )
            except RuntimeError as e:
                print(f"[skinny] preview pipeline build failed: {e}")
                self._preview_pipeline = None
                return False
        return True

    def render_material_preview(
        self,
        material_id: int,
        prim_kind: int,
        *,
        size: int = 256,
        yaw: float = 0.6,
        pitch: float = 0.4,
        distance: float = 3.0,
        fov_tan: float = 0.55,
    ) -> "tuple[bytes, int] | None":
        """Dispatch the preview compute shader for `material_id` on the
        chosen primitive and return (rgba_float32_bytes, size).

        Reuses the main descriptor set 0 (so all material SSBOs / texture
        bindings / per-graph buffers are visible). Push constants carry
        the per-call inputs. Synchronous: submits, waits idle, reads back.
        Returns None when the renderer is not ready (no scene loaded, or
        slangc failed on preview_pass.slang).
        """
        if self.is_metal:
            # Bound the single-command-buffer Metal preview dispatch under the
            # GPU watchdog (codex #2) — clamp before allocating the output image
            # so image size, push `size`, and the returned size stay consistent.
            size = min(int(size), _METAL_PREVIEW_MAX_SIZE)
        if not self._ensure_preview_resources(size):
            return None
        if self._usd_scene is None:
            return None
        if material_id <= 0 or material_id >= len(self._usd_scene.materials):
            return None

        graph_id = int(self._material_graph_ids.get(material_id, 0))

        if self.is_metal:
            # Native-Metal preview dispatch (change metal-tool-dock-render P1).
            return self._render_material_preview_metal(
                material_id, graph_id, prim_kind, size,
                yaw, pitch, distance, fov_tan,
            )

        from skinny.vk_compute import PreviewPipeline

        if self.descriptor_sets is None or not self.descriptor_sets:
            return None

        # Allocate a one-shot command buffer (same pattern as the existing
        # screenshot path). We submit on the compute queue and wait idle.
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]

        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

        pp = self._preview_pipeline
        vk.vkCmdBindPipeline(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pp.pipeline,
        )
        # Bind set 0 (main material descriptors) + set 1 (preview output).
        # Issued as two single-set calls because python-vulkan's cffi
        # binding fails ("array item of unknown size void") when given a
        # multi-element list — every other call site in this file uses a
        # single-element list, matching that limitation.
        vk.vkCmdBindDescriptorSets(
            cmd,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pp.pipeline_layout,
            0, 1, [self.descriptor_sets[0]],
            0, None,
        )
        vk.vkCmdBindDescriptorSets(
            cmd,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pp.pipeline_layout,
            1, 1, [pp.descriptor_set],
            0, None,
        )
        push_bytes = PreviewPipeline.pack_push(
            material_id, graph_id, prim_kind, size,
            yaw, pitch, distance, fov_tan,
        )
        # python-vulkan binds `const void* pValues` via cffi; pass a typed
        # char buffer so cffi can size the array correctly.
        import cffi as _cffi
        _ffi = _cffi.FFI()
        push_buf = _ffi.new("char[]", push_bytes)
        vk.vkCmdPushConstants(
            cmd, pp.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, len(push_bytes), push_buf,
        )
        groups = (size + 7) // 8
        vk.vkCmdDispatch(cmd, groups, groups, 1)

        # GENERAL → TRANSFER_SRC, copy, → GENERAL.
        sub = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        )
        b_to_src = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image=self._preview_image.image,
            subresourceRange=sub,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, [b_to_src],
        )
        self._preview_readback.record_copy_from(cmd, self._preview_image.image)
        b_to_general = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self._preview_image.image,
            subresourceRange=sub,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [b_to_general],
        )
        vk.vkEndCommandBuffer(cmd)

        submit = vk.VkSubmitInfo(
            commandBufferCount=1, pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1, [submit], vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(
            self.ctx.device, self.ctx.command_pool, 1, [cmd],
        )
        return self._preview_readback.read(), size

    def _render_material_preview_metal(
        self, material_id: int, graph_id: int, prim_kind: int, size: int,
        yaw: float, pitch: float, distance: float, fov_tan: float,
    ) -> "tuple[bytes, int]":
        """Metal path of `render_material_preview` (change metal-tool-dock-render
        P1). Mirrors `_render_megakernel_metal`: build the set-0 material bind
        dict (`_build_metal_binds`) + the bindless texture pool, pack the same
        32-byte push block, dispatch `PreviewPipelineMetal` over `size×size`,
        and read back the RGBA32F output image (float32, matching the Vulkan
        `(pixels, size)` contract the Material Graph dock reshapes)."""
        from skinny.vk_compute import PreviewPipeline

        push_bytes = PreviewPipeline.pack_push(
            material_id, graph_id, prim_kind, size,
            yaw, pitch, distance, fov_tan,
        )
        binds = self._build_metal_binds()
        bindless = (
            "flatMaterialTextures",
            [(s.texture if s is not None else None)
             for s in self.texture_pool._slots],
        )
        self._preview_pipeline.dispatch(
            size,
            push_bytes=push_bytes,
            # Pack `fc` against the preview program's own reflected layout so the
            # preview works before any megakernel/wavefront layout source exists
            # (Metal wavefront mode, codex #1).
            uniform_blob=self._pack_uniforms_msl(self._preview_pipeline),
            binds=binds,
            output_image=self._preview_image.texture,
            bindless=bindless,
        )
        # Read the float32 output directly — the Metal `ReadbackBuffer` would
        # down-convert to RGBA8, but the dock consumer reshapes float32 RGBA.
        arr = self._preview_image.texture.to_numpy()
        return np.ascontiguousarray(arr, dtype=np.float32).tobytes(), size

    def apply_light_override(
        self, light_type: str, light_index: int, key: str, value: object,
    ) -> None:
        """Mutate a scene light parameter and re-upload."""
        if light_type == "env":
            if key == "intensity":
                if (
                    self._is_usd_active()
                    and self._usd_scene is not None
                    and self._usd_scene.environment is not None
                ):
                    self._usd_scene.environment.intensity = float(value)
                else:
                    self.env_intensity = float(value)
                prim = self._find_dome_light_prim(light_index)
                if prim is not None:
                    try:
                        from pxr import UsdLux
                        UsdLux.DomeLight(prim).CreateIntensityAttr().Set(
                            float(value),
                        )
                    except Exception:
                        pass
                self._material_version += 1
            return

        if light_type == "dir":
            has_usd_light = (
                self._is_usd_active()
                and self._usd_scene is not None
                and 0 <= light_index < len(self._usd_scene.lights_dir)
            )
            if has_usd_light:
                light = self._usd_scene.lights_dir[light_index]
                radiance = np.asarray(light.radiance, np.float32)
                intensity = float(np.max(radiance))
                color = (
                    radiance / intensity
                    if intensity > 1e-6
                    else np.ones(3, np.float32)
                )
                direction = np.asarray(light.direction, np.float32)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    direction = direction / norm
                elevation = float(
                    np.degrees(np.arcsin(np.clip(direction[1], -1.0, 1.0)))
                )
                azimuth = float(
                    np.degrees(np.arctan2(direction[0], direction[2]))
                )
                if key == "color":
                    color = _light_value_to_vec3(value)
                    light.radiance = (color * intensity).astype(np.float32)
                elif key == "intensity":
                    light.radiance = (
                        color * float(value)
                    ).astype(np.float32)
                elif key in ("elevation", "azimuth"):
                    if key == "elevation":
                        elevation = float(value)
                    else:
                        azimuth = float(value)
                    el = np.radians(elevation)
                    az = np.radians(azimuth)
                    direction = np.array([
                        np.cos(el) * np.sin(az),
                        np.sin(el),
                        np.cos(el) * np.cos(az),
                    ], dtype=np.float32)
                    light.direction = direction / np.linalg.norm(direction)
                self._upload_distant_lights(self._usd_scene.lights_dir)
                self._material_version += 1
                return

            # Synthesized fallback light: renderer slider state is the source
            # of truth and remains available for the next light-less scene.
            if key == "color":
                color = _light_value_to_vec3(value)
                self.light_color_r = float(color[0])
                self.light_color_g = float(color[1])
                self.light_color_b = float(color[2])
            elif key == "intensity":
                self.light_intensity = float(value)
            elif key == "elevation":
                self.light_elevation = float(value)
            elif key == "azimuth":
                self.light_azimuth = float(value)
            self._update_light()
            self._material_version += 1
            return

        if self._usd_scene is None:
            return

        if light_type == "sphere":
            lights = self._usd_scene.lights_sphere
            if light_index < 0 or light_index >= len(lights):
                return
            light = lights[light_index]
            if key == "color":
                color = _light_value_to_vec3(value)
                light.color = color.astype(np.float32)
                light.radiance = (color * float(light.intensity)).astype(np.float32)
            elif key == "intensity":
                light.intensity = float(value)
                light.radiance = (
                    np.asarray(light.color, np.float32) * float(value)
                ).astype(np.float32)
            elif key == "radius":
                light.radius = float(value)
            self._upload_sphere_lights(lights)
            self._material_version += 1

    def _resolve_renderer_ref(self, prim_path: str):
        """Map a USD prim path to its ``RendererRef(kind, index)`` via the scene
        graph node tree.

        Returns ``None`` when no scene graph is built (e.g. the headless
        ``set_usd_scene`` path) or the path carries no renderer binding.
        """
        sg = self._scene_graph
        if sg is None:
            return None
        from skinny.scene_graph import find_node_by_path
        node = find_node_by_path(sg, prim_path)
        return node.renderer_ref if node is not None else None

    def _instance_indices_for_path(self, prim_path: str) -> list[int]:
        """Indices into ``_usd_scene.instances`` baked from ``prim_path``.

        Uses the prim-path index (works headless); falls back to the scene
        graph's instance ref when the index has no entry.
        """
        idxs = self._prim_to_instances.get(prim_path)
        if idxs:
            return list(idxs)
        ref = self._resolve_renderer_ref(prim_path)
        if ref is not None and ref.kind == "instance" and self._usd_scene is not None:
            if 0 <= ref.index < len(self._usd_scene.instances):
                return [ref.index]
        return []

    def apply_instance_transform(
        self,
        prim_path: str,
        translate: tuple[float, float, float],
        rotate_deg: tuple[float, float, float],
        scale: tuple[float, float, float],
    ) -> None:
        """Recompose TRS into a 4x4 and re-upload the scene.

        Prim-path keyed: targets every instance baked from ``prim_path`` (a
        single prim may expand to several instances).
        """
        if self._usd_scene is None:
            return
        indices = self._instance_indices_for_path(prim_path)
        if not indices:
            return
        from skinny.scene_graph import compose_trs_matrix
        m = compose_trs_matrix(translate, rotate_deg, scale)
        for i in indices:
            self._usd_scene.instances[i].transform = m
        self._upload_usd_scene()
        self._material_version += 1

    def apply_node_enabled(self, prim_path: str, enabled: bool) -> None:
        """Toggle a single scene node on/off by prim path and re-upload buffers."""
        enabled = bool(enabled)
        # Instances resolve through the prim-path index so this works headless
        # (no scene graph). One prim may map to several instances.
        inst_idxs = self._prim_to_instances.get(prim_path)
        if inst_idxs:
            if self._usd_scene is None:
                return
            for i in inst_idxs:
                self._usd_scene.instances[i].enabled = enabled
            self._upload_usd_scene()
            self._material_version += 1
            return
        # Lights / camera / environment resolve via the scene-graph node ref.
        ref = self._resolve_renderer_ref(prim_path)
        kind = ref.kind if ref is not None else None
        index = ref.index if ref is not None else -1
        scene = self._usd_scene
        if kind == "light_dir":
            if scene is not None and 0 <= index < len(scene.lights_dir):
                scene.lights_dir[index].enabled = enabled
                self._upload_distant_lights(scene.lights_dir)
            else:
                # Synthesised default direct light: drives the renderer-owned
                # direct_light_index toggle (0=On, 1=Off). The buffer is
                # re-uploaded by update()'s per-frame mirror.
                self.direct_light_index = 0 if enabled else 1
            self._material_version += 1
            return
        if scene is None:
            return
        if kind == "instance":
            if 0 <= index < len(scene.instances):
                scene.instances[index].enabled = enabled
                self._upload_usd_scene()
        elif kind == "light_sphere":
            if 0 <= index < len(scene.lights_sphere):
                scene.lights_sphere[index].enabled = enabled
                self._upload_sphere_lights(scene.lights_sphere)
        elif kind in ("environment", "light_env"):
            if scene.environment is not None:
                scene.environment.enabled = enabled
        elif kind == "camera":
            if scene.camera_override is not None:
                scene.camera_override.enabled = enabled
        else:
            return
        self._material_version += 1

    def apply_subtree_enabled(self, usd_path: str, enabled: bool) -> None:
        """Toggle every renderer-bound leaf in the subtree rooted at ``usd_path``.

        Walks the SceneGraphNode tree (which mirrors the USD hierarchy) and
        flips the ``enabled`` flag on every instance / light / camera below
        the node, then issues one GPU re-upload for each affected buffer.
        """
        if self._usd_scene is None or self._scene_graph is None:
            return
        from skinny.scene_graph import find_node_by_path
        root = find_node_by_path(self._scene_graph, usd_path)
        if root is None:
            return
        flags = {"instance": False, "light_sphere": False, "light_dir": False}
        self._walk_apply_enabled(root, bool(enabled), flags)
        if flags["instance"]:
            self._upload_usd_scene()
        if flags["light_sphere"]:
            self._upload_sphere_lights(self._usd_scene.lights_sphere)
        if flags["light_dir"]:
            self._upload_distant_lights(self._usd_scene.lights_dir)
        self._material_version += 1

    # ── Gizmo (Phase D) ─────────────────────────────────────────────

    def set_gizmo_target(self, instance_index: int) -> None:
        """Select a mesh instance for the rotate gizmo. Pass -1 to clear."""
        if instance_index < 0 or self._usd_scene is None:
            self.gizmo.clear_target()
            return
        instances = self._usd_scene.instances
        if not (0 <= instance_index < len(instances)):
            self.gizmo.clear_target()
            return
        inst = instances[instance_index]
        # Pivot = instance origin in world (last row of the row-vector-
        # convention transform). Good enough for now; mesh centroid would
        # be marginally nicer for off-origin geometry.
        pivot = np.array(inst.transform[3, :3], dtype=np.float32)
        self.gizmo.set_target(
            instance_index, pivot, _instance_local_basis(inst.transform),
        )

    def gizmo_cycle_mode(self):
        """Advance the gizmo to the next mode (space-key cycle). No-op while a
        drag is in progress. Returns the new ``GizmoMode``."""
        return self.gizmo.cycle_mode()

    def gizmo_hit_test(self, mouse_x: float, mouse_y: float) -> str | None:
        if not self.gizmo.has_target:
            return None
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(self.width / max(self.height, 1))
        return self.gizmo.hit_test(
            mouse_x, mouse_y, view, proj, self.width, self.height,
        )

    def gizmo_begin_drag(
        self, axis: str, mouse_x: float, mouse_y: float,
    ) -> bool:
        if not self.gizmo.has_target or self._usd_scene is None:
            return False
        idx = self.gizmo.target_index
        if not (0 <= idx < len(self._usd_scene.instances)):
            return False
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(self.width / max(self.height, 1))
        self.gizmo.begin_drag(
            axis, mouse_x, mouse_y, view, proj, self.width, self.height,
            self._usd_scene.instances[idx].transform,
        )
        return True

    def gizmo_update_drag(self, mouse_x: float, mouse_y: float) -> bool:
        if not self.gizmo.is_dragging or self._usd_scene is None:
            return False
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(self.width / max(self.height, 1))
        result = self.gizmo.update_drag(
            mouse_x, mouse_y, view, proj, self.width, self.height,
        )
        if result is None:
            return False
        t, r, s = result
        idx = self.gizmo.target_index
        if self._usd_scene is None or not (0 <= idx < len(self._usd_scene.instances)):
            return False
        self.apply_instance_transform(
            self._usd_scene.instances[idx].prim_path, t, r, s,
        )
        return True

    def gizmo_end_drag(self) -> None:
        self.gizmo.end_drag()

    # ── BXDF visualizer scene pick ──────────────────────────────────

    def request_scene_pick(
        self, mouse_x: float, mouse_y: float, callback,
    ) -> None:
        """Capture the HitInfo of the pixel under (mouse_x, mouse_y).

        Sets ``pickArmed`` in the next frame's UBO so main_pass writes the
        hit into ``toolBuffer``. After at least one full frame completes
        the result is forwarded to ``callback(dict | None)`` from
        ``poll_pick_result`` (called every frame from ``render``). The
        callback receives None when the ray missed the scene.
        """
        # GLFW pixel coordinates have origin at the upper-left, matching
        # the shader's `dispatchThreadID.xy` mapping, so no Y flip needed.
        px = max(0, min(int(mouse_x), self.width - 1))
        py = max(0, min(int(mouse_y), self.height - 1))
        self._pick_pixel = (px, py)
        self._pick_armed = True
        # Reset the frame counter so we wait a full pipeline length
        # (MAX_FRAMES_IN_FLIGHT) before reading; the write may live in a
        # frame still queued on the GPU.
        self._pick_frame_count = 0
        self._pending_pick_callbacks.append(callback)

    def poll_pick_result(self) -> None:
        """Drain any pending pick callbacks once their frame has retired."""
        if not self._pending_pick_callbacks:
            return
        # Defer reads until at least one full frame has been submitted and
        # waited on after arming, so the buffer write is visible.
        self._pick_frame_count += 1
        if self._pick_frame_count < MAX_FRAMES_IN_FLIGHT + 1:
            return

        # Pick output sits at toolBuffer slots 8..11 (byte offset 128);
        # slots 0..7 are reserved for the BXDF eval header. Layout matches
        # `main_pass.slang` pick write:
        #   [8]  = float4(position.xyz, t)
        #   [9]  = float4(normal.xyz, asfloat(materialId))
        #   [10] = float4(tangent.xyz, hitFlag)
        #   [11] = float4(uv.xy, hasTangent, _pad)
        raw = self.tool_buffer.read(64, offset=128)
        px = np.frombuffer(raw[0:12], dtype=np.float32)
        t = float(np.frombuffer(raw[12:16], dtype=np.float32)[0])
        n = np.frombuffer(raw[16:28], dtype=np.float32)
        mat_bits = np.frombuffer(raw[28:32], dtype=np.uint32)[0]
        tan = np.frombuffer(raw[32:44], dtype=np.float32)
        hit_flag = float(np.frombuffer(raw[44:48], dtype=np.float32)[0])
        uv = np.frombuffer(raw[48:56], dtype=np.float32)
        has_tangent = float(np.frombuffer(raw[56:60], dtype=np.float32)[0])

        if hit_flag > 0.5:
            result = {
                "position": np.array(px, dtype=np.float32).copy(),
                "normal": np.array(n, dtype=np.float32).copy(),
                "tangent": np.array(tan, dtype=np.float32).copy(),
                "uv": np.array(uv, dtype=np.float32).copy(),
                "t": t,
                "material_id": int(mat_bits),
                "has_tangent": has_tangent > 0.5,
                "pixel": tuple(self._pick_pixel),
            }
        else:
            result = None

        callbacks = self._pending_pick_callbacks
        self._pending_pick_callbacks = []
        self._pick_armed = False
        for cb in callbacks:
            try:
                cb(result)
            except Exception as exc:
                print(f"[skinny] pick callback raised: {exc}")

    # ── Autofocus (thick-lens camera) ───────────────────────────────

    def autofocus_at_pixel(self, mouse_x: float, mouse_y: float) -> None:
        """Set ``camera.focus_distance`` to the axial scene depth at the
        pixel under (mouse_x, mouse_y). No-op when no lens is loaded or
        the ray misses the scene. Result lands asynchronously after the
        scene pick retires (``poll_pick_result``).
        """
        lens = getattr(self.camera, "lens", None)
        if lens is None or not getattr(lens, "active_elements", None):
            return
        self.request_scene_pick(mouse_x, mouse_y, self._on_autofocus_hit)

    def _on_autofocus_hit(self, result: dict | None) -> None:
        if result is None:
            return
        if not hasattr(self.camera, "focus_distance"):
            return
        fwd = np.asarray(self.camera.forward(), dtype=np.float32)
        hit = np.asarray(result["position"], dtype=np.float32)
        pos = np.asarray(self.camera.position, dtype=np.float32)
        d = float(np.dot(hit - pos, fwd))
        d = max(d, 1e-3)
        self.camera.focus_distance = d
        self.accum_frame = 0

    def request_bssrdf_eval(
        self, params: dict, callback,
    ) -> None:
        """Dispatch a GPU BSSRDF (skin) lobe eval.

        ``params`` mirrors ``request_bxdf_eval`` plus
        ``entrance_position`` (vec3). The shader uses ``r = ||xo - xi||``
        with ``mmPerUnit`` to evaluate the Burley diffusion profile.
        Only meaningful for MATERIAL_TYPE_SKIN; non-skin materials read
        back as zero.
        """
        params = dict(params)
        params["_tool_mode"] = 3  # TOOL_MODE_BSSRDF
        self.request_bxdf_eval(params, callback)

    def request_bxdf_eval(
        self, params: dict, callback,
    ) -> None:
        """Dispatch a GPU BXDF lobe eval at the picked shading frame.

        ``params`` must contain:
            material_id (int), position (vec3), normal (vec3),
            tangent (vec3), uv (vec2), locked_dir (vec3, tangent space),
            lock_mode (int 0=lock wi, 1=lock wo),
            n_theta (int), n_phi (int).

        Synchronous: writes the tool header, runs a one-shot compute
        submit on the compute queue (sized to the grid only, not the
        full screen), waits for completion, disarms, reads the grid,
        and invokes ``callback`` before returning. The main render loop
        is untouched — no viewport flicker, no wasted re-evals.
        """
        n_theta = int(params["n_theta"])
        n_phi = int(params["n_phi"])
        n_theta = max(1, min(n_theta, 128))
        n_phi = max(1, min(n_phi, 128))

        # The BXDF/BSSRDF visualiser reuses the megakernel main_pass tool-mode
        # dispatch. In wavefront mode that pipeline isn't compiled; on Metal the
        # one-shot readback below is Vulkan-only (command_pool / vkDeviceWaitIdle /
        # descriptor_sets, none of which exist on the compute-only Metal context —
        # and `self.pipeline` is non-None there, so the None check alone wouldn't
        # catch it). Degrade gracefully to a zeroed grid in both cases instead of
        # crashing. (A native Metal tool-dispatch sibling is a later phase.)
        if self.pipeline is None or self.is_metal:
            try:
                callback(np.zeros((n_theta, n_phi, 3), dtype=np.float32))
            except Exception as exc:
                print(f"[skinny] bxdf eval callback raised: {exc}")
            return
        if n_theta * n_phi > 128 * 64:
            raise ValueError(
                f"BXDF grid {n_theta}×{n_phi} exceeds tool buffer capacity"
            )

        mat_id = int(params["material_id"])
        P = np.asarray(params["position"], dtype=np.float32).reshape(3)
        N = np.asarray(params["normal"], dtype=np.float32).reshape(3)
        T = np.asarray(params["tangent"], dtype=np.float32).reshape(3)
        UV = np.asarray(params["uv"], dtype=np.float32).reshape(2)
        dLocked = np.asarray(params["locked_dir"], dtype=np.float32).reshape(3)
        lock_mode = int(params["lock_mode"])

        # Pack 8 × float4 (128 bytes) header. Mixed uint / float fields
        # are written through the float4 view using `struct` so the
        # shader can asuint() the uint slots and treat the rest as float.
        # `_tool_mode` is an internal escape hatch used by
        # `request_bssrdf_eval` to switch to TOOL_MODE_BSSRDF and stuff
        # the entrance position into slot 7.
        tool_mode = int(params.get("_tool_mode", 2))
        entrance = params.get("entrance_position")
        if entrance is not None:
            xi = np.asarray(entrance, dtype=np.float32).reshape(3)
        else:
            xi = np.zeros(3, dtype=np.float32)
        header = bytearray(128)
        struct.pack_into(
            "IIII", header, 0,
            tool_mode,
            lock_mode,
            n_theta,
            n_phi,
        )
        struct.pack_into("IIII", header, 16, mat_id, 0, 0, 0)
        struct.pack_into("ffff", header, 32, float(dLocked[0]), float(dLocked[1]), float(dLocked[2]), 0.0)
        struct.pack_into("ffff", header, 48, float(P[0]), float(P[1]), float(P[2]), 0.0)
        struct.pack_into("ffff", header, 64, float(N[0]), float(N[1]), float(N[2]), 0.0)
        struct.pack_into("ffff", header, 80, float(T[0]), float(T[1]), float(T[2]), 0.0)
        struct.pack_into("ffff", header, 96, float(UV[0]), float(UV[1]), 0.0, 0.0)
        struct.pack_into("ffff", header, 112, float(xi[0]), float(xi[1]), float(xi[2]), 0.0)
        self.tool_buffer.write(bytes(header), offset=0)

        # Zero the previous grid so partial reads don't show stale data.
        grid_bytes = n_theta * n_phi * 16
        self.tool_buffer.write(b"\x00" * grid_bytes, offset=256)

        # One-shot synchronous dispatch on the compute queue. Reuses the
        # main pipeline + descriptor_sets[0] (same layout, all scene
        # bindings live). vkDeviceWaitIdle covers descriptor-in-use
        # validation since the main render loop also runs on the
        # compute queue.
        vk.vkDeviceWaitIdle(self.ctx.device)

        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        vk.vkCmdBindPipeline(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline.pipeline_layout, 0, 1, [self.descriptor_sets[0]],
            0, None,
        )
        groups_x = (n_theta + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        groups_y = (n_phi + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        vk.vkCmdDispatch(cmd, groups_x, groups_y, 1)
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

        # Disarm so the next normal frame's main_pass mainImage reads
        # toolMode = 0 and renders normally.
        self.tool_buffer.write(b"\x00" * 16, offset=0)

        raw = self.tool_buffer.read(grid_bytes, offset=256)
        grid = np.frombuffer(raw, dtype=np.float32).reshape(n_phi, n_theta, 4)
        # Drop alpha channel; transpose to (n_theta, n_phi, 3) to match
        # the visualizer's grid convention.
        result = np.array(grid[..., :3].transpose(1, 0, 2), dtype=np.float32)
        try:
            callback(result)
        except Exception as exc:
            print(f"[skinny] bxdf eval callback raised: {exc}")

    def read_structural_aov(self) -> np.ndarray:
        """Dispatch one megakernel frame in ``TOOL_MODE_STRUCTURAL`` and read the
        per-primary-ray structural channel back from the tool buffer.

        Returns an ``(H, W, 4)`` float32 array; the last axis is
        ``(hit_mask, instance_id, material_id, depth)`` — hit_mask is 1.0 on a
        hit else 0.0, the ids are integer-valued floats, and depth is the
        world-space ray parameter at the hit (0.0 on a miss).

        Backend-agnostic: the dispatch goes through the active backend's headless
        frame path and the tool buffer (binding 30) is host-visible on both
        backends. Used by the Metal↔Vulkan structural-parity test (6.1); the
        ``runFrame`` structural write reuses the real primary ray, so the channel
        reflects exactly what the renderer traces. Determinism: the frame is
        dispatched from ``accum_frame == 0`` so both backends use the same
        ``createRNG(pixel, frameIndex)`` AA jitter.

        Raises ``RuntimeError`` if the megakernel pipeline is not built (e.g.
        wavefront mode, where the tool-mode dispatch is not compiled) and
        ``ValueError`` if the resolution overflows the tool buffer.
        """
        if self.pipeline is None or not self._backend_render_ready:
            raise RuntimeError(
                "read_structural_aov requires a built megakernel pipeline"
            )
        w, h = self.width, self.height
        n = w * h
        base_bytes = TOOL_STRUCT_AOV_BASE * 16
        need = base_bytes + n * 16
        if need > self.tool_buffer.size:
            raise ValueError(
                f"structural AOV needs {need} B but tool buffer is "
                f"{self.tool_buffer.size} B — lower the render resolution"
            )

        # Arm: TOOL_MODE_STRUCTURAL at tool-buffer slot 0.x. Only .x is read.
        header = bytearray(16)
        struct.pack_into("Ifff", header, 0, TOOL_MODE_STRUCTURAL, 0.0, 0.0, 0.0)
        self.tool_buffer.write(bytes(header), offset=0)
        # Zero the structural region so a partial/missed write reads as a miss.
        self.tool_buffer.write(b"\x00" * (n * 16), offset=base_bytes)

        # Deterministic single frame: same frameIndex on both backends ⇒ same
        # AA jitter ⇒ same primary rays.
        self.accum_frame = 0
        self.render_headless()

        # Read the GPU's structural output BEFORE disarming. On Metal,
        # HostStorageBuffer.write() re-uploads the full host shadow (which never
        # saw the GPU writes), so disarming first would clobber the structural
        # region back to the pre-dispatch zeros — the read must come first.
        raw = self.tool_buffer.read(n * 16, offset=base_bytes)

        # Disarm so the next normal frame renders.
        self.tool_buffer.write(b"\x00" * 16, offset=0)

        return np.frombuffer(raw, dtype=np.float32).reshape(h, w, 4).copy()

    def _refresh_gizmo_segments(self) -> None:
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(self.width / max(self.height, 1))

        # Refresh pivot (and the local basis) from the live instance transform
        # so the idle gizmo follows the geometry. Skipped mid-drag — begin_drag
        # froze both, and a live read would feed back into the manipulation.
        if (
            self.gizmo.has_target
            and self._usd_scene is not None
            and not self.gizmo.is_dragging
        ):
            idx = self.gizmo.target_index
            if 0 <= idx < len(self._usd_scene.instances):
                xf = self._usd_scene.instances[idx].transform
                self.gizmo.pivot_world = np.array(xf[3, :3], dtype=np.float32)
                self.gizmo.target_basis = _instance_local_basis(xf)

        segs: list = []
        if self.gizmo.has_target:
            segs.extend(
                self.gizmo.build_segments(view, proj, self.width, self.height)
            )
        if self._zoom_drag_overlay is not None:
            from skinny.gizmo import GizmoSegment
            x0, y0, x1, y1 = self._zoom_drag_overlay
            color = (0.95, 0.85, 0.20)
            for ax, ay, bx, by in (
                (x0, y0, x1, y0), (x1, y0, x1, y1),
                (x1, y1, x0, y1), (x0, y1, x0, y0),
            ):
                segs.append(GizmoSegment(
                    ax=ax, ay=ay, bx=bx, by=by,
                    r=color[0], g=color[1], b=color[2], width=1.5,
                ))

        if not segs and self._num_gizmo_segments == 0:
            return
        self._upload_gizmo_segments(segs)

    def _aspect_constrain_pixels(
        self,
        start_px: tuple[float, float],
        end_px: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Snap a drag rect so its pixel aspect matches the window's.

        Keeps the longest side (relative to window aspect) and the
        start corner fixed; the short side is computed from the window
        aspect. If the resulting rect would exceed the window from the
        start corner, both sides are scaled down uniformly so the
        aspect remains intact.
        """
        sx, sy = float(start_px[0]), float(start_px[1])
        ex, ey = float(end_px[0]), float(end_px[1])
        raw_w = abs(ex - sx)
        raw_h = abs(ey - sy)
        if raw_w == 0.0 and raw_h == 0.0:
            return (sx, sy), (ex, ey)
        W = max(self.width, 1)
        H = max(self.height, 1)
        aspect = W / H
        sign_x = 1.0 if ex >= sx else -1.0
        sign_y = 1.0 if ey >= sy else -1.0
        if raw_w >= raw_h * aspect:
            w, h = raw_w, raw_w / aspect
        else:
            w, h = raw_h * aspect, raw_h
        max_w = (W - sx) if sign_x > 0 else sx
        max_h = (H - sy) if sign_y > 0 else sy
        scale = 1.0
        if w > 0:
            scale = min(scale, max_w / w)
        if h > 0:
            scale = min(scale, max_h / h)
        scale = max(scale, 0.0)
        w *= scale
        h *= scale
        return (sx, sy), (sx + sign_x * w, sy + sign_y * h)

    def set_zoom_drag_overlay(
        self, rect: Optional[tuple[float, float, float, float]],
    ) -> None:
        """Display (or clear) a yellow rectangle over the viewport while
        the user is picking a zoom region."""
        if rect is None:
            self._zoom_drag_overlay = None
            return
        (sx, sy), (ex, ey) = self._aspect_constrain_pixels(
            (rect[0], rect[1]), (rect[2], rect[3])
        )
        self._zoom_drag_overlay = (sx, sy, ex, ey)

    def commit_zoom_rect(
        self,
        start_px: tuple[float, float],
        end_px: tuple[float, float],
    ) -> None:
        """Commit a pixel-space rectangle as the new viewport zoom
        sub-region. Tiny drags are ignored to avoid jumping into a
        single-pixel zoom by accident; degenerate inputs reset to
        full-frame.
        """
        if abs(end_px[0] - start_px[0]) < 8 or abs(end_px[1] - start_px[1]) < 8:
            return  # treat as a click, not a drag
        start_px, end_px = self._aspect_constrain_pixels(start_px, end_px)
        x0 = min(start_px[0], end_px[0])
        x1 = max(start_px[0], end_px[0])
        y0 = min(start_px[1], end_px[1])
        y1 = max(start_px[1], end_px[1])
        u0 = float(np.clip(x0 / max(self.width, 1), 0.0, 1.0))
        u1 = float(np.clip(x1 / max(self.width, 1), 0.0, 1.0))
        v0 = float(np.clip(y0 / max(self.height, 1), 0.0, 1.0))
        v1 = float(np.clip(y1 / max(self.height, 1), 0.0, 1.0))
        # GLFW pixel y goes top→bottom; UBO zoom uv expects 0=top to
        # match the existing pinhole's `ndc.y = -ndc.y` convention.
        self.zoom_rect = [u0, v0, u1, v1]
        self._material_version += 1   # force accumulation reset

    def reset_zoom_rect(self) -> None:
        self.zoom_rect = [0.0, 0.0, 1.0, 1.0]
        self._material_version += 1

    def apply_dome_light_texture(self, env_index: int, path: str) -> bool:
        """Swap the HDR texture for the DomeLight at ``env_index`` and
        update both the env library slot and the source USD prim
        attribute. Mirrors `apply_camera_lens_file`'s pattern.

        Returns True on success.
        """
        from pathlib import Path as _Path
        from skinny.environment import make_environment_from_path
        p = _Path(path)
        if not p.is_file():
            print(f"[skinny] HDR load failed: {p} not found", flush=True)
            return False
        try:
            env = make_environment_from_path(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[skinny] HDR load failed: {exc}", flush=True)
            return False

        # Branch on the active lighting authority, NOT on whether an
        # environment already exists. A dome authored via `add_light` starts
        # with no `texture:file`, so `_usd_scene.environment` is None even
        # though the authored scene owns the authority — keying on
        # `environment is not None` would (wrongly) route the edit to the
        # fallback default-lights library, which the authored authority never
        # reads, leaving the dome dark until a full stage resync.
        authored_scene = (
            self._is_usd_active()
            and not self.uses_default_lights
            and self._usd_scene is not None
        )
        if authored_scene:
            from skinny.scene import LightEnvHDR
            cur = self._usd_scene.environment
            if cur is None:
                # No prior environment (freshly added dome): construct one so
                # the authority-selected environment is non-null and
                # contributes. Fold the dome prim's color×intensity×exposure
                # into a scalar intensity, matching `_extract_dome_light`.
                intensity = 1.0
                prim = self._find_dome_light_prim(env_index)
                if prim is not None:
                    try:
                        from pxr import UsdLux
                        from skinny.usd_loader import _light_color_radiance
                        rad = _light_color_radiance(UsdLux.LightAPI(prim))
                        intensity = float(np.dot(
                            rad, np.array([0.2126, 0.7152, 0.0722], np.float32)))
                    except Exception:
                        pass
                self._usd_scene.environment = LightEnvHDR(
                    name=env.name, data=env.data, intensity=intensity)
            else:
                cur.name = env.name
                cur.data = env.data
        else:
            if 0 <= env_index < len(self.environments):
                self.environments[env_index] = env
            else:
                self.environments.append(env)
                env_index = len(self.environments) - 1
            self.env_index = env_index
        # Invalidate the env-upload cache so the texture genuinely
        # re-uploads. `_ensure_env_uploaded` keys on (env_index, furnace)
        # and short-circuits on a hit — replacing the slot at the same
        # index would otherwise leave the previous HDR on the GPU.
        self._last_env_index = (-1, -1)

        # Mirror the change onto the source USD DomeLight prim so the
        # scene-graph view stays in sync with the actual env.
        prim = self._find_dome_light_prim(env_index)
        if prim is not None:
            try:
                from pxr import Sdf, UsdLux
                dome = UsdLux.DomeLight(prim)
                dome.CreateTextureFileAttr().Set(Sdf.AssetPath(str(p)))
                dome.CreateTextureFormatAttr().Set("latlong")
            except Exception:
                pass

        # Upload the new HDR straight away. `_ensure_env_uploaded` reads
        # from `self.scene.environment`, which is rebuilt on the next
        # `update()` tick — for this in-flight swap we have the bytes in
        # hand, so upload them directly to avoid a one-frame lag.
        try:
            self.env_image.upload_sync(env.data)
            self._last_env_index = object()
        except Exception as exc:  # noqa: BLE001
            print(f"[skinny] env upload failed for {p.name}: {exc}")
        # Bump the scene-graph version so the dock repopulates with the
        # new texture path on its next tick.
        self._scene_graph_version = getattr(self, "_scene_graph_version", 0) + 1
        self._material_version += 1
        return True

    def _find_dome_light_prim(self, env_index: int):
        """Return the ``env_index``-th UsdLuxDomeLight prim across the
        active USD stages (loaded model stage first, then the synthesised
        default stage). Returns None if not found.
        """
        try:
            from pxr import UsdLux
        except Exception:
            return None
        idx = 0
        stages = (
            (self._default_light_stage,)
            if self.uses_default_lights
            else (self._usd_stage,)
        )
        for stage in stages:
            if stage is None:
                continue
            for prim in stage.Traverse():
                if not prim.IsActive() or prim.IsAbstract():
                    continue
                if prim.IsA(UsdLux.DomeLight):
                    if idx == env_index:
                        return prim
                    idx += 1
        return None

    def apply_camera_lens_file(self, path: str) -> bool:
        """Load a `.usda` lens definition and attach it to the active
        camera, replacing any current lens stack. The file may be a
        bare lens prim (a top-level `Xform` with `skinny:lens:*` child
        prims) or any prim path; we walk the children until we find
        the first child carrying a `skinny:lens:role` attribute.
        Returns True on success.
        """
        from pathlib import Path
        try:
            from pxr import Usd
            from skinny.usd_loader import _extract_lens_system
        except Exception as exc:
            print(f"[skinny] lens load failed (USD unavailable): {exc}", flush=True)
            return False
        p = Path(path)
        if not p.is_file():
            print(f"[skinny] lens load failed: {p} not found", flush=True)
            return False
        try:
            stage = Usd.Stage.Open(str(p))
        except Exception as exc:
            print(f"[skinny] lens load failed to open stage: {exc}", flush=True)
            return False
        # Walk every prim and use the first one whose children include
        # at least one lens-element child.
        ls = None
        for prim in stage.Traverse():
            if not prim.IsActive() or prim.IsAbstract():
                continue
            ls = _extract_lens_system(prim, Usd.TimeCode.Default())
            if ls is not None:
                break
        if ls is None:
            print(f"[skinny] lens load: no skinny:lens:* prims found in {p}", flush=True)
            return False
        self.orbit_camera.lens = ls
        self.free_camera.lens = ls
        self._material_version += 1
        self._refresh_camera_node()
        print(
            f"[skinny] lens loaded: {p.name} ({len(ls.elements)} elements)",
            flush=True,
        )
        return True

    def _focus_plane_state(self) -> tuple[bool, np.ndarray, np.ndarray]:
        """Return (enabled, origin, normal) for the focal-plane visualiser.

        Origin = camera_position + forward · focus_distance.
        Normal = forward (so the plane faces the camera and ray-plane
        intersection is well-defined). Disabled state still returns
        valid arrays so the UBO layout stays fixed.
        """
        cam = self.camera
        if hasattr(cam, "forward") and callable(cam.forward):
            fwd = np.asarray(cam.forward(), dtype=np.float32)
        else:
            tgt = np.asarray(cam.target, dtype=np.float32)
            pos = np.asarray(cam.position, dtype=np.float32)
            fwd = tgt - pos
        n = float(np.linalg.norm(fwd))
        if n < 1e-9:
            fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            fwd = (fwd / n).astype(np.float32)

        focus = float(getattr(cam, "focus_distance", 0.0))
        if focus <= 1e-3:
            focus = float(getattr(cam, "distance", 5.0))
        origin = (np.asarray(cam.position, dtype=np.float32) + fwd * focus).astype(np.float32)
        enabled = bool(getattr(self, "show_focus_overlay", False))
        return enabled, origin, fwd

    def _upload_gizmo_segments(self, segments: list) -> None:
        n = min(len(segments), self.gizmo_segment_capacity)
        data = bytearray()
        for i in range(self.gizmo_segment_capacity):
            if i < n:
                s = segments[i]
                data += struct.pack(
                    "ff ff fff f",
                    float(s.ax), float(s.ay),
                    float(s.bx), float(s.by),
                    float(s.r), float(s.g), float(s.b),
                    float(s.width),
                )
            else:
                data += b"\x00" * self.gizmo_segment_stride
        self.gizmo_segments_buffer.upload_sync(bytes(data))
        self._num_gizmo_segments = n

    def _ensure_default_scene_graph(self) -> None:
        """Build a minimal SceneGraphNode tree off the in-memory default-light
        stage so the editor shows ``/Skinny/DefaultLight`` even before any USD
        scene is loaded. Idempotent — bails if a real graph already exists.
        """
        if self._scene_graph is not None:
            return
        if self._default_light_stage is None:
            return
        if not self.scene.lights_dir:
            return
        try:
            from skinny.scene_graph import build_scene_graph
            self._scene_graph = build_scene_graph(
                self._default_light_stage, self.scene,
            )
            self._refresh_camera_node()
        except Exception as exc:
            print(f"[skinny] default scene graph build failed: {exc}")

    def _refresh_camera_node(self) -> None:
        """(Re)attach the synthetic ``/Skinny/MainCamera`` node so the UI sees
        the active camera with current values."""
        if self._scene_graph is None:
            return
        from skinny.scene_graph import inject_renderer_camera
        inject_renderer_camera(self._scene_graph, self.camera, self.camera_mode)
        # In-place mutation of the existing tree — bump the version so the Scene
        # Graph dock repopulates (its change gate is version + live-tree id; a
        # reshape of the same tree object changes neither id nor content hash).
        # reset_camera / toggle_camera_mode / apply_camera_lens_file rely on this.
        self._scene_graph_version = getattr(self, "_scene_graph_version", 0) + 1

    def _inject_default_lights_into_scene_graph(self) -> None:
        """Project or remove the fallback light pair in the scene graph."""
        if self._scene_graph is None or self._default_light_stage is None:
            return
        # Refresh the default dome prim from current env state before we
        # mirror it into the scene graph.
        self._sync_default_dome_prim()
        from skinny.scene_graph import inject_default_lights
        inject_default_lights(
            self._scene_graph,
            self._default_light_stage,
            enabled=self.uses_default_lights,
        )
        self._last_projected_default_lights = self.uses_default_lights
        # Bump the version so the Scene Graph dock repopulates its tree — the
        # graph object is mutated in place, so an `id()` comparison alone
        # wouldn't trigger a redraw.
        self._scene_graph_version = getattr(self, "_scene_graph_version", 0) + 1

    def apply_camera_param(self, key: str, value: object) -> None:
        """Mutate a single parameter on the active camera.

        Logs every write so the scene-graph window's slider/checkbox
        edits are observable on the console (`[skinny] camera.<key> =
        <value>`). Filtered to one line per call to keep the noise
        bounded.

        Keys:
          - ``fov`` (degrees)
          - ``near`` / ``far`` (world units)
          - ``fstop`` / ``focus_distance`` (DOF — inert until DOF pass lands)
          - ``focal_length_mm`` / ``vertical_aperture_mm`` (USD camera units;
            converted to vertical FOV: ``fov = 2·atan(0.5·va / fl)`` deg)
          - orbit only: ``distance``, ``yaw``, ``pitch``,
            ``target_x`` / ``target_y`` / ``target_z``
          - free only: ``yaw``, ``pitch``,
            ``position_x`` / ``position_y`` / ``position_z``
        """
        cam = self.camera
        v = float(value) if not isinstance(value, bool) else float(value)

        if key == "fov":
            cam.fov = float(np.clip(v, 1.0, 170.0))
        elif key == "near":
            cam.near = max(1e-4, v)
        elif key == "far":
            cam.far = max(cam.near + 1e-3, v)
        elif key == "fstop":
            cam.fstop = max(0.0, v)
        elif key == "focus_distance":
            cam.focus_distance = max(0.0, v)
        elif key == "lens_enabled":
            if cam.lens is not None:
                cam.lens.enabled = bool(value)
        elif key == "focal_length_mm":
            va = 24.0  # default vertical aperture if not set elsewhere
            cam.fov = float(np.degrees(2.0 * np.arctan(0.5 * va / max(v, 1e-3))))
        elif key == "vertical_aperture_mm":
            # Treat current fov as fixed by previous focal length; only used
            # when the UI surfaces both. Re-derive fov assuming 50mm if no
            # focal length is tracked.
            fl = 50.0
            cam.fov = float(np.degrees(2.0 * np.arctan(0.5 * v / max(fl, 1e-3))))
        elif key == "yaw":
            cam.yaw = v
        elif key == "pitch":
            cam.pitch = float(np.clip(v, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))
        elif self.camera_mode == "orbit":
            if key == "distance":
                cam.set_distance(v)
            elif key in ("target_x", "target_y", "target_z"):
                axis = "xyz".index(key[-1])
                cam.target[axis] = v
        else:  # free
            if key in ("position_x", "position_y", "position_z"):
                axis = "xyz".index(key[-1])
                cam.position[axis] = v

        self._material_version += 1
        try:
            print(f"[skinny] camera.{key} = {value!r}", flush=True)
        except Exception:
            pass
        # Hard reset the accumulation when the camera *model* toggles,
        # so the previous frames' pinhole / lens samples don't bleed
        # through the running mean while the state-hash detection
        # catches up.
        if key == "lens_enabled":
            self.accum_frame = 0
            if hasattr(self, "light_splat_buffer"):
                try:
                    self.light_splat_buffer.fill_zero_sync()
                except Exception:
                    pass
        # Note: do *not* bump _scene_graph_version here — the property
        # widgets are bound to the live SceneGraphProperty objects, so
        # rebuilding the tree mid-drag would destroy the slider the
        # user is interacting with. Structural refreshes happen via
        # _refresh_camera_node only.

    def _walk_apply_enabled(self, node, enabled: bool, flags: dict) -> None:
        scene = self._usd_scene
        ref = node.renderer_ref
        if ref is not None:
            if ref.kind == "instance" and 0 <= ref.index < len(scene.instances):
                scene.instances[ref.index].enabled = enabled
                flags["instance"] = True
            elif ref.kind == "light_dir" and 0 <= ref.index < len(scene.lights_dir):
                scene.lights_dir[ref.index].enabled = enabled
                flags["light_dir"] = True
            elif ref.kind == "light_sphere" and 0 <= ref.index < len(scene.lights_sphere):
                scene.lights_sphere[ref.index].enabled = enabled
                flags["light_sphere"] = True
            elif ref.kind == "camera" and scene.camera_override is not None:
                scene.camera_override.enabled = enabled
        for child in node.children:
            self._walk_apply_enabled(child, enabled, flags)

    def _update_texture_pool_descriptors(self) -> None:
        """Push the current TexturePool slots into binding 14 (bindless
        textures) for every descriptor set. PARTIALLY_BOUND lets unfilled
        slots stay invalid — the shader gates reads behind a sentinel idx.
        """
        # Metal has no Vulkan descriptor sets (`descriptor_sets is None`); the
        # bindless pool is bound by name straight from `texture_pool` at dispatch
        # (see `_build_pipeline_for_current_graphs`), so this Vulkan descriptor-
        # write is a no-op. The pipeline-build path already skips it, but
        # `_upload_flat_materials` calls it unconditionally on every material
        # upload — without this guard the Metal leg crashes there with
        # `TypeError: 'NoneType' object is not iterable`.
        if self.is_metal:
            return
        filled = self.texture_pool.filled_slots()
        if not filled:
            return
        writes: list = []
        for ds in self.descriptor_sets:
            for slot_idx, sampled in filled:
                info = vk.VkDescriptorImageInfo(
                    sampler=sampled.sampler,
                    imageView=sampled.view,
                    imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                )
                writes.append(
                    vk.VkWriteDescriptorSet(
                        dstSet=ds,
                        dstBinding=14,
                        dstArrayElement=slot_idx,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        pImageInfo=[info],
                    )
                )
        if writes:
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _sync_volume_grid(self, scene) -> None:
        """Upload the scene's density grid (nanovdb-volume-rendering) once per
        scene load — NOT per frame.

        Normalizes the grid to [0, 1] by dividing by its value max (skipped when
        already 1.0 — e.g. the wdas cloud; mandatory for bunny_cloud's 2.792),
        converts to float16, and uploads in the GPU's (depth, height, width) =
        (nz, ny, nx) order. The previous texture is destroyed and replaced; a
        scene with no grid restores the 1×1×1 zero fallback. Also caches the
        world→uvw rows + value max for `pack_flat_material`'s σ folds, and the
        grid identity key for `_current_state_hash` (accumulation reset).
        Call BEFORE the scene's material upload so volume materials pack against
        the fresh grid state.
        """
        vg = getattr(scene, "volume_grid", None) if scene is not None else None
        key = (str(vg.asset_path), float(vg.value_max)) if vg is not None else None
        if key == self._volume_grid_key:
            return
        self._volume_grid_key = key
        if self.volume_density_image is not None:
            self.volume_density_image.destroy()
        if vg is None:
            self.volume_density_image = self._gpu.SampledImage3D(self.ctx, 1, 1, 1)
            self.volume_density_image.upload_sync(np.zeros((1, 1, 1), np.float16))
            self._volume_world_to_uvw = None
            self._volume_value_max = 1.0
        else:
            nx, ny, nz = vg.dims
            dens = vg.density
            vmax = float(vg.value_max) if vg.value_max > 0.0 else 1.0
            if vmax != 1.0:
                dens = dens / np.float32(vmax)
            # Reader layout is (nx, ny, nz); GPU upload wants (D, H, W) = (nz, ny, nx).
            voxels = np.ascontiguousarray(
                np.transpose(dens, (2, 1, 0)).astype(np.float16))
            self.volume_density_image = self._gpu.SampledImage3D(self.ctx, nx, ny, nz)
            self.volume_density_image.upload_sync(voxels)
            self._volume_world_to_uvw = np.asarray(vg.world_to_uvw, np.float32)
            self._volume_value_max = vmax
            print(
                f"[skinny] volume grid uploaded: {nx}x{ny}x{nz} R16F "
                f"({nx * ny * nz * 2 / 1e6:.0f} MB), value_max {vmax:.4g}"
            )
        self._rebind_volume_descriptor()

    def _rebind_volume_descriptor(self) -> None:
        """Vulkan-only: rewrite binding 26 (volumeDensity) after the 3D texture
        was replaced. Metal binds the live `volume_density_image` by name at
        every dispatch (`_build_metal_binds`), so a swap is picked up
        automatically there; before the Vulkan descriptor sets exist the initial
        `_create_descriptors` writes the binding against the current image."""
        if self.is_metal or self.descriptor_sets is None:
            return
        info = vk.VkDescriptorImageInfo(
            sampler=self.volume_density_image.sampler,
            imageView=self.volume_density_image.view,
            imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        writes = [
            vk.VkWriteDescriptorSet(
                dstSet=ds,
                dstBinding=26,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[info],
            )
            for ds in self.descriptor_sets
        ]
        vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    # Shared-buffer element strides (bytes). Must match the packers in mesh.py
    # and the grow math in _ensure_mesh_buffer_capacity.
    _SLAB_V_STRIDE = 32  # vertex
    _SLAB_T_STRIDE = 12  # triangle (3 × uint32 index)
    _SLAB_N_STRIDE = 32  # BVH node

    def _mesh_fingerprint(self, mesh: Mesh) -> str:
        """Stable content fingerprint of a baked mesh (cached on the object).
        Detects when a resident slab's geometry changed (e.g. a displacement
        rebake) so it is re-uploaded rather than silently reused."""
        fp = getattr(mesh, "_slab_fp", None)
        if fp is None:
            import hashlib
            h = hashlib.blake2b(digest_size=16)
            h.update(mesh.vertex_bytes)
            h.update(mesh.index_bytes)
            h.update(mesh.bvh_bytes)
            fp = h.hexdigest()
            try:
                mesh._slab_fp = fp  # dataclass is mutable; cache to skip rehash
            except Exception:  # noqa: BLE001 — frozen/slotted variant: just recompute
                pass
        return fp

    def _upload_meshes_suballocated(
        self, instances: "list"
    ) -> list[tuple[int, int, int]]:
        """Slab-allocate every instance mesh in the shared vertex/index/BVH
        buffers and return one (node_offset, triangle_offset, vertex_offset)
        per instance — the triple each shader Instance record holds.

        Replaces the whole-scene concatenation: each instance keeps a slab with
        **stable** offsets keyed by (prim_path, sub-index), so an add/remove
        touches only the changed mesh. Departed meshes are freed to the
        free-list (reused by later adds); resident, content-unchanged meshes are
        not re-uploaded; content changes (rebake) free + re-allocate. Buffers
        grow in place (existing slab offsets preserved), so TLAS BLAS offsets
        stay valid without reindexing. Mode-independent — both execution modes
        read these buffers.
        """
        alloc = self._slab_alloc
        from skinny.slab_allocator import Counts

        # Stable per-instance key: (prim_path, sub-index among same path). The
        # loader yields a prim's sub-meshes in a deterministic order, so the
        # key survives a resync for unchanged geometry. Anonymous instances fall
        # back to their position (re-uploaded if the set shifts — no regression
        # over the old concat-everything path).
        keys: list = []
        seen_path: dict[str, int] = {}
        for i, inst in enumerate(instances):
            pp = getattr(inst, "prim_path", None) or getattr(inst, "name", None)
            if pp:
                sub = seen_path.get(pp, 0)
                seen_path[pp] = sub + 1
                keys.append((pp, sub))
            else:
                keys.append(("__anon__", i))

        # Content-change check: a resident key whose mesh bytes changed must be
        # freed so it re-allocates (and re-uploads) fresh.
        fps = [self._mesh_fingerprint(inst.mesh) for inst in instances]
        for key, fp in zip(keys, fps):
            if alloc.is_resident(key) and self._slab_content_fp.get(key) != fp:
                alloc.free(key)
                self._slab_content_fp.pop(key, None)

        # Drop meshes that left the scene (free-list reclaims their regions).
        for gone in alloc.retain_only(keys):
            self._slab_content_fp.pop(gone, None)

        # Allocate slabs (resident → reuse; else free-list best-fit or append).
        offsets: list[tuple[int, int, int]] = []
        plan: list[tuple] = []  # (off, is_new, vbytes, ibytes, bbytes)
        for key, fp, inst in zip(keys, fps, instances):
            mesh = inst.mesh
            res = alloc.allocate(key, Counts(mesh.num_vertices,
                                             mesh.num_triangles,
                                             mesh.num_nodes))
            off = res.offsets  # (v, t, n)
            offsets.append((off.n, off.t, off.v))  # shader order: node, tri, vert
            self._slab_content_fp[key] = fp
            plan.append((off, res.is_new,
                         mesh.vertex_bytes, mesh.index_bytes, mesh.bvh_bytes))

        # Grow the backing buffers if the high-water mark outran them. Growth
        # preserves slab offsets, so a grown buffer is repopulated from every
        # resident slab; non-grown buffers keep their data and take only new
        # slabs. _ensure_mesh_buffer_capacity rebinds descriptors + re-seeds the
        # dummy mesh, which the slab writes below then overwrite at real offsets.
        hw = alloc.high_water
        v_size_before = self.vertex_buffer.size
        i_size_before = self.index_buffer.size
        b_size_before = self.bvh_buffer.size
        self._ensure_mesh_buffer_capacity(hw.v, hw.t, hw.n)
        grew_v = self.vertex_buffer.size != v_size_before
        grew_i = self.index_buffer.size != i_size_before
        grew_n = self.bvh_buffer.size != b_size_before

        for off, is_new, vb, ib, bb in plan:
            if grew_v or is_new:
                self.vertex_buffer.upload_range(vb, off.v * self._SLAB_V_STRIDE)
            if grew_i or is_new:
                self.index_buffer.upload_range(ib, off.t * self._SLAB_T_STRIDE)
            if grew_n or is_new:
                self.bvh_buffer.upload_range(bb, off.n * self._SLAB_N_STRIDE)
        return offsets

    def compact_geometry(self) -> int:
        """Defragment the shared geometry buffers: pack live slabs contiguously,
        move their GPU bytes, and rewrite every referencing TLAS BLAS offset so
        the rendered image is unchanged. Opt-in (fragmentation is otherwise
        tolerated via the free-list). Returns the number of slabs relocated.

        Safe to skip; safe to call repeatedly (a no-op when already packed)."""
        scene = self._usd_scene
        if scene is None or not scene.instances:
            return 0
        moves = self._slab_alloc.compact()
        if not moves:
            return 0
        # Re-upload every live slab's bytes at its new offset. Build a key→mesh
        # map from the current instances (the live set == current instances).
        seen_path: dict[str, int] = {}
        key_mesh: dict = {}
        for inst in scene.instances:
            pp = getattr(inst, "prim_path", None) or getattr(inst, "name", None)
            if pp:
                sub = seen_path.get(pp, 0)
                seen_path[pp] = sub + 1
                key_mesh[(pp, sub)] = inst.mesh
        for key, _old, new, _counts in moves:
            mesh = key_mesh.get(key)
            if mesh is None:
                continue
            self.vertex_buffer.upload_range(mesh.vertex_bytes, new.v * self._SLAB_V_STRIDE)
            self.index_buffer.upload_range(mesh.index_bytes, new.t * self._SLAB_T_STRIDE)
            self.bvh_buffer.upload_range(mesh.bvh_bytes, new.n * self._SLAB_N_STRIDE)
        # Rewrite TLAS instance records with the post-compaction offsets.
        self._upload_usd_scene()
        return len(moves)

    def _upload_instances(
        self,
        transforms: list[np.ndarray],
        *,
        material_ids: list[int] | None = None,
        blas_offsets: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Pack and upload TLAS instance records.

        Each entry packs (worldFromLocal, localFromWorld, blasNodeOffset,
        blasIndexOffset, blasVertexOffset, materialId) — see
        mesh_head.slang::Instance. blas_offsets is one (node_off, tri_off,
        vert_off) per instance, defaulting to (0, 0, 0) for the legacy
        single-BLAS case.
        """
        if material_ids is None:
            material_ids = [0] * len(transforms)
        if blas_offsets is None:
            blas_offsets = [(0, 0, 0)] * len(transforms)
        if len(material_ids) != len(transforms):
            raise ValueError("material_ids must have one entry per transform")
        if len(blas_offsets) != len(transforms):
            raise ValueError("blas_offsets must have one entry per transform")
        if len(transforms) > self.instance_capacity:
            new_cap = max(len(transforms), self.instance_capacity * 2)
            self.instance_capacity = new_cap
            self.instance_buffer.destroy()
            self.instance_buffer = self._gpu.StorageBuffer(
                self.ctx, self.instance_capacity * INSTANCE_STRIDE + 256
            )
            self._rebind_scene_descriptors()

        data = bytearray()
        for xform, mat_id, (n_off, t_off, v_off) in zip(
            transforms, material_ids, blas_offsets
        ):
            world_from_local = np.asarray(xform, dtype=np.float32)
            if world_from_local.shape != (4, 4):
                raise ValueError(
                    f"instance transform must be 4x4, got {world_from_local.shape}"
                )
            local_from_world = np.linalg.inv(world_from_local).astype(np.float32)
            data += world_from_local.tobytes()
            data += local_from_world.tobytes()
            # blasNodeOffset, blasIndexOffset (in triangles),
            # blasVertexOffset, materialId
            data += struct.pack("4I", int(n_off), int(t_off), int(v_off), int(mat_id))

        self.instance_buffer.upload_sync(bytes(data))
        self._num_instances = len(transforms)

    def _upload_detail_maps(self, src_idx: int | None) -> None:
        """Upload this source's detail maps (or blanks when absent).

        Caches decoded bytes for both the normal and displacement maps so the
        CPU bake (which needs normal-into-vertex baking) doesn't have to re-read
        TIF/TGA files on every rebake. SDF mode (src_idx=None) restores blanks.
        """
        if src_idx is None:
            self.normal_image.upload_sync(blank_normal_bytes())
            self.roughness_image.upload_sync(blank_roughness_bytes())
            self.displacement_image.upload_sync(blank_displacement_bytes())
            self._detail_available = (False, False, False)
            return

        from concurrent.futures import ThreadPoolExecutor

        src = self._mesh_sources[src_idx]
        to_load: dict[str, Path | None] = {}
        if src_idx not in self._normal_cache:
            to_load["nrm"] = src.normal_map
        if src_idx not in self._displacement_cache:
            to_load["dsp"] = src.displacement_map
        to_load["rgh"] = src.roughness_map

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {k: pool.submit(load_texture_bytes, p) for k, p in to_load.items()}
            loaded = {k: f.result() for k, f in futures.items()}

        if "nrm" in loaded:
            self._normal_cache[src_idx] = loaded["nrm"]
        nrm = self._normal_cache.get(src_idx)
        rgh = loaded["rgh"]
        if "dsp" in loaded:
            self._displacement_cache[src_idx] = loaded["dsp"]
        dsp = self._displacement_cache.get(src_idx)

        self.normal_image.upload_sync(nrm if nrm is not None else blank_normal_bytes())
        self.roughness_image.upload_sync(rgh if rgh is not None else blank_roughness_bytes())
        self.displacement_image.upload_sync(dsp if dsp is not None else blank_displacement_bytes())

        self._detail_available = (nrm is not None, rgh is not None, dsp is not None)

    def _current_scale_world(self) -> float:
        """Convert the mm-valued displacement slider into world units."""
        mm_per_unit = max(float(self.mm_per_unit), 1e-6)
        return float(self.displacement_scale_mm) / mm_per_unit

    def _bake_and_upload(self, src_idx: int) -> None:
        """Displace + (optionally) bake normals + rebuild BVH, with disk cache."""
        src = self._mesh_sources[src_idx]
        disp_bytes = self._displacement_cache.get(src_idx)
        nrm_bytes  = self._normal_cache.get(src_idx)
        scale_world = self._current_scale_world()
        nrm_strength = float(self.normal_map_strength)

        cache_key = make_cache_key(
            src.content_hash, disp_bytes, DETAIL_TEX_RES, scale_world,
            nrm_bytes, DETAIL_TEX_RES, nrm_strength,
        )
        mesh = lookup_cached_mesh(self._mesh_cache_index, cache_key, src)
        if mesh is None:
            t0 = time.monotonic()
            mesh = bake_mesh(
                src,
                displacement_bytes=disp_bytes,
                displacement_res=DETAIL_TEX_RES,
                displacement_scale_world=scale_world,
                normal_bytes=nrm_bytes,
                normal_res=DETAIL_TEX_RES,
                normal_map_strength=nrm_strength,
            )
            dt = time.monotonic() - t0
            print(f"[skinny] mesh bake '{src.name}' ({dt:.1f}s)")
            save_cached_mesh(self._mesh_cache_index, cache_key, mesh)

        self._upload_mesh(mesh)
        # Reset the instance buffer to a single identity-transform record
        # at offsets (0, 0, 0). Necessary when the previous active slot
        # was the USD scene (which leaves N records with non-zero BLAS
        # offsets); without this reset the shader would walk into wrong
        # buffer slices after the toggle.
        self._upload_instances(
            [np.eye(4, dtype=np.float32)],
            material_ids=[0],
        )
        self._baked_source_idx      = src_idx
        self._baked_scale_mm        = float(self.displacement_scale_mm)
        self._baked_scale_world     = scale_world
        self._baked_mm_per_unit     = float(self.mm_per_unit)
        self._baked_normals         = bool(mesh.normals_baked)
        self._baked_normal_strength = float(self.normal_map_strength)
        self._dirty_since           = None

    def _rebake_if_needed(self, now: float) -> None:
        """Decide whether the mesh buffers need rebuilding this frame.

        Rebakes immediately on model change, and after a 300 ms debounce
        on the displacement-scale slider.
        """
        if not self.models:
            if self._baked_source_idx != -1:
                self._upload_detail_maps(None)
                self._baked_source_idx = -1
            return

        self.model_index = int(np.clip(self.model_index, 0, len(self.models) - 1))

        # USD mode: meshes ship pre-baked from the loader.
        if self._usd_model_index >= 0 and self.model_index == self._usd_model_index:
            if self._usd_scene is not None and self._usd_scene.instances:
                if self._baked_source_idx != -2:
                    print("[skinny] switching to USD scene")
                    self._upload_usd_scene()
                    self._upload_detail_maps(None)
                    self._baked_source_idx = -2
                    self._usd_uploaded_count = len(self._usd_scene.instances)
            return

        src_idx = self.model_index
        if self._usd_model_index >= 0 and self.model_index > self._usd_model_index:
            src_idx = self.model_index - 1
        if not (0 <= src_idx < len(self._mesh_sources)):
            return

        # Source change: upload this source's detail maps, then force a bake.
        if src_idx != self._baked_source_idx:
            self._upload_detail_maps(src_idx)
            self._bake_and_upload(src_idx)
            return

        target_scale_world = self._current_scale_world()

        scale_changed = abs(target_scale_world - self._baked_scale_world) > 1e-9
        strength_changed = (
            self._baked_normals
            and abs(float(self.normal_map_strength) - self._baked_normal_strength) > 1e-9
        )

        if scale_changed or strength_changed:
            if self._dirty_since is None:
                self._dirty_since = now
            elif now - self._dirty_since > 0.3:
                self._bake_and_upload(src_idx)
        else:
            self._dirty_since = None

    def _ensure_tattoo_uploaded(self) -> None:
        """Re-upload the tattoo texture on selection change."""
        if self.tattoo_index == self._last_tattoo_index:
            return
        self.tattoo_index = int(np.clip(self.tattoo_index, 0, len(self.tattoos) - 1))
        self.tattoo_image.upload_sync(self.tattoos[self.tattoo_index].data)
        self._last_tattoo_index = self.tattoo_index

    def _sync_lens_buffer(self) -> None:
        """Repack lens_elements_buffer if the active camera's lens has
        changed since the last upload. Sets self._lens_* fields used by
        _pack_uniforms.

        f-stop coupling: when ``camera.fstop > 0`` the aperture-stop
        element's clear-aperture diameter is overridden to
        ``focal_length_mm / fstop`` (clamped not to exceed the authored
        design aperture, since a real iris can stop down but not open
        wider than the lens design allows).
        """
        lens = getattr(self.camera, "lens", None)
        fstop = float(getattr(self.camera, "fstop", 0.0))
        focal = float(getattr(self.camera, "focal_length_mm", 50.0))
        focus_d = float(getattr(self.camera, "focus_distance", 0.0))
        # Combined cache key: lens identity + iris-driving inputs +
        # focus distance (drives the rear-element-to-film gap below).
        sig = (
            lens.signature() if lens is not None else None,
            round(fstop, 6),
            round(focal, 6),
            round(focus_d, 6),
        )
        if sig == self._packed_lens_signature:
            return

        active = lens.active_elements if lens is not None else []
        n = min(len(active), self.lens_element_capacity)
        if n == 0:
            if self._lens_active_count != 0:
                zeros = b"\x00" * (self.lens_element_capacity * self.lens_element_stride)
                self.lens_elements_buffer.upload_sync(zeros)
            self._lens_active_count = 0
            self._lens_film_distance_world = 0.0
            self._lens_rear_z_world = 0.0
            self._lens_rear_aperture_world = 0.0
            self._lens_front_z_world = 0.0
            self._packed_lens_signature = sig
            return

        # mm → world via Scene.mm_per_unit (1 world unit = N mm).
        mm_per_unit = float(self.scene.mm_per_unit) if self.scene.mm_per_unit > 0 else 1.0
        scale = 1.0 / mm_per_unit  # world units per mm

        # Element thicknesses in world units, in PBRT order (index 0 =
        # front, index N-1 = rear).
        thicknesses_world = [float(e.thickness_mm) * scale for e in active]

        # PBRT FocusThickLens — paraxial-trace through the actual lens
        # to find the principal-plane positions, then solve for the
        # rear-element-to-film gap that images `focus_distance` onto
        # the film. This is exact for the lens design (subject to the
        # paraxial approximation), unlike the naive `F²/(s−F)`
        # thin-lens shortcut which assumes the authored
        # focalLength = effective F (often false for real designs).
        if focus_d > 0.0:
            from skinny.lens_optics import LensInterface, focus_thick_lens
            elems_for_focus = [
                LensInterface(
                    radius=float(e.radius_mm),
                    thickness=float(e.thickness_mm),
                    ior=float(e.ior),
                    half_aperture=float(e.aperture_mm) * 0.5,
                    is_stop=bool(e.is_aperture_stop),
                )
                for e in active
            ]
            va_mm_focus = float(getattr(self.camera, "vertical_aperture_mm", 24.0))
            film_diag_focus = math.sqrt(va_mm_focus * va_mm_focus
                                        + (va_mm_focus * 1.5) ** 2)
            try:
                new_rear_mm = focus_thick_lens(
                    elems_for_focus, film_diag_focus, focus_d * mm_per_unit,
                )
                thicknesses_world[-1] = new_rear_mm * scale
            except Exception:
                pass    # paraxial trace failed — keep authored back focal length

        film_distance = thicknesses_world[-1]
        front_z = sum(thicknesses_world)

        # Iris diameter (mm) implied by the user's f-stop. Stops down
        # the aperture-stop element only; PBRT additionally precomputes
        # the *exit pupil* (image of the iris seen through any lens
        # elements between it and the film) and samples within that
        # bound to keep almost every sample valid. We approximate
        # without the per-film-position bound by linearly projecting
        # the iris through the in-between elements onto the rear plane:
        #     r_rear ≈ irisHalfAp · (rearZ / irisZ)
        # This is exact for axial film points and a thin air gap; for
        # off-axis points it is conservative enough that most samples
        # survive the iris clip rather than getting averaged in as
        # zeros. Without this, large fstops vignette > 99 % of rays and
        # the image reads as black/noise instead of sharp pinhole.
        iris_diameter_mm = (focal / fstop) if fstop > 1e-6 else 0.0
        authored_rear_half = 0.5 * float(active[-1].aperture_mm) * scale
        # Locate the iris element to size the cone.
        iris_idx = next(
            (k for k, e in enumerate(active) if e.is_aperture_stop),
            None,
        )
        if iris_idx is None:
            rear_aperture_world = authored_rear_half
        else:
            iris_half_world = 0.5 * float(active[iris_idx].aperture_mm) * scale
            if iris_diameter_mm > 0.0:
                iris_half_world = min(iris_half_world, 0.5 * iris_diameter_mm * scale)
            # Distance from rear surface to iris (sum of thicknesses
            # between iris and rear inclusive of iris's own thickness).
            iris_to_rear = sum(thicknesses_world[iris_idx:])
            iris_z_abs = iris_to_rear  # |irisZ| in PBRT-speak (rearZ is at thickness[-1])
            rear_z_abs = thicknesses_world[-1]
            if iris_z_abs > 1e-9:
                projected = iris_half_world * (rear_z_abs / iris_z_abs)
                rear_aperture_world = min(authored_rear_half, projected)
            else:
                rear_aperture_world = authored_rear_half

        # Pack float4 per element: (radius, thickness, ior, halfAperture).
        # Matches PBRT-v3's LensElementInterface; the shader walks
        # rear→front decrementing a running `elementZ` by `thickness`.
        buf = bytearray(self.lens_element_capacity * self.lens_element_stride)
        for k, e in enumerate(active[:n]):
            radius_world = float(e.radius_mm) * scale
            thickness_world = thicknesses_world[k]
            aperture_mm = float(e.aperture_mm)
            if e.is_aperture_stop and iris_diameter_mm > 0.0:
                aperture_mm = min(aperture_mm, iris_diameter_mm)
            half_ap_world = 0.5 * aperture_mm * scale
            struct.pack_into(
                "ffff", buf, k * self.lens_element_stride,
                radius_world, thickness_world, float(e.ior), half_ap_world,
            )
        self.lens_elements_buffer.upload_sync(bytes(buf))

        # PBRT exit-pupil bounds — pre-compute the rear-plane rectangle
        # of valid (non-vignetting) lens samples per film radius, so
        # closing the iris doesn't shrink the rendered area to a
        # central pinhole at the cost of off-axis pixels.
        from skinny.lens_optics import LensInterface, compute_exit_pupil_bounds
        lens_in_mm = [
            LensInterface(
                radius=float(e.radius_mm),
                thickness=float(e.thickness_mm),
                ior=float(e.ior),
                half_aperture=float(e.aperture_mm) * 0.5,
                is_stop=bool(e.is_aperture_stop),
            )
            for e in active
        ]
        if iris_diameter_mm > 0.0:
            for li in lens_in_mm:
                if li.is_stop:
                    li.half_aperture = min(li.half_aperture, 0.5 * iris_diameter_mm)
                    break
        # Mirror the autofocus rear-thickness adjustment so the bounds
        # are computed against the same lens geometry the shader sees.
        # Ignored if the focus_thick_lens helper isn't available.
        if focus_d > 0.0:
            try:
                from skinny.lens_optics import focus_thick_lens
                lens_in_mm[-1].thickness = focus_thick_lens(
                    lens_in_mm,
                    math.sqrt(24.0 * 24.0 + 36.0 * 36.0),
                    focus_d * mm_per_unit,
                )
            except Exception:
                pass
        va_mm = float(getattr(self.camera, "vertical_aperture_mm", 24.0))
        film_diag_mm = math.sqrt(va_mm * va_mm + (va_mm * 1.5) ** 2)
        n_bins = 16
        bounds_mm = compute_exit_pupil_bounds(
            lens_in_mm, film_diag_mm, num_bounds=n_bins, samples_per_bound=64,
        )
        bounds_world = bounds_mm * float(scale)
        upload = np.zeros((self.lens_pupil_capacity, 4), dtype=np.float32)
        upload[:n_bins] = bounds_world
        self.lens_pupil_buffer.upload_sync(upload.tobytes())

        self._lens_active_count = n
        self._lens_film_distance_world = float(film_distance)
        self._lens_rear_z_world = float(film_distance)   # |LensRearZ()| in PBRT-speak
        self._lens_rear_aperture_world = float(rear_aperture_world)
        self._lens_front_z_world = float(front_z)
        self._lens_iris_z_world = 0.0   # legacy; no longer consumed
        self._lens_film_diag_world = float(film_diag_mm) * float(scale)
        self._lens_num_pupil_bounds = int(n_bins)
        self._packed_lens_signature = sig

        # Throttle the diagnostic — slider drags re-sign the lens every
        # frame, and a print per frame on the main thread compounds with
        # GLFW/Tk message-pump pressure during a window resize.
        now = time.perf_counter()
        last = getattr(self, "_last_lens_print_t", 0.0)
        if now - last > 0.5:
            iris_mm = (focal / fstop) if fstop > 1e-6 else float("inf")
            print(
                f"[skinny] lens repack: N={n} "
                f"filmDist={film_distance:.3f}wu "
                f"frontZ={front_z:.3f}wu "
                f"rearAp={rear_aperture_world:.3f}wu "
                f"fstop={fstop:.2f} iris={iris_mm:.2f}mm "
                f"mm_per_unit={mm_per_unit:.2f}",
                flush=True,
            )
            self._last_lens_print_t = now

    # ── Scene-sampling seam: resolve preset indices → plugin instances ──

    def _active_proposals(self) -> list:
        """Active directional proposals for the current preset index."""
        from skinny.sampling import parse_proposals
        n = len(self._PROPOSAL_PRESETS)
        idx = max(0, min(int(self.proposal_preset_index), n - 1))
        token = self._PROPOSAL_PRESETS[idx][1]
        # Spectral path tracing supports the analytic BSDF/environment subset.
        # Proposal selection is persisted and runtime-switchable, so strip the
        # unsupported neural bit here as well as refusing it on explicit CLI
        # startup. BDPT/SPPM/MLT do not consume the proposal seam, so they also
        # resolve to native BSDF sampling. An empty subset safely falls back to
        # the BSDF baseline.
        if self._spectral:
            token = _spectral_analytic_proposal_token(
                token,
                allow_environment=(int(self.integrator_index) == 0),
            )
        return parse_proposals(token)

    def _active_integrator_index(self) -> int:
        """The integrator actually dispatched. Spectral now spans PATH (0),
        BDPT (1), SPPM (2), and MLT (3) — like RGB. Under the megakernel,
        main_pass.slang
        under SKINNY_SPECTRAL dispatches SpectralBDPTIntegrator when
        fc.integratorType == INTEGRATOR_BDPT on a flat first hit, else
        SpectralPathTracer (so SPPM, which has no megakernel path, falls to
        PATH there — but resolve_execution_mode sends sppm → wavefront, so a
        spectral SPPM session runs the wavefront photon+gather passes). Under the
        wavefront execution mode all four integrators dispatch spectrally. So we
        report the selected integrator verbatim, matching RGB — this drives
        fc.integratorType and the config matrix. integrator_index is persisted and
        runtime-switchable on the interactive front-ends."""
        return int(self.integrator_index)

    def _neural_active(self) -> bool:
        """True when the neural proposal (bit2) is selected AND the backend can
        run it (wavefront only). Drives lazy pass build + the scene-set bind."""
        if self.effective_execution_mode_index != EXECUTION_WAVEFRONT:
            return False
        return any(getattr(p, "mask_bit", 0) == 0x4 for p in self._active_proposals())

    def _warn_neural_megakernel_once(self) -> None:
        """One-shot notice that the neural proposal is unsupported on the
        megakernel (reported, not silently dropped — the bit is stripped + the
        mixture falls back to its analytic subset)."""
        if not self._neural_warned:
            self._neural_warned = True
            print("[skinny] neural proposal is wavefront-only; ignored on the "
                  "megakernel backend (falling back to the analytic proposals)")

    def _effective_neural_config(self):
        """The active NeuralBuildConfig with precision degraded to what the device
        supports — graceful fp32 fallback (study change
        neural-precision-size-study). An fp16 mode whose required capability
        (16-bit storage and/or shaderFloat16) is absent runs at fp32 instead of
        failing; the downgrade is logged once. The size dims are unaffected."""
        from dataclasses import replace

        from skinny.sampling.neural_weights import NeuralPrecision

        cfg = self._neural_config
        prec = cfg.precision
        need_storage = prec.needs_device_fp16_storage
        need_compute = prec.needs_device_fp16_compute
        supported = (
            (not need_storage or getattr(self.ctx, "supports_fp16_storage", False))
            and (not need_compute or getattr(self.ctx, "supports_fp16_compute", False))
        )
        if supported:
            return cfg
        if not self._fp16_fallback_warned:
            self._fp16_fallback_warned = True
            print(f"[fp16] precision '{prec.value}' unsupported on this device "
                  f"(storage={getattr(self.ctx, 'supports_fp16_storage', False)}, "
                  f"compute={getattr(self.ctx, 'supports_fp16_compute', False)}); "
                  "falling back to fp32")
        return replace(cfg, precision=NeuralPrecision.FP32)

    def _sync_neural_weights(self) -> None:
        """Upload the active neural weights to bindings 33/34/35. Loads the
        resolved per-scene NFW1 file when ``_neural_weights_path`` is set;
        otherwise keeps the dummy (zero) net seeded at init — the 1a bring-up
        network. Idempotent: re-uploads only when the path changes. NFW1 is fp32
        on disk; the bytes are cast to the effective precision's storage dtype at
        upload, and the load asserts the baked arch matches the built dims (study
        change neural-precision-size-study)."""
        path = self._neural_weights_path
        if path == self._neural_weights_loaded:
            return
        if path is None:
            self._neural_weights_loaded = None
            return
        cfg = self._effective_neural_config()
        try:
            from skinny.sampling.neural_weights import load_neural_weights
            nw = load_neural_weights(path, expect=cfg.arch, expect_mlp_in=cfg.mlp_in,
                                     expect_chart=cfg.chart)
        except Exception as exc:  # noqa: BLE001 - fall back to the dummy net
            print(f"[skinny] neural weights load failed for {path}: {exc}; "
                  "keeping the dummy net")
            self._neural_weights_path = None
            self._neural_weights_loaded = None
            return
        self.neural_weights_buffer.upload_sync(nw.weight_bytes_for(cfg.precision))
        self.neural_biases_buffer.upload_sync(nw.bias_bytes_for(cfg.precision))
        self.neural_layers_buffer.upload_sync(nw.header_bytes)
        self._neural_weights_loaded = path

    # ── Online neural training (Stage 2, change neural-online-training) ──

    @property
    def online_training_active(self) -> bool:
        return self._online_training

    def _current_neural_weights(self):
        """The ``NeuralWeights`` currently driving inference (the loaded per-scene
        net, or the dummy bring-up net) — the warm-start for online training."""
        from skinny.sampling.neural_weights import (
            load_neural_weights,
            make_dummy_weights,
        )
        cfg = self._effective_neural_config()
        path = self._neural_weights_path
        if path is not None:
            try:
                return load_neural_weights(path, expect=cfg.arch,
                                           expect_mlp_in=cfg.mlp_in,
                                           expect_chart=cfg.chart)
            except Exception:  # noqa: BLE001 — fall back to the dummy net
                pass
        # make_dummy_weights takes the build config itself; cfg.arch is the
        # (layers, bins, hidden, cond) tuple used only as a load-time expectation.
        return make_dummy_weights(cfg)

    def enable_online_training(self, *, handoff: str | None = None,
                               trainer_backend: str | None = None,
                               train_precision: str | None = None,
                               replay=None, trainer=None,
                               capacity: int = 1_000_000, **publisher_kwargs):
        """Start the online training loop (change neural-online-training).

        A recency-weighted ``ReplayBuffer`` feeds a warm-started ``NeuralTrainer``
        whose new weights a ``NeuralWeightPublisher`` double-buffers into the
        render buffers at the frame boundary, bumping ``networkVersion``.
        ``handoff`` overrides the renderer's ``--neural-handoff`` publisher
        (``file`` | ``interop``); ``trainer_backend`` overrides the
        ``--neural-trainer`` compute backend (``cpu`` | ``cuda`` | ``mlx`` |
        ``auto``) and ``train_precision`` the ``--train-precision`` optimizer
        precision (change neural-trainer-backends). Returns the publisher.
        """
        from skinny.sampling.neural_handoff import make_publisher
        from skinny.sampling.neural_replay import ReplayBuffer
        from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig

        cfg = self._effective_neural_config()
        init = self._current_neural_weights()
        self._neural_replay = (replay if replay is not None
                               else ReplayBuffer(capacity=capacity))
        # Inference precision defaults to match the chosen training precision
        # (post-training quantization); both feed the trainer's TrainerConfig.
        backend_kind = trainer_backend or self._neural_trainer_kind
        precision = train_precision or self._train_precision
        kind = handoff or self._neural_handoff_kind
        # Condition the online trainer on the SAME scene-AABB position
        # normalisation the shader's neuralCondition uses (fc.sceneBoundsMin/
        # Extent). Without this the trainer's TrainerConfig.bounds defaults to
        # None -> _bounds() falls back to (0,1) raw world position, so the net is
        # trained on a different position scale than inference queries it with
        # (e.g. Cornell world coords ~[-0.7,4] -> train p~[-2.4,7] vs infer
        # p~[-1,1]); the position channel becomes off-distribution at inference
        # and the learned proposal collapses to a generic, scene-agnostic (near
        # no-op) distribution. Same AABB as _pack_uniforms (inference) and
        # dump_path_records (the offline training header).
        train_bounds = tuple(self._neural_scene_bounds())
        self._neural_trainer = (trainer if trainer is not None
                                else NeuralTrainer(TrainerConfig(
                                    arch=cfg, backend=backend_kind,
                                    train_precision=precision, handoff=kind,
                                    bounds=train_bounds),
                                    initial=init))
        # The interop publisher writes weights+biases straight into the GPU-shared
        # binding-33/34 buffers; hand it those buffers plus the active storage
        # precision. `make_publisher` resolves the mechanism per backend (change
        # metal-neural-interop, design D2): on Vulkan the CUDA publisher imports
        # the exported memory and signals the exported timeline semaphore (task
        # 5.2); on Metal the UMA publisher writes the shared-storage buffers in
        # place at the frame-boundary swap (the semaphore kwarg is None here and
        # dropped by the factory — no exported semaphores on Metal).
        if kind == "interop":
            publisher_kwargs.setdefault(
                "weights_buffer", getattr(self, "neural_weights_buffer", None))
            publisher_kwargs.setdefault(
                "biases_buffer", getattr(self, "neural_biases_buffer", None))
            publisher_kwargs.setdefault(
                "timeline_semaphore", getattr(self, "neural_timeline_semaphore", None))
            publisher_kwargs.setdefault("precision", cfg.precision)
        self._neural_publisher = make_publisher(
            kind, initial=init, expect_arch=cfg.arch, **publisher_kwargs)
        self._online_training = True
        # Reset the STOPPED-summary guard for this session and register an atexit
        # fallback so a run summary still prints if the process exits while
        # training is active (change online-training-observability). The guard is
        # shared with disable_online_training so it fires at most once.
        self._train_summary_printed = False
        if not self._atexit_registered:
            import atexit
            atexit.register(self._print_train_summary)
            self._atexit_registered = True
        # Resolve the record source. Wavefront-native emission removes the
        # megakernel record dispatch (the 2 s-TDR / ~400 s-compile seam on
        # NVIDIA/Windows): the normal wavefront render fills bindings 36/37 and
        # the drain just reads them. When the source is the megakernel, the
        # wavefront render stays byte-identical (recordMode 0) and the drain
        # dispatches `mainImageRecord` as before.
        self._wf_record_active = (self._resolve_record_source() == "wavefront")
        # Arm the wavefront drain on either backend (change metal-record-drain):
        # Vulkan rebinds descriptor 36; Metal allocates the header+records drain
        # target and routes it through the bind-by-name dict. Either way the
        # records build flavor lands via the pass rebuild key (wf_record), and
        # the accumulation restarts cleanly under the new pipeline.
        if self._wf_record_active and (
                self.descriptor_sets is not None or self.is_metal):
            self._ensure_wf_record_drain()
        self._last_state_hash = None
        self._start_trainer_thread()
        return self._neural_publisher

    def disable_online_training(self) -> None:
        self._print_train_summary()
        self._online_training = False
        self._wf_record_active = False
        self._last_state_hash = None  # records-off pass rebuild → restart accum
        self._stop_trainer_thread()

    def _start_trainer_thread(self) -> None:
        """Spin up the daemon trainer thread (change online-training-trigger).

        The thread loops ``online_train_and_publish`` + a short sleep on the host
        replay buffer; the render thread keeps draining + swapping. The publisher
        double-buffer is the only trainer→render handoff, so no extra weight
        locking is needed. Idempotent — a no-op if a thread is already running."""
        if self._trainer_thread is not None and self._trainer_thread.is_alive():
            return
        self._trainer_stop = threading.Event()
        rng = np.random.default_rng()

        def _loop(stop=self._trainer_stop):
            while not stop.is_set():
                if self._online_training and self._neural_trainer is not None:
                    try:
                        self.online_train_and_publish(rng)
                    except Exception as exc:  # noqa: BLE001 — keep the loop alive
                        print(f"[neural] trainer cycle failed: {exc}")
                stop.wait(self._trainer_cadence_s)

        self._trainer_thread = threading.Thread(
            target=_loop, name="skinny-neural-trainer", daemon=True)
        self._trainer_thread.start()

    def _stop_trainer_thread(self) -> None:
        """Signal the trainer thread to stop and join it (change
        online-training-trigger). Safe to call when no thread is running."""
        if self._trainer_stop is not None:
            self._trainer_stop.set()
        thread = self._trainer_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._trainer_thread = None
        self._trainer_stop = None

    def online_train_execution_supported(self) -> bool:
        """Whether this session's execution mode permits online training at all
        (wavefront-only; the record drain + neural pre-pass are wavefront-only).

        The execution axis is fixed for the session, so a False here is a
        *permanent* refusal. Distinct from :meth:`can_online_train`, which also
        requires a neural proposal to be *currently* active — that half is
        runtime-selectable (e.g. the skinny-gui Proposals combobox), so a False
        there is transient, not permanent."""
        return self.effective_execution_mode_index == EXECUTION_WAVEFRONT

    def can_online_train(self) -> tuple[bool, str]:
        """Prerequisite check for online training (change online-training-trigger).

        True only when the execution mode is wavefront (the record drain + neural
        pre-pass are wavefront-only) AND a neural proposal is active in the
        mixture. Returns ``(ok, reason)``; ``reason`` names the missing
        prerequisite when ``ok`` is False, and is empty when ``ok`` is True."""
        if self.effective_execution_mode_index != EXECUTION_WAVEFRONT:
            return (False, "online training requires --execution-mode wavefront")
        # BDPT never consumes the neural proposal (bdpt.slang does not import the
        # proposal seam) and has no wavefront record source, so training under it
        # would guide nothing and the drain would fall back to the megakernel
        # records (absent on Metal). Refuse so the GUI runtime path (where the
        # integrator is selected after startup) reports it cleanly rather than
        # crashing mid-frame (change bdpt-neural-incompatibility). The CLI
        # front-ends additionally hard-exit on the flag combo at startup.
        if self.integrator_index == 1:  # 1 = bdpt
            return (False, "online training requires --integrator path — BDPT "
                           "does not consume the neural proposal")
        if not self._neural_active():
            return (False, "online training requires a neural proposal in the "
                           "mixture (--proposals …,neural)")
        return (True, "")

    # ── Configuration matrix + lifecycle logging (change
    #    online-training-observability) ──────────────────────────────────────

    def _collect_config_rows(self) -> list:
        """Assemble the configuration-matrix rows from the resolved render state.
        Requested values come from the CLI/env/persisted selections; resolved
        values are what the renderer actually runs; the online-training row's
        status names the missing prerequisite when it is not approved."""
        from skinny import config_report as cr

        rows: list = []
        # backend: requested (front-end-set, default "auto") vs resolved device.
        req_b = self._requested_backend or "auto"
        res_b = "metal" if self.is_metal else "vulkan"
        rows.append(cr.ConfigRow("backend", req_b, res_b, cr.ON))

        # execution mode: the Metal/bdpt capability gate can pin the resolved
        # value away from the requested one.
        req_e = str(self._requested_execution_mode)
        res_e = ("wavefront"
                 if self.effective_execution_mode_index == EXECUTION_WAVEFRONT
                 else "megakernel")
        est = (f"{cr.ON} (pinned from {req_e})"
               if self.execution_mode_fallback_active else cr.ON)
        rows.append(cr.ConfigRow("execution-mode", req_e, res_e, est))

        # integrator. Spectral now spans path (0), bdpt (1), sppm (2), and mlt
        # (3) — like RGB (see _active_integrator_index). SPPM/MLT have no
        # megakernel path, but resolve_execution_mode sends them to wavefront, so
        # the resolved integrator equals the requested one. Reads only fields —
        # `_collect_config_rows` runs against a plain namespace in the
        # observability tests.
        req_integ = self.integrator_modes[self.integrator_index].lower()
        rows.append(cr.ConfigRow("integrator", req_integ, req_integ, cr.ON))

        # proposals: requested = the selected preset token; resolved = what the
        # renderer actually samples. Spectral retains the analytic BSDF/env
        # subset and strips neural (see _active_proposals), so the resolved
        # column reports a pin only when the requested token actually changes.
        idx = max(0, min(int(self.proposal_preset_index),
                         len(self._PROPOSAL_PRESETS) - 1))
        prop_tok = self._PROPOSAL_PRESETS[idx][1]
        resolved_prop_tok = (
            _spectral_analytic_proposal_token(
                prop_tok,
                allow_environment=(int(self.integrator_index) == 0),
            )
            if self._spectral
            else prop_tok
        )
        if resolved_prop_tok != prop_tok:
            # Fold the requested token into the STATUS (not just the requested
            # column): matrix_signature dedups re-prints on resolved+status only,
            # so a runtime switch between pinned presets must also change status.
            rows.append(cr.ConfigRow("proposals", prop_tok, resolved_prop_tok,
                                     f"{cr.ON} (spectral pin; requested {prop_tok})"))
        else:
            prop_status = "neural ACTIVE" if self._neural_active() else cr.ON
            rows.append(cr.ConfigRow(
                "proposals", prop_tok, resolved_prop_tok, prop_status,
            ))

        # The training stack rows only matter when online training is requested;
        # mark them n/a otherwise so the matrix reads cleanly with the loop off.
        ot_req = self._online_training_requested
        requested_ot = (bool(ot_req) if ot_req is not None
                        else bool(self._online_training))
        train_status = cr.ON if requested_ot else cr.NA

        tr = self._neural_trainer
        res_trainer = tr.backend_name if tr is not None else self._neural_trainer_kind
        rows.append(cr.ConfigRow("neural-trainer", self._neural_trainer_kind,
                                 res_trainer, train_status))

        res_handoff = self._neural_handoff_kind
        if res_handoff == "interop":
            res_handoff = "interop(UMA)" if self.is_metal else "interop(CUDA)"
        rows.append(cr.ConfigRow("neural-handoff", self._neural_handoff_kind,
                                 res_handoff, train_status))

        res_prec = (tr.config.train_precision if tr is not None
                    else self._train_precision)
        rows.append(cr.ConfigRow("train-precision", self._train_precision,
                                 res_prec, train_status))

        # online-training: the payoff row — OFF / REFUSED / WAITING / APPROVED.
        if not requested_ot:
            ot_status = cr.OFF
        elif self.effective_execution_mode_index != EXECUTION_WAVEFRONT:
            ot_status = cr.refused("requires --execution-mode wavefront")
        elif self.integrator_index == 1:  # bdpt ignores the neural proposal
            ot_status = cr.refused("requires --integrator path (bdpt ignores neural)")
        elif not self._neural_active():
            ot_status = cr.waiting("select a neural proposal")
        else:
            ot_status = cr.APPROVED
        rows.append(cr.ConfigRow("online-training",
                                 "on" if requested_ot else "off", "—", ot_status))
        return rows

    def _emit_config_matrix(self, reason: str = "") -> None:
        """Print the configuration matrix, but only when its status signature
        differs from the last print (startup + every runtime flip). Cheap enough
        to call once per frame from update(): it builds a few rows and dedups."""
        from skinny import config_report as cr

        rows = self._collect_config_rows()
        sig = cr.matrix_signature(rows)
        if sig == self._last_config_sig:
            return
        self._last_config_sig = sig
        print(cr.build_config_matrix(rows))

    def _print_train_summary(self) -> None:
        """Emit the one-shot ``online training STOPPED`` run summary. Guarded so
        the explicit-disable and atexit paths never double-print; a no-op if the
        trainer never actually ran a cycle."""
        if self._train_summary_printed:
            return
        tr = self._neural_trainer
        if tr is None:
            return
        s = tr.summary()
        if s["cycles"] == 0:
            return  # armed but never trained — nothing worth summarizing
        self._train_summary_printed = True
        fl = f"{s['final_loss']:.4f}" if s["final_loss"] is not None else "n/a"
        print(f"[neural] online training STOPPED: ran {s['duration_s']:.1f} s, "
              f"{s['cycles']} cycles, {s['steps']} steps, {s['samples']} samples, "
              f"final loss={fl}, backend={s['backend']}")

    def online_training_status(self) -> dict:
        """Cheap, lock-free snapshot of the online-training state for the GUI to
        poll each frame (change online-training-observability). Plain attribute
        reads — never blocks the render or trainer thread."""
        tr = self._neural_trainer
        active = bool(self._online_training and tr is not None
                      and getattr(tr, "_started_t", None) is not None)
        return {
            "armed": bool(self._online_training),
            "active": active,
            "last_loss": getattr(tr, "last_loss", None) if tr is not None else None,
            "cycles": getattr(tr, "_trained_cycles", 0) if tr is not None else 0,
            "backend": tr.backend_name if tr is not None else None,
        }

    def online_training_tick(self) -> int:
        """Per-frame driver the render loop calls once per frame (change
        online-training-trigger). Drains GPU path records into the replay buffer
        on the render thread (cheap; must touch the GPU/queue here) so the
        background trainer thread has fresh data. No-op — returns 0 — when online
        training is off. The actual per-cycle training runs on the trainer thread;
        the frame-end swap in render()/render_headless() promotes new weights."""
        if not self._online_training:
            return 0
        # The drain reads GPU records, so the scene must be built; skip the frame
        # while it isn't (USD streams in async, a rebake transiently nulls these).
        # descriptor_sets is Vulkan-only; the Metal drain binds by name
        # (change metal-record-drain).
        if self._scene_bindings is None or (
                self.descriptor_sets is None and not self.is_metal):
            return 0
        return self.online_drain()

    def online_drain(self, **kw) -> int:
        """Drain one frame of GPU records into the replay buffer (the live feed,
        task 1.2). HARDWARE SEAM: the record entry is a megakernel that
        device-losts under the 2 s Windows TDR, so this runs on the NVIDIA box
        (task 7.3), not the Mac/wavefront suite. No-op when online training is
        off."""
        if not self._online_training or self._neural_replay is None:
            return 0
        return self.drain_path_records_to_replay(self._neural_replay, **kw)

    def online_train_and_publish(self, rng=None):
        """Run one warm-started training cycle on the replay buffer and stage the
        result for the next frame-end swap. Returns the staged version, or
        ``None`` when there is nothing to train on / online training is off."""
        if not self._online_training or self._neural_trainer is None:
            return None
        if self._neural_replay is None or len(self._neural_replay) == 0:
            return None
        new_w = self._neural_trainer.train_cycle(self._neural_replay, rng)
        return self._neural_publisher.publish(new_w)

    def _apply_render_weights(self, nw, version: int) -> None:
        """Upload ``nw`` to the inference buffers (33/34/35) and stamp ``version``
        everywhere the per-sample density key is read — ``FrameConstants`` and the
        neural pass push-constant — so a swapped sample is always evaluated
        against the version that drew it (task 4.3)."""
        cfg = self._effective_neural_config()
        self.neural_weights_buffer.upload_sync(nw.weight_bytes_for(cfg.precision))
        self.neural_biases_buffer.upload_sync(nw.bias_bytes_for(cfg.precision))
        self.neural_layers_buffer.upload_sync(nw.header_bytes)
        self._neural_network_version = int(version)
        if self._neural_pass is not None:
            self._neural_pass.network_version = int(version)

    def _online_frame_end_swap(self) -> bool:
        """Frame-end double-buffer swap point (task 4.2): promote any pending
        weights to the render buffers and increment ``networkVersion``. Render
        weights stay frozen during a frame; the swap happens only here, at the
        boundary, so the per-sample version stamp is consistent within a frame and
        unbiasedness holds across an async swap (task 4.3). Returns True on swap.
        """
        if not self._online_training or self._neural_publisher is None:
            return False
        if not self._neural_publisher.swap():
            return False
        nw, version = self._neural_publisher.acquire_for_render()
        if nw is not None:
            self._apply_render_weights(nw, version)
        else:
            self._neural_network_version = int(version)
            if self._neural_pass is not None:
                self._neural_pass.network_version = int(version)
        return True

    def _active_reuse(self):
        """Active reuse plugin for the current reuse index."""
        from skinny.sampling import parse_reuse
        n = len(self._REUSE_TOKENS)
        idx = max(0, min(int(self.reuse_index), n - 1))
        return parse_reuse(self._REUSE_TOKENS[idx])

    def proposal_preset_from_token(self, token: str) -> int:
        """Map a `--proposals` CLI token (e.g. ``'bsdf,env'``) to a preset index.

        Matches by token *set* so order doesn't matter; unknown combinations
        fall back to preset 0 (``bsdf``)."""
        want = frozenset(t.strip() for t in str(token).split(",") if t.strip())
        for i, (_, tok) in enumerate(self._PROPOSAL_PRESETS):
            if frozenset(tok.split(",")) == want:
                return i
        return 0

    def _neural_scene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World AABB ``(min, extent)`` for the neural condition's position
        normalisation. The per-frame ``Scene`` snapshot has no instances for USD
        scenes (their geometry streams straight to the GPU and is not mirrored
        into ``Scene.instances``), so fall back to the streamed ``_usd_scene``
        instances — otherwise the condition normalises against the degenerate
        ``(0,1)`` default and the proposal sees un-normalised positions. Used by
        BOTH ``_pack_uniforms`` (inference) and ``dump_path_records`` (the
        training header) so train↔infer share the exact same AABB.
        """
        wb = self.scene.world_bounds() if self.scene is not None else None
        if wb is None and self._usd_scene is not None and self._usd_scene.instances:
            mins, maxs = [], []
            for inst in self._usd_scene.instances:
                wmin, wmax = inst.world_bounds()
                mins.append(wmin)
                maxs.append(wmax)
            if mins:
                wb = (np.minimum.reduce(mins).astype(np.float32),
                      np.maximum.reduce(maxs).astype(np.float32))
        if wb is None:
            return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
        bmin = np.asarray(wb[0], dtype=np.float32)
        bext = np.maximum(np.asarray(wb[1], dtype=np.float32) - bmin, 1e-6).astype(np.float32)
        return bmin, bext

    def _pack_uniforms_msl(self, layout_source=None) -> bytes:
        """Pack the `fc` uniform block to the Metal Shading Language struct layout
        (design D3). Reuses `_pack_uniforms` (the Vulkan scalar blob) verbatim and
        relocates every field to its reflected MSL offset, so the field *values*
        can never drift between backends — only their placement differs (Slang pads
        `float3` to 16 B on Metal, making the struct 592 B vs the 512 B scalar
        blob). Offsets come from the compiled module's reflection
        (`pipeline.uniform_layout`), never a hand-maintained table. Uploaded via
        `set_data` byte blobs only (design D4).

        `layout_source` overrides the default `_msl_layout_source` — the material
        preview passes its own `PreviewPipelineMetal` so it packs against that
        program's reflected `fc` layout, independent of whether the megakernel /
        wavefront layout source exists yet (wavefront-mode preview, codex #1)."""
        src = layout_source if layout_source is not None else self._msl_layout_source
        layout = src.uniform_layout
        size = src.uniform_size
        # The scalar tail must match the TARGET layout, not the session state:
        # only a `SKINNY_MLT` program's reflected `fc` has the MLT fields, so an
        # explicit non-MLT `layout_source` (e.g. the material preview's
        # `PreviewPipelineMetal`) needs the base 568 B blob even while MLT is the
        # active integrator — otherwise the tail bytes have no destination and
        # the drift guard fires (codex pre-merge review). Drive `_pack_uniforms`
        # off `"mltSigma" in layout` so the blob and the field table always agree.
        has_tail = "mltSigma" in layout
        scalar = self._pack_uniforms(mlt_tail=has_tail)
        fields = _FC_SCALAR_FIELDS_MLT if has_tail else _FC_SCALAR_FIELDS
        out = bytearray(size)
        off = 0
        for name, sz in fields:
            moff, _msz = layout[name]
            out[moff:moff + sz] = scalar[off:off + sz]
            off += sz
        # Drift guard (task 3.3): the scalar field table must cover the whole blob,
        # and the packed MSL length must equal the reflected struct size.
        assert off == len(scalar), (
            f"MSL field table covers {off}B but scalar blob is {len(scalar)}B")
        assert len(out) == size, (len(out), size)
        return bytes(out)

    def _build_metal_binds(self) -> dict:
        """Map every Metal megakernel shader global → its native SlangPy resource
        (bind-by-name, design D2). The pipeline filters to the names the compiled
        module actually references, so binding an unused name is harmless and a
        dead-stripped one is simply skipped."""
        b = {
            # Storage buffers (bindings 5-7, 12-24, 30-37).
            "meshVertices": self.vertex_buffer.buffer,
            "meshIndices": self.index_buffer.buffer,
            "bvhNodes": self.bvh_buffer.buffer,
            "instances": self.instance_buffer.buffer,
            "flatMaterials": self.flat_material_buffer.buffer,
            "materialTypes": self.material_types_buffer.buffer,
            "mtlxSkin": self.mtlx_skin_buffer.buffer,
            "sphereLights": self.sphere_lights_buffer.buffer,
            "emissiveTriangles": self.emissive_tri_buffer.buffer,
            "stdSurfaceParams": self.std_surface_buffer.buffer,
            "distantLights": self.distant_lights_buffer.buffer,
            "lightSplatBuffer": self.light_splat_buffer.buffer,
            "gizmoSegments": self.gizmo_segments_buffer.buffer,
            "lensElements": self.lens_elements_buffer.buffer,
            "exitPupilBounds": self.lens_pupil_buffer.buffer,
            "toolBuffer": self.tool_buffer.buffer,
            "envDistCdf": self.env_dist_buffer.buffer,
            "neuralWeights": self.neural_weights_buffer.buffer,
            "neuralBiases": self.neural_biases_buffer.buffer,
            "neuralLayers": self.neural_layers_buffer.buffer,
            "recordBuf": self.record_buffer.buffer,
            "recordCounter": self.record_counter.buffer,
            # Storage images (bindings 1-3).
            "outputBuffer": self._offscreen_output.texture,
            "accumBuffer": self.accum_image.texture,
            "hudMask": self.hud_overlay.texture,
            # Shared bindless-pool sampler (binding 38, design D8).
            "commonSampler": self._metal_common_sampler,
        }
        # Discrete maps: combined `Sampler2D`/`Sampler3D` is unsupported on Metal,
        # so each is a `Texture2D`/`Texture3D` + its own `SamplerState`
        # (bindings 4/8-11/26 + 39-44, design D8). `volumeDensity` is the
        # heterogeneous-medium density grid (nanovdb-volume-rendering) — always
        # bound (1×1×1 zero fallback), re-read fresh here every dispatch so a
        # per-scene texture swap needs no descriptor bookkeeping on Metal.
        for name, img in (
            ("envMap", self.env_image), ("tattooMap", self.tattoo_image),
            ("normalMap", self.normal_image), ("roughnessMap", self.roughness_image),
            ("displacementMap", self.displacement_image),
            ("volumeDensity", self.volume_density_image),
        ):
            b[name] = img.texture
            b[name + "Sampler"] = img.sampler
        # Per-graph MaterialX param SSBOs (globals `graphParams_<sanitized>`,
        # binding GRAPH_BINDING_BASE). Bind-by-name like every other resource;
        # one combined buffer shared by every graph (the Slang global is
        # `graphParamsCombined`, change combine-graph-param-buffers). Contents
        # are scalar-packed in `_upload_graph_param_buffers` — `Load<T>` reads
        # the same scalar layout on Metal and SPIR-V, so no MSL relocation.
        combined = getattr(self, "_graph_params_combined", None)
        if combined is not None:
            b["graphParamsCombined"] = combined.buffer
        # Spectral upsample tables (bindings 45/46/47) — only the spectral
        # megakernel variant references these names; Metal silently skips
        # unbound names, but only add when the buffers exist.
        if self._spectral and self._spectral_scale_buffer is not None:
            b["spectralScale"] = self._spectral_scale_buffer.buffer
            b["spectralData"] = self._spectral_data_buffer.buffer
            b["spectralD65"] = self._spectral_d65_buffer.buffer
            b["spectralMetals"] = self._spectral_metals_buffer.buffer
            if self._spectral_emitters_buffer is not None:
                b["spectralEmitters"] = self._spectral_emitters_buffer.buffer
            if self._spectral_light_spd_buffer is not None:
                b["spectralLightSpd"] = self._spectral_light_spd_buffer.buffer
            if self._spectral_mat_emission_buffer is not None:
                b["spectralMatEmission"] = self._spectral_mat_emission_buffer.buffer
        return b

    def _render_megakernel_metal(self) -> None:
        """Bind every megakernel resource and dispatch one frame on the Metal
        pipeline (design D2/D4). Writes the display image into `_offscreen_output`."""
        mtlx_bytes = self._pack_mtlx_skin_array_msl()
        if mtlx_bytes:
            self.mtlx_skin_buffer.upload_sync(mtlx_bytes)
        binds = self._build_metal_binds()
        bindless = (
            "flatMaterialTextures",
            [(s.texture if s is not None else None) for s in self.texture_pool._slots],
        )
        # Row-band tiling bounds each committed command buffer under the macOS GPU
        # watchdog (change metal-megakernel-watchdog-tiling). The tileOriginY u32 is
        # patched at its reflected MSL offset per band.
        tile_off, _ = self._msl_layout_source.uniform_layout["tileOriginY"]
        self.pipeline.dispatch(
            self.width, self.height,
            uniform_blob=self._pack_uniforms_msl(),
            binds=binds, bindless=bindless,
            bands=self._metal_megakernel_bands(),
            tile_origin_offset=tile_off,
        )

    def _metal_megakernel_bands(self) -> int:
        """Row-band count for the Metal megakernel dispatch so no single command
        buffer exceeds the GPU watchdog budget. Integrator-aware and resolution-
        scaled; `SKINNY_METAL_MEGAKERNEL_BANDS` overrides for tuning. Vulkan never
        calls this (it dispatches the full frame in one buffer)."""
        import os
        override = os.environ.get("SKINNY_METAL_MEGAKERNEL_BANDS")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        budget = _METAL_MEGAKERNEL_BAND_PIXELS.get(
            int(self.integrator_index), _METAL_MEGAKERNEL_BAND_PIXELS_DEFAULT)
        pixels = int(self.width) * int(self.height)
        bands = (pixels + budget - 1) // budget
        return max(1, min(int(self.height), bands))

    def _has_heavy_nonflat(self) -> bool:
        """True when the scene has a non-terminal non-flat material (VOLUME /
        PYTHON). Under wavefront BDPT/SPPM their non-flat first-hit path fallback
        runs a full multi-bounce `PathTracer.estimateRadiance` in the eye kernel,
        so the Metal eye submit is bounded per tile to stay within the GPU
        watchdog (change wavefront-nonflat-tiled-fallback). The terminal types
        (SUBSURFACE / SKIN) evaluate one vertex, so they need no bounding."""
        types = getattr(self, "_material_types", None) or ()
        return any(
            int(t) in (MATERIAL_TYPE_PYTHON, MATERIAL_TYPE_VOLUME) for t in types)

    def _render_wavefront_metal(self, staged) -> None:
        """Dispatch one staged wavefront frame on the Metal backend (change
        metal-wavefront-parity phases 3/4): the shared `record_path_loop` /
        `record_bdpt_loop` stage order, encoded into one frame command encoder
        by the pass (path or bdpt — same `dispatch_frame` surface). The resolve
        stage writes both the accumulation image and the display image, exactly
        like the Vulkan wavefront path."""
        mtlx_bytes = self._pack_mtlx_skin_array_msl()
        if mtlx_bytes:
            self.mtlx_skin_buffer.upload_sync(mtlx_bytes)
        # Bound the heavy per-tile eye submit for BDPT when the scene has a
        # non-terminal non-flat material (change wavefront-nonflat-tiled-fallback).
        staged.bound_heavy_eye = self._has_heavy_nonflat()
        staged.dispatch_frame(
            binds=self._build_metal_binds(),
            uniform_blob=self._pack_uniforms_msl(),
            bindless_textures=[
                (s.texture if s is not None else None)
                for s in self.texture_pool._slots
            ],
        )

    def _render_scene_metal(self) -> None:
        """Render one Metal frame into `_offscreen_output` through the active
        execution mode: the staged wavefront tracer when selected (and
        buildable) — bdpt when the bidirectional integrator is active (phase
        4), else the path tracer — falling back to the megakernel."""
        if self.effective_execution_mode_index == EXECUTION_WAVEFRONT:
            if self.integrator_index == 3:  # MLT → staged wavefront mlt
                mlt = self._ensure_wavefront_mlt_pass_metal()
                if mlt is not None:
                    self._render_wavefront_mlt_metal(mlt)
                    return
                # unbuildable → fall back to the path tracer below
            if self.integrator_index == 2:  # SPPM → staged wavefront sppm
                sppm = self._ensure_wavefront_sppm_pass()
                if sppm is not None:
                    self._render_wavefront_sppm_metal(sppm)
                    return
                # unbuildable → fall back to the path tracer below
            if self.integrator_index == 1:  # BDPT → staged wavefront bdpt
                staged = self._ensure_wavefront_bdpt_pass()
            else:
                staged = self._ensure_wavefront_path_pass()
            if staged is not None:
                self._render_wavefront_metal(staged)
                return
        self._render_megakernel_metal()

    def _render_wavefront_sppm_metal(self, sppm) -> None:
        """Dispatch one staged SPPM pass on the Metal backend (change
        photon-mapping-sppm) — record_sppm_loop over one MetalFrameEncoder,
        mirroring _render_wavefront_metal but with the per-frame photon count +
        first-frame flag the SPPM pass needs."""
        mtlx_bytes = self._pack_mtlx_skin_array_msl()
        if mtlx_bytes:
            self.mtlx_skin_buffer.upload_sync(mtlx_bytes)
        # Bound the heavy per-tile eye submit when the scene has a non-terminal
        # non-flat material (change wavefront-nonflat-tiled-fallback).
        sppm.bound_heavy_eye = self._has_heavy_nonflat()
        sppm.dispatch_frame(
            binds=self._build_metal_binds(),
            uniform_blob=self._pack_uniforms_msl(),
            bindless_textures=[
                (s.texture if s is not None else None)
                for s in self.texture_pool._slots
            ],
            photons=self._sppm_photons_emitted,
            first_frame=(self.accum_frame == 0),
            photon_batch=self._sppm_metal_photon_batch,
        )

    def _run_wavefront_mlt_bootstrap_metal(self, mlt) -> None:
        """Synchronous MLT (re)seed at an accumulation reset on Metal (design
        D3) — the Metal sibling of `_run_wavefront_mlt_bootstrap`. Identical
        host round-trip; the two submits are the pass's own
        `dispatch_bootstrap` / `dispatch_init` encoders (each ends in a
        `MetalFrameEncoder.submit`, which drains), so the weight readback
        between them sees finished GPU work without an explicit wait."""
        from skinny.mlt_bootstrap import resample_chain_seeds

        self._mlt_seed = self._next_mlt_seed()
        mlt.b = 0.0
        mlt.seeded = False
        binds = self._build_metal_binds()
        textures = [(s.texture if s is not None else None)
                    for s in self.texture_pool._slots]
        batch = self._mlt_metal_chain_batch()
        # Packed AFTER _mlt_seed is set: the bootstrap/init kernels read
        # fc.mltSeed, and on Metal the blob is a per-dispatch argument (no
        # persistent uniform buffer to re-upload as on Vulkan).
        mlt.dispatch_bootstrap(
            binds=binds, uniform_blob=self._pack_uniforms_msl(),
            bindless_textures=textures, chain_batch=batch)
        weights = mlt.read_bootstrap_weights()
        b, seeds = resample_chain_seeds(weights, mlt.num_chains, self._mlt_seed)
        mlt.upload_chain_seeds(seeds)
        mlt.dispatch_init(
            binds=binds, uniform_blob=self._pack_uniforms_msl(),
            bindless_textures=textures, chain_batch=batch)
        mlt.b = b
        mlt.seeded = True

    def _mlt_metal_chain_batch(self) -> int:
        """Per-dispatch chain breadth for the Metal MLT phases (design D7,
        codex pre-merge review). 0 on Vulkan (no watchdog) and off by env; on
        Metal, `SKINNY_MLT_METAL_CHAIN_BATCH` overrides the default so a large
        `--chains` stays under the macOS GPU watchdog."""
        if not self.is_metal:
            return 0
        import os
        return int(os.environ.get("SKINNY_MLT_METAL_CHAIN_BATCH",
                                  str(_MLT_METAL_CHAIN_BATCH_DEFAULT)))

    def _render_wavefront_mlt_metal(self, mlt) -> None:
        """Dispatch one staged MLT frame on the Metal backend (change
        mlt-integrator, task 5.6) — the `record_mlt_frame` sequence over one
        MetalFrameEncoder, preceded at an accumulation reset by the synchronous
        bootstrap round-trip. The frame's uniform blob is packed AFTER the
        bootstrap so the resolve reads the freshly measured `fc.mltB`."""
        mtlx_bytes = self._pack_mtlx_skin_array_msl()
        if mtlx_bytes:
            self.mtlx_skin_buffer.upload_sync(mtlx_bytes)
        if self.accum_frame == 0 or not mlt.seeded:
            self._run_wavefront_mlt_bootstrap_metal(mlt)
        mlt.dispatch_frame(
            binds=self._build_metal_binds(),
            uniform_blob=self._pack_uniforms_msl(),
            bindless_textures=[
                (s.texture if s is not None else None)
                for s in self.texture_pool._slots
            ],
            iterations=self._mlt_iterations_per_frame(),
            chain_batch=self._mlt_metal_chain_batch(),
        )

    def _render_headless_metal(self) -> bytes:
        """Metal headless render → raw RGBA8 bytes (the structural-parity test
        path). Mirrors the Vulkan `render_headless` minus the Vulkan
        command-buffer/descriptor machinery — resources bind at dispatch."""
        self._render_scene_metal()
        arr = self._offscreen_output.read_rgba()  # (H, W, 4)
        from skinny.metal_compute import _rgba_f32_to_rgba8
        return _rgba_f32_to_rgba8(arr).tobytes()

    def _render_windowed_metal(self) -> None:
        """Windowed Metal frame: dispatch the megakernel into `_offscreen_output`
        (rgba8) and blit it onto the acquired slang-rhi surface image, then
        present. The blit converts the surface's native format (typically
        `bgra8_unorm` on macOS) and scales to the window extent. A `None` image
        means the surface had nothing ready this tick — skip the frame. The device
        is drained each frame so the present fence signals (no per-field cursor
        writes anywhere on the Metal path, design D4)."""
        surface = getattr(self.ctx, "surface", None)
        if surface is None:
            return
        self.poll_pick_result()
        self._render_scene_metal()
        image = surface.acquire_next_image()
        if image is None:
            return
        enc = self.ctx.device.create_command_encoder()
        enc.blit(image, self._offscreen_output.texture)
        self.ctx.device.submit_command_buffer(enc.finish())
        surface.present()
        self.ctx.device.wait_for_idle()

    def _pack_uniforms(self, *, mlt_tail: bool | None = None) -> bytes:
        """Assemble the `fc` scalar blob. ``mlt_tail`` overrides whether the
        `#if defined(SKINNY_MLT)` tail is appended; ``None`` (the Vulkan direct
        path, which has no reflected layout to key off) defers to
        ``_mlt_uniform_tail_active()``. ``_pack_uniforms_msl`` passes an explicit
        bool derived from the target layout so the blob matches it exactly."""
        self._sync_lens_buffer()
        aspect = self.width / self.height
        view_fwd = self.camera.view_matrix()
        proj_fwd = self.camera.projection_matrix(aspect)
        view_inv = np.linalg.inv(view_fwd)
        proj_inv = np.linalg.inv(proj_fwd)

        data = bytearray()
        # FrameConstants: viewInverse (mat4), projInverse (mat4),
        # view (mat4), proj (mat4), position (vec3), fov, frameIndex, ...
        data += view_inv.astype(np.float32).tobytes()       # 64 bytes
        data += proj_inv.astype(np.float32).tobytes()        # 64 bytes
        data += view_fwd.astype(np.float32).tobytes()        # 64 bytes
        data += proj_fwd.astype(np.float32).tobytes()        # 64 bytes
        data += self.camera.position.tobytes()               # 12 bytes
        data += struct.pack("f", self.camera.fov)            # 4 bytes
        data += struct.pack("I", self.frame_index)           # 4 bytes
        data += struct.pack("I", self.accum_frame)           # 4 bytes
        data += struct.pack("f", self.time_elapsed)          # 4 bytes
        data += struct.pack("II", self.width, self.height)   # 8 bytes
        # Active distant-light count (bounds every NEE loop in the
        # integrators). Replaces the legacy useDirectLight boolean —
        # `direct_light_index` is folded into `_upload_distant_lights`
        # (uploads zero records when Off).
        data += struct.pack("I", int(self._num_distant_lights))  # 4 bytes
        use_mesh = 1
        data += struct.pack("I", use_mesh)                   # 4 bytes
        # Pigment density: today's tattoo slider, surfaced via the scene's
        # active material so a future per-instance override falls out for free.
        primary_material = self.scene.primary_material()
        pigment_density = (
            primary_material.pigment_density if primary_material is not None
            else float(self.tattoo_density)
        )
        data += struct.pack("f", float(pigment_density))    # 4 bytes
        # E-2: scatterMode is no longer in FrameConstants — the per-material
        # entry in materialTypes[i] carries scatter flags in bits 8-9.
        from skinny.scene import environment_contribution_intensity
        env_intensity = environment_contribution_intensity(
            self.scene.environment,
        )
        data += struct.pack("f", float(env_intensity))      # 4 bytes
        data += struct.pack("I", 1 if self.scene.furnace_mode else 0)  # 4 bytes
        data += struct.pack("f", float(self.scene.mm_per_unit))        # 4 bytes
        # Detail-map controls — single `detailFlags` bitfield + two strengths.
        # Bit 0: master enable (mirror of the UI toggle, AND-ed with the
        #        per-map availability bits below so a missing map is always
        #        treated as off, even when the user toggles on).
        # Bit 1: normal map available
        # Bit 2: roughness map available
        # Bit 3: displacement map available
        # Bit 4: normal map already baked into vertex normals (shader skips
        #        its own normal-map sample for mesh hits so we don't
        #        double-apply the same perturbation).
        master = 1 if self.detail_maps_index == 0 else 0
        nrm_ok, rgh_ok, dsp_ok = self._detail_available
        flags = (
            master
            | ((1 if nrm_ok else 0) << 1)
            | ((1 if rgh_ok else 0) << 2)
            | ((1 if dsp_ok else 0) << 3)
            | ((1 if self._baked_normals else 0) << 4)
        )
        data += struct.pack("I", flags)                              # 4 bytes
        data += struct.pack("f", float(self.normal_map_strength))    # 4 bytes
        data += struct.pack("f", float(self.displacement_scale_mm))  # 4 bytes
        # TLAS instance count consumed by mesh_head.slang::marchHeadMesh.
        # When useMesh==0 the shader skips marchHeadMesh entirely; we still
        # write the count for completeness so the field is always defined.
        data += struct.pack("I", int(self._num_instances))           # 4 bytes
        # Active sphere-light count (bounds the shader's loop).
        data += struct.pack("I", int(self._num_sphere_lights))        # 4 bytes
        # Active emissive-triangle count (bounds the shader's NEE loop).
        data += struct.pack("I", int(self._num_emissive_tris))        # 4 bytes
        # Integrator selector — 0 = path, 1 = BDPT. main_pass.slang dispatches
        # on this; PathTracer codepath is byte-identical for value 0. Under
        # --spectral this is pinned to 0 (path) — the only integrator the spectral
        # megakernel runs — so fc.integratorType never disagrees with the render.
        data += struct.pack("I", int(self._active_integrator_index()))  # 4 bytes
        # Active gizmo-segment count (bounds main_pass's overlay loop).
        data += struct.pack("I", int(self._num_gizmo_segments))       # 4 bytes
        # Thick-lens parameters. numLensElements > 0 swaps the pinhole ray
        # generator for cameras/thick_lens.slang::generateLensRay. All
        # distances are in world units (Scene.mm_per_unit applied).
        data += struct.pack("I", int(self._lens_active_count))        # 4 bytes
        data += struct.pack("f", float(self._lens_film_distance_world))   # 4 bytes
        data += struct.pack("f", float(self._lens_rear_z_world))          # 4 bytes
        data += struct.pack("f", float(self._lens_rear_aperture_world))   # 4 bytes
        data += struct.pack("f", float(self._lens_front_z_world))         # 4 bytes
        # Sensor half-height in world units. Lens path frames the image
        # off this (verticalAperture/2 / mm_per_unit), making `camera.fov`
        # inert when a lens is active — the lens stack alone determines
        # field of view.
        # Sensor half-height adjusted so the lens path frames the same
        # field of view as the pinhole. Pinhole's fov derives from the
        # idealised image distance F (focal length); the realistic lens
        # actually images onto a plane at the back focal length BFL ≠ F
        # for a thick lens, which would otherwise widen or narrow the
        # frame on lens enable. Scale by `filmDistance / F` so a unit
        # NDC at the lens path projects to the same world angle as it
        # does through the pinhole.
        va_mm = float(getattr(self.camera, "vertical_aperture_mm", 24.0))
        focal_mm = float(getattr(self.camera, "focal_length_mm", 50.0))
        mm_per_unit = max(float(self.scene.mm_per_unit), 1e-6)
        film_half_h_world = 0.5 * va_mm / mm_per_unit
        if self._lens_active_count > 0 and focal_mm > 1e-3:
            ratio = self._lens_film_distance_world / (focal_mm / mm_per_unit)
            film_half_h_world *= ratio
        data += struct.pack("f", film_half_h_world)                        # 4 bytes
        # emissiveTotalPower (reuses the retired irisZ slot): Σ(area·Rec709-lum)
        # over emissive triangles, read by the path tracer's BSDF-hit MIS weight.
        data += struct.pack("f", float(getattr(self, "_emissive_total_power", 0.0)))  # 4 bytes
        data += struct.pack("I", int(self._lens_num_pupil_bounds))         # 4 bytes
        data += struct.pack("f", float(self._lens_film_diag_world * 0.5))  # 4 bytes
        # Focal-plane visualiser: a translucent infinite plane main_pass.slang
        # alpha-composites over the integrator output when `focusOverlay`==1.
        # Plane is defined by a world-space origin and unit normal — origin
        # sits at camera + forward · focus_distance, normal = forward.
        focus_on, fp_origin, fp_normal = self._focus_plane_state()
        data += struct.pack("I", 1 if focus_on else 0)                  # 4 bytes
        data += fp_origin.tobytes()                                      # 12 bytes
        data += fp_normal.tobytes()                                      # 12 bytes
        # Viewport zoom-rect — sub-region of the output in [0, 1]² that
        # gets stretched to fill the window.
        zr = self.zoom_rect
        data += struct.pack("ff", float(zr[0]), float(zr[1]))            # 8 bytes (zoomMin)
        data += struct.pack("ff", float(zr[2]), float(zr[3]))            # 8 bytes (zoomMax)
        data += struct.pack("I", 1 if getattr(self, "lens_vignette_debug", False) else 0)  # 4 bytes
        # BXDF visualizer scene-pick. When pick is armed the main pass
        # snapshots the HitInfo of the matching pixel into toolBuffer
        # (binding 30); the CPU then disarms via `poll_pick_result`.
        pick_px = getattr(self, "_pick_pixel", (0, 0))
        pick_armed = 1 if getattr(self, "_pick_armed", False) else 0
        data += struct.pack("II", int(pick_px[0]), int(pick_px[1]))  # 8 bytes
        data += struct.pack("I", pick_armed)                          # 4 bytes
        # Display exposure (EV stops, applied as 2^EV multiplier before
        # tonemapping) and tonemap operator selector consumed by
        # main_pass.slang::applyTonemap. The pbrt film imaging ratio
        # (exposure_time·iso/100, change pbrt-radiometric-parity) is a linear
        # output gain; fold it in as log2(ratio) stops so the on-screen path
        # reproduces it with no shader/UBO change. The linear-HDR readback applies
        # the same ratio multiplicatively (read_accumulation_hdr consumers), so
        # display and linear stay consistent. ratio 1.0 ⇒ +0 stops ⇒ unchanged.
        ratio = self.film.imaging_ratio()
        exposure_ev = float(self.exposure) + (math.log2(ratio) if ratio > 0.0 else 0.0)
        data += struct.pack("f", exposure_ev)                         # 4 bytes
        data += struct.pack("I", int(self.tonemap_index))             # 4 bytes
        # Pluggable scene-sampling seam — proposalMask + reuseMode + the
        # float4 one-sample-MIS selection weights. Scalar layout: these append
        # tightly, matching the FrameConstants tail in common.slang. Default
        # {bsdf}/none folds to mask=1, alpha=(1,0,0,0), reuseMode=0 — the
        # bounce takes the BSDF fast path, bit-identical to the pre-seam build.
        from skinny.sampling import proposal_mask_and_alpha
        prop_mask, prop_alpha = proposal_mask_and_alpha(self._active_proposals())
        # Reuse capability gate: ReSTIR DI is wavefront-only (multi-pass). On
        # the megakernel (either device) the reuseMode folds to 0 (identity) so
        # the shader's depth-0 reuseDirect gate stays inert — stock NEE. On the
        # wavefront backend both Vulkan and Metal run ReSTIR (phase 5); the
        # pass builders construct the ReSTIR sub-pass under the same condition.
        reuse_mode = int(self._active_reuse().reuse_mode)
        if self.effective_execution_mode_index != EXECUTION_WAVEFRONT:
            reuse_mode = 0
        # Neural proposal (bit2) is wavefront-only — the MLP runs as a compute
        # pre-pass (vk_wavefront.WavefrontNeuralProposalPass), infeasible inline
        # in the megakernel (MoltenVK big-kernel limit). On the megakernel strip
        # the bit and renormalise the mixture over the analytic remainder, then
        # warn once so the request is reported, not silently dropped.
        if (prop_mask & 0x4) and self.effective_execution_mode_index != EXECUTION_WAVEFRONT:
            self._warn_neural_megakernel_once()
            prop_mask &= ~0x4
            a = list(prop_alpha)
            a[2] = 0.0
            s = sum(a) or 1.0
            prop_alpha = (a[0] / s, a[1] / s, a[2] / s, a[3] / s)
            if prop_mask == 0:
                # Neural-only on the megakernel: stripping the bit leaves an empty
                # mixture. Fold back to the {bsdf} fast path so the bounce still has
                # a valid proposal (rather than a zero mask / zero alpha).
                prop_mask = 0x1
                prop_alpha = (1.0, 0.0, 0.0, 0.0)
        data += struct.pack("II", int(prop_mask), reuse_mode)         # 8 bytes
        data += struct.pack("4f", *prop_alpha)                        # 16 bytes
        # Per-lobe sampler selection (flatLobeSamplers): one uint, 8 bits/lobe
        # (coat | spec<<8 | diff<<16), folded from the three selection indices.
        # All-native (0) is the default — bit-identical draw path to pre-change.
        from skinny.sampling import fold_lobe_samplers

        flat_lobe_samplers = fold_lobe_samplers(
            self.coat_sampler_index, self.spec_sampler_index, self.diff_sampler_index
        )
        data += struct.pack("I", int(flat_lobe_samplers))             # 4 bytes
        # Neural directional proposal — scene AABB (for the condition's position
        # normalisation) + active frozen-net version (baseline 0). Scalar tail
        # matching FrameConstants in common.slang; read only when the NEURAL bit
        # is active, so default {bsdf} stays bit-identical. The encoding here MUST
        # match the offline trainer's (spline_flow) condition.
        bmin, bext = self._neural_scene_bounds()
        data += bmin.tobytes()                                        # 12 bytes
        data += bext.tobytes()                                        # 12 bytes
        data += struct.pack("I", int(self._neural_network_version))   # 4 bytes
        # Wavefront-native path-record emission (change wavefront-native-path-
        # records): 1 only while the wavefront record drain is active, so the
        # default render packs 0 and stays byte-identical. Scalar tail matching
        # FrameConstants.recordMode in common.slang.
        record_mode = 1 if getattr(self, "_wf_record_active", False) else 0
        data += struct.pack("I", int(record_mode))                   # 4 bytes
        # Improper (mirrored) pbrt camera: 1 ⇒ zoomedNDC negates ndc.x for a
        # horizontal screen-space mirror (change pbrt-mirrored-camera-flip).
        # Default 0 ⇒ a non-mirrored render is byte-identical. Scalar tail
        # matching FrameConstants.cameraMirror in common.slang.
        camera_mirror = 1 if getattr(self, "_camera_mirror", False) else 0
        data += struct.pack("I", int(camera_mirror))                 # 4 bytes
        # SPPM per-pass photon-mapping tail (change photon-mapping-sppm). Zero
        # unless the SPPM integrator is active. Initial radius = the pbrt `radius`
        # override when imported, else ~0.1% of the scene bbox diagonal; cell size
        # tracks the initial radius (a valid upper bound as radii shrink).
        # sppmGridRes is unused by the kernels (they hash from width*height) and
        # is packed zero. Photons/pass default to one per pixel. The chosen photon
        # count is stashed for the SPPM pass's record_dispatch this frame.
        if self.integrator_index == 2:  # INTEGRATOR_SPPM
            _bmin, _bext = self._neural_scene_bounds()
            _diag = float(np.linalg.norm(_bext))
            sppm_radius = float(getattr(self, "_sppm_radius_override", 0.0)) \
                or max(_diag * 0.001, 1e-4)
            # Photon-emission group selection pmf, proportional to each group's
            # emitted power (change sppm-power-proportional-photon-groups).
            # Presence predicates mirror the shader's hasE/hasS/hasD/hasEnv
            # byte-for-byte (the packed counts below feed the fc fields the
            # shader reads). R is the same bounding-sphere radius the emission
            # geometry uses; envIntensity is folded exactly once (the CDF's
            # luminance integral is unscaled). Emitted powers:
            #   Φ_E = π·Σ(area·lum)      (cosine-hemisphere area emitters)
            #   Φ_S = 4π²·Σ(lum·r²)      (= π·Σ(lum·4πr²), full-sphere emitters)
            #   Φ_D = πR²·Σlum           (parallel beam through the bbox disc)
            #   Φ_env = πR²·envIntensity·∫L dω   (pbrt ImageInfiniteLight::Phi)
            _R = max(0.5 * _diag, 1e-4)
            _furnace = 1 if self.scene.furnace_mode else 0
            _present = (
                int(self._num_emissive_tris) > 0,
                int(self._num_sphere_lights) > 0,
                int(self._num_distant_lights) > 0,
                _furnace == 0 and float(env_intensity) > 0.0,
            )
            _powers = (
                math.pi * float(getattr(self, "_emissive_total_power", 0.0)),
                4.0 * math.pi ** 2 * float(self._sphere_power_sum),
                math.pi * _R * _R * float(self._distant_lum_sum),
                math.pi * _R * _R * float(env_intensity) * float(self._env_lum_integral),
            )
            sppm_pmf = self._sppm_group_pmf_override \
                or _sppm_photon_group_pmf(_powers, _present)
            # Env-aware per-pass photon budget (change sppm-env-photon-budget):
            # the expected non-env photon count stays exactly width*height, the
            # env group's photons ride on top (capped ×8) — env deposits are
            # sparse (depth≥1 only, whole-bounding-disc emission), so the flat
            # one-per-pixel budget left env-lit scenes speckled. pmfEnv == 0 ⇒
            # width*height exactly (env-free renders bit-identical).
            sppm_photons = int(getattr(self, "_sppm_photons_override", 0)) \
                or _sppm_photon_budget(self.width * self.height, float(sppm_pmf[3]))
            # Metal GPU-watchdog handling (change sppm-photon-dispatch-tiling):
            # the phase-3 photon dispatch is the heaviest SPPM command buffer —
            # one thread per photon, each depositing into every visible point in
            # radius, per-λ under spectral. A caustic scene clusters visible
            # points into the focus cell, so photons × VPs-in-cell wedges the GPU.
            # The driver now BOUNDS this by BREADTH — it tiles the photon dispatch
            # into flushed sub-batches of `_sppm_metal_photon_batch` photons — so
            # photons/pass stays the full width*height (no dark-starvation bias).
            # SKINNY_SPPM_METAL_PHOTON_BATCH sets the per-dispatch breadth (0 =
            # single dispatch). SKINNY_SPPM_METAL_PHOTON_CAP is retained as an
            # OPTIONAL per-pass ceiling (default 0 = unlimited) for pathological
            # scenes; prefer lowering the batch over capping photons.
            self._sppm_metal_photon_batch = 0
            if self.is_metal:
                import os
                _cap = int(os.environ.get("SKINNY_SPPM_METAL_PHOTON_CAP", "0"))
                if _cap > 0:
                    sppm_photons = min(sppm_photons, _cap)
                self._sppm_metal_photon_batch = int(
                    os.environ.get("SKINNY_SPPM_METAL_PHOTON_BATCH",
                                   str(_SPPM_METAL_PHOTON_BATCH_DEFAULT)))
            _glossy = getattr(self, "_sppm_glossy_roughness_override", None)
            sppm_glossy = float(_glossy) if _glossy is not None else _SPPM_GLOSSY_ROUGHNESS_DEFAULT
        else:
            sppm_radius = 0.0
            sppm_photons = 0
            sppm_glossy = 0.0
            sppm_pmf = (0.0, 0.0, 0.0, 0.0)
        self._sppm_photons_emitted = sppm_photons
        data += struct.pack("f", sppm_radius)                        # sppmInitialRadius
        data += struct.pack("f", sppm_radius)                        # sppmCellSize
        data += struct.pack("III", 0, 0, 0)                          # sppmGridRes (unused)
        data += struct.pack("I", int(sppm_photons))                  # sppmPhotonsEmitted
        data += struct.pack("f", sppm_glossy)                        # sppmGlossyContinueRoughness
        data += struct.pack("f", float(self.film_max_component))      # filmMaxComponent
        data += struct.pack("4f", *(float(p) for p in sppm_pmf))     # sppmGroupPmfE/S/D/Env
        # MLT chain constants (change mlt-integrator): the SKINNY_MLT
        # FrameConstants tail (`_FC_MLT_FIELDS`), appended ONLY when the MLT
        # wavefront pass is the consumer — every other pipeline's struct ends
        # above. The tail goes BEFORE the trailing tileOriginY word: in the
        # Vulkan MLT SPIR-V `tileOriginY` does not exist at all (it is
        # `#if defined(SKINNY_METAL)`-gated), so `mltSigma` sits at offset 564
        # immediately after sppmGroupPmfEnv — exactly where the filler would
        # land — and the trailing word is harmless slack in the oversized UBO.
        # On Metal BOTH fields exist and the packer relocates by reflected name
        # (`_pack_uniforms_msl` over `_FC_SCALAR_FIELDS_MLT`), so this scalar
        # order needs no backend split: only the MSL placement differs. Gated on
        # `_mlt_uniform_tail_active()` (not just integrator 3): on Metal the tail
        # is packed ONLY when the MLT wavefront pass is the real consumer, so a
        # megakernel-fallback MLT selection or a runtime switch to another
        # integrator can't desync the blob from the reflected `fc` (codex
        # pre-merge review).
        emit_tail = (self._mlt_uniform_tail_active() if mlt_tail is None
                     else bool(mlt_tail))
        if emit_tail:  # INTEGRATOR_MLT, active consumer
            chains = max(1, int(self.mlt_num_chains))
            pixels = max(1, self.width * self.height)
            iterations = self._mlt_iterations_per_frame()
            mpp_actual = iterations * chains / pixels
            mlt_pass = self._wavefront_mlt_pass
            mlt_b = float(getattr(mlt_pass, "b", 0.0)) if mlt_pass is not None else 0.0
            data += struct.pack(
                "ffff", float(self.mlt_sigma), float(self.mlt_large_step_prob),
                mlt_b, float(mpp_actual))
            data += struct.pack(
                "IIII", chains, 0,  # mltChainBase: both backends tile via wfTile
                int(self.mlt_max_depth), int(self._mlt_seed) & 0xFFFFFFFF)
        data += struct.pack("I", 0)                                  # tileOriginY (Metal band loop patches)

        # Directional lights are no longer in the UBO — they live in the
        # `distantLights` SSBO at binding 20 (uploaded by
        # _upload_distant_lights). The shader iterates `numDistantLights`
        # entries via DirectionalLightImpl (ILight).
        return bytes(data)

    def _ensure_env_uploaded(self) -> None:
        """Upload current env to GPU if it has changed (called once per switch).

        Reads environment data from `self.scene.environment`. Authored scenes
        without a DomeLight deliberately keep the already allocated texture
        as an inert backing resource while uniforms set its contribution to
        zero.
        """
        self.env_index = int(np.clip(self.env_index, 0, len(self.environments) - 1))
        env_hdr = self.scene.environment
        if self.scene.furnace_mode:
            cache_key = ("furnace", id(env_hdr.data) if env_hdr is not None else None)
        elif self.uses_default_lights:
            cache_key = (
                "fallback",
                int(self.env_index),
                id(env_hdr.data) if env_hdr is not None else None,
            )
        elif env_hdr is not None:
            cache_key = ("authored", id(self._usd_scene), id(env_hdr.data))
        else:
            cache_key = ("authored-black", id(self._usd_scene))
        if cache_key == self._last_env_index:
            return
        if env_hdr is None:
            # The GPU binding remains valid, but environment intensity is zero
            # for this authored no-Dome path.
            env_hdr_data = self.environments[self.env_index].data
        else:
            env_hdr_data = env_hdr.data
        self.env_image.upload_sync(env_hdr_data)
        # Rebuild + upload the importance-sampling distribution to match the
        # newly-uploaded env texture, so env NEE samples the right directions.
        from skinny.environment import build_env_distribution
        marg, cond, lum_integral = build_env_distribution(env_hdr_data)
        # ∫L dω of the (unscaled) map → SPPM photon-group power
        # Φ_env = πR²·envIntensity·∫L dω (intensity applied at pack time).
        self._env_lum_integral = float(lum_integral)
        # Concatenate into the one combined buffer: marginal then conditional,
        # matching envDistCdf's [marginal | conditional] layout (the shader reads
        # the conditional at ENV_COND_CDF_BASE = ENV_DIST_H+1 elements in).
        self.env_dist_buffer.upload_sync(marg + cond)
        self._last_env_index = cache_key

    @property
    def env_name(self) -> str:
        if 0 <= self.env_index < len(self.environments):
            return self.environments[self.env_index].name
        return "(none)"

    @staticmethod
    def _load_hud_font() -> ImageFont.ImageFont:
        """Try a common monospace TTF; fall back to Pillow's bitmap default."""
        for candidate in (
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/lucon.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ):
            try:
                return ImageFont.truetype(candidate, 14)
            except OSError:
                continue
        return ImageFont.load_default()

    def _build_hud_bytes(self) -> bytes:
        """Rasterise hud_text_lines into an R8 alpha mask.

        Pixel value encoding consumed by main_pass.slang:
          0     : transparent
          150   : dim background panel (alpha ≈ 0.59)
          255   : text ink (white)

        TTF anti-aliasing is thresholded away so edges don't fall into the
        panel-alpha range (which would render text edges as dim panel).
        """
        base = Image.new("L", (self.width, self.height), 0)
        if not self.show_hud or not self.hud_text_lines:
            return base.tobytes()

        draw = ImageDraw.Draw(base)
        font = self._hud_font

        line_height = 18
        padding = 10
        max_text_w = 0
        for line in self.hud_text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            max_text_w = max(max_text_w, bbox[2] - bbox[0])

        panel_w = max_text_w + padding * 2
        panel_h = line_height * len(self.hud_text_lines) + padding * 2

        # Panel background
        draw.rectangle((0, 0, panel_w, panel_h), fill=150)

        # Text rendered onto a separate mask, then thresholded to binary so
        # AA edges become crisp ink rather than falling into the panel range.
        text_mask = Image.new("L", (self.width, self.height), 0)
        tdraw = ImageDraw.Draw(text_mask)
        for i, line in enumerate(self.hud_text_lines):
            tdraw.text(
                (padding, padding + i * line_height),
                line,
                fill=255,
                font=font,
            )
        text_mask = text_mask.point(lambda v: 255 if v >= 128 else 0)
        base.paste(255, (0, 0), mask=text_mask)

        return base.tobytes()

    def _current_state_hash(self) -> int:
        """Hash camera + material + light state. Changes reset the accumulation."""
        parts = (
            self.camera.state_signature(),
            float(self.light_elevation), float(self.light_azimuth),
            float(self.light_intensity),
            float(self.light_color_r), float(self.light_color_g), float(self.light_color_b),
            int(self.env_index),
            int(self.direct_light_index),
            int(self.model_index),
            int(self.tattoo_index),
            float(self.tattoo_density),
            int(self.scatter_index),
            int(self.integrator_index),
            # Scene-sampling seam: changing the proposal mixture or reuse mode
            # resets accumulation so the new configuration converges cleanly.
            int(self.proposal_preset_index),
            int(self.reuse_index),
            # Per-lobe sampler selection (flat BSDF) — same reset semantics.
            int(self.coat_sampler_index),
            int(self.spec_sampler_index),
            int(self.diff_sampler_index),
            int(self.restir_regime_index),
            bool(self.restir_biased),
            int(self.restir_m_light), int(self.restir_m_bsdf),
            int(self.restir_spatial_k), float(self.restir_spatial_radius),
            int(self.restir_m_cap),
            # execution_mode_index is fixed for the session (CLI-selected), so
            # it never changes mid-session and is omitted from the hash.
            float(self.env_intensity),
            int(self.furnace_index),
            float(self.mm_per_unit),
            # Heterogeneous-medium grid identity (nanovdb-volume-rendering):
            # swapping the density grid (scene change) changes every volume
            # pixel, so reset accumulation. Material σ/g edits ride on
            # `_material_version` like every other override edit.
            self._volume_grid_key,
            # Film per-sample radiance clamp (change film-maxcomponent-clamp):
            # changing it changes every pixel, so reset accumulation.
            float(self.film_max_component),
            # Improper-camera mirror: changing it flips the image, so reset accum.
            bool(self._camera_mirror),
            # pbrt film exposure controls (change pbrt-radiometric-parity): retuning
            # ISO / exposure time rescales every pixel, so reset accumulation.
            float(self.film.iso), float(self.film.exposure_time),
            int(self.detail_maps_index),
            float(self.normal_map_strength),
            float(self.displacement_scale_mm),
            int(self.preset_index),
            int(self._material_version),
            # USD playback time — while playing this changes every frame so
            # accumulation resets (1 spp in motion); stable when paused.
            float(self.clock.current_time_code),
            # E-4: user-direct MaterialX field overrides — sort for stable hash
            tuple(sorted(
                (k, _hashable_value(v)) for k, v in self.mtlx_overrides.items()
            )),
            # SPPM per-pass tuning overrides (change sppm-glossy-final-gather):
            # changing any resets accumulation so an A/B (e.g. glossy threshold 0
            # vs the tuned default on one reused renderer) converges cleanly
            # instead of accumulating across configurations.
            getattr(self, "_sppm_radius_override", None),
            getattr(self, "_sppm_photons_override", None),
            getattr(self, "_sppm_glossy_roughness_override", None),
        )
        return hash(parts)

    def update(self, dt: float) -> None:
        self.time_elapsed += dt
        self.frame_index += 1

        # Emit the configuration matrix (change online-training-observability).
        # Dedup-guarded, so this prints once at startup and again only when a
        # status flips (e.g. selecting a neural proposal arms online training).
        # Driven from the shared per-frame entry so every front-end gets it.
        self._emit_config_matrix()

        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps_smooth = (
                inst_fps if self._fps_smooth == 0 else self._fps_smooth * 0.9 + inst_fps * 0.1
            )

        # Apply newly-arrived USD metadata before building the per-frame scene
        # snapshot. This makes the authority transition atomic: authored lights
        # and environment state become visible in the same frame.
        self._poll_usd_streaming()

        # Advance USD playback and re-evaluate animated prims before any
        # light/scene upload below reads from _usd_scene. No-op when paused or
        # when the loaded stage has no animation.
        self.clock.advance(dt)
        self._apply_animation_frame()

        # A `usd:` control edited a stage attribute → re-read live state.
        if self._usd_live_dirty:
            self._refresh_usd_live_state()
            self._usd_live_dirty = False

        # Recompute light direction + radiance from current slider state so
        # _build_scene_from_state picks up intensity / colour / angle changes.
        self._update_light()

        # Refresh the per-frame Scene snapshot before any GPU upload path
        # reads from it. Cheap (a few attribute copies); rebuilt every
        # frame so UI changes propagate without an explicit notification.
        self.scene = self._build_scene_from_state()
        if self._scene_graph is None:
            self._ensure_default_scene_graph()
        if (
            self._scene_graph is not None
            and self.uses_default_lights
            != self._last_projected_default_lights
        ):
            self._inject_default_lights_into_scene_graph()

        # Mirror the active authority's distant lights into binding 20 every
        # frame. Any authored USD lighting suppresses the fallback sun, even
        # when the authored source is a dome, area light, emissive material,
        # disabled light, or zero-power light. A retained inactive USD scene
        # cannot affect the active default/OBJ model.
        if self.uses_default_lights:
            self._upload_distant_lights(
                self.scene.lights_dir,
                fallback_controls=True,
            )
        else:
            self._upload_distant_lights(self._usd_scene.lights_dir)
        self._sync_auxiliary_light_authority()

        # If the environment selection changed, re-upload the HDR texture.
        self._ensure_env_uploaded()
        # Rebake the head mesh if source or displacement-scale drifted from
        # whatever we last built. Uses wall-clock time so slider drags get
        # debounced cleanly regardless of frame rate.
        self._rebake_if_needed(time.monotonic())
        # And for the tattoo texture.
        self._ensure_tattoo_uploaded()

        # Re-upload material types if scatter mode changed (per-material
        # scatter bits live in the upper bits of materialTypes[i]).
        cur_scatter = int(np.clip(
            self.scatter_index, 0, len(self._scatter_mode_bits) - 1
        ))
        if cur_scatter != self._last_scatter_index:
            self._upload_material_types()

        state = self._current_state_hash()
        if state != self._last_state_hash:
            self.accum_frame = 0
            self._last_state_hash = state
            # Zero the BDPT light-tracer splat buffer so the running mean
            # restarts cleanly. Cheap on integrated/dedicated GPUs (single
            # FillBuffer command + queue wait) and only fires on state change.
            if hasattr(self, 'light_splat_buffer'):
                self.light_splat_buffer.fill_zero_sync()
        else:
            self.accum_frame += 1

        # Refresh the gizmo overlay each frame (cheap CPU-side rebuild +
        # one storage-buffer upload). Camera moves and instance edits
        # both shift the on-screen ring, so building per-frame keeps it
        # synced without an explicit dirty signal.
        self._refresh_gizmo_segments()

    def render(self) -> None:
        # The selected backend is built lazily once a scene's MaterialX
        # fragments are gen'd (USD metadata arrival, OBJ load). Until then the
        # window has nothing to draw — skip the whole frame.
        if not self._backend_render_ready:
            return
        if self.is_metal:
            # Metal has no Vulkan swapchain/fence machinery — dispatch the
            # megakernel and blit the offscreen frame to the slang-rhi surface.
            self._render_windowed_metal()
            # Frame-end double-buffer swap for online neural training (change
            # metal-neural-interop, design D1): _render_windowed_metal drains
            # the device before returning, so no in-flight command buffer reads
            # bindings 33/34 while the interop publisher's swap writes the
            # shared weight buffers in place (and the file publisher's
            # upload_sync path is equally safe).
            if self._online_training:
                self._online_frame_end_swap()
            return
        f = self.current_frame

        vk.vkWaitForFences(
            self.ctx.device, 1, [self.in_flight_fences[f]], vk.VK_TRUE, 2**64 - 1
        )
        vk.vkResetFences(self.ctx.device, 1, [self.in_flight_fences[f]])

        # Drain BXDF visualizer pick callback once its frame is fence-
        # visible. Must run BEFORE _pack_uniforms below so disarming on
        # a satisfied pick lands in this frame's UBO. BXDF / BSSRDF eval
        # uses a synchronous out-of-band dispatch — no per-frame poll
        # needed there.
        self.poll_pick_result()

        image_index = self.ctx.vkAcquireNextImageKHR(
            self.ctx.device,
            self.ctx.swapchain_info.swapchain,
            2**64 - 1,
            self.image_available[f],
            vk.VK_NULL_HANDLE,
        )

        # Upload uniforms and HUD staging
        self.uniform_buffer.upload(self._pack_uniforms())
        # Re-pack the per-material skin UBO array so SkinParameters slider
        # changes (and any per-material override mutations) are visible
        # to the shader on the next frame. Skipped when the runtime
        # didn't load.
        mtlx_bytes = self._pack_mtlx_skin_array()
        if mtlx_bytes:
            self.mtlx_skin_buffer.upload_sync(mtlx_bytes)
        self.hud_overlay.upload(self._build_hud_bytes())

        # Compute writes binding 1 (offscreen) at the user-chosen render
        # resolution; we then blit that into the acquired swapchain image
        # (which is locked to the window's surface extent). The descriptor
        # for binding 1 was already written to ``_offscreen_output`` at
        # init / resize, so no per-frame rewrite is needed here.
        swap_extent = self.ctx.swapchain_info.extent
        swap_w = int(swap_extent.width)
        swap_h = int(swap_extent.height)
        swap_image = self.ctx.swapchain_info.images[image_index]
        sub_color = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        )

        # Record command buffer
        cmd = self.command_buffers[f]
        vk.vkResetCommandBuffer(cmd, 0)
        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

        # Cross-frame memory dependency on the accumulation image: previous
        # frame's writes must be visible to this frame's reads.
        accum_mem_barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [accum_mem_barrier], 0, None, 0, None,
        )

        # Copy this frame's HUD bytes from staging into the device-local image.
        self.hud_overlay.record_copy(cmd)

        # Execution-mode gate (mirrors render_headless): in wavefront mode the
        # staged compute pipeline writes the offscreen image; megakernel binds
        # main_pass directly. The offscreen image is then blitted to the
        # swapchain below, identically for both backends.
        if self.effective_execution_mode_index == EXECUTION_WAVEFRONT:
            if self._wavefront_debug_pass is not None:
                self._wavefront_debug_pass.record_dispatch(cmd)
            else:
                self._record_wavefront_dispatch(cmd, self.descriptor_sets[f])
        else:
            # Bind pipeline and descriptors
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline)
            vk.vkCmdBindDescriptorSets(
                cmd,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self.pipeline.pipeline_layout,
                0, 1, [self.descriptor_sets[f]],
                0, None,
            )

            # Dispatch into the offscreen image at render resolution.
            groups_x = (self.width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
            groups_y = (self.height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
            vk.vkCmdDispatch(cmd, groups_x, groups_y, 1)

        # Offscreen GENERAL → TRANSFER_SRC for the blit source.
        offscreen_to_src = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image=self._offscreen_output.image,
            subresourceRange=sub_color,
        )
        # Swapchain UNDEFINED → TRANSFER_DST for the blit destination.
        swap_to_dst = vk.VkImageMemoryBarrier(
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            image=swap_image,
            subresourceRange=sub_color,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None,
            2, [offscreen_to_src, swap_to_dst],
        )

        # Blit offscreen → swapchain image (linear filter scales when sizes differ).
        blit = vk.VkImageBlit(
            srcSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            srcOffsets=[
                vk.VkOffset3D(x=0, y=0, z=0),
                vk.VkOffset3D(x=int(self.width), y=int(self.height), z=1),
            ],
            dstSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            ),
            dstOffsets=[
                vk.VkOffset3D(x=0, y=0, z=0),
                vk.VkOffset3D(x=swap_w, y=swap_h, z=1),
            ],
        )
        vk.vkCmdBlitImage(
            cmd,
            self._offscreen_output.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swap_image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, [blit],
            vk.VK_FILTER_LINEAR,
        )

        # Offscreen TRANSFER_SRC → GENERAL for the next compute dispatch.
        offscreen_to_general = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self._offscreen_output.image,
            subresourceRange=sub_color,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None,
            1, [offscreen_to_general],
        )
        # Swapchain TRANSFER_DST → PRESENT_SRC for present.
        swap_to_present = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=0,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            image=swap_image,
            subresourceRange=sub_color,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, None, 0, None,
            1, [swap_to_present],
        )

        vk.vkEndCommandBuffer(cmd)

        # Submit
        submit_info = vk.VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.image_available[f]],
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT],
            commandBufferCount=1,
            pCommandBuffers=[cmd],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.render_finished[image_index]],
        )
        vk.vkQueueSubmit(self.ctx.compute_queue, 1, [submit_info], self.in_flight_fences[f])

        # Present
        present_info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.render_finished[image_index]],
            swapchainCount=1,
            pSwapchains=[self.ctx.swapchain_info.swapchain],
            pImageIndices=[image_index],
        )
        self.ctx.vkQueuePresentKHR(self.ctx.present_queue, present_info)

        # Frame-end double-buffer swap for online neural training (task 4.2):
        # _apply_render_weights' upload_sync waits the device idle, so the
        # in-flight frame's reads of bindings 33/34/35 complete before the swap
        # overwrites them.
        if self._online_training:
            self._online_frame_end_swap()

        self.current_frame = (f + 1) % MAX_FRAMES_IN_FLIGHT

    def render_headless(self) -> bytes:
        """Render one frame to an offscreen image and return raw RGBA8 pixels.

        Works in both headless and windowed modes — binding 1 (storage
        image output) is rewritten to ``_offscreen_output`` here so a
        windowed session that just rebound binding 1 to a swapchain image
        in render() doesn't corrupt the screenshot.
        """
        # Backend not built yet — caller asked for a screenshot before any
        # scene/model was loaded. Return a fully-zeroed RGBA8 frame so the
        # web/screenshot path stays well-defined.
        if not self._backend_render_ready:
            return b"\x00" * (self.width * self.height * 4)
        if self.is_metal:
            self.poll_pick_result()
            frame = self._render_headless_metal()
            # Frame-end swap (metal-neural-interop): the frame's readback has
            # drained the device, so promoting pending weights now only affects
            # the NEXT frame — render weights stayed frozen this frame.
            if self._online_training:
                self._online_frame_end_swap()
            return frame
        f = self.current_frame

        # Drain BXDF visualiser scene-pick callbacks once their frame has
        # retired. Matches render() so the Qt + web entry points get the
        # same pick behaviour as the legacy GLFW path.
        self.poll_pick_result()

        vk.vkWaitForFences(
            self.ctx.device, 1, [self.in_flight_fences[f]], vk.VK_TRUE, 2**64 - 1
        )
        vk.vkResetFences(self.ctx.device, 1, [self.in_flight_fences[f]])

        # Point binding 1 at the offscreen image. In headless mode this is
        # already its initial value; in windowed mode render() points it at
        # the acquired swapchain image, so we restore it here.
        offscreen_info = vk.VkDescriptorImageInfo(
            imageView=self._offscreen_output.view,
            imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
        )
        vk.vkUpdateDescriptorSets(
            self.ctx.device, 1,
            [vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_sets[f],
                dstBinding=1, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[offscreen_info],
            )],
            0, None,
        )

        self.uniform_buffer.upload(self._pack_uniforms())
        mtlx_bytes = self._pack_mtlx_skin_array()
        if mtlx_bytes:
            self.mtlx_skin_buffer.upload_sync(mtlx_bytes)

        cmd = self.command_buffers[f]
        vk.vkResetCommandBuffer(cmd, 0)
        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

        accum_mem_barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [accum_mem_barrier], 0, None, 0, None,
        )

        # Push (pre-zeroed) HUD staging into the device-local image. Without
        # this the GPU side of hud_overlay has UNDEFINED contents after a
        # fresh allocation (driver-dependent garbage) and the shader's
        # binding 3 sample reads garbage alpha — visible as smeared/banded
        # artefacts in the rendered frame after a resize.
        self.hud_overlay.record_copy(cmd)

        # Execution-mode gate: in wavefront mode a staged compute pipeline writes
        # the accumulation image; megakernel is the default in-kernel path. A
        # test/debug pass overrides the staged path tracer when set; otherwise
        # the real staged generate→bounce→resolve loop runs (reusing the
        # megakernel scene descriptor set as set 0).
        if self.effective_execution_mode_index == EXECUTION_WAVEFRONT:
            if self._wavefront_debug_pass is not None:
                self._wavefront_debug_pass.record_dispatch(cmd)
            else:
                self._record_wavefront_dispatch(cmd, self.descriptor_sets[f])
        else:
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline)
            vk.vkCmdBindDescriptorSets(
                cmd,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self.pipeline.pipeline_layout,
                0, 1, [self.descriptor_sets[f]],
                0, None,
            )
            groups_x = (self.width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
            groups_y = (self.height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
            vk.vkCmdDispatch(cmd, groups_x, groups_y, 1)

        # Transition offscreen output: GENERAL → TRANSFER_SRC for readback
        barrier_to_src = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image=self._offscreen_output.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, [barrier_to_src],
        )

        self._readback.record_copy_from(cmd, self._offscreen_output.image)

        # Transition back: TRANSFER_SRC → GENERAL for next frame's compute write
        barrier_to_general = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self._offscreen_output.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier_to_general],
        )

        vk.vkEndCommandBuffer(cmd)

        submit_info = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(self.ctx.compute_queue, 1, [submit_info], self.in_flight_fences[f])

        vk.vkWaitForFences(
            self.ctx.device, 1, [self.in_flight_fences[f]], vk.VK_TRUE, 2**64 - 1
        )

        # Frame-end double-buffer swap for online neural training (task 4.2):
        # the frame's GPU work is complete, so promoting pending weights now only
        # affects the NEXT frame — render weights stayed frozen this frame.
        if self._online_training:
            self._online_frame_end_swap()

        self.current_frame = (f + 1) % MAX_FRAMES_IN_FLIGHT
        return self._readback.read()

    # ── Resolution + screenshot ─────────────────────────────────────

    def resize(self, width: int, height: int) -> None:
        """Change *render* resolution at runtime. Recreates the offscreen
        output, readback buffer, accumulation image, and HUD overlay.

        The window-side swapchain is intentionally not touched — surface
        capabilities lock its extent to the OS window size. In windowed
        mode the compute shader writes into the offscreen image at the
        new render resolution and ``render()`` blits that to the swapchain
        (with scaling) for present, so render and present resolutions
        stay decoupled.
        """
        width = max(64, min(8192, int(width)))
        height = max(64, min(8192, int(height)))
        # Round up to a workgroup-aligned extent so the dispatch grid
        # covers exactly the image with no waste.
        width = ((width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE) * WORKGROUP_SIZE
        height = ((height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE) * WORKGROUP_SIZE
        if width == self.width and height == self.height:
            return


        self.ctx.wait_idle()

        self._offscreen_output.destroy()
        self._readback.destroy()
        self.accum_image.destroy()
        self.hud_overlay.destroy()

        self.width = width
        self.height = height
        self.ctx.width = width
        self.ctx.height = height

        self._offscreen_output = self._gpu.StorageImage(
            self.ctx, width, height,
            format="rgba8_unorm",
            transfer_src=True,
        )
        self._readback = self._gpu.ReadbackBuffer(self.ctx, width, height)
        self.accum_image = self._gpu.StorageImage(
            self.ctx, width, height, transfer_src=True,
        )
        self.hud_overlay = self._gpu.HudOverlay(self.ctx, width, height)
        self.hud_overlay.upload(bytes(width * height))

        # Vulkan binds these images through a persistent descriptor set, so the
        # set-0 entries must be re-pointed at the freshly recreated images. Metal
        # has no persistent descriptor set — the megakernel dispatch binds
        # `accum_image.texture` / `_offscreen_output.texture` by reference each
        # frame, so the new images are picked up automatically.
        if not self.is_metal:
            self._rewrite_size_dependent_descriptors()

        self.accum_frame = 0
        self._last_state_hash = None

    def _rewrite_size_dependent_descriptors(self) -> None:
        """Re-write the descriptor entries that point at images recreated
        by resize(): binding 1 (offscreen output), binding 2 (accumulation),
        binding 3 (HUD overlay).
        """
        for ds in self.descriptor_sets:
            output_info = vk.VkDescriptorImageInfo(
                imageView=self._offscreen_output.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            accum_info = vk.VkDescriptorImageInfo(
                imageView=self.accum_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            hud_info = vk.VkDescriptorImageInfo(
                imageView=self.hud_overlay.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=1, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[output_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=2, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[accum_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds, dstBinding=3, dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[hud_info],
                ),
            ]
            vk.vkUpdateDescriptorSets(
                self.ctx.device, len(writes), writes, 0, None,
            )

    def read_accumulation_hdr(self) -> tuple[np.ndarray, int]:
        """Copy the float32 RGBA accumulation image to the host. Returns
        ``(array, sample_count)`` where ``array`` is shape (H, W, 4) and
        the caller divides by ``sample_count`` to get linear mean radiance.
        """

        if self.is_metal:
            # Metal `StorageImage` is rgba32_float; `read_rgba()` drains the
            # texture straight to an (H, W, 4) float32 host array — no transfer
            # command buffer, barriers, or fence needed. Match the Vulkan path's
            # shape/dtype and sample-count exactly.
            self.ctx.wait_idle()
            arr = np.asarray(
                self.accum_image.read_rgba(), dtype=np.float32,
            ).reshape(self.height, self.width, 4).copy()
            samples = max(1, int(self.accum_frame) + 1)
            return arr, samples

        vk.vkDeviceWaitIdle(self.ctx.device)

        rb = self._gpu.ReadbackBuffer(
            self.ctx, self.width, self.height, bytes_per_pixel=16,
        )

        alloc = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc)[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )

        sub = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )
        to_src = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image=self.accum_image.image,
            subresourceRange=sub,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, [to_src],
        )
        rb.record_copy_from(cmd, self.accum_image.image)
        to_general = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self.accum_image.image,
            subresourceRange=sub,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [to_general],
        )
        vk.vkEndCommandBuffer(cmd)

        fence = vk.vkCreateFence(
            self.ctx.device, vk.VkFenceCreateInfo(), None,
        )
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            fence,
        )
        vk.vkWaitForFences(
            self.ctx.device, 1, [fence], vk.VK_TRUE, 2**64 - 1,
        )
        vk.vkDestroyFence(self.ctx.device, fence, None)
        vk.vkFreeCommandBuffers(
            self.ctx.device, self.ctx.command_pool, 1, [cmd],
        )

        raw = rb.read()
        rb.destroy()

        arr = np.frombuffer(raw, dtype=np.float32).reshape(
            self.height, self.width, 4,
        ).copy()
        samples = max(1, int(self.accum_frame) + 1)
        return arr, samples

    def save_screenshot(self, path_or_file, fmt: str) -> None:
        """Save the current render to disk (or a file-like object).

        Supported ``fmt``:
        - ``"png"`` / ``"jpeg"`` / ``"bmp"``: tonemapped LDR via the same
          compute pass as live rendering, captured from the offscreen
          output. HUD is suppressed for this dispatch.
        - ``"exr"`` / ``"hdr"``: linear HDR from the accumulation image
          divided by sample count. Alpha is dropped.
        """
        fmt = fmt.lower().lstrip(".")
        if fmt == "jpg":
            fmt = "jpeg"

        if fmt in ("png", "jpeg", "bmp"):
            raw = self.render_headless()
            img = Image.frombuffer(
                "RGBA", (self.width, self.height), raw, "raw", "RGBA", 0, 1,
            )
            if fmt == "jpeg":
                img = img.convert("RGB")
                img.save(path_or_file, format="JPEG", quality=95)
            elif fmt == "png":
                img.save(path_or_file, format="PNG")
            else:
                img.save(path_or_file, format="BMP")
            return

        if fmt in ("exr", "hdr"):
            arr, samples = self.read_accumulation_hdr()
            # Apply the pbrt film imaging ratio as a linear output scale so the
            # saved linear-HDR carries pbrt-equivalent absolute radiance (change
            # pbrt-radiometric-parity). ratio 1.0 ⇒ unchanged.
            ratio = self.film.imaging_ratio()
            rgb = (arr[..., :3] / float(samples) * ratio).astype(np.float32)
            writer = _write_exr if fmt == "exr" else _write_hdr_rgbe
            ext = ".exr" if fmt == "exr" else ".hdr"
            if hasattr(path_or_file, "write"):
                # OpenEXR.File.write() takes a path, not a file-like
                # object — round-trip through a tempfile for the web
                # download path. The RGBE writer takes a path too.
                import os
                import tempfile
                fd, tmp_path = tempfile.mkstemp(suffix=ext)
                os.close(fd)
                try:
                    writer(tmp_path, rgb)
                    with open(tmp_path, "rb") as fh:
                        path_or_file.write(fh.read())
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            else:
                writer(str(path_or_file), rgb)
            return

        raise ValueError(f"Unsupported screenshot format: {fmt!r}")

    @staticmethod
    def screenshot_format_options() -> list[str]:
        """User-facing format names — order is also the GUI dropdown order."""
        return ["PNG", "JPEG", "BMP", "EXR", "HDR"]

    # ── Neural training-record dump (task 5.1) ──────────────────────

    def _bind_record_buffer(self, buf) -> None:
        """Point descriptor binding 36 (the PathRecord append buffer) at ``buf``
        across all frames-in-flight descriptor sets."""
        vk.vkDeviceWaitIdle(self.ctx.device)
        for ds in self.descriptor_sets:
            write = vk.VkWriteDescriptorSet(
                dstSet=ds, dstBinding=36, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(
                    buffer=buf.buffer, offset=0, range=buf.size)])
            vk.vkUpdateDescriptorSets(self.ctx.device, 1, [write], 0, None)

    def _dispatch_record(self, descriptor_set) -> None:
        """One-shot dispatch of the ``mainImageRecord`` entry over the frame."""
        cmd = vk.vkAllocateCommandBuffers(
            self.ctx.device, vk.VkCommandBufferAllocateInfo(
                commandPool=self.ctx.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                             self._record_pipeline.pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self._record_pipeline.pipeline_layout, 0, 1, [descriptor_set], 0, None)
        gx = (self.width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        gy = (self.height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        vk.vkCmdDispatch(cmd, gx, gy, 1)
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(self.ctx.compute_queue, 1,
                         [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
                         vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def dump_path_records(self, out_path, *, num_frames: int = 256,
                          max_records_per_frame: int | None = None,
                          frame_seed_base: int = 0) -> int:
        """Emit per-vertex neural training records to a ``.nrec`` file (task 5.1).

        Runs the ``mainImageRecord`` megakernel entry for ``num_frames``
        independent frames (each a fresh ``frameIndex`` → fresh RNG → fresh
        paths) and streams the records out. Backend-independent: it builds its
        own record pipeline and reuses the scene descriptor set, so it works in
        either execution mode. The file header carries the scene AABB so the
        offline ``spline_flow`` trainer normalises the condition byte-for-byte
        like the renderer. The records reflect the CURRENT proposal set as the
        sample generator (the MIS weight is divided out, so any generator is
        unbiased; ``{bsdf}`` or ``{bsdf,env}`` are the usual choices). Returns
        the total number of records written.
        """
        import struct as _struct

        from skinny.sampling.path_records import RECORD_STRIDE, pack_header

        if self._scene_bindings is None or self.descriptor_sets is None:
            raise RuntimeError("dump_path_records: scene not built yet (pump update())")

        rec_max_bounces = 6  # lockstep with path_record.slang REC_MAX_BOUNCES
        capacity = int(self.width) * int(self.height) * rec_max_bounces
        if max_records_per_frame is not None:
            capacity = min(capacity, int(max_records_per_frame))

        bmin, bext = self._neural_scene_bounds()

        self._ensure_record_pipeline()

        saved_frame_index = self.frame_index
        big = self._gpu.StorageBuffer(self.ctx, capacity * RECORD_STRIDE)
        self._bind_record_buffer(big)
        total = 0
        try:
            with open(out_path, "wb") as f:
                f.write(pack_header(bmin, bext))
                for i in range(int(num_frames)):
                    self.frame_index = int(frame_seed_base) + i
                    self.uniform_buffer.upload(self._pack_uniforms())
                    self.record_counter.upload_sync(_struct.pack("<II", 0, capacity))
                    self._dispatch_record(self.descriptor_sets[0])
                    count = _struct.unpack("<I", self.record_counter.download_sync(4))[0]
                    count = min(count, capacity)
                    if count:
                        f.write(big.download_sync(count * RECORD_STRIDE))
                        total += count
        finally:
            self._bind_record_buffer(self.record_buffer)  # restore the dummy
            big.destroy()
            self.frame_index = saved_frame_index
        return total

    def _ensure_record_pipeline(self):
        """Lazily build the ``mainImageRecord`` megakernel pipeline (shared by the
        ``.nrec`` dump and the live drain). Backend-independent — it reuses the
        scene descriptor set, so it runs in either execution mode."""
        if self._record_pipeline is None:
            from skinny.vk_compute import ComputePipeline
            self._record_pipeline = ComputePipeline(
                self.ctx, self.shader_dir,
                entry_module="main_pass", entry_point="mainImageRecord",
                graph_fragments=list(self._scene_graph_fragments),
                # The record pipeline reuses the scene descriptor sets, which under
                # --spectral carry the spectral-only bindings 45-51; build its layout
                # with the same flag so binding a spectral set against it is valid
                # (record dumps are otherwise wavefront/neural-only, refused under
                # spectral — this keeps the layout consistent as defense-in-depth).
                spectral=self._spectral)
        return self._record_pipeline

    def _resolve_record_source(self) -> str:
        """Resolve the live-drain record source to ``'wavefront'`` or
        ``'megakernel'``. ``'auto'`` (default) picks the wavefront-native emitter
        when running the wavefront *path* integrator on a GPU-compute backend —
        no megakernel dispatch, so it sidesteps the 2 s Windows TDR / ~400 s
        pipeline compile that loses the device for the megakernel record entry.
        It falls back to the megakernel for bdpt (wavefront record emission is
        out of scope for bdpt) and for megakernel execution. ``'megakernel'`` /
        ``'wavefront'`` force a source."""
        src = str(getattr(self, "_record_source", "auto")).lower()
        if src in ("megakernel", "wavefront"):
            return src
        if (self.effective_execution_mode_index == EXECUTION_WAVEFRONT
                and self.integrator_index != 1):  # 1 = bdpt → out of scope
            return "wavefront"
        return "megakernel"

    def _ensure_wf_record_drain(self, max_records_per_frame: int | None = None) -> int:
        """Allocate + bind the persistent wavefront-native drain target.
        Idempotent: only (re)allocates when the capacity grows. Returns the
        capacity (in records). The wavefront render's `emitRecord` appends here
        directly; the drain reads it back without dispatching the megakernel.

        Backend layouts (change metal-record-drain): on Vulkan the target is
        binding 36 (records from byte 0) with the counter in binding 37,
        ``[count, capacity]``. On Metal the records build merges the counter
        into the record buffer (slot cap): a 64-byte header — capacity (uint at
        byte 0), atomic count (uint at byte 60) — with packed 64-byte records
        from byte 64; the buffer replaces ``self.record_buffer`` so
        ``_build_metal_binds`` routes it to the ``recordBuf`` global by name."""
        import struct as _struct

        from skinny.sampling.path_records import RECORD_STRIDE

        rec_max_bounces = 6  # lockstep with REC_MAX_BOUNCES
        capacity = int(self.width) * int(self.height) * rec_max_bounces
        cap_limit = (1 << 20) if max_records_per_frame is None else int(max_records_per_frame)
        capacity = max(1, min(capacity, cap_limit))
        if self.is_metal:
            size = 64 + capacity * RECORD_STRIDE   # header + records
            if self._drain_buffer is None or self._drain_buffer.size < size:
                if self._drain_buffer is not None:
                    self._drain_buffer.destroy()
                self._drain_buffer = self._gpu.StorageBuffer(self.ctx, size)
                header = bytearray(64)
                header[0:4] = _struct.pack("<I", capacity)   # capacity @ 0
                self._drain_buffer.upload_sync(bytes(header))  # count @ 60 = 0
                # Route through the bind-by-name dict: every dispatch binds
                # `recordBuf` from self.record_buffer.
                if self.record_buffer is not None:
                    self.record_buffer.destroy()
                self.record_buffer = self._drain_buffer
            self._wf_record_capacity = capacity
            return capacity
        if (self._drain_buffer is None
                or self._drain_buffer.size < capacity * RECORD_STRIDE):
            if self._drain_buffer is not None:
                self._drain_buffer.destroy()
            self._drain_buffer = self._gpu.StorageBuffer(self.ctx, capacity * RECORD_STRIDE)
            self._bind_record_buffer(self._drain_buffer)
            self.record_counter.upload_sync(_struct.pack("<II", 0, capacity))
        self._wf_record_capacity = capacity
        return capacity

    def _drain_wavefront_records(self, replay, max_records_per_frame: int | None) -> int:
        """Drain the records the wavefront render already produced — no
        megakernel dispatch. Reads the counter, appends the records to
        ``replay``, then resets the counter for the next frame's render. The
        Metal branch reads the merged header+records layout (see
        ``_ensure_wf_record_drain``) and resets only the 4-byte count word."""
        import struct as _struct

        from skinny.sampling.path_records import RECORD_STRIDE, records_from_buffer

        cap = self._ensure_wf_record_drain(max_records_per_frame)
        if self.is_metal:
            header = self._drain_buffer.download_sync(64)
            count = min(_struct.unpack("<I", header[60:64])[0], cap)
            if count:
                raw = self._drain_buffer.download_sync(64 + count * RECORD_STRIDE)[64:]
                replay.add(records_from_buffer(raw, count))
            self._drain_buffer.upload_range(_struct.pack("<I", 0), 60)
            return count
        count = _struct.unpack("<I", self.record_counter.download_sync(4))[0]
        count = min(count, cap)
        if count:
            raw = self._drain_buffer.download_sync(count * RECORD_STRIDE)
            replay.add(records_from_buffer(raw, count))
        # Reset the counter so the next render's records start at index 0.
        self.record_counter.upload_sync(_struct.pack("<II", 0, cap))
        return count

    def drain_path_records_to_replay(self, replay, *,
                                     max_records_per_frame: int | None = None,
                                     frame_seed: int | None = None) -> int:
        """Drain one frame of GPU path records straight into ``replay`` (the live
        online-training feed, task 1.2) instead of streaming them to a ``.nrec``
        file. Runs the ``mainImageRecord`` entry once over the frame, reads the
        GPU counter (binding 37), and appends the produced records (the shipped
        ``RECORD_DTYPE`` layout, binding 36) to the recency-weighted buffer.
        Returns the number of records drained this frame.

        The drain buffer is allocated once and kept bound to binding 36 (only the
        record entry reads it; ``mainImage`` dead-strips it), so steady-state
        online training pays no per-frame (re)allocation. ``replay.add`` stamps a
        fresh generation, so repeated drains accumulate with recency weighting.

        HARDWARE SEAM: the record entry is a *megakernel*. On NVIDIA/Windows the
        8 MB megakernel runs longer than the 2 s GPU watchdog (TDR) and the
        dispatch loses the device — so this live GPU drain is exercised on a box
        without that watchdog (the NVIDIA-box benchmark, task 7.3), not in the
        Mac/wavefront test suite. The reader contract it depends on
        (``records_from_buffer``) is validated off-GPU. A wavefront-native record
        path that avoids the megakernel entirely is proposed separately
        (``openspec/changes/wavefront-native-path-records``).
        """
        import struct as _struct

        from skinny.sampling.path_records import RECORD_STRIDE, records_from_buffer

        if self._scene_bindings is None or (
                self.descriptor_sets is None and not self.is_metal):
            raise RuntimeError(
                "drain_path_records_to_replay: scene not built yet (pump update())")

        # Wavefront-native source: the normal wavefront render already appended
        # this frame's records (FrameConstants.recordMode on), so just read them
        # back — no megakernel `mainImageRecord` dispatch (the 2 s-TDR /
        # ~400 s-compile seam this change removes). Works on both backends
        # (change metal-record-drain).
        if self._resolve_record_source() == "wavefront":
            return self._drain_wavefront_records(replay, max_records_per_frame)

        if self.is_metal:
            raise RuntimeError(
                "the megakernel record source is unavailable on the Metal "
                "backend — use the wavefront path integrator (record source "
                "'wavefront') for online training there")

        rec_max_bounces = 6  # lockstep with path_record.slang REC_MAX_BOUNCES
        capacity = int(self.width) * int(self.height) * rec_max_bounces
        cap_limit = (1 << 20) if max_records_per_frame is None else int(max_records_per_frame)
        capacity = max(1, min(capacity, cap_limit))

        self._ensure_record_pipeline()

        # (Re)allocate the persistent drain target only when capacity grows.
        if self._drain_buffer is None or self._drain_buffer.size < capacity * RECORD_STRIDE:
            if self._drain_buffer is not None:
                self._drain_buffer.destroy()
            self._drain_buffer = self._gpu.StorageBuffer(self.ctx, capacity * RECORD_STRIDE)
            self._bind_record_buffer(self._drain_buffer)

        saved_frame_index = self.frame_index
        try:
            if frame_seed is not None:
                self.frame_index = int(frame_seed)
            self.uniform_buffer.upload(self._pack_uniforms())
            self.record_counter.upload_sync(_struct.pack("<II", 0, capacity))
            self._dispatch_record(self.descriptor_sets[0])
            count = _struct.unpack("<I", self.record_counter.download_sync(4))[0]
            count = min(count, capacity)
            if count:
                raw = self._drain_buffer.download_sync(count * RECORD_STRIDE)
                replay.add(records_from_buffer(raw, count))
            return count
        finally:
            self.frame_index = saved_frame_index

    def cleanup(self) -> None:
        # Backend-neutral device drain (see `_build_pipeline_for_current_graphs`).
        # Metal routes through slang-rhi's `Device.wait_for_idle`; the semaphore /
        # fence / descriptor-pool teardown below is all Vulkan-only but no-ops on
        # Metal (empty sync lists, `descriptor_pool is None`).
        self.ctx.wait_idle()

        self._destroy_wavefront_env_pass()
        for sem in self.image_available + self.render_finished:
            vk.vkDestroySemaphore(self.ctx.device, sem, None)
        for fence in self.in_flight_fences:
            vk.vkDestroyFence(self.ctx.device, fence, None)

        if self.descriptor_pool is not None:
            vk.vkDestroyDescriptorPool(self.ctx.device, self.descriptor_pool, None)
        self.texture_pool.destroy()
        if getattr(self, "_graph_params_combined", None) is not None:
            self._graph_params_combined.destroy()
            self._graph_params_combined = None
        self.std_surface_buffer.destroy()
        self.emissive_tri_buffer.destroy()
        self.sphere_lights_buffer.destroy()
        self.distant_lights_buffer.destroy()
        self.material_types_buffer.destroy()
        self.flat_material_buffer.destroy()
        self.instance_buffer.destroy()
        self.bvh_buffer.destroy()
        self.index_buffer.destroy()
        self.vertex_buffer.destroy()
        self.displacement_image.destroy()
        self.roughness_image.destroy()
        self.normal_image.destroy()
        self.tattoo_image.destroy()
        self.env_image.destroy()
        # Heterogeneous-medium density grid (binding 26).
        if getattr(self, "volume_density_image", None) is not None:
            self.volume_density_image.destroy()
            self.volume_density_image = None
        # Env importance-sampling CDFs (bindings 31/32) — allocated once in
        # _init_gpu; were previously never freed (surfaced by the record-dump's
        # clean-teardown check).
        self.env_dist_buffer.destroy()
        # Spectral buffers (bindings 45/46/47 upsample + 48 conductor eta/k) —
        # only allocated for the spectral megakernel variant.
        for _sb in (self._spectral_scale_buffer, self._spectral_data_buffer,
                    self._spectral_d65_buffer, self._spectral_metals_buffer,
                    self._spectral_emitters_buffer, self._spectral_light_spd_buffer,
                    self._spectral_mat_emission_buffer):
            if _sb is not None:
                _sb.destroy()
        self.hud_overlay.destroy()
        self.accum_image.destroy()
        self._offscreen_output.destroy()
        self._readback.destroy()
        self.uniform_buffer.destroy()
        self.mtlx_skin_buffer.destroy()
        # Interop handoff (task 5.2): release the CUDA imports (external memory +
        # semaphore + stream) BEFORE freeing the Vulkan objects they reference.
        if getattr(self, "_neural_publisher", None) is not None:
            _close = getattr(self._neural_publisher, "close", None)
            if _close is not None:
                _close()
        self.neural_weights_buffer.destroy()
        self.neural_biases_buffer.destroy()
        self.neural_layers_buffer.destroy()
        if getattr(self, "neural_timeline_semaphore", None) is not None:
            self.neural_timeline_semaphore.destroy()
            self.neural_timeline_semaphore = None
        if getattr(self, "_record_pipeline", None) is not None:
            self._record_pipeline.destroy()
            self._record_pipeline = None
        if getattr(self, "_drain_buffer", None) is not None:
            self._drain_buffer.destroy()
            self._drain_buffer = None
        self.record_buffer.destroy()
        self.record_counter.destroy()
        self.light_splat_buffer.destroy()
        self.gizmo_segments_buffer.destroy()
        self.lens_elements_buffer.destroy()
        self.lens_pupil_buffer.destroy()
        self.tool_buffer.destroy()
        if getattr(self, "_preview_pipeline", None) is not None:
            self._preview_pipeline.destroy()
            self._preview_pipeline = None
        if getattr(self, "_preview_readback", None) is not None:
            self._preview_readback.destroy()
            self._preview_readback = None
        if getattr(self, "_preview_image", None) is not None:
            self._preview_image.destroy()
            self._preview_image = None
        # The scene bindings own the set-0 layout (+ the megakernel pipeline in
        # megakernel mode, where `self.pipeline is self._scene_bindings`). Tear
        # the wavefront stage passes down first, then the scene bindings once.
        self._destroy_wavefront_path_pass()
        self._destroy_wavefront_bdpt_pass()
        self._destroy_wavefront_mlt_pass()
        if self._scene_bindings is not None:
            self._scene_bindings.destroy()
            self._scene_bindings = None
            self.pipeline = None
