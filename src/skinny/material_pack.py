"""Flat-material and standard-surface record packing — device-free.

Split out of ``renderer.py`` (change renderer-pure-core-extraction). These
packers produce the bytes BOTH backends upload, so they must be testable on a
Metal-only host; before the split they could only be imported where the Vulkan
SDK was present.

The strides are DERIVED from the Slang declarations by ``slang_layout`` (change
reflection-owned-byte-layouts) — read the per-field offsets there, not from a
comment map.
"""

from __future__ import annotations

import struct

import numpy as np

from skinny import slang_layout
from skinny.pbrt.data import CONDUCTOR_METAL_ID

# Per-material flat-shading record consumed by main_pass.slang's non-skin BSDF
# dispatch. The struct is `FlatMaterialParams` (common.slang), declared as
# float4-wrapped rows so the MSL and SPIR-V layouts are byte-identical (no Metal
# repack). Its per-row byte offsets are DERIVED from that declaration by
# `slang_layout` (change reflection-owned-byte-layouts) — read them there rather
# than from a comment map:
#     slang_layout.scalar_layout("FlatMaterialParams").offsets
# The sub-offsets *inside* each float4 row (e.g. diffuseColor.xyz + roughness in
# row 0) are packer-internal, not struct fields — see `pack_flat_material`'s
# docstring for that map.
FLAT_MATERIAL_STRIDE = slang_layout.scalar_stride("FlatMaterialParams")  # 256 B
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

# StdSurfaceParams record (binding 19): full MaterialX standard_surface
# parameters packed in scalar layout matching the Slang struct in
# mtlx_std_surface.slang (256 B / record) — stride DERIVED from that declaration
# (change reflection-owned-byte-layouts).
STD_SURFACE_STRIDE = slang_layout.scalar_stride("StdSurfaceParams")
STD_SURFACE_CAPACITY = FLAT_MATERIAL_CAPACITY_INIT

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


# StdSurfaceParams scalar layout — ordered (field-name, byte-size), DERIVED from
# the Slang struct in mtlx_std_surface.slang (change
# reflection-owned-byte-layouts). Matches `pack_std_surface_params`'s scalar
# (std430) packing: each field's scalar offset is the running byte sum (float3 =
# 12 B, no 16-B promotion). `pack_std_surface_params_msl` walks it to relocate
# the scalar record into Metal's MSL layout (where float3 → 16 B), keyed by the
# reflected field names.
_STD_SURFACE_SCALAR_ENTRIES: tuple[tuple[str, int], ...] = tuple(
    slang_layout.scalar_layout("StdSurfaceParams").entries)


def pack_std_surface_params_msl(
    scalar: bytes, layout: dict[str, tuple[int, int]], stride: int
) -> bytes:
    """Relocate a scalar-packed `pack_std_surface_params` record (256 B, float3 =
    12 B) into Metal's reflected MSL element layout for
    `StructuredBuffer<StdSurfaceParams>` (binding 19), where Slang pads every
    `float3` to 16 B and grows the element stride past 256 B (≈400). Each field's
    bytes move from its scalar offset (the running sum over the derived
    `_STD_SURFACE_SCALAR_ENTRIES`) to its reflected MSL offset (`layout[name]`). Same design-D3 repack the skin
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
    for name, size in _STD_SURFACE_SCALAR_ENTRIES:
        moff = layout.get(name)
        if moff is not None:
            rec[moff[0]:moff[0] + size] = scalar[off:off + size]
        off += size
    assert off == len(scalar), (
        f"StdSurfaceParams field table covers {off} B but the scalar record is "
        f"{len(scalar)} B")
    return bytes(rec)
