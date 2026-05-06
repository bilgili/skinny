"""Core renderer — orchestrates Vulkan compute dispatch for skin ray tracing."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import vulkan as vk

from PIL import Image, ImageDraw, ImageFont

from skinny.environment import Environment, load_environments
from skinny.scene import Scene, build_default_scene
from skinny.head_textures import (
    DETAIL_TEX_RES,
    blank_displacement_bytes,
    blank_normal_bytes,
    blank_roughness_bytes,
    expected_bytes as detail_expected_bytes,
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
from skinny.presets import PRESETS, Preset
from skinny.settings import load_user_presets
from skinny.tattoos import TATTOO_HEIGHT, TATTOO_WIDTH, Tattoo, blank_tattoo_data, load_tattoos
from skinny.vk_context import VulkanContext
from skinny.vk_compute import (
    BINDLESS_TEXTURE_CAPACITY,
    ComputePipeline,
    HudOverlay,
    SampledImage,
    StorageBuffer,
    StorageImage,
    UniformBuffer,
)

WORKGROUP_SIZE = 8
MAX_FRAMES_IN_FLIGHT = 2

# Per-instance storage record consumed by mesh_head.slang::Instance.
# Layout (std430-compatible): worldFromLocal (mat4x4, 64 B), localFromWorld
# (mat4x4, 64 B), four uints (blasNodeOffset, blasIndexOffset,
# blasVertexOffset, materialId; 16 B). Total 144 B, naturally 16-byte
# aligned so consecutive instances don't need padding.
INSTANCE_STRIDE = 144

# Per-material flat-shading record consumed by main_pass.slang's
# non-skin BSDF dispatch. Layout (std430-compatible):
#    0: diffuseColor (vec3, 12) + roughness (float, packs into trailing 4)
#   16: metallic + specular + opacity + diffuseTextureIdx
#   32: roughnessTextureIdx + metallicTextureIdx + normalTextureIdx + emissiveTextureIdx
#   48: emissiveColor (vec3, 12) + ior (float)
#   64: coat + coatRoughness + coatIOR + pad0
#   80: coatColor (vec3, 12) + pad1
# 96 B / record, naturally 16-byte aligned.
FLAT_MATERIAL_STRIDE = 96
FLAT_MATERIAL_CAPACITY = 16


def _hashable_value(v: object) -> object:
    """Coerce mtlx_overrides values into something hash()-friendly."""
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    if isinstance(v, (int, float)):
        return float(v)
    return v

# Material type codes consumed by main_pass.slang's dispatcher.
MATERIAL_TYPE_SKIN = 0  # legacy skin slot 0, or any mtlx_target_name pointing
                         # at the layered-skin material — routes to the inline
                         # skin BSSRDF/specular path.
MATERIAL_TYPE_FLAT = 1  # UsdPreviewSurface-style standard surface — routes
                         # to evalFlatMaterial's bounded path tracer.

# Sphere-light record (binding 17): vec3 position, float radius, vec3
# radiance, float pad. 32 B / record, naturally 16-byte aligned.
SPHERE_LIGHT_STRIDE = 32
SPHERE_LIGHT_CAPACITY = 16

# Emissive-triangle record (binding 18): vec3 v0 + pad, vec3 v1 + pad,
# vec3 v2 + pad, vec3 emission + float area. 64 B / record.
EMISSIVE_TRI_STRIDE = 64
EMISSIVE_TRI_CAPACITY = 256

# StdSurfaceParams record (binding 19): full MaterialX standard_surface
# parameters packed in scalar layout matching the Slang struct in
# mtlx_std_surface.slang.  256 B / record.
STD_SURFACE_STRIDE = 256
STD_SURFACE_CAPACITY = FLAT_MATERIAL_CAPACITY

# ProceduralParams record (binding 20): per-material procedural evaluation
# parameters (marble 3D noise, etc.).  96 B / record, scalar layout.
PROCEDURAL_PARAMS_STRIDE = 96
PROCEDURAL_PARAMS_CAPACITY = FLAT_MATERIAL_CAPACITY

# Default diffuse for materials whose UsdPreviewSurface diffuseColor is
# texture-connected rather than constant — mid-grey keeps unbound prims
# visible until bindless textures (Phase C-4) actually sample the file.
_FLAT_DEFAULT_DIFFUSE = (0.72, 0.72, 0.72)


def _extract_procedural_params(cm) -> dict | None:
    """Detect a procedural material from its CompiledMaterial uniform_block.

    Returns a dict suitable for `pack_procedural_params` when the material
    has fractal-noise uniforms (e.g. marble 3D), or None otherwise.
    """
    if not cm.uniform_block:
        return None
    field_map = {f.name: f for f in cm.uniform_block}
    if "noise_amplitude" not in field_map or "noise_octaves" not in field_map:
        return None

    def _f(name, default=0.0):
        f = field_map.get(name)
        if f is None or f.default is None:
            return float(default)
        try:
            return float(f.default)
        except (TypeError, ValueError):
            return float(default)

    def _i(name, default=0):
        f = field_map.get(name)
        if f is None or f.default is None:
            return int(default)
        try:
            return int(f.default)
        except (TypeError, ValueError):
            return int(default)

    def _c3(name, default=(0.0, 0.0, 0.0)):
        f = field_map.get(name)
        if f is None or f.default is None:
            return tuple(float(c) for c in default)
        v = f.default
        try:
            if hasattr(v, "__getitem__"):
                return float(v[0]), float(v[1]), float(v[2])
            if hasattr(v, "asTuple"):
                t = v.asTuple()
                return float(t[0]), float(t[1]), float(t[2])
        except (TypeError, IndexError, ValueError):
            pass
        return tuple(float(c) for c in default)

    return {
        "type": PROC_TYPE_MARBLE_3D,
        "add_xyz_in2": _c3("add_xyz_in2", (1.0, 1.0, 1.0)),
        "scale_pos_in2": _f("scale_pos_in2", 6.0),
        "scale_xyz_in2": _f("scale_xyz_in2", 4.0),
        "noise_amplitude": _f("noise_amplitude", 1.0),
        "noise_octaves": _i("noise_octaves", 3),
        "noise_lacunarity": _f("noise_lacunarity", 2.0),
        "noise_diminish": _f("noise_diminish", 0.5),
        "scale_noise_in2": _f("scale_noise_in2", 3.0),
        "scale_in2": _f("scale_in2", 0.5),
        "bias_in2": _f("bias_in2", 0.5),
        "power_in2": _f("power_in2", 3.0),
        "color_mix_fg": _c3("color_mix_fg", (0.1, 0.1, 0.3)),
        "color_mix_bg": _c3("color_mix_bg", (0.8, 0.8, 1.0)),
    }


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

    def __init__(self, ctx: VulkanContext) -> None:
        self.ctx = ctx
        self._slots: list[SampledImage | None] = [None] * BINDLESS_TEXTURE_CAPACITY
        self._by_path: dict[str, int] = {}
        self._next_slot = 0

    def add_or_get(self, path: Path) -> int:
        """Decode the file at `path` and return the array slot it lives in.

        Subsequent calls with the same path return the cached slot. Returns
        SENTINEL when the file can't be loaded (missing/corrupt) — callers
        should treat that as "no texture; use the constant fallback".
        """
        key = str(path.resolve()) if path.is_absolute() else str(path)
        cached = self._by_path.get(key)
        if cached is not None:
            return cached
        try:
            img = Image.open(path).convert("RGBA")
        except (FileNotFoundError, OSError):
            return self.SENTINEL
        if self._next_slot >= BINDLESS_TEXTURE_CAPACITY:
            return self.SENTINEL
        w, h = img.size
        slot = SampledImage(
            self.ctx, w, h,
            format=vk.VK_FORMAT_R8G8B8A8_SRGB,
            bytes_per_pixel=4,
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


def pack_flat_material(
    material,
    diffuse_texture_idx: int = 0xFFFFFFFF,
    roughness_texture_idx: int = 0xFFFFFFFF,
    metallic_texture_idx: int = 0xFFFFFFFF,
    normal_texture_idx: int = 0xFFFFFFFF,
    emissive_texture_idx: int = 0xFFFFFFFF,
) -> bytes:
    """Pack a Material's overrides into 96 bytes (FlatMaterialParams).

    Layout (std430-compatible):
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
      76: pad0                    (float)
      80: coatColor.r/g/b         (vec3 → 12 B)
      92: pad1                    (float)
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
    return struct.pack(
        "fff f f f f I I I I I fff f  f f f f  fff f",
        diffuse[0], diffuse[1], diffuse[2],
        roughness, metallic, specular, opacity,
        int(diffuse_texture_idx) & 0xFFFFFFFF,
        int(roughness_texture_idx) & 0xFFFFFFFF,
        int(metallic_texture_idx) & 0xFFFFFFFF,
        int(normal_texture_idx) & 0xFFFFFFFF,
        int(emissive_texture_idx) & 0xFFFFFFFF,
        emissive[0], emissive[1], emissive[2],
        ior,
        coat, coat_roughness, coat_ior, 0.0,
        coat_color[0], coat_color[1], coat_color[2], 0.0,
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


# Procedural type codes matching PROC_TYPE_* in common.slang.
PROC_TYPE_NONE = 0
PROC_TYPE_MARBLE_3D = 1


def pack_procedural_params(params: dict | None) -> bytes:
    """Pack procedural evaluation parameters into 96 bytes (ProceduralParams).

    `params` is a dict with keys matching the ProceduralParams struct fields.
    When None or type==0, returns a zeroed record (no procedural evaluation).
    """
    if params is None or params.get("type", 0) == 0:
        return b"\x00" * PROCEDURAL_PARAMS_STRIDE

    proc_type = int(params.get("type", 0))
    add_xyz = params.get("add_xyz_in2", (1.0, 1.0, 1.0))
    scale_pos = float(params.get("scale_pos_in2", 6.0))
    scale_xyz = float(params.get("scale_xyz_in2", 4.0))
    noise_amp = float(params.get("noise_amplitude", 1.0))
    noise_oct = int(params.get("noise_octaves", 3))
    noise_lac = float(params.get("noise_lacunarity", 2.0))
    noise_dim = float(params.get("noise_diminish", 0.5))
    scale_noise = float(params.get("scale_noise_in2", 3.0))
    scale = float(params.get("scale_in2", 0.5))
    bias = float(params.get("bias_in2", 0.5))
    power = float(params.get("power_in2", 3.0))
    fg = params.get("color_mix_fg", (0.1, 0.1, 0.3))
    bg = params.get("color_mix_bg", (0.8, 0.8, 1.0))

    return struct.pack(
        "I fff ff fi ffffff fff fff ffff",
        proc_type,
        float(add_xyz[0]), float(add_xyz[1]), float(add_xyz[2]),
        scale_pos, scale_xyz,
        noise_amp, noise_oct,
        noise_lac, noise_dim, scale_noise, scale, bias, power,
        float(fg[0]), float(fg[1]), float(fg[2]),
        float(bg[0]), float(bg[1]), float(bg[2]),
        0.0, 0.0, 0.0, 0.0,
    )


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


def _perspective(fov_deg: float, aspect: float) -> np.ndarray:
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
    near, far = 0.1, 100.0
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = far / (near - far)
    proj[2, 3] = -1.0
    proj[3, 2] = (near * far) / (near - far)
    return proj


def _look_at(pos: np.ndarray, forward: np.ndarray) -> np.ndarray:
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
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / max(np.linalg.norm(right), 1e-6)
    up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float32)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = -forward
    view[3, 0] = -np.dot(right, pos)
    view[3, 1] = -np.dot(up, pos)
    view[3, 2] = np.dot(forward, pos)
    return view


@dataclass
class OrbitCamera:
    """Camera that rotates around a target point (default: centre of the SDF head).

    The head's y-extent is roughly [-0.94, +1.15] and its z-extent [-0.80, +0.97],
    so target=(0, 0.1, 0.05) pins the pivot to its visual centroid. If you orbit
    around the world origin instead, the head drifts noticeably around the frame
    because that origin sits near the jaw/throat rather than the head's middle.
    """

    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.1, 0.05], dtype=np.float32)
    )
    distance: float = 3.0
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0

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

    def zoom(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance * (1.0 - delta * 0.1), 0.5, 50.0))

    def pan(self, dx: float, dy: float) -> None:
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        scale = self.distance * 0.002
        self.target = self.target + (-right * dx + up * dy) * scale

    def view_matrix(self) -> np.ndarray:
        pos = self.position
        f = self.target - pos
        return _look_at(pos, f / np.linalg.norm(f))

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fov, aspect)

    def state_signature(self) -> tuple:
        return (
            "orbit",
            float(self.yaw), float(self.pitch), float(self.distance), float(self.fov),
            float(self.target[0]), float(self.target[1]), float(self.target[2]),
        )


@dataclass
class FreeCamera:
    """FPS-style camera: WASD translates, mouse look rotates yaw/pitch."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 3.0], dtype=np.float32)
    )
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0
    move_speed: float = 1.5   # world units / second

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

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fov, aspect)

    def state_signature(self) -> tuple:
        return (
            "free",
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.yaw), float(self.pitch), float(self.fov),
        )


class Renderer:
    """Sets up Vulkan resources and dispatches Slang compute shaders each frame."""

    def __init__(
        self,
        vk_ctx: VulkanContext,
        shader_dir: Path,
        hdr_dir: Path | None = None,
        tattoo_dir: Path | None = None,
        usd_scene_path: Path | None = None,
        use_usd_mtlx_plugin: bool = False,
    ) -> None:
        self.ctx = vk_ctx
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
        self.camera_mode: str = "orbit"

        self.frame_index = 0
        self.time_elapsed = 0.0

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

        # Sampling / integration strategy consumed by main_pass.slang.
        # MIS affects surface IBL. "Bidirectional" keeps the camera-path
        # renderer but also adds explicit light-connection samples in volume.
        # "Stored BDPT" additionally builds a light-side surface vertex and
        # connects the camera hit to it with a visibility-tested geometry term.
        self.integrator_modes: list[str] = [
            "Path tracing", "MIS", "Bidirectional", "Stored BDPT",
        ]
        self.integrator_index = 0

        # Scalar applied to every sampleEnvironment() lookup. With many HDR
        # environments the raw luminance swamps skin albedo once multiplied
        # through the SSS estimator; this lets the user rebalance direct vs.
        # indirect contribution.
        self.env_intensity = 1.0

        # Furnace / energy-conservation probe. In this mode the shader swaps
        # the head for a unit sphere, clamps the environment to white (L=1)
        # in every direction, disables analytic direct light, and paints any
        # pixel whose accumulated radiance exceeds 1.0 per channel in a loud
        # pink — so energy violations are visible by eye. Exposed as a
        # discrete UI slider (On/Off) instead of a CLI flag so it can be
        # toggled during a session without restarting.
        self.furnace_modes: list[str] = ["Off", "On"]
        self.furnace_index: int = 0

        # Scene-scale bridge between mm-valued skin params and world-unit
        # ray distances. 1 world unit = mm_per_unit millimetres. The SDF
        # Loomis head is roughly unit-scale (~2 units tall), so with a ~240
        # mm real head height the default is 120. Exposed as a slider so
        # mesh heads of other sizes can be dialled in without editing code.
        self.mm_per_unit = 120.0

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

        # Load the MaterialX library and generate Slang for the canonical
        # skin material. The CompiledMaterial drives the per-material
        # MtlxSkinParams buffer (binding 15) that the shader reads via
        # skinParamsFromMtlx().
        self._mtlx_library: object | None = None
        self._mtlx_skin_material: object | None = None
        self._procedural_params: dict[int, dict] = {}
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
        return self.orbit_camera if self.camera_mode == "orbit" else self.free_camera

    def reset_camera(self) -> None:
        """Snap both cameras back to a known-good frame on the head."""
        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        self.camera_mode = "orbit"

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
            self.orbit_camera.distance = dist
            self.orbit_camera.pitch = float(np.arcsin(offset[1] / dist))
            self.orbit_camera.yaw = float(np.arctan2(offset[0], offset[2]))
            self.camera_mode = "orbit"

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
        env: Environment | None = (
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
        active_tattoo = (
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
            env_for_scene = env
            env_intensity_for_scene = float(self.env_intensity)
            direct_enabled = (self.direct_light_index == 0)
        return build_default_scene(
            environment=env_for_scene,
            env_intensity=env_intensity_for_scene,
            mesh=None,
            light_direction=self.light_direction,
            light_radiance=self.light_radiance,
            direct_light_enabled=direct_enabled,
            mm_per_unit=float(self.mm_per_unit),
            furnace_mode=is_furnace,
            pigment=active_tattoo,
            pigment_density=float(self.tattoo_density),
        )

    def _frame_camera_to_scene(self, scene: Scene) -> None:
        """Position the orbit camera from a USD authored UsdGeom.Camera
        when present, falling back to an auto-frame around the scene's
        world AABB.

        Auto-frame: target = bounds centre, distance fits the bounding
        sphere inside the vertical FOV with a small margin. Override:
        target = position + forward·focus_distance (or auto-distance when
        focusDistance is unauthored), yaw/pitch derived from forward.
        """
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

        fov_v_rad = np.radians(self.orbit_camera.fov)
        margin = 1.4
        distance = radius / np.tan(fov_v_rad * 0.5) * margin
        # Respect OrbitCamera.zoom's clamp range so user wheel-zoom stays
        # consistent with the seeded value.
        distance = float(np.clip(distance, 0.5, 50.0))

        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = 0.0
        self.orbit_camera.pitch = 0.0

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
        # Pick a focus distance: authored value if present, else aim at
        # bounds centre, else fall back to a 5-unit default.
        d = ov.focus_distance
        if d is None or d <= 1e-6:
            bounds = scene.world_bounds()
            if bounds is not None:
                amin, amax = bounds
                center = (amin + amax) * 0.5
                d = float(np.linalg.norm(center.astype(np.float32) - ov.position))
            if not d:
                d = 5.0
        d = float(np.clip(d, 0.5, 50.0))

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

        self.orbit_camera.target = target
        self.orbit_camera.distance = d
        self.orbit_camera.yaw = yaw
        self.orbit_camera.pitch = pitch
        self.orbit_camera.fov = fov_v_deg

    def _frame_camera_to_mesh(self, source: MeshSource) -> None:
        """Auto-fit orbit camera to a MeshSource's bounding box."""
        amin = source.positions.min(axis=0)
        amax = source.positions.max(axis=0)
        diag = amax - amin
        radius = float(np.linalg.norm(diag) * 0.5)
        if radius < 1e-6:
            return
        center = ((amin + amax) * 0.5).astype(np.float32)

        fov_v_rad = np.radians(self.orbit_camera.fov)
        margin = 1.4
        distance = radius / np.tan(fov_v_rad * 0.5) * margin
        distance = float(np.clip(distance, 0.5, 50.0))

        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = 0.0
        self.orbit_camera.pitch = 0.0

    def _clear_model_state(self) -> None:
        """Reset all model/scene state so a fresh load starts clean."""
        self.models.clear()
        self._mesh_sources.clear()
        self.model_index = -1
        self._usd_scene = None
        self._usd_model_index = -1
        self._usd_bake_done = None
        self._usd_uploaded_count = 0
        self._baked_source_idx = -1
        self._displacement_cache.clear()
        self._normal_cache.clear()
        self._dirty_since = None
        # Reset GPU mesh to dummy so stale geometry doesn't render
        self._upload_mesh(self._dummy_mesh)
        self._upload_detail_maps(None)

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
            from skinny.usd_loader import _read_usd_stage, bake_usd_prim
            scene, prim_data = _read_usd_stage(
                path, use_usd_mtlx_plugin=self._use_usd_mtlx_plugin,
            )
            self._usd_metadata_queue.put(scene)
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

        Slot 0 uses the current mtlx_overrides dict (seeded from
        SkinParameters defaults, edited live via mtlx.* sliders).
        Slots 1+ layer each USD-bound skin material's
        `parameter_overrides` dict on top of the global overrides.
        Slots typed FLAT or unused are zeroed.
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
        for slot in range(FLAT_MATERIAL_CAPACITY):
            if slot == 0 or slot >= len(scene_mats):
                # Slot 0 is the legacy SDF skin; padding slots get the
                # global defaults too so any stray dispatch is sane.
                out += pack_material_values(cm.uniform_block, base)
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

        # Milestone 4-a: when a USD scene was loaded, run each non-skin
        # material through the gen as a standard_surface network. This
        # builds a per-material CompiledMaterial cache the future GPU
        # integration (4-c) will use to dispatch on hit.materialId. No
        # GPU effect today; failures log but don't abort startup.
        self._mtlx_scene_materials: dict[int, object] = {}
        if self._usd_scene is not None and self._usd_scene.materials:
            ok_std = 0
            ok_mtlx = 0
            fail = 0
            total_kb = 0.0
            for i, mat in enumerate(self._usd_scene.materials):
                if i == 0:
                    continue  # legacy skin slot
                try:
                    if mat.mtlx_target_name:
                        if mat.mtlx_document is not None:
                            lib.import_document(mat.mtlx_document)
                        cm = lib.generate(
                            mat.mtlx_target_name, compile_check=False
                        )
                    else:
                        cm = lib.compile_for_scene_material(mat)
                except Exception as e:  # noqa: BLE001 - gen raises generic
                    print(f"[skinny] mat[{i}] {mat.name!r}: gen FAIL  "
                          f"{type(e).__name__}: {e}")
                    fail += 1
                    continue
                self._mtlx_scene_materials[i] = cm
                # Detect procedural materials by checking if the gen
                # emitted fractal-noise uniform fields (marble 3D, etc.).
                proc = _extract_procedural_params(cm)
                if proc is not None:
                    self._procedural_params[i] = proc
                if mat.mtlx_target_name:
                    ok_mtlx += 1
                else:
                    ok_std += 1
                total_kb += len(cm.pixel_source) / 1024.0
            if ok_std or ok_mtlx or fail:
                n_proc = len(self._procedural_params)
                print(
                    f"[skinny] MaterialX per-scene-material gen: "
                    f"{ok_std} std_surface, {ok_mtlx} mtlx-targeted, "
                    f"{fail} fail, {n_proc} procedural, "
                    f"total {total_kb:.1f}KB slang"
                )

    def _apply_usd_lights(self, scene: Scene) -> None:
        """Seed the renderer's light + environment state from a USD scene.

        Runs once at __init__ time. Picks the first DistantLight and the
        DomeLight (if any) the loader extracted, converts them into the
        renderer's UI-friendly representation (elevation/azimuth + colour
        + intensity for the analytic light, an Environment entry for the
        dome), and leaves the user's sliders fully in charge afterwards.
        """
        if scene.lights_dir:
            light = scene.lights_dir[0]
            d = np.asarray(light.direction, dtype=np.float32)
            norm = float(np.linalg.norm(d))
            if norm > 1e-6:
                d = d / norm
                # Inverse of _update_light's spherical→cartesian:
                #   d = [cos(el)·sin(az), sin(el), cos(el)·cos(az)]
                self.light_elevation = float(
                    np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0)))
                )
                self.light_azimuth = float(
                    np.degrees(np.arctan2(d[0], d[2]))
                )
            r = np.asarray(light.radiance, dtype=np.float32)
            intensity = float(np.max(r))
            if intensity > 1e-6:
                self.light_intensity = intensity
                self.light_color_r = float(r[0] / intensity)
                self.light_color_g = float(r[1] / intensity)
                self.light_color_b = float(r[2] / intensity)

        if scene.environment is not None:
            from skinny.environment import Environment
            env_hdr = scene.environment
            self.environments.append(Environment(
                name=f"USD: {env_hdr.name}",
                _data=env_hdr.data,
            ))
            self.env_index = len(self.environments) - 1
            if env_hdr.intensity > 0:
                self.env_intensity = float(env_hdr.intensity)
        else:
            # USD scene without a DomeLight: assume the scene is self-
            # contained (Cornell-style closed interior, emissive lights,
            # or just a dim direct-only setup). Mute the built-in noon-
            # sky so indirect bounces don't get swamped. The user can
            # bump env_intensity via the UI when a scene like test_scene
            # wants chrome reflections of the default env.
            self.env_intensity = 0.0

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

    def _init_gpu(self) -> None:
        # Compute pipeline from Slang
        self.pipeline = ComputePipeline(
            self.ctx,
            self.shader_dir,
            entry_module="main_pass",
            entry_point="mainImage",
        )

        # Uniform buffer — FrameConstants + SkinParams + light
        self.uniform_size = 512  # generous, std140 aligned
        self.uniform_buffer = UniformBuffer(self.ctx, self.uniform_size)

        # Per-material skin UBO array (binding 15). StructuredBuffer of
        # MtlxSkinParams, one per material slot — slot 0 mirrors the
        # legacy SDF-head SkinParameters, slots 1+ each USD-bound skin
        # material's overrides on top of the global defaults. Filled
        # per-frame in render() via _pack_mtlx_skin_array.
        # Each record is 164 scalar-layout bytes (27 fields, no vec3 padding).
        # _init_materialx_runtime may have set this already from reflection.
        if not hasattr(self, 'mtlx_skin_record_size') or self.mtlx_skin_record_size == 0:
            self.mtlx_skin_record_size = 164
        self.mtlx_skin_buffer = StorageBuffer(
            self.ctx,
            FLAT_MATERIAL_CAPACITY * self.mtlx_skin_record_size + 256,
        )
        # Seed with current SkinParameters → MaterialX defaults so the
        # buffer is valid on frame 0.
        seed = self._pack_mtlx_skin_array()
        if seed:
            self.mtlx_skin_buffer.upload_sync(seed)

        # Persistent HDR accumulation image (progressive convergence).
        # transfer_src=True so screenshot path can copy raw float radiance
        # to a host-visible staging buffer for EXR/HDR export.
        self.accum_image = StorageImage(
            self.ctx, self.width, self.height, transfer_src=True,
        )

        # Per-frame HUD overlay (R8 alpha mask rasterised by Pillow).
        # Pre-zero the staging buffer so the GPU image starts clean even if
        # render() never gets to upload before render_headless() / a
        # screenshot dispatch reads it.
        self.hud_overlay = HudOverlay(self.ctx, self.width, self.height)
        self.hud_overlay.upload(bytes(self.width * self.height))

        # HDR environment texture (RGBA32F, equirectangular).
        from skinny.environment import ENV_HEIGHT, ENV_WIDTH
        self.env_image = SampledImage(self.ctx, ENV_WIDTH, ENV_HEIGHT)
        self._ensure_env_uploaded()

        # Tattoo texture (RGBA32F, spherical UV). Seeded with a blank so the
        # descriptor is valid even before the user flips off "None".
        self.tattoo_image = SampledImage(self.ctx, TATTOO_WIDTH, TATTOO_HEIGHT)
        self.tattoo_image.upload_sync(blank_tattoo_data())
        self._ensure_tattoo_uploaded()

        # Per-model detail maps — RGBA8, 2K square. Three images cover
        # normal / roughness / displacement respectively. Seeded with
        # blanks so the descriptors are valid on frame 1.
        self.normal_image = SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM, bytes_per_pixel=4,
        )
        self.roughness_image = SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM, bytes_per_pixel=4,
        )
        self.displacement_image = SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM, bytes_per_pixel=4,
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
        self.vertex_buffer = StorageBuffer(self.ctx, v_size)
        self.index_buffer = StorageBuffer(self.ctx, i_size)
        self.bvh_buffer = StorageBuffer(self.ctx, b_size)
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
        self.instance_buffer = StorageBuffer(
            self.ctx, self.instance_capacity * INSTANCE_STRIDE + 256
        )
        self._upload_instances([np.eye(4, dtype=np.float32)], material_ids=[0])
        self._num_instances = 1

        # Flat-material parameter buffer — one record per scene material,
        # consumed when hit.materialId > 0 (skin still owns slot 0). Sized
        # for FLAT_MATERIAL_CAPACITY entries up front.
        self.flat_material_buffer = StorageBuffer(
            self.ctx, FLAT_MATERIAL_CAPACITY * FLAT_MATERIAL_STRIDE + 256
        )
        # Initialize with one zeroed record so the buffer is valid even
        # before any USD scene is loaded.
        self.flat_material_buffer.upload_sync(b"\x00" * FLAT_MATERIAL_STRIDE)
        self._num_flat_materials = 0

        # Bindless texture array (binding 14). Slots are populated lazily by
        # `_upload_flat_materials` from each Material.texture_paths entry.
        self.texture_pool = TexturePool(self.ctx)

        # Per-material type-code buffer (binding 16). One uint per slot,
        # written each time _upload_flat_materials runs. Default is all
        # MATERIAL_TYPE_SKIN so an unmapped slot routes to the skin path.
        self.material_types_buffer = StorageBuffer(
            self.ctx, FLAT_MATERIAL_CAPACITY * 4 + 16
        )
        self.material_types_buffer.upload_sync(
            b"\x00" * (FLAT_MATERIAL_CAPACITY * 4)
        )
        self._material_types: list[int] = [MATERIAL_TYPE_SKIN]

        # Sphere-light buffer (binding 17). Filled from
        # scene.lights_sphere; fc.numSphereLights bounds the active range.
        self.sphere_lights_buffer = StorageBuffer(
            self.ctx, SPHERE_LIGHT_CAPACITY * SPHERE_LIGHT_STRIDE + 16
        )
        self.sphere_lights_buffer.upload_sync(
            b"\x00" * (SPHERE_LIGHT_CAPACITY * SPHERE_LIGHT_STRIDE)
        )
        self._num_sphere_lights: int = 0

        # Emissive-triangle buffer (binding 18). Built from scene instances
        # whose material has non-zero emissiveColor. The shader samples one
        # triangle per pixel per frame for next-event estimation.
        self.emissive_tri_buffer = StorageBuffer(
            self.ctx, EMISSIVE_TRI_CAPACITY * EMISSIVE_TRI_STRIDE + 16
        )
        self.emissive_tri_buffer.upload_sync(
            b"\x00" * (EMISSIVE_TRI_CAPACITY * EMISSIVE_TRI_STRIDE)
        )
        self._num_emissive_tris: int = 0

        # StdSurfaceParams buffer (binding 19). One 256-byte record per
        # material slot, carrying the full MaterialX standard_surface input
        # set for evalStdSurfaceBSDF in mtlx_std_surface.slang.
        self.std_surface_buffer = StorageBuffer(
            self.ctx, STD_SURFACE_CAPACITY * STD_SURFACE_STRIDE + 16
        )
        self.std_surface_buffer.upload_sync(
            b"\x00" * (STD_SURFACE_CAPACITY * STD_SURFACE_STRIDE)
        )

        # ProceduralParams buffer (binding 20). One 96-byte record per
        # material slot. type==0 means no procedural eval; non-zero
        # triggers per-pixel noise evaluation in the shader.
        self.procedural_params_buffer = StorageBuffer(
            self.ctx, PROCEDURAL_PARAMS_CAPACITY * PROCEDURAL_PARAMS_STRIDE + 16
        )
        self.procedural_params_buffer.upload_sync(
            b"\x00" * (PROCEDURAL_PARAMS_CAPACITY * PROCEDURAL_PARAMS_STRIDE)
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
        from skinny.vk_compute import ReadbackBuffer
        self._offscreen_output = StorageImage(
            self.ctx, self.width, self.height,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            transfer_src=True,
        )
        self._readback = ReadbackBuffer(self.ctx, self.width, self.height)

        # Descriptor pool and sets
        self._create_descriptors()

        # Command buffers
        self.command_buffers = self.ctx.allocate_command_buffers(MAX_FRAMES_IN_FLIGHT)

        # Synchronisation
        if self.ctx.swapchain_info is not None:
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
                # env + tattoo + n/r/d (5) + bindless flat-material array.
                descriptorCount=MAX_FRAMES_IN_FLIGHT * (5 + BINDLESS_TEXTURE_CAPACITY),
            )
        )
        # Storage buffers per frame: vertices, indices, BVH nodes, TLAS
        # instances, flat-material params, material type codes,
        # per-material skin UBO array, sphere lights, emissive triangles,
        # StdSurfaceParams, ProceduralParams.
        pool_sizes.append(
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=MAX_FRAMES_IN_FLIGHT * 11,
            )
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            maxSets=MAX_FRAMES_IN_FLIGHT,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.ctx.device, pool_info, None
        )

        layouts = [self.pipeline.descriptor_set_layout] * MAX_FRAMES_IN_FLIGHT
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
            procedural_info = vk.VkDescriptorBufferInfo(
                buffer=self.procedural_params_buffer.buffer,
                offset=0,
                range=self.procedural_params_buffer.size,
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
                    dstBinding=20,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[procedural_info],
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
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _rebind_mesh_descriptors(self) -> None:
        """Re-write descriptor bindings 5, 6, 7 after buffer reallocation."""
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

        vk.vkDeviceWaitIdle(self.ctx.device)

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

        self.vertex_buffer = StorageBuffer(self.ctx, v_new)
        self.index_buffer = StorageBuffer(self.ctx, i_new)
        self.bvh_buffer = StorageBuffer(self.ctx, b_new)

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
                scene = self._usd_metadata_queue.get_nowait()
            except _queue.Empty:
                return
            self._usd_scene = scene
            self._apply_usd_lights(scene)
            self._frame_camera_to_scene(scene)
            if scene.mm_per_unit != 120.0:
                self.mm_per_unit = scene.mm_per_unit
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

        if self._usd_bake_done.is_set() and self._usd_instance_queue.empty():
            self._usd_bake_done = None
            print(
                f"[skinny] USD streaming complete — "
                f"{len(self._usd_scene.instances)} instance(s)"
            )

    def _is_usd_active(self) -> bool:
        return (
            self._usd_model_index >= 0
            and self.model_index == self._usd_model_index
        )

    def _upload_mesh(self, mesh: Mesh) -> None:
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
        if scene is None or not scene.instances:
            return
        meshes = [inst.mesh for inst in scene.instances]
        offsets = self._upload_meshes_concatenated(meshes)
        transforms = [inst.transform for inst in scene.instances]
        material_ids = [inst.material_id for inst in scene.instances]
        self._upload_instances(
            transforms,
            material_ids=material_ids,
            blas_offsets=offsets,
        )
        self._upload_flat_materials(scene.materials)
        self._upload_sphere_lights(scene.lights_sphere)
        self._upload_emissive_triangles(scene)

    def _upload_emissive_triangles(self, scene: Scene) -> None:
        """Build the emissive triangle buffer (binding 18) from scene instances.

        Walks every instance whose material has non-zero emissiveColor,
        world-transforms its source triangles, and packs each into a 64-byte
        record. The shader samples one triangle per pixel per frame for NEE.
        """
        records: list[tuple] = []
        for inst in scene.instances:
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
                records.append((p0, p1, p2, emissive, area))

        n = min(len(records), EMISSIVE_TRI_CAPACITY)
        data = bytearray()
        for i in range(n):
            p0, p1, p2, em, area = records[i]
            data += struct.pack(
                "fff f fff f fff f fff f",
                float(p0[0]), float(p0[1]), float(p0[2]), 0.0,
                float(p1[0]), float(p1[1]), float(p1[2]), 0.0,
                float(p2[0]), float(p2[1]), float(p2[2]), 0.0,
                float(em[0]), float(em[1]), float(em[2]), float(area),
            )
        while len(data) < EMISSIVE_TRI_CAPACITY * EMISSIVE_TRI_STRIDE:
            data += b"\x00" * EMISSIVE_TRI_STRIDE
        self.emissive_tri_buffer.upload_sync(
            bytes(data[: EMISSIVE_TRI_CAPACITY * EMISSIVE_TRI_STRIDE])
        )
        self._num_emissive_tris = n

    def _upload_sphere_lights(self, lights: list) -> None:
        """Pack each LightSphere into binding 17. Active count goes to
        FrameConstants.numSphereLights for the shader to bound its loop.
        """
        n = min(len(lights), SPHERE_LIGHT_CAPACITY)
        data = bytearray()
        for i in range(SPHERE_LIGHT_CAPACITY):
            if i < n:
                light = lights[i]
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

    def _upload_flat_materials(self, materials: list) -> None:
        """Pack each scene Material into the FLAT_MATERIAL_STRIDE record format
        and upload. Slot 0 is the legacy skin material — its record is never
        consumed by the shader (skin path branches before the buffer read), so
        we still emit a zeroed record to keep slot indices stable.

        Also walks each Material.texture_paths.diffuseColor (when present)
        through the bindless TexturePool so the GPU has the actual image and
        the packed record carries the resolved array slot.
        """
        if len(materials) > FLAT_MATERIAL_CAPACITY:
            raise ValueError(
                f"material count {len(materials)} exceeds "
                f"capacity {FLAT_MATERIAL_CAPACITY}"
            )
        data = bytearray()
        types: list[int] = []
        for i, mat in enumerate(materials):
            # Slot 0 is the legacy SDF-head skin material. Slots > 0 are
            # USD-bound materials; classify by mtlx_target_name (skin if
            # the author hinted at a layered-skin MaterialX target, else
            # flat-material).
            if i == 0:
                types.append(MATERIAL_TYPE_SKIN)
                data += b"\x00" * FLAT_MATERIAL_STRIDE
                continue
            if mat.mtlx_target_name == "M_skinny_skin_default":
                types.append(MATERIAL_TYPE_SKIN)
                # Skin materials don't read the flat-material record; pad
                # the slot so indexing stays aligned.
                data += b"\x00" * FLAT_MATERIAL_STRIDE
                continue
            types.append(MATERIAL_TYPE_FLAT)
            indices: dict[str, int] = {
                "diffuseColor":  TexturePool.SENTINEL,
                "roughness":     TexturePool.SENTINEL,
                "metallic":      TexturePool.SENTINEL,
                "normal":        TexturePool.SENTINEL,
                "emissiveColor": TexturePool.SENTINEL,
            }
            for input_name in indices:
                tex_path = mat.texture_paths.get(input_name)
                if tex_path is not None:
                    indices[input_name] = self.texture_pool.add_or_get(tex_path)
            data += pack_flat_material(
                mat,
                diffuse_texture_idx=indices["diffuseColor"],
                roughness_texture_idx=indices["roughness"],
                metallic_texture_idx=indices["metallic"],
                normal_texture_idx=indices["normal"],
                emissive_texture_idx=indices["emissiveColor"],
            )
        if not data:
            data += b"\x00" * FLAT_MATERIAL_STRIDE
            types.append(MATERIAL_TYPE_SKIN)
        self.flat_material_buffer.upload_sync(bytes(data))
        self._num_flat_materials = len(materials)
        self._material_types = types
        self._upload_material_types()
        # Pack StdSurfaceParams for every material slot into binding 19.
        # Skin-typed slots get zeroed records (the shader dispatches to the
        # skin path before reading stdSurfaceParams); flat-typed slots get
        # the full standard_surface parameter set.
        ss_data = bytearray()
        for i, mat in enumerate(materials):
            if i < len(types) and types[i] == MATERIAL_TYPE_FLAT:
                ss_data += pack_std_surface_params(mat)
            else:
                ss_data += b"\x00" * STD_SURFACE_STRIDE
        while len(ss_data) < STD_SURFACE_CAPACITY * STD_SURFACE_STRIDE:
            ss_data += b"\x00" * STD_SURFACE_STRIDE
        self.std_surface_buffer.upload_sync(
            bytes(ss_data[: STD_SURFACE_CAPACITY * STD_SURFACE_STRIDE])
        )
        # Pack ProceduralParams for every material slot into binding 20.
        proc_data = bytearray()
        for i in range(PROCEDURAL_PARAMS_CAPACITY):
            params = self._procedural_params.get(i)
            proc_data += pack_procedural_params(params)
        self.procedural_params_buffer.upload_sync(
            bytes(proc_data[: PROCEDURAL_PARAMS_CAPACITY * PROCEDURAL_PARAMS_STRIDE])
        )
        # Refresh binding-14 descriptor writes so newly populated slots
        # are visible to the shader.
        self._update_texture_pool_descriptors()

    def _upload_material_types(self) -> None:
        """Pack per-material type+flags into binding 16.

        Encoding per slot (uint):
          bits  0-7 : material type code (skin=0, flat=1)
          bits  8-9 : scatter mode for skin slots (bit0=BSSRDF, bit1=volume)
          bits 10-31: reserved.

        Re-uploaded whenever the global scatter mode changes (cheap — the
        buffer is FLAT_MATERIAL_CAPACITY uints).
        """
        scatter_idx = int(np.clip(self.scatter_index, 0, len(self._scatter_mode_bits) - 1))
        scatter_bits = int(self._scatter_mode_bits[scatter_idx]) & 0x3
        types = self._material_types
        type_bytes = bytearray()
        for i in range(FLAT_MATERIAL_CAPACITY):
            t = int(types[i]) if i < len(types) else MATERIAL_TYPE_SKIN
            packed = (t & 0xFF)
            if t == MATERIAL_TYPE_SKIN:
                packed |= (scatter_bits & 0x3) << 8
            type_bytes += struct.pack("I", packed)
        self.material_types_buffer.upload_sync(bytes(type_bytes))
        self._last_scatter_index = scatter_idx

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
        self._upload_flat_materials(mats)
        self._material_version += 1

    def _update_texture_pool_descriptors(self) -> None:
        """Push the current TexturePool slots into binding 14 (bindless
        textures) for every descriptor set. PARTIALLY_BOUND lets unfilled
        slots stay invalid — the shader gates reads behind a sentinel idx.
        """
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

    def _upload_meshes_concatenated(
        self, meshes: list[Mesh]
    ) -> list[tuple[int, int, int]]:
        """Pack multiple meshes back-to-back into the unified GPU buffers.

        Returns one (node_offset, triangle_offset, vertex_offset) per mesh
        — the same triple the shader's Instance record holds. The shader
        adds these offsets to the BLAS-local indices stored in each mesh's
        BVH / index / vertex bytes.

        Used by the USD path; the legacy single-mesh OBJ rebake stays on
        `_upload_mesh` since it always lives at offsets (0, 0, 0).
        """
        v_chunks: list[bytes] = []
        i_chunks: list[bytes] = []
        b_chunks: list[bytes] = []
        offsets: list[tuple[int, int, int]] = []
        v_off = 0
        t_off = 0
        n_off = 0
        for mesh in meshes:
            offsets.append((n_off, t_off, v_off))
            v_chunks.append(mesh.vertex_bytes)
            i_chunks.append(mesh.index_bytes)
            b_chunks.append(mesh.bvh_bytes)
            v_off += mesh.num_vertices
            t_off += mesh.num_triangles
            n_off += mesh.num_nodes
        self._ensure_mesh_buffer_capacity(v_off, t_off, n_off)
        self.vertex_buffer.upload_sync(b"".join(v_chunks))
        self.index_buffer.upload_sync(b"".join(i_chunks))
        self.bvh_buffer.upload_sync(b"".join(b_chunks))
        return offsets

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
            raise ValueError(
                f"instance count {len(transforms)} exceeds "
                f"capacity {self.instance_capacity}"
            )

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

    def _pack_uniforms(self) -> bytes:
        aspect = self.width / self.height
        view_inv = np.linalg.inv(self.camera.view_matrix())
        proj_inv = np.linalg.inv(self.camera.projection_matrix(aspect))

        data = bytearray()
        # FrameConstants: viewInverse (mat4), projInverse (mat4), position (vec3),
        # fov, frameIndex, accumFrame, time, width, height
        data += view_inv.astype(np.float32).tobytes()       # 64 bytes
        data += proj_inv.astype(np.float32).tobytes()        # 64 bytes
        data += self.camera.position.tobytes()               # 12 bytes
        data += struct.pack("f", self.camera.fov)            # 4 bytes
        data += struct.pack("I", self.frame_index)           # 4 bytes
        data += struct.pack("I", self.accum_frame)           # 4 bytes
        data += struct.pack("f", self.time_elapsed)          # 4 bytes
        data += struct.pack("II", self.width, self.height)   # 8 bytes
        use_direct = 1 if self.direct_light_index == 0 else 0
        data += struct.pack("I", use_direct)                 # 4 bytes
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
        env_intensity = (
            self.scene.environment.intensity if self.scene.environment is not None
            else float(self.env_intensity)
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
        data += struct.pack(
            "I",
            int(np.clip(self.integrator_index, 0, len(self.integrator_modes) - 1)),
        )                                                            # 4 bytes
        # TLAS instance count consumed by mesh_head.slang::marchHeadMesh.
        # When useMesh==0 the shader skips marchHeadMesh entirely; we still
        # write the count for completeness so the field is always defined.
        data += struct.pack("I", int(self._num_instances))           # 4 bytes
        # Active sphere-light count (bounds the shader's loop).
        data += struct.pack("I", int(self._num_sphere_lights))        # 4 bytes
        # Active emissive-triangle count (bounds the shader's NEE loop).
        data += struct.pack("I", int(self._num_emissive_tris))        # 4 bytes
        # No padding here — scalar layout (`-fvk-use-scalar-layout`) aligns
        # the next field (float3 lightDirection) at 4 bytes, not 16.

        # Light — pulled from the per-frame Scene so direct-light disable
        # cleanly emits zero-radiance (the shader's
        # `dot(lightRadiance, lightRadiance) > eps` gate then no-ops the
        # analytic NEE). Multi-light scenes will iterate
        # `self.scene.lights_dir` here once the shader is updated to
        # consume a light array (Phase E).
        if self.scene.lights_dir:
            primary = self.scene.lights_dir[0]
            data += primary.direction.tobytes()  # 12 bytes (float3, scalar-aligned)
            data += primary.radiance.tobytes()   # 12 bytes (float3, scalar-aligned)
        else:
            data += np.zeros(3, dtype=np.float32).tobytes()
            data += np.zeros(3, dtype=np.float32).tobytes()

        return bytes(data)

    def _ensure_env_uploaded(self) -> None:
        """Upload current env to GPU if it has changed (called once per switch).

        Reads the environment data from `self.scene.environment` (populated
        by `_build_scene_from_state`); env_index continues to drive the
        change-detection cache key so we don't re-upload every frame.
        """
        # Cache key combines env selection AND furnace toggle so the
        # constant-white furnace IBL gets uploaded when the user toggles
        # furnace on/off (env_index alone wouldn't trip the change check).
        cache_key = (int(self.env_index), int(self.furnace_index))
        if cache_key == self._last_env_index:
            return
        self.env_index = int(np.clip(self.env_index, 0, len(self.environments) - 1))
        env_hdr = self.scene.environment
        if env_hdr is None:
            # Should not happen at runtime — _build_scene_from_state always
            # produces an env when env_index is valid. Defensive fallback.
            env_hdr_data = self.environments[self.env_index].data
        else:
            env_hdr_data = env_hdr.data
        self.env_image.upload_sync(env_hdr_data)
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
            float(self.env_intensity),
            int(self.furnace_index),
            float(self.mm_per_unit),
            int(self.detail_maps_index),
            float(self.normal_map_strength),
            float(self.displacement_scale_mm),
            int(self.preset_index),
            int(self._material_version),
            # E-4: user-direct MaterialX field overrides — sort for stable hash
            tuple(sorted(
                (k, _hashable_value(v)) for k, v in self.mtlx_overrides.items()
            )),
        )
        return hash(parts)

    def update(self, dt: float) -> None:
        self.time_elapsed += dt
        self.frame_index += 1

        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps_smooth = (
                inst_fps if self._fps_smooth == 0 else self._fps_smooth * 0.9 + inst_fps * 0.1
            )

        # Recompute light direction + radiance from current slider state so
        # _build_scene_from_state picks up intensity / colour / angle changes.
        self._update_light()

        # Refresh the per-frame Scene snapshot before any GPU upload path
        # reads from it. Cheap (a few attribute copies); rebuilt every
        # frame so UI changes propagate without an explicit notification.
        self.scene = self._build_scene_from_state()

        # If the environment selection changed, re-upload the HDR texture.
        self._ensure_env_uploaded()
        # Pick up USD meshes that finished baking in the background.
        self._poll_usd_streaming()
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
        else:
            self.accum_frame += 1

    def render(self) -> None:
        f = self.current_frame

        vk.vkWaitForFences(
            self.ctx.device, 1, [self.in_flight_fences[f]], vk.VK_TRUE, 2**64 - 1
        )
        vk.vkResetFences(self.ctx.device, 1, [self.in_flight_fences[f]])

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

        self.current_frame = (f + 1) % MAX_FRAMES_IN_FLIGHT

    def render_headless(self) -> bytes:
        """Render one frame to an offscreen image and return raw RGBA8 pixels.

        Works in both headless and windowed modes — binding 1 (storage
        image output) is rewritten to ``_offscreen_output`` here so a
        windowed session that just rebound binding 1 to a swapchain image
        in render() doesn't corrupt the screenshot.
        """
        f = self.current_frame

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

        from skinny.vk_compute import ReadbackBuffer

        vk.vkDeviceWaitIdle(self.ctx.device)

        self._offscreen_output.destroy()
        self._readback.destroy()
        self.accum_image.destroy()
        self.hud_overlay.destroy()

        self.width = width
        self.height = height
        self.ctx.width = width
        self.ctx.height = height

        self._offscreen_output = StorageImage(
            self.ctx, width, height,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            transfer_src=True,
        )
        self._readback = ReadbackBuffer(self.ctx, width, height)
        self.accum_image = StorageImage(
            self.ctx, width, height, transfer_src=True,
        )
        self.hud_overlay = HudOverlay(self.ctx, width, height)
        self.hud_overlay.upload(bytes(width * height))

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
        from skinny.vk_compute import ReadbackBuffer

        vk.vkDeviceWaitIdle(self.ctx.device)

        rb = ReadbackBuffer(
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
            rgb = (arr[..., :3] / float(samples)).astype(np.float32)
            import imageio.v3 as iio
            ext = ".exr" if fmt == "exr" else ".hdr"
            if hasattr(path_or_file, "write"):
                # FreeImage backend can't write directly to a file-like
                # object; round-trip through a tempfile so EXR/HDR works
                # for the web download path.
                import os
                import tempfile
                fd, tmp_path = tempfile.mkstemp(suffix=ext)
                os.close(fd)
                try:
                    iio.imwrite(tmp_path, rgb, extension=ext)
                    with open(tmp_path, "rb") as fh:
                        path_or_file.write(fh.read())
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            else:
                iio.imwrite(str(path_or_file), rgb, extension=ext)
            return

        raise ValueError(f"Unsupported screenshot format: {fmt!r}")

    @staticmethod
    def screenshot_format_options() -> list[str]:
        """User-facing format names — order is also the GUI dropdown order."""
        return ["PNG", "JPEG", "BMP", "EXR", "HDR"]

    def cleanup(self) -> None:
        vk.vkDeviceWaitIdle(self.ctx.device)

        for sem in self.image_available + self.render_finished:
            vk.vkDestroySemaphore(self.ctx.device, sem, None)
        for fence in self.in_flight_fences:
            vk.vkDestroyFence(self.ctx.device, fence, None)

        vk.vkDestroyDescriptorPool(self.ctx.device, self.descriptor_pool, None)
        self.texture_pool.destroy()
        self.procedural_params_buffer.destroy()
        self.std_surface_buffer.destroy()
        self.emissive_tri_buffer.destroy()
        self.sphere_lights_buffer.destroy()
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
        self.hud_overlay.destroy()
        self.accum_image.destroy()
        self._offscreen_output.destroy()
        self._readback.destroy()
        self.uniform_buffer.destroy()
        self.mtlx_skin_buffer.destroy()
        self.pipeline.destroy()
