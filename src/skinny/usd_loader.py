"""USD stage → skinny Scene loader.

Phase D-1 scope (geometry only):
  - Open a USD stage with usd-core (pxr.Usd / pxr.UsdGeom).
  - Walk every UsdGeom.Mesh, pull points / faceVertexCounts /
    faceVertexIndices / normals / primvars:st and bake the result through
    `mesh.bake_mesh` (no displacement) into a `Mesh`.
  - Compute each prim's local-to-world transform via UsdGeom.Xformable.
  - Wrap one `MeshInstance` per mesh with a single placeholder `Material`.
  - Convert the stage's metersPerUnit into the renderer's mm_per_unit.

Materials, lights, cameras, and displacement come in subsequent Phase D
steps. The default material is a flat fallback (matId=0); meshes without
an explicit material binding share it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel, UsdVol

from skinny.environment import _load_radiance_hdr, _resize_equirect
from skinny.mesh import MeshSource, bake_mesh, compute_source_hash
from skinny.usd_gprims import tessellate_gprim
from skinny.mesh_cache import (
    load_cache_index,
    lookup_cached_mesh,
    make_cache_key,
    save_cached_mesh,
)
from skinny.scene import (
    LensElement,
    LensSystem,
    LightDir,
    LightEnvHDR,
    Material,
    MeshInstance,
    Scene,
    TextureBinding,
    VolumeGrid,
)

try:
    import MaterialX as mx

    _HAS_MATERIALX = True
except ImportError:
    _HAS_MATERIALX = False


# ─── Geometry extraction ──────────────────────────────────────────────


def _triangulate(
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
) -> np.ndarray:
    """Fan-triangulate a polygon mesh.

    A face with N vertices (v0, v1, ..., v_{N-1}) becomes (N-2) triangles
    (v0, v1, v2), (v0, v2, v3), ..., (v0, v_{N-2}, v_{N-1}).
    Returns an (T, 3) int32 array.
    """
    tris: list[tuple[int, int, int]] = []
    cursor = 0
    for count in face_vertex_counts:
        n = int(count)
        if n < 3:
            cursor += n
            continue
        v0 = int(face_vertex_indices[cursor])
        for i in range(1, n - 1):
            v1 = int(face_vertex_indices[cursor + i])
            v2 = int(face_vertex_indices[cursor + i + 1])
            tris.append((v0, v1, v2))
        cursor += n
    return np.asarray(tris, dtype=np.int32) if tris else np.zeros((0, 3), np.int32)


def _read_mesh_attrs(prim: Usd.Prim, time: Usd.TimeCode) -> Optional[MeshSource]:
    """Pull a `MeshSource` out of a UsdGeom.Mesh prim.

    Returns None for malformed prims (missing points or empty face list)
    so the caller can skip them.
    """
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(time)
    if points is None or len(points) == 0:
        return None
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time)
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    if face_vertex_counts is None or face_vertex_indices is None:
        return None
    if len(face_vertex_counts) == 0 or len(face_vertex_indices) == 0:
        return None

    positions = np.asarray(points, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        return None

    fvc = np.asarray(face_vertex_counts, dtype=np.int32)
    fvi = np.asarray(face_vertex_indices, dtype=np.int32)
    tri_idx = _triangulate(fvc, fvi)
    if tri_idx.shape[0] == 0:
        return None

    # Normals: prefer authored values, fall back to triangle-area-weighted
    # smooth normals (mesh.py already has the helper, but it's underscore-
    # prefixed; we replicate the simple variant inline to avoid a private
    # import). Authored UsdGeom normals can be per-vertex or per-face-vertex
    # depending on interpolation; for now we accept only per-vertex.
    normals_attr = mesh.GetNormalsAttr().Get(time)
    interpolation = mesh.GetNormalsInterpolation()
    normals: np.ndarray
    if normals_attr is not None and interpolation == UsdGeom.Tokens.vertex:
        normals = np.asarray(normals_attr, dtype=np.float32)
        if normals.shape != positions.shape:
            normals = _smooth_normals(positions, tri_idx)
    else:
        normals = _smooth_normals(positions, tri_idx)

    # UVs: primvars:st by convention. Vertex interpolation (one UV per
    # mesh vertex) is the simple case. faceVarying interpolation (one UV
    # per face-vertex slot) requires expanding the mesh: each face-vertex
    # becomes its own (position, normal, uv) so adjacent faces can carry
    # different UVs at a shared corner. After expansion we recompute
    # normals from the new positions + tri_idx, which yields faceted
    # shading on hard-edged geometry like cubes.
    pv_api = UsdGeom.PrimvarsAPI(prim)
    st_pv = pv_api.GetPrimvar("st")
    uvs: np.ndarray = np.zeros((positions.shape[0], 2), dtype=np.float32)
    if st_pv:
        st_interp = st_pv.GetInterpolation()
        # USD primvars can be authored as (values + indices); the value array
        # then has fewer entries than the consumer expects. `ComputeFlattened`
        # de-indexes so we always see one value per vertex / face-vertex slot.
        st_vals = st_pv.ComputeFlattened(time)
        if st_vals is None:
            st_vals = st_pv.Get(time)
        if (st_interp == UsdGeom.Tokens.vertex
            and st_vals is not None
            and len(st_vals) == positions.shape[0]):
            arr = np.asarray(st_vals, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                uvs = arr
        elif (st_interp == UsdGeom.Tokens.faceVarying
              and st_vals is not None
              and len(st_vals) == fvi.shape[0]):
            arr = np.asarray(st_vals, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                # Expand: each face-vertex slot becomes its own vertex.
                positions = positions[fvi].astype(np.float32)
                # Re-triangulate in face-vertex space (sequential indices).
                expanded_tris: list[tuple[int, int, int]] = []
                cursor = 0
                for count in fvc:
                    n = int(count)
                    if n < 3:
                        cursor += n
                        continue
                    for i in range(1, n - 1):
                        expanded_tris.append((cursor, cursor + i, cursor + i + 1))
                    cursor += n
                tri_idx = (
                    np.asarray(expanded_tris, dtype=np.int32)
                    if expanded_tris
                    else np.zeros((0, 3), np.int32)
                )
                normals = _smooth_normals(positions, tri_idx)
                uvs = arr

    # USD puts V=0 at the bottom of the texture; Vulkan + image loaders
    # put V=0 at the top.  Flip to match the OBJ loader convention.
    if uvs is not None and uvs.size > 0:
        uvs = uvs.copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]

    return MeshSource(
        name=str(prim.GetPath()),
        positions=positions,
        normals=normals,
        uvs=uvs,
        tri_idx=tri_idx,
    )


def _smooth_normals(positions: np.ndarray, tri_idx: np.ndarray) -> np.ndarray:
    """Triangle-area-weighted vertex normals.

    Self-contained mirror of mesh.py's private `_smooth_normals` — we don't
    reach into private helpers from another module.
    """
    n = positions.shape[0]
    accum = np.zeros((n, 3), dtype=np.float32)
    for i0, i1, i2 in tri_idx:
        a = positions[i0]
        b = positions[i1]
        c = positions[i2]
        face = np.cross(b - a, c - a)
        accum[i0] += face
        accum[i1] += face
        accum[i2] += face
    lengths = np.linalg.norm(accum, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-8, 1.0, lengths)
    return (accum / lengths).astype(np.float32)


# ─── Transform extraction ─────────────────────────────────────────────


_VALID_CHANNELS = {"rgb", "r", "g", "b", "a"}
_VALID_WRAP_MODES = {"repeat", "clamp", "mirror", "black", "useMetadata"}


def _normalize_channel(name: str) -> str:
    """Map a UsdUVTexture `outputs:<name>` connection name to our canonical
    channel string. UsdShade exposes a vector output as `rgb` and scalars
    as `r/g/b/a`; MaterialX-style `xyz` collapses to `rgb`."""
    if not name:
        return "rgb"
    lower = name.lower()
    if lower in _VALID_CHANNELS:
        return lower
    if lower in ("xyz", "rgba"):
        return "rgb"
    if lower in ("x",): return "r"
    if lower in ("y",): return "g"
    if lower in ("z",): return "b"
    if lower in ("w",): return "a"
    return "rgb"


def _normalize_wrap(name: str) -> str:
    if not name:
        return "repeat"
    if name in _VALID_WRAP_MODES:
        return name
    # USD also recognises `mirror` / `black`; default unknown values to repeat.
    return "repeat"


def _read_vec4_input(shader: UsdShade.Shader, name: str,
                     default: tuple[float, float, float, float]
                     ) -> tuple[float, float, float, float]:
    """Read a float4 (or float3 promoted) shader input as a 4-tuple. USD's
    UsdUVTexture authors `bias` / `scale` as float4 — we tolerate float3
    by appending the default's w component."""
    inp = shader.GetInput(name)
    if inp is None:
        return default
    v = inp.Get()
    if v is None:
        return default
    try:
        if len(v) >= 4:
            return float(v[0]), float(v[1]), float(v[2]), float(v[3])
        if len(v) == 3:
            return float(v[0]), float(v[1]), float(v[2]), default[3]
        if len(v) == 2:
            return float(v[0]), float(v[1]), default[2], default[3]
        if len(v) == 1:
            return float(v[0]), default[1], default[2], default[3]
    except (TypeError, ValueError):
        pass
    return default


def _resolve_texture_binding(input_obj: UsdShade.Input) -> Optional[TextureBinding]:
    """Walk a connected shader input back to a UsdUVTexture and capture all
    sampler-relevant parameters: file path, scale/bias remap, channel
    selector (from `outputs:<name>`), `sourceColorSpace`, and per-axis
    wrap modes.

    Returns None when the input is unconnected or the source shader has no
    resolvable `file` asset. Defaults align with the UsdUVTexture spec so
    older authoring (or MaterialX `<image>` fallbacks) keeps working.
    """
    if not input_obj.HasConnectedSource():
        return None

    src_info = input_obj.GetConnectedSource()
    if not src_info:
        return None
    src_api, src_output_name, _src_type = src_info
    src_prim = src_api.GetPrim()
    if not src_prim:
        return None

    src_shader = UsdShade.Shader(src_prim)
    if not src_shader:
        return None

    file_input = src_shader.GetInput("file")
    if not file_input:
        return None

    asset = file_input.Get()
    if asset is None:
        return None
    asset_path = (
        getattr(asset, "resolvedPath", None) or getattr(asset, "path", None) or str(asset)
    )
    if not asset_path:
        return None

    bias = _read_vec4_input(src_shader, "bias", (0.0, 0.0, 0.0, 0.0))
    scale = _read_vec4_input(src_shader, "scale", (1.0, 1.0, 1.0, 1.0))

    # sourceColorSpace is authored as a token on UsdUVTexture. Anything
    # outside the standard set ("sRGB" / "raw") is treated as "auto".
    cs_input = src_shader.GetInput("sourceColorSpace")
    cs_val = cs_input.Get() if cs_input is not None else None
    if cs_val in ("sRGB", "raw", "auto"):
        color_space = str(cs_val)
    else:
        color_space = "auto"

    def _wrap_attr(name: str) -> str:
        a = src_shader.GetInput(name)
        return _normalize_wrap(str(a.Get())) if (a is not None and a.Get() is not None) else "repeat"

    return TextureBinding(
        path=Path(asset_path),
        bias=bias,
        scale=scale,
        channel=_normalize_channel(str(src_output_name) if src_output_name else "rgb"),
        source_color_space=color_space,
        wrap_s=_wrap_attr("wrapS"),
        wrap_t=_wrap_attr("wrapT"),
    )


def _resolve_texture_input(input_obj: UsdShade.Input) -> Optional[Path]:
    """Back-compat wrapper: returns just the resolved file path."""
    binding = _resolve_texture_binding(input_obj)
    return binding.path if binding is not None else None


def _resolve_connected_value(inp: UsdShade.Input) -> object:
    """Resolve the constant a connected shader input ultimately produces.

    Returns None when the connection has no authored constant upstream
    (e.g. a procedural node graph output), so the caller falls back to the
    shader's default for that input.
    """
    try:
        producers = inp.GetValueProducingAttributes()
    except Exception:
        producers = []
    for attr in producers:
        v = attr.Get()
        if v is not None:
            return v
    return None


def _store_shader_override(overrides: dict, name: str, value: object) -> None:
    """Record a shader-input value under its raw (MaterialX) name and, when
    one exists, its FlatMaterialParams/UsdPreviewSurface alias.

    The dual-author pattern matches `_load_mtlx_materials`: `pack_flat_material`
    reads UsdPreviewSurface names (`diffuseColor`/`roughness`/…) while
    `pack_std_surface_params` reads the canonical standard_surface names.

    OpenPBR inputs are first folded onto their standard_surface equivalent
    (`transmission_weight`→`transmission`, …) so the downstream packers — which
    only understand standard_surface — see the authored value instead of a
    default.
    """
    overrides[name] = value
    std_name = _OPENPBR_TO_STD_SURFACE.get(name, name)
    if std_name != name:
        overrides[std_name] = value
    flat_key = _STD_SURFACE_TO_FLAT.get(std_name)
    if flat_key:
        overrides[flat_key] = value


def _opacity_is_fully_opaque(value: object) -> bool:
    """True when an authored `opacity` is the default fully-opaque 1.

    A `standard_surface`/OpenPBR shader ALWAYS authors `opacity` — its MaterialX
    default is the fully-opaque `1` (scalar) / `(1, 1, 1)` (color3). That default
    must not be mistaken for a genuine cutout, so the transmission→opacity bridge
    can still fire on it. A scalar `>= 1`, or a per-channel value whose every
    channel is `>= 1`, counts as fully opaque; any channel `< 1` is a cutout.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) >= 1.0
    try:
        return all(float(c) >= 1.0 for c in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _derive_opacity_from_transmission(overrides: dict) -> None:
    """Lower `opacity` to `1 - transmission` when a material is transmissive.

    The flat/std-surface render path only refracts through surfaces whose
    `opacity < 1` (flat_material.slang). standard_surface / OpenPBR express
    glass via a `transmission` weight instead, so without this bridge the
    surface stays opaque. No-op when `transmission` is absent/zero.

    A `standard_surface` shader ALWAYS authors `opacity` (its MaterialX default
    is the fully-opaque `(1, 1, 1)`), so the bridge must NOT skip on the mere
    presence of an authored opacity — that default would block every glass
    (the MaterialX `glass_caustics_test` sphere rendered opaque). It skips only
    when a GENUINE cutout opacity is authored (some channel `< 1`, e.g. an
    OpenPBR `geometry_opacity` alpha): that cutout wins over transmission. A
    default-opaque opacity is overwritten by `1 - transmission`.
    """
    t = overrides.get("transmission")
    if not isinstance(t, (int, float)) or isinstance(t, bool):
        return
    if float(t) <= 0.0:
        return
    existing = overrides.get("opacity")
    if existing is not None and not _opacity_is_fully_opaque(existing):
        return
    overrides["opacity"] = max(0.0, 1.0 - float(t))


def _has_subsurface_medium(overrides: dict) -> bool:
    """True when a non-zero subsurface interior medium (σ_a/σ_s, mm⁻¹) is set.

    Mirrors ``renderer._material_is_subsurface`` so the opacity-bridge and the
    MATERIAL_TYPE_SUBSURFACE routing agree on what a "subsurface material" is.
    """
    def _nonzero(v: object) -> bool:
        if v is None or isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return float(v) > 0.0
        try:
            return any(float(c) > 0.0 for c in v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    return (_nonzero(overrides.get("subsurface_sigma_a"))
            or _nonzero(overrides.get("subsurface_sigma_s")))


def _derive_opacity_from_subsurface(overrides: dict) -> None:
    """Open the flat refraction gate (`opacity = 0`) for a subsurface material.

    pbrt's `subsurface` material is a transmissive dielectric boundary around a
    homogeneous interior; it routes to MATERIAL_TYPE_SUBSURFACE (the volumetric
    interior walk) and carries σ_a/σ_s coefficients. skinny's flat path only
    refracts through surfaces whose `opacity < 1` (flat_material.slang), so the
    boundary must drop to opacity 0 or it renders solid (the SSS dragon was
    opaque white). Mirrors the UsdPreviewSurface export, which authors
    `opacity = 0` for such a material directly.

    Crucially this fires ONLY for a genuine volumetric medium (σ_a/σ_s present).
    A plain Autodesk standard_surface `subsurface` *weight* (e.g. the
    three_materials_demo marble: `subsurface = 0.4` with no medium) is a
    diffuse-SSS shading term routed to MATERIAL_TYPE_FLAT — forcing opacity=0
    there would turn it into clear glass that refracts the environment (the
    "marble totally broken" regression). No-op when `subsurface` is absent/zero,
    when no interior medium is set, or when an explicit `opacity` was authored.
    """
    s = overrides.get("subsurface")
    if not isinstance(s, (int, float)) or isinstance(s, bool):
        return
    if float(s) <= 0.0 or "opacity" in overrides:
        return
    if not _has_subsurface_medium(overrides):
        return
    overrides["opacity"] = 0.0


def _canonicalize_coat(overrides: dict) -> None:
    """Fold UsdPreviewSurface `clearcoat`/`clearcoatRoughness` onto the canonical
    `coat`/`coat_roughness` slots the FlatMaterial packer reads.

    `pack_flat_material` / `pack_std_surface_params` read `coat`/`coat_roughness`
    (the standard_surface names). The UsdPreviewSurface export authors a coat as
    `clearcoat`/`clearcoatRoughness`, so without this fold the coat lobe is
    silently dropped on the UsdPreviewSurface load path while the `-mtlx` path
    (which authors `coat` directly) keeps it — the two exports then disagree.
    No-op when no clearcoat was authored, and never clobbers an explicit `coat`.
    """
    cc = overrides.get("clearcoat")
    if isinstance(cc, (int, float)) and not isinstance(cc, bool) and "coat" not in overrides:
        overrides["coat"] = float(cc)
    ccr = overrides.get("clearcoatRoughness")
    if (
        isinstance(ccr, (int, float))
        and not isinstance(ccr, bool)
        and "coat_roughness" not in overrides
    ):
        overrides["coat_roughness"] = float(ccr)


def _extract_material(shade_mat: UsdShade.Material) -> Material:
    """Build a skinny Material from a bound UsdShade.Material.

    Captures:
      - the material prim's leaf name,
      - any authored shader-input scalars/colours that the connected
        surface shader exposes (stored verbatim in `parameter_overrides`),
      - texture file paths for any input connected to a UsdUVTexture (or
        another shader carrying a `file` asset input — covers MaterialX
        `<image>` nodes), keyed by input name.

    Phase D-3+ extends `mtlx_document` to carry the parsed surface graph;
    Phase C-4 maps `texture_paths` to bindless sampler slots.
    """
    name = shade_mat.GetPath().name

    # Prefer a MaterialX-flavored surface output when present. The render-
    # context-aware lookup picks `outputs:mtlx:surface` over the generic
    # `outputs:surface` when both exist, which matches USD's standard
    # render-context priority.
    surface_output = shade_mat.GetSurfaceOutput("mtlx")
    if not surface_output:
        surface_output = shade_mat.GetSurfaceOutput()

    overrides: dict[str, object] = {}
    textures: dict[str, Path] = {}
    bindings: dict[str, TextureBinding] = {}
    if surface_output and surface_output.HasConnectedSource():
        # GetConnectedSource() returns (source, sourceName, sourceType) or
        # None on older PyUSD; ComputeSurfaceSource is the modern
        # convenience that walks NodeGraph indirections too.
        connected_shader, _output_name, _output_type = (
            shade_mat.ComputeSurfaceSource("mtlx") or (None, None, None)
        )
        if not connected_shader:
            connected_shader, _output_name, _output_type = (
                shade_mat.ComputeSurfaceSource() or (None, None, None)
            )
        if connected_shader:
            shader = UsdShade.Shader(connected_shader)
            for inp in shader.GetInputs():
                base = inp.GetBaseName()
                if inp.HasConnectedSource():
                    binding = _resolve_texture_binding(inp)
                    if binding is not None:
                        # Remap the shader-input name to the flat/UsdPreviewSurface
                        # key the renderer's flat texture binder reads. A composed
                        # standard_surface (usdMtlx-plugin path) names its inputs
                        # `base_color`/`specular_roughness`/… (MaterialX), but
                        # `_upload_flat_materials` only looks up `diffuseColor`/
                        # `roughness`/… — so an unremapped texture is silently
                        # dropped and the input falls back to its constant default
                        # (a diffuse imagemap rendered as flat grey). Mirror the
                        # OpenPBR→std→flat fold `_store_shader_override` applies to
                        # constants. Plain UsdPreviewSurface names aren't in either
                        # table, so `.get(name, name)` leaves them untouched.
                        std_name = _OPENPBR_TO_STD_SURFACE.get(base, base)
                        flat_key = _STD_SURFACE_TO_FLAT.get(std_name, std_name)
                        textures[flat_key] = binding.path
                        bindings[flat_key] = binding
                        continue
                    # Non-texture connection: the shader input is wired up to
                    # a Material-level interface input that carries the
                    # authored constant. This is the OpenPBR /
                    # standard_surface convention the materialxusd exporter
                    # emits — every input `.connect`s to the Material, so
                    # `inp.Get()` is None and the value lives upstream.
                    value = _resolve_connected_value(inp)
                else:
                    value = inp.Get()
                if value is None:
                    continue
                _store_shader_override(overrides, base, value)

    # Author-controlled hint for routing this material through skinny's
    # MaterialX library instead of the auto-generated standard_surface.
    # Authors set:
    #   customData = {
    #       string skinnyMaterialX = "M_skinny_skin_default"
    #       dictionary skinnyOverrides = {
    #           float layer_top_melanin = 0.5
    #       }
    #   }
    # on the Material prim. skinnyMaterialX names the target in the
    # MaterialLibrary doc; skinnyOverrides supplies per-material values
    # keyed by MaterialX input name (merged into parameter_overrides so
    # the renderer's mtlxSkin-array packer picks them up).
    mtlx_target_name: Optional[str] = None
    python_module: Optional[str] = None
    cd = shade_mat.GetPrim().GetCustomData()
    if cd:
        hint = cd.get("skinnyMaterialX")
        if isinstance(hint, str) and hint:
            mtlx_target_name = hint
        py_hint = cd.get("python_module")
        if isinstance(py_hint, str) and py_hint:
            python_module = py_hint
        skinny_overrides = cd.get("skinnyOverrides")
        # USD VtDictionary surfaces as a Python dict (or pxr.Vt.Dictionary).
        if hasattr(skinny_overrides, "items"):
            for k, v in skinny_overrides.items():
                overrides[str(k)] = v

    # Bridge transmission → opacity so the flat path's delta-dielectric
    # refraction branch (flat_material.slang: `if (m.opacity < 1.0)`) actually
    # fires. The renderer's only refraction mechanism is opacity-gated; a
    # standard_surface/OpenPBR `transmission` weight that never lowers opacity
    # leaves the surface fully opaque (glass/oil rendered solid white). Mirrors
    # the .mtlx fallback loader (`_load_mtlx_materials`). Skip when opacity is
    # already authored (e.g. OpenPBR `geometry_opacity` cutout alpha).
    _derive_opacity_from_transmission(overrides)
    # Same gate for a standard_surface `subsurface` weight (the usdMtlx-plugin
    # intake of a `-mtlx` subsurface material); idempotent when the
    # UsdPreviewSurface export already authored `opacity = 0`.
    _derive_opacity_from_subsurface(overrides)
    # Fold UsdPreviewSurface clearcoat onto the canonical coat slot so a coated
    # material's coat lobe survives the flat-material pack.
    _canonicalize_coat(overrides)

    return Material(
        name=name,
        parameter_overrides=overrides,
        texture_paths=textures,
        texture_bindings=bindings,
        mtlx_target_name=mtlx_target_name,
        python_module=python_module,
    )


# ─── Direct MaterialX loading (bypass for missing usdMtlx plugin) ────


_STD_SURFACE_TO_FLAT: dict[str, str] = {
    "base_color": "diffuseColor",
    "specular_roughness": "roughness",
    "metalness": "metallic",
    "specular": "specular",
    "specular_IOR": "ior",
}

# OpenPBR surface shader-input names → Autodesk standard_surface names.
# The materialxusd exporter emits OpenPBR (`open_pbr_surface`) materials whose
# inputs carry `_weight`/`_metalness`/`_ior` suffixes (`transmission_weight`,
# `base_metalness`, `specular_ior`, …). skinny's only uber-closure is
# `mtlx_std_surface.slang`, whose packer (`pack_std_surface_params`) reads the
# standard_surface names. Without this translation every OpenPBR weight falls
# back to its default — `transmission`→0 makes glass opaque, `metalness`→0
# makes metals render as plastic. color/roughness inputs that share a name
# (base_color, specular_color, specular_roughness, transmission_color,
# coat_color, coat_roughness, emission_color, subsurface_color, …) need no
# entry. `emission_luminance` is intentionally omitted: OpenPBR authors it in
# nits, not as standard_surface's 0..1 `emission` weight, so a 1:1 alias would
# blow out emissive materials.
_OPENPBR_TO_STD_SURFACE: dict[str, str] = {
    "base_weight": "base",
    "base_metalness": "metalness",
    "base_diffuse_roughness": "diffuse_roughness",
    "specular_weight": "specular",
    "specular_ior": "specular_IOR",
    "specular_roughness_anisotropy": "specular_anisotropy",
    "transmission_weight": "transmission",
    "transmission_dispersion_scale": "transmission_dispersion",
    "subsurface_weight": "subsurface",
    "subsurface_scatter_anisotropy": "subsurface_anisotropy",
    "coat_weight": "coat",
    "coat_ior": "coat_IOR",
    "coat_roughness_anisotropy": "coat_anisotropy",
    "fuzz_weight": "sheen",
    "fuzz_color": "sheen_color",
    "fuzz_roughness": "sheen_roughness",
    "thin_film_ior": "thin_film_IOR",
    "geometry_thin_walled": "thin_walled",
}


def _parse_mtlx_value_str(type_name: str, value_str: str) -> object:
    """Convert a MaterialX value string to a Python scalar or tuple."""
    if type_name == "float":
        return float(value_str)
    if type_name == "integer":
        return int(value_str)
    if type_name in ("color3", "vector3"):
        return tuple(float(x.strip()) for x in value_str.split(","))
    if type_name in ("color4", "vector4"):
        return tuple(float(x.strip()) for x in value_str.split(","))
    if type_name == "vector2":
        return tuple(float(x.strip()) for x in value_str.split(","))
    if type_name == "boolean":
        return value_str.lower() in ("true", "1")
    return value_str


def _find_image_file_in_nodegraph(
    doc: object,
    ng_name: str,
    output_name: str,
    mtlx_dir: Path,
) -> Optional[Path]:
    """Walk a MaterialX nodegraph output back to an image file path."""
    ng = doc.getNodeGraph(ng_name)
    if ng is None:
        return None
    out = ng.getOutput(output_name)
    if out is None:
        return None
    source_name = out.getAttribute("nodename")
    if not source_name:
        return None
    node = ng.getNode(source_name)
    if node is None:
        return None
    if node.getCategory() not in ("image", "tiledimage"):
        return None
    file_input = node.getInput("file")
    if file_input is None:
        return None
    file_val = file_input.getValueString()
    if not file_val:
        return None
    return (mtlx_dir / file_val).resolve()


def _evaluate_nodegraph_constant(
    doc: object,
    ng_name: str,
    output_name: str,
    inp_type: str,
) -> Optional[object]:
    """Try to extract a representative constant from a procedural node graph.

    Handles two patterns:
    - Output directly references an interface input with a default value
    - Output is a mix/lerp node whose bg/fg are interface inputs (returns
      the 50/50 blend as a representative average)

    Returns a parsed Python value (float or tuple) or None.
    """
    ng = doc.getNodeGraph(ng_name)
    if ng is None:
        return None
    out = ng.getOutput(output_name)
    if out is None:
        return None
    source_name = out.getAttribute("nodename")
    if not source_name:
        iface = out.getAttribute("interfacename")
        if iface:
            ng_input = ng.getInput(iface)
            if ng_input:
                val_str = ng_input.getValueString()
                if val_str:
                    return _parse_mtlx_value_str(inp_type, val_str)
        return None
    node = ng.getNode(source_name)
    if node is None:
        return None

    if node.getCategory() == "mix" and inp_type in ("color3", "vector3"):
        bg_val = _resolve_ng_input_value(ng, node, "bg", inp_type)
        fg_val = _resolve_ng_input_value(ng, node, "fg", inp_type)
        if bg_val is not None and fg_val is not None:
            return tuple(0.5 * b + 0.5 * f for b, f in zip(bg_val, fg_val))
        return bg_val or fg_val

    if node.getCategory() == "multiply" and inp_type in ("color3", "vector3"):
        in1 = _resolve_ng_input_value(ng, node, "in1", inp_type)
        in2 = _resolve_ng_input_value(ng, node, "in2", inp_type)
        if in1 is not None and in2 is not None:
            return tuple(a * b for a, b in zip(in1, in2))
        return in1 or in2

    return None


def _resolve_ng_input_value(
    ng: object, node: object, input_name: str, type_name: str,
) -> Optional[tuple]:
    """Resolve a node input to a constant value, following interface refs."""
    inp = node.getInput(input_name)
    if inp is None:
        return None
    iface = inp.getAttribute("interfacename")
    if iface:
        ng_input = ng.getInput(iface)
        if ng_input:
            val_str = ng_input.getValueString()
            if val_str:
                v = _parse_mtlx_value_str(type_name, val_str)
                if isinstance(v, tuple):
                    return v
    val_str = inp.getValueString()
    if val_str:
        v = _parse_mtlx_value_str(type_name, val_str)
        if isinstance(v, tuple):
            return v
    return None


def _collect_mtlx_asset_paths(stage: Usd.Stage) -> set[str]:
    """Find .mtlx asset paths referenced by the stage's root layer."""
    root_layer = stage.GetRootLayer()
    paths: set[str] = set()

    def _visit(spec: "Sdf.PrimSpec") -> None:
        ref_list = spec.referenceList
        for ref in (
            list(ref_list.prependedItems)
            + list(ref_list.appendedItems)
            + list(ref_list.explicitItems)
        ):
            if ref.assetPath.endswith(".mtlx"):
                paths.add(ref.assetPath)
        for child_spec in spec.nameChildren:
            _visit(child_spec)

    for prim_spec in root_layer.rootPrims:
        _visit(prim_spec)
    return paths


def _resolve_layer_asset(
    asset_path: str, stage_dir: Path, root_layer: "Sdf.Layer",
) -> Optional[Path]:
    """Resolve a layer-relative asset path to an absolute filesystem path."""
    resolved = root_layer.ComputeAbsolutePath(asset_path)
    if resolved:
        p = Path(resolved)
        if p.exists():
            return p
    p = (stage_dir / asset_path).resolve()
    if p.exists():
        return p
    log.warning("could not resolve asset %r (stage_dir=%s)", asset_path, stage_dir)
    return None


def _load_mtlx_materials(
    stage: Usd.Stage, stage_dir: Path,
) -> dict[str, Material]:
    """Load .mtlx files referenced by the stage that USD couldn't resolve.

    Returns a mapping from surfacematerial element name to Material.
    Matched against binding-target leaf names when ComputeBoundMaterial
    fails (typical symptom of the missing usdMtlx file-format plugin).
    """
    if not _HAS_MATERIALX:
        log.warning("MaterialX Python package not installed — cannot load .mtlx fallback materials")
        return {}

    asset_paths = _collect_mtlx_asset_paths(stage)
    if not asset_paths:
        return {}

    result: dict[str, Material] = {}
    root_layer = stage.GetRootLayer()

    for asset_path in sorted(asset_paths):
        mtlx_file = _resolve_layer_asset(asset_path, stage_dir, root_layer)
        if mtlx_file is None:
            continue
        try:
            doc = mx.createDocument()
            mx.readFromXmlFile(doc, str(mtlx_file))
        except Exception as e:
            log.warning("failed to load MaterialX %s: %s", mtlx_file.name, e)
            continue

        mtlx_dir = mtlx_file.parent
        for node in doc.getMaterialNodes():
            mat_name = node.getName()
            overrides: dict[str, object] = {}
            textures: dict[str, Path] = {}

            ss_input = node.getInput("surfaceshader")
            if ss_input is None:
                continue
            ss_node_name = ss_input.getNodeName()
            if not ss_node_name:
                continue
            ss_node = doc.getNode(ss_node_name)
            if ss_node is None:
                continue

            for inp in ss_node.getInputs():
                inp_name = inp.getName()
                inp_type = inp.getType()

                ng_name = inp.getAttribute("nodegraph")
                if ng_name:
                    out_name = inp.getAttribute("output")
                    if out_name:
                        tex = _find_image_file_in_nodegraph(
                            doc, ng_name, out_name, mtlx_dir,
                        )
                        if tex is not None:
                            flat_key = _STD_SURFACE_TO_FLAT.get(inp_name, inp_name)
                            textures[flat_key] = tex
                        else:
                            avg = _evaluate_nodegraph_constant(
                                doc, ng_name, out_name, inp_type,
                            )
                            if avg is not None:
                                flat_key = _STD_SURFACE_TO_FLAT.get(inp_name, inp_name)
                                # Store under BOTH the flat alias (read by
                                # pack_flat_material) and the canonical
                                # std-surface name (read by
                                # pack_std_surface_params).
                                overrides[flat_key] = avg
                                overrides[inp_name] = avg
                    continue

                if inp.getNodeName():
                    continue

                val_str = inp.getValueString()
                if not val_str:
                    continue
                value = _parse_mtlx_value_str(inp_type, val_str)
                flat_key = _STD_SURFACE_TO_FLAT.get(inp_name, inp_name)
                # Same dual-author pattern: flat_key for FlatMaterialParams,
                # raw inp_name for StdSurfaceParams. Lets brass's
                # metalness=1 / specular=0 / coat=1 reach both packs even
                # though pack_flat_material understands UsdPreviewSurface
                # names and pack_std_surface_params understands MaterialX
                # standard_surface names.
                overrides[flat_key] = value
                overrides[inp_name] = value

            base = overrides.pop("base", None)
            if isinstance(base, (int, float)):
                dc = overrides.get("diffuseColor")
                if isinstance(dc, tuple):
                    overrides["diffuseColor"] = tuple(
                        float(base) * c for c in dc
                    )

            emission = overrides.pop("emission", None)
            emission_color = overrides.pop("emission_color", None)
            if isinstance(emission, (int, float)) and float(emission) > 0:
                if isinstance(emission_color, tuple):
                    overrides["emissiveColor"] = tuple(
                        float(emission) * c for c in emission_color
                    )

            transmission = overrides.pop("transmission", None)
            if isinstance(transmission, (int, float)) and float(transmission) > 0:
                overrides["opacity"] = max(0.0, 1.0 - float(transmission))
                # Stage-2 (flat-lobes-rich-inputs): keep transmission_color as a
                # first-class override. pack_flat_material reads it into
                # FlatHitMat.transmissionColor and the delta-transmission lobe
                # tints by it — so it is NO LONGER folded into diffuseColor. The
                # diffuse albedo stays decoupled from the transmission tint; a
                # transmissive dielectric with no authored base color defaults to
                # white so its reflection lobe is neutral. (White glass —
                # transmission_color = (1,1,1) — stays byte-identical, since
                # transmissionColor then equals the white albedo as before.)
                if overrides.get("diffuseColor") in (None, (0.0, 0.0, 0.0)):
                    overrides["diffuseColor"] = (1.0, 1.0, 1.0)

            # Subsurface boundary: open the flat refraction gate, mirroring the
            # UsdPreviewSurface export (and _derive_opacity_from_subsurface on the
            # plugin-present intake). Guarded so a non-subsurface standard_surface
            # material is untouched and an explicit opacity is preserved.
            _derive_opacity_from_subsurface(overrides)

            opacity_val = overrides.get("opacity")
            if isinstance(opacity_val, tuple):
                overrides["opacity"] = float(opacity_val[0])

            result[mat_name] = Material(
                name=mat_name,
                parameter_overrides=overrides,
                texture_paths=textures,
                mtlx_target_name=mat_name,
                mtlx_document=doc,
            )

    if result:
        log.info(
            "loaded %d material(s) from %d .mtlx file(s) via MaterialX API fallback",
            len(result), len(asset_paths),
        )
    else:
        log.warning(
            "found %d .mtlx asset path(s) but loaded 0 materials "
            "(MaterialX API fallback produced nothing)",
            len(asset_paths),
        )
    return result


def _merge_prim_overrides(
    stage: Usd.Stage, material_path: "Sdf.Path", mtlx_mat: Material,
) -> Material:
    """Return *mtlx_mat* with the bound Material prim's `skinnyOverrides`
    customData merged into its `parameter_overrides`.

    Mirrors the `skinnyOverrides` merge `_extract_material` does on the
    plugin-present intake. Returns a copy (never mutates the shared
    `mtlx_materials` entry); a no-op when the prim carries no overrides.
    """
    target_prim = stage.GetPrimAtPath(material_path)
    if not (target_prim and target_prim.IsValid()):
        return mtlx_mat
    cd = target_prim.GetCustomData()
    skinny_overrides = cd.get("skinnyOverrides") if cd else None
    if not hasattr(skinny_overrides, "items"):
        return mtlx_mat
    merged = dict(mtlx_mat.parameter_overrides)
    for k, v in skinny_overrides.items():
        merged[str(k)] = v
    # The subsurface medium coefficients (subsurface_sigma_*) live in
    # skinnyOverrides, not the .mtlx shader, so the transmissive-boundary gate
    # cannot be derived until the customData is merged here — `_load_mtlx_materials`
    # ran `_derive_opacity_from_subsurface` before the medium was available.
    # Re-derive now that the interior is present (no-op when already opaque-gated
    # or when there is no medium). Mirrors `_extract_material`, which merges the
    # customData first and then derives.
    _derive_opacity_from_subsurface(merged)
    return replace(mtlx_mat, parameter_overrides=merged)


def _prim_has_mtlx_reference(stage: Usd.Stage, prim_path: "Sdf.Path") -> bool:
    """True when the root-layer spec at *prim_path* carries a `.mtlx` reference.

    Identifies an exporter-authored MaterialX-backed material `over` (the
    exporter authors the `.mtlx` reference on the root layer — see
    `mtlx_emit.author_mtlx_reference`). Used to scope the preloaded-table
    preemption so a coincidentally same-named plain-USD material is never
    shadowed by an unrelated sidecar entry.
    """
    spec = stage.GetRootLayer().GetPrimAtPath(prim_path)
    if spec is None:
        return False
    ref_list = spec.referenceList
    for ref in (
        list(ref_list.prependedItems)
        + list(ref_list.appendedItems)
        + list(ref_list.explicitItems)
    ):
        if ref.assetPath.endswith(".mtlx"):
            return True
    return False


def _resolve_binding_from_mtlx_table(
    prim: Usd.Prim,
    materials: list[Material],
    material_index: dict[str, int],
    mtlx_materials: dict[str, Material],
) -> Optional[int]:
    """Resolve *prim*'s material binding against the preloaded `.mtlx` table.

    Walks the prim and its ancestors for a `material:binding` relationship and
    matches each target's leaf name against *mtlx_materials* (the sidecar
    table). A leaf-name hit is accepted ONLY when the bound target prim actually
    carries a `.mtlx` reference (`_prim_has_mtlx_reference`) — so a plain-USD
    material that coincidentally shares a leaf name with a sidecar entry is left
    to `ComputeBoundMaterial` instead of being shadowed. On an accepted hit,
    merges the bound Material prim's `skinnyOverrides` customData (the
    homogeneous SSS interior that is not a standard_surface input) and appends
    the Material, returning its `Scene.materials` index. Returns a cached index
    for an already-resolved target, or None when no ancestor binding maps to an
    exported `.mtlx` material.
    """
    stage = prim.GetStage()
    ancestor = prim
    while ancestor and ancestor != stage.GetPseudoRoot():
        binding_rel = ancestor.GetRelationship("material:binding")
        if binding_rel:
            for target_path in binding_rel.GetTargets():
                target_str = str(target_path)
                cached = material_index.get(target_str)
                if cached is not None:
                    return cached
                mtlx_mat = mtlx_materials.get(target_path.name)
                if mtlx_mat is not None and _prim_has_mtlx_reference(
                    stage, target_path
                ):
                    mtlx_mat = _merge_prim_overrides(stage, target_path, mtlx_mat)
                    idx = len(materials)
                    material_index[target_str] = idx
                    materials.append(mtlx_mat)
                    return idx
        ancestor = ancestor.GetParent()
    return None


def _resolve_material_binding(
    prim: Usd.Prim,
    materials: list[Material],
    material_index: dict[str, int],
    mtlx_materials: Optional[dict[str, Material]] = None,
) -> int:
    """Return a Scene.materials index for the prim's bound material.

    Materials list invariant: index 0 is always the flat fallback
    material. Bound USD materials are appended at indices 1..N and cached
    by prim path so multi-instance bindings share a single entry.

    When ComputeBoundMaterial fails (e.g. the usdMtlx plugin is missing
    and .mtlx references couldn't be resolved), falls back to matching
    the binding target's leaf name against pre-loaded mtlx_materials.
    """
    if not prim.HasAPI(UsdShade.MaterialBindingAPI):
        UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api = UsdShade.MaterialBindingAPI(prim)

    # Preloaded `.mtlx` sidecar table is authoritative when populated. It is
    # loaded only when the caller did NOT opt into use_usd_mtlx_plugin, so
    # resolving against it FIRST keeps the default load path on the sidecar
    # fallback even on a host where the usdMtlx plugin IS present and
    # ComputeBoundMaterial would otherwise compose the referenced `.mtlx`
    # material. Without this the intake path would silently differ by host
    # plugin availability (the exported reference now composes explicitly —
    # `/MaterialX/Materials/<name>`). The plugin-present opt-in path keeps this
    # table empty and falls through to ComputeBoundMaterial below.
    if mtlx_materials:
        idx = _resolve_binding_from_mtlx_table(
            prim, materials, material_index, mtlx_materials
        )
        if idx is not None:
            return idx

    bound, _relationship = binding_api.ComputeBoundMaterial()

    if not bound:
        stage = prim.GetStage()
        ancestor = prim
        while ancestor and ancestor != stage.GetPseudoRoot():
            binding_rel = ancestor.GetRelationship("material:binding")
            if binding_rel:
                for target_path in binding_rel.GetTargets():
                    target_str = str(target_path)
                    cached = material_index.get(target_str)
                    if cached is not None:
                        return cached
                    if mtlx_materials:
                        leaf = target_path.name
                        mtlx_mat = mtlx_materials.get(leaf)
                        if mtlx_mat is not None:
                            # The .mtlx document carries only the standard_surface
                            # inputs; per-material state that can't be expressed as
                            # a shader input (the homogeneous SSS interior) is
                            # authored as `skinnyOverrides` customData on the bound
                            # Material prim. _extract_material reads it on the
                            # plugin-present path; merge it here for the fallback so
                            # the `-mtlx` subsurface interior is not lost.
                            mtlx_mat = _merge_prim_overrides(
                                stage, target_path, mtlx_mat
                            )
                            idx = len(materials)
                            material_index[target_str] = idx
                            materials.append(mtlx_mat)
                            return idx
                    target_prim = stage.GetPrimAtPath(target_path)
                    if target_prim and target_prim.IsValid():
                        bound = UsdShade.Material(target_prim)
                        if bound:
                            break
            if bound:
                break
            ancestor = ancestor.GetParent()

    if not bound:
        log.debug("prim %s: no material binding resolved — using flat fallback", prim.GetPath())
        return 0  # fallback skin material

    mat_path = str(bound.GetPath())
    cached = material_index.get(mat_path)
    if cached is not None:
        return cached

    extracted = _extract_material(bound)
    material_index[mat_path] = len(materials)
    materials.append(extracted)
    return material_index[mat_path]


def _light_color_radiance(light_api: UsdLux.LightAPI) -> np.ndarray:
    """Combine `inputs:color * inputs:intensity * 2^inputs:exposure` into a
    linear-HDR radiance vec3 used by skinny's renderer."""
    color_attr = light_api.GetColorAttr()
    intensity_attr = light_api.GetIntensityAttr()
    exposure_attr = light_api.GetExposureAttr()
    color = color_attr.Get() if color_attr else (1.0, 1.0, 1.0)
    intensity = float(intensity_attr.Get()) if intensity_attr else 1.0
    exposure = float(exposure_attr.Get()) if exposure_attr else 0.0
    scaled = float(intensity) * (2.0 ** exposure)
    color_arr = np.asarray(color, dtype=np.float32).reshape(3)
    return (color_arr * scaled).astype(np.float32)


def _extract_distant_light(
    prim: Usd.Prim, time: Usd.TimeCode
) -> Optional[LightDir]:
    """Pull a UsdLux.DistantLight into a `LightDir`.

    USD convention: a distant light shines down its local -Z axis. The
    direction skinny wants is *toward* the light source — i.e. the +Z of
    the light's local frame, transformed into world space. We grab the
    third column of the world matrix (post-transpose, that's local +Z).
    """
    distant = UsdLux.DistantLight(prim)
    if not distant:
        return None

    world_mat = _world_transform(prim, time)
    # world_mat stores math-transposed: row k of world_mat is column k of
    # the math matrix (USD's row-major / row-vector convention). Local +Z
    # → world: math M * (0,0,1,0) = column 2 of math M = row 2 of stored.
    light_dir = world_mat[2, :3].astype(np.float32)
    norm = float(np.linalg.norm(light_dir))
    if norm < 1e-8:
        return None
    light_dir /= norm

    radiance = _light_color_radiance(UsdLux.LightAPI(prim))
    return LightDir(direction=light_dir, radiance=radiance,
                    prim_path=str(prim.GetPath()),
                    spectral_spd=_extract_light_spd(prim))


def _extract_light_spd(prim: Usd.Prim) -> "np.ndarray | None":
    """Authored illuminant SPD (spectral mode, Group 6.3) from a light prim's
    `skinnyOverrides["spectral"]` (kind `spectrum_samples`), as a 95-sample
    360-830/5 nm array. None for a blackbody / named / plain-RGB light — the
    spectral path upsamples the RGB radiance in those cases."""
    cd = prim.GetCustomData()
    payload = cd.get("skinnyOverrides", {}).get("spectral") if cd else None
    if payload is None or not hasattr(payload, "get"):
        return None
    if payload.get("kind") != "spectrum_samples":
        return None
    values = payload.get("values")
    if values is None:
        return None
    arr = np.asarray(list(values), dtype=np.float32)
    # The renderer's SPD buffer (binding 50) is a fixed 95-sample/light block; a
    # mis-sized payload can't be indexed, so reject here (→ RGB upsample) rather
    # than let a wrong-length array reach LightDir.spectral_spd.
    if arr.shape != (95,):
        return None
    return arr


def _extract_dome_light(
    prim: Usd.Prim, time: Usd.TimeCode
) -> Optional[LightEnvHDR]:
    """Pull a UsdLux.DomeLight's HDR texture into `LightEnvHDR`.

    Tries to resolve `inputs:texture:file`. .hdr files are loaded via
    skinny.environment._load_radiance_hdr; other formats (.exr, .png)
    return None for now — the renderer's existing legacy environment
    library is the user's escape hatch until Phase D-4.
    """
    dome = UsdLux.DomeLight(prim)
    if not dome:
        return None

    file_input = dome.GetTextureFileAttr()
    asset = file_input.Get() if file_input else None
    if asset is None:
        return None

    asset_path = (
        getattr(asset, "resolvedPath", None) or getattr(asset, "path", None) or ""
    )
    if not asset_path:
        return None
    p = Path(asset_path)
    if not p.exists() or p.suffix.lower() != ".hdr":
        # .exr / unresolved path: skip rather than throw — the renderer's
        # built-in HDR library still works, just no DomeLight upload.
        return None

    try:
        data = _load_radiance_hdr(p)
        data = _resize_equirect(data)
    except (OSError, ValueError):
        return None

    radiance = _light_color_radiance(UsdLux.LightAPI(prim))
    # DomeLight intensity scales the env radiance multiplicatively. We
    # collapse color×intensity×2^exposure into one scalar by taking the
    # luminance — the texture itself carries the chromatic detail.
    intensity = float(np.dot(radiance, np.array([0.2126, 0.7152, 0.0722], np.float32)))
    return LightEnvHDR(name=prim.GetPath().name, data=data, intensity=intensity)


def _extract_sphere_light(
    prim: Usd.Prim, time: Usd.TimeCode
) -> Optional["LightSphere"]:
    """Pull a UsdLux.SphereLight prim into a skinny LightSphere.

    Position is the prim's world translation; radius from inputs:radius;
    radiance derived from inputs:color × inputs:intensity × 2^exposure.
    """
    from skinny.scene import LightSphere

    sphere = UsdLux.SphereLight(prim)
    if not sphere:
        return None
    world_mat = _world_transform(prim, time)
    position = world_mat[3, :3].astype(np.float32)  # row-vector convention
    radius_attr = sphere.GetRadiusAttr().Get(time)
    radius = float(radius_attr) if radius_attr is not None else 0.5
    light_api = UsdLux.LightAPI(prim)
    radiance = _light_color_radiance(light_api)
    # Stash the authored colour + intensity so the scene-graph editor
    # can mutate either dimension without losing the other (radiance is
    # the combined product, irreversible from a single edit).
    color_attr = light_api.GetColorAttr()
    color = (
        np.asarray(color_attr.Get(), np.float32).reshape(3)
        if color_attr and color_attr.HasAuthoredValue()
        else np.ones(3, np.float32)
    )
    intensity_attr = light_api.GetIntensityAttr()
    intensity = (
        float(intensity_attr.Get())
        if intensity_attr and intensity_attr.HasAuthoredValue()
        else 1.0
    )
    exposure_attr = light_api.GetExposureAttr()
    exposure = (
        float(exposure_attr.Get())
        if exposure_attr and exposure_attr.HasAuthoredValue()
        else 0.0
    )
    intensity_eff = float(intensity) * (2.0 ** exposure)
    return LightSphere(
        position=position, radius=radius, radiance=radiance,
        color=color, intensity=intensity_eff, prim_path=str(prim.GetPath()),
    )


def _area_light_to_instance(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    positions: np.ndarray,
    tri_idx: np.ndarray,
    material_id: int,
    radiance: np.ndarray,
) -> MeshInstance:
    """Build a renderable MeshInstance from synthetic area-light geometry.

    The caller supplies local-space positions and triangle indices for the
    light shape (quad for RectLight, fan for DiskLight). This function
    synthesises smooth normals, bakes the source into a GPU Mesh, and
    world-transforms it via the prim's Xformable stack.
    """
    normals = np.zeros_like(positions)
    for tri in tri_idx:
        e1 = positions[tri[1]] - positions[tri[0]]
        e2 = positions[tri[2]] - positions[tri[0]]
        fn = np.cross(e1, e2)
        n = float(np.linalg.norm(fn))
        if n > 1e-8:
            fn /= n
        normals[tri[0]] += fn
        normals[tri[1]] += fn
        normals[tri[2]] += fn
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-8)
    normals /= lens
    uvs = np.zeros((len(positions), 2), dtype=np.float32)
    source = MeshSource(
        name=prim.GetPath().name,
        positions=positions.astype(np.float32),
        normals=normals.astype(np.float32),
        uvs=uvs,
        tri_idx=tri_idx.astype(np.int32),
    )
    source.content_hash = compute_source_hash(source)
    mesh = bake_mesh(source, displacement_bytes=None,
                     displacement_res=0, displacement_scale_world=0.0)
    transform = _world_transform(prim, time)
    return MeshInstance(
        mesh=mesh, transform=transform, material_id=material_id,
        name=source.name, source=source, prim_path=str(prim.GetPath()),
    )


def _rect_light_to_instance(
    prim: Usd.Prim, time: Usd.TimeCode, material_id: int,
) -> Optional[tuple[MeshInstance, Material]]:
    """Convert a UsdLux.RectLight into a 2-triangle emissive MeshInstance."""
    rect = UsdLux.RectLight(prim)
    if not rect:
        return None
    w = float(rect.GetWidthAttr().Get(time) or 1.0) * 0.5
    h = float(rect.GetHeightAttr().Get(time) or 1.0) * 0.5
    positions = np.array([
        [-w, -h, 0.0],
        [ w, -h, 0.0],
        [ w,  h, 0.0],
        [-w,  h, 0.0],
    ], dtype=np.float32)
    tri_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    radiance = _light_color_radiance(UsdLux.LightAPI(prim))
    inst = _area_light_to_instance(
        prim, time, positions, tri_idx, material_id, radiance,
    )
    mat = Material(
        name=prim.GetPath().name,
        parameter_overrides={
            "diffuseColor": (1.0, 1.0, 1.0),
            "emissiveColor": tuple(float(c) for c in radiance),
            "roughness": 1.0,
            "metallic": 0.0,
            "specular": 0.0,
        },
    )
    return inst, mat


_DISK_SEGMENTS = 24


def _disk_light_to_instance(
    prim: Usd.Prim, time: Usd.TimeCode, material_id: int,
) -> Optional[tuple[MeshInstance, Material]]:
    """Convert a UsdLux.DiskLight into a fan of emissive triangles."""
    disk = UsdLux.DiskLight(prim)
    if not disk:
        return None
    r = float(disk.GetRadiusAttr().Get(time) or 0.5)
    n = _DISK_SEGMENTS
    positions = np.zeros((n + 1, 3), dtype=np.float32)
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        positions[i + 1] = [r * np.cos(angle), r * np.sin(angle), 0.0]
    tri_idx = np.array(
        [[0, i + 1, (i % n) + 2] for i in range(n)], dtype=np.int32
    )
    radiance = _light_color_radiance(UsdLux.LightAPI(prim))
    inst = _area_light_to_instance(
        prim, time, positions, tri_idx, material_id, radiance,
    )
    mat = Material(
        name=prim.GetPath().name,
        parameter_overrides={
            "diffuseColor": (1.0, 1.0, 1.0),
            "emissiveColor": tuple(float(c) for c in radiance),
            "roughness": 1.0,
            "metallic": 0.0,
            "specular": 0.0,
        },
    )
    return inst, mat


def _extract_lights(
    stage: Usd.Stage, time: Usd.TimeCode,
    materials: list[Material], material_index: dict[str, int],
) -> tuple[list[LightDir], list, Optional[LightEnvHDR], list[MeshInstance]]:
    """Walk the stage and return (distant, sphere, dome, emissive instances).

    RectLight and DiskLight prims are converted to emissive mesh instances
    so they flow through the existing emissive triangle NEE path.
    """
    from skinny.scene import LightSphere

    lights_dir: list[LightDir] = []
    lights_sphere: list[LightSphere] = []
    environment: Optional[LightEnvHDR] = None
    emissive_instances: list[MeshInstance] = []
    for prim in stage.Traverse():
        if not prim.IsActive() or prim.IsAbstract():
            continue
        if prim.IsA(UsdLux.DistantLight):
            ld = _extract_distant_light(prim, time)
            if ld is not None:
                lights_dir.append(ld)
        elif prim.IsA(UsdLux.SphereLight):
            ls = _extract_sphere_light(prim, time)
            if ls is not None:
                lights_sphere.append(ls)
        elif prim.IsA(UsdLux.DomeLight):
            if environment is None:
                environment = _extract_dome_light(prim, time)
        elif prim.IsA(UsdLux.RectLight):
            mat_id = len(materials)
            result = _rect_light_to_instance(prim, time, mat_id)
            if result is not None:
                inst, mat = result
                materials.append(mat)
                material_index[str(prim.GetPath())] = mat_id
                emissive_instances.append(inst)
        elif prim.IsA(UsdLux.DiskLight):
            mat_id = len(materials)
            result = _disk_light_to_instance(prim, time, mat_id)
            if result is not None:
                inst, mat = result
                materials.append(mat)
                material_index[str(prim.GetPath())] = mat_id
                emissive_instances.append(inst)
    return lights_dir, lights_sphere, environment, emissive_instances


def _extract_camera(
    stage: Usd.Stage, time: Usd.TimeCode
) -> Optional["CameraOverride"]:
    """Pull the first authored UsdGeom.Camera prim into a CameraOverride.

    USD camera convention: the camera looks down its local -Z axis with
    +Y up. We grab focalLength + verticalAperture (both in tenths of a
    scene unit per the schema, but pxr returns real mm) and the world
    transform; the renderer turns those into yaw/pitch/distance/fov.
    """
    from skinny.scene import CameraOverride

    for prim in stage.Traverse():
        if not prim.IsActive() or prim.IsAbstract():
            continue
        if not prim.IsA(UsdGeom.Camera):
            continue
        cam = UsdGeom.Camera(prim)
        world = _world_transform(prim, time)
        # USD row-vector convention: position = (origin) * world = world's row 3.
        position = world[3, :3].astype(np.float32)
        # Camera looks down local -Z. world stores math-transpose so we
        # apply via row-vector multiplication: world_dir = local_dir @ world.
        local_forward = np.array([0.0, 0.0, -1.0, 0.0], np.float32)
        forward = (local_forward @ world)[:3].astype(np.float32)
        # Camera local +Y is up; transform to world like forward so the authored
        # roll survives. The renderer builds the view basis from this instead of
        # assuming world-up = +Y, fixing non-Y-up (pbrt Z-up) cameras. Mirrored
        # cameras are handled separately (ndc.x flip); up comes from the same
        # matrix, so both corrections compose.
        local_up = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
        up = (local_up @ world)[:3].astype(np.float32)
        un = float(np.linalg.norm(up))
        up = up / un if un > 1e-6 else np.array([0.0, 1.0, 0.0], np.float32)
        focal = float(cam.GetFocalLengthAttr().Get(time) or 50.0)
        v_ap  = float(cam.GetVerticalApertureAttr().Get(time) or 24.0)
        focus_attr = cam.GetFocusDistanceAttr().Get(time)
        focus = float(focus_attr) if focus_attr is not None else None
        fstop_attr = cam.GetFStopAttr().Get(time)
        fstop = float(fstop_attr) if fstop_attr is not None else 0.0
        lens = _extract_lens_system(prim, time)
        # Improper (orientation-reversing) pbrt cameras are flagged by the
        # importer in customData["pbrt"]["mirrored"]; the renderer reproduces
        # the reflection as a horizontal screen mirror at ray-gen.
        pbrt_md = prim.GetCustomDataByKey("pbrt") or {}
        mirrored = bool(pbrt_md.get("mirrored", False))
        # pbrt film exposure controls (change pbrt-radiometric-parity). ISO and
        # exposure time live on the camera prim as custom skinny:film:* attrs; the
        # renderer derives the imaging ratio exposure_time·iso/100 and applies it as
        # a live output scale. exposure_time falls back to the standard UsdGeom
        # `exposure` attr (interpreted as seconds here, NOT 2^EV stops) when the
        # skinny attr is absent. Defaults (100 / 1.0) ⇒ ratio 1.0 ⇒ unchanged.
        iso = _read_float(prim, "skinny:film:iso", time, 100.0)
        exp_attr = prim.GetAttribute("skinny:film:exposureTime")
        if exp_attr and exp_attr.IsValid() and exp_attr.HasAuthoredValue():
            exposure_time = float(exp_attr.Get(time) or 1.0)
        else:
            # Fallback: the standard UsdGeom `exposure` attr is a log2 (stops)
            # gain, so 2^exposure is the equivalent linear ratio. Fold it into
            # exposure_time (iso stays at its own default) so the imaging ratio
            # exposure_time·iso/100 reproduces the authored brightness.
            std_exp = cam.GetExposureAttr().Get(time)
            exposure_time = float(2.0 ** float(std_exp)) if std_exp is not None else 1.0
        return CameraOverride(
            position=position,
            forward=forward,
            up=up,
            focal_length_mm=focal,
            vertical_aperture_mm=v_ap,
            focus_distance=focus,
            fstop=fstop,
            lens=lens,
            mirrored=mirrored,
            iso=iso,
            exposure_time=exposure_time,
        )
    return None


def _extract_lens_system(
    cam_prim: Usd.Prim, time: Usd.TimeCode
) -> Optional[LensSystem]:
    """Walk camera children for skinny:lens:* attributes.

    Returns None if no child carries a `skinny:lens:role` attribute, so a
    plain authored camera collapses to the existing pinhole path. When at
    least one element is found, sort by `skinny:lens:order` and honour
    each child's USD `visibility` (invisible ⇒ element disabled).
    """
    elements: list[tuple[int, LensElement]] = []
    for child in cam_prim.GetChildren():
        if not child.IsActive() or child.IsAbstract():
            continue
        role_attr = child.GetAttribute("skinny:lens:role")
        if not role_attr or not role_attr.IsValid() or not role_attr.HasAuthoredValue():
            continue
        role_val = role_attr.Get(time)
        role = str(role_val) if role_val is not None else "element"

        radius    = _read_float(child, "skinny:lens:radius",    time, 0.0)
        thickness = _read_float(child, "skinny:lens:thickness", time, 0.0)
        ior       = _read_float(child, "skinny:lens:ior",       time, 1.0)
        aperture  = _read_float(child, "skinny:lens:aperture",  time, 0.0)
        order_attr = child.GetAttribute("skinny:lens:order")
        order = int(order_attr.Get(time)) if order_attr and order_attr.IsValid() and order_attr.HasAuthoredValue() else len(elements)

        if aperture <= 0.0:
            log.warning(
                "lens element %s has non-positive aperture %.3f; skipping",
                child.GetPath(), aperture,
            )
            continue

        enabled = True
        imageable = UsdGeom.Imageable(child)
        if imageable:
            try:
                vis = imageable.GetVisibilityAttr().Get(time)
                enabled = (vis != UsdGeom.Tokens.invisible)
            except Exception:
                enabled = True

        elements.append((order, LensElement(
            radius_mm=radius,
            thickness_mm=thickness,
            ior=ior,
            aperture_mm=aperture,
            is_aperture_stop=(role == "aperture") or (radius == 0.0),
            enabled=enabled,
        )))

    if not elements:
        return None
    elements.sort(key=lambda t: t[0])
    return LensSystem(elements=[e for _, e in elements])


def _read_float(prim: Usd.Prim, name: str, time: Usd.TimeCode, default: float) -> float:
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid() or not attr.HasAuthoredValue():
        return float(default)
    val = attr.Get(time)
    return float(val) if val is not None else float(default)


def _world_transform(prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
    """Compute the prim's local-to-world transform as a 4x4 numpy float32.

    USD stores matrices in row-major order with row-vector multiplication
    semantics (`v_row * M = math M^T * v_col`). The renderer's GPU
    convention is "numpy stored as math-transpose; GLSL reads as column-
    major; GPU sees math". Since USD's storage already matches that
    transpose, np.array(gf_matrix) drops in directly.
    """
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return np.eye(4, dtype=np.float32)
    gf_xform = xformable.ComputeLocalToWorldTransform(time)
    arr = np.array(gf_xform, dtype=np.float32)
    if arr.shape != (4, 4):
        # USD sometimes returns a flat 16-element list; reshape.
        arr = arr.reshape(4, 4).astype(np.float32)
    return arr


def _up_axis_rt(up_axis: str) -> "Optional[np.ndarray]":
    """Stored-form (transpose) rotation that maps a stage's up axis to +Y.

    Returns ``None`` for a Y-up stage (no correction needed). For a Z-up
    stage returns ``Rᵀ`` for the −90°-about-X rotation, ready to
    right-multiply this codebase's row-vector/stored transforms:
    ``M_new = M_stored @ rt`` and ``v_new = v_row @ rt``.

    Rᵀ maps +Z→+Y and +Y→−Z.
    """
    if up_axis != "Z":
        return None
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )


def _apply_up_axis_correction(
    prim_data: "list[tuple[MeshSource, np.ndarray, int]]",
    scene: Scene,
    up_axis: str,
) -> "list[tuple[MeshSource, np.ndarray, int]]":
    """Rotate a Z-up stage's geometry, lights, and camera to scene +Y.

    Returns prim_data with corrected transforms (mesh instances are baked
    from it after this returns). Lights, emissive instances, and the camera
    override already live on ``scene`` and are mutated in place. Y-up stages
    are returned untouched.
    """
    rt = _up_axis_rt(up_axis)
    if rt is None:
        return prim_data

    rt4 = np.eye(4, dtype=np.float32)
    rt4[:3, :3] = rt

    prim_data = [
        (src, (xf @ rt4).astype(np.float32), mat) for (src, xf, mat) in prim_data
    ]
    for inst in scene.instances:  # emissive-light instances
        inst.transform = (inst.transform @ rt4).astype(np.float32)
    for ls in scene.lights_sphere:
        ls.position = (ls.position @ rt).astype(np.float32)
    for ld in scene.lights_dir:
        ld.direction = (ld.direction @ rt).astype(np.float32)
    ov = scene.camera_override
    if ov is not None:
        ov.position = (ov.position @ rt).astype(np.float32)
        ov.forward = (ov.forward @ rt).astype(np.float32)
    return prim_data


# ─── Public entry point ───────────────────────────────────────────────

_USD_POOL_SIZE = 4


def _film_max_component(stage: "Usd.Stage") -> float:
    """The pbrt film `maxcomponentvalue` carried in stage metadata, or 0.0.

    The pbrt→USD importer (`pbrt/metadata.scene_metadata`) preserves the whole
    film verbatim under `customLayerData["pbrt"]["film"]["params"]`. A positive
    value arms the renderer's per-sample radiance clamp (change
    film-maxcomponent-clamp), matching pbrt's `RGBFilm` firefly suppression so an
    imported pbrt scene reproduces its reference EXR. 0.0 (default / absent /
    non-finite / non-positive) leaves the clamp disabled. Any read failure is
    non-fatal — the clamp simply stays off.
    """
    try:
        cld = dict(stage.GetRootLayer().customLayerData)
        params = cld.get("pbrt", {}).get("film", {}).get("params", {})
        val = float(params.get("maxcomponentvalue", 0.0))
    except Exception:  # noqa: BLE001 - metadata is best-effort; never block a load
        return 0.0
    if not math.isfinite(val) or val <= 0.0:
        return 0.0
    return val


# ─── UsdVol.Volume ingest (nanovdb-volume-rendering) ─────────────────

# pbrt→USD point change of basis. The importer authors a Volume prim's xform as
# `to_skinny(ctm) = B @ ctm @ B` (pbrt/emit.add_volume), i.e. the prim transform
# maps *B-converted* medium-space points to USD world:
#   p_usd = primXform @ (B @ p_medium),   p_medium = M_grid @ ijk.
# The grid's own map (`NanoVdbGrid.index_to_world`) lives in pbrt/medium axes,
# so composing the full index→USD-world chain needs one explicit B between the
# prim xform and the grid map — the same convention the geometry bake uses
# (`bake_world_mesh` transforms points by `B @ CTM`; here
# primXform @ B @ M_grid == B @ ctm @ M_grid, the identical point chain).
_PBRT_TO_USD_B = np.diag([1.0, 1.0, -1.0, 1.0])


def compute_volume_world_to_uvw(
    prim_transform_stored: np.ndarray,
    index_to_world: np.ndarray,
    index_min: tuple[int, int, int],
    dims: tuple[int, int, int],
) -> np.ndarray:
    """Fold the whole world→texture chain into one (3, 4) affine map.

    `prim_transform_stored` is the Volume prim's local-to-world in this
    codebase's *stored* (row-vector / math-transpose) form, as returned by
    `_world_transform` (with any up-axis correction already right-multiplied
    on, like every mesh instance transform). `index_to_world` is the grid's
    math-convention map of ijk voxel centers to medium space
    (`NanoVdbGrid.index_to_world`, world = M @ ijk).

    Returns rows 0..2 of the math (column-vector) matrix A with
    ``uvw = A @ [p_world, 1]`` where ``uvw = ((X⁻¹ @ p) − index_min + 0.5) / dims``
    and ``X = primXform_math @ B @ index_to_world`` — so the shader does one
    affine transform per density lookup and samples the texture at the exact
    voxel centers (`density[a, b, c]` ↔ uvw = ((a,b,c) + 0.5) / dims).
    """
    prim_math = np.asarray(prim_transform_stored, np.float64).T
    x = prim_math @ _PBRT_TO_USD_B @ np.asarray(index_to_world, np.float64)
    world_to_index = np.linalg.inv(x)
    shift = np.eye(4, dtype=np.float64)
    shift[:3, 3] = 0.5 - np.asarray(index_min, np.float64)
    scale = np.diag([1.0 / dims[0], 1.0 / dims[1], 1.0 / dims[2], 1.0])
    a = scale @ shift @ world_to_index
    return a[:3, :].astype(np.float32)


def _resolve_volume_asset_path(file_path_attr, stage: "Usd.Stage") -> Optional[str]:
    """Resolve an OpenVDBAsset ``filePath`` to an on-disk path string."""
    if file_path_attr is None:
        return None
    resolved = getattr(file_path_attr, "resolvedPath", "") or ""
    if resolved:
        return str(resolved)
    raw = getattr(file_path_attr, "path", None)
    if raw is None:
        raw = str(file_path_attr)
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        real = stage.GetRootLayer().realPath
        if real:
            p = Path(real).parent / p
    return str(p)


def _extract_volume_grid(
    stage: "Usd.Stage", time: Usd.TimeCode,
) -> "Optional[tuple[object, np.ndarray, str, str, str]]":
    """Find the stage's ``UsdVol.Volume`` prim and decode its density grid.

    Returns ``(NanoVdbGrid, stored_prim_transform, prim_path, asset_path,
    field_name)`` or None when the stage has no volume. Exactly ONE grid per
    scene is supported: additional Volume prims are warned about and skipped.
    Decode failures (unsupported codec / grid class / missing file) propagate —
    the nanovdb reader's errors name the file and feature, and a loud failure
    is required over silently rendering an empty medium (heterogeneous-media
    spec).
    """
    from skinny.pbrt.nanovdb import read_nanovdb

    found = None
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or prim.IsAbstract() or not prim.IsA(UsdVol.Volume):
            continue
        if found is not None:
            log.warning(
                "multiple UsdVol.Volume prims; only one grid per scene is "
                "supported — keeping %s, skipping %s",
                found[2], prim.GetPath(),
            )
            print(
                f"[skinny] WARNING: multiple UsdVol.Volume prims — keeping "
                f"{found[2]}, skipping {prim.GetPath()}"
            )
            continue
        volume = UsdVol.Volume(prim)
        # GetFieldPathMap is missing from some usd-core builds (the repo runs two
        # venvs with different pxr versions) — fall back to the raw `field:*`
        # relationships, which every build exposes.
        if hasattr(volume, "GetFieldPathMap"):
            field_map = volume.GetFieldPathMap()
        else:
            field_map = {}
            for rel in prim.GetRelationships():
                rel_name = rel.GetName()
                if not rel_name.startswith("field:"):
                    continue
                targets = rel.GetTargets()
                if targets:
                    field_map[rel_name.split(":", 1)[1]] = targets[0]
        asset_path = None
        field_name = "density"
        for _token, field_path in field_map.items():
            field_prim = stage.GetPrimAtPath(field_path)
            if not field_prim or not field_prim.IsA(UsdVol.OpenVDBAsset):
                continue
            vdb = UsdVol.OpenVDBAsset(field_prim)
            asset_path = _resolve_volume_asset_path(vdb.GetFilePathAttr().Get(), stage)
            name_attr = vdb.GetFieldNameAttr().Get()
            if name_attr:
                field_name = str(name_attr)
            break
        if not asset_path:
            log.warning("UsdVol.Volume %s has no resolvable OpenVDBAsset field; skipped",
                        prim.GetPath())
            continue
        grid = read_nanovdb(asset_path, field=field_name)
        xf = _world_transform(prim, time)
        found = (grid, xf, str(prim.GetPath()), asset_path, field_name)
    return found


def _read_open_stage(
    stage: "Usd.Stage",
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
    keep_stage: bool = False,
    source_label: Optional[str] = None,
) -> tuple[Scene, list[tuple[MeshSource, np.ndarray, int]], Optional["Usd.Stage"]]:
    """Serial read of an already-open USD stage. See `_read_usd_stage`."""
    label = source_label or (stage.GetRootLayer().identifier or "<anonymous stage>")
    eval_time = time if time is not None else Usd.TimeCode.Default()

    mtlx_materials: dict[str, Material] = {}
    if not use_usd_mtlx_plugin:
        real = stage.GetRootLayer().realPath
        stage_dir = Path(real).parent if real else Path.cwd()
        mtlx_materials = _load_mtlx_materials(stage, stage_dir)

    materials: list[Material] = [Material(name="default")]
    material_index: dict[str, int] = {}

    prim_data: list[tuple[MeshSource, np.ndarray, int]] = []
    # TraverseInstanceProxies so meshes inside instanceable references
    # (USD prototypes) are visited; the default predicate stops at the
    # instance boundary and would yield zero meshes for such scenes.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or prim.IsAbstract():
            continue

        if prim.IsA(UsdGeom.Mesh):
            source = _read_mesh_attrs(prim, eval_time)
        else:
            # Analytic gprims (Sphere/Cube/Cylinder/Cone/Capsule/Plane) carry
            # no point data; tessellate them into a mesh. Returns None for any
            # other prim type, which we skip.
            source = tessellate_gprim(prim, eval_time)
        if source is None:
            continue

        source.content_hash = compute_source_hash(source)
        transform = _world_transform(prim, eval_time)
        material_id = _resolve_material_binding(
            prim, materials, material_index, mtlx_materials,
        )
        prim_data.append((source, transform, material_id))

    if not prim_data:
        raise ValueError(
            f"USD stage {label} contains no usable mesh or gprim geometry"
        )

    lights_dir, lights_sphere, environment, emissive_instances = _extract_lights(
        stage, eval_time, materials, material_index,
    )
    camera_override = _extract_camera(stage, eval_time)

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    mm_per_unit = max(meters_per_unit * 1000.0, 1e-6)
    film_max_component = _film_max_component(stage)

    partial_scene = Scene(
        instances=list(emissive_instances),
        materials=materials,
        lights_dir=lights_dir,
        lights_sphere=lights_sphere,
        environment=environment,
        camera_override=camera_override,
        mm_per_unit=mm_per_unit,
        film_max_component=film_max_component,
    )
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    prim_data = _apply_up_axis_correction(prim_data, partial_scene, up_axis)

    # UsdVol.Volume ingest (nanovdb-volume-rendering): decode the density grid
    # and fold the full index→world chain (prim xform, up-axis correction, B,
    # grid map, index_min, voxel-center offset) into one world→uvw affine map.
    vol = _extract_volume_grid(stage, eval_time)
    if vol is not None:
        grid, vol_xf, vol_prim_path, asset_path, field_name = vol
        rt = _up_axis_rt(up_axis)
        if rt is not None:
            rt4 = np.eye(4, dtype=np.float32)
            rt4[:3, :3] = rt
            vol_xf = vol_xf @ rt4  # same stored-form correction as prim_data
        dims = tuple(int(d) for d in grid.density.shape)
        vmax = float(grid.value_max) if grid.value_max > 0.0 else 1.0
        partial_scene.volume_grid = VolumeGrid(
            density=grid.density,
            value_max=vmax,
            index_min=tuple(int(i) for i in grid.index_min),
            dims=dims,
            world_to_uvw=compute_volume_world_to_uvw(
                vol_xf, grid.index_to_world, grid.index_min, dims),
            asset_path=str(asset_path),
            field_name=field_name,
            prim_path=vol_prim_path,
        )
        print(
            f"[skinny] USD volume {vol_prim_path}: grid {grid.grid_name!r} "
            f"{dims[0]}x{dims[1]}x{dims[2]} value_max {vmax:.4g} ({asset_path})"
        )
    return partial_scene, prim_data, (stage if keep_stage else None)


def _read_usd_stage(
    stage_path: Path,
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
    keep_stage: bool = False,
) -> tuple[Scene, list[tuple[MeshSource, np.ndarray, int]], Optional["Usd.Stage"]]:
    """Open a USD stage from disk, then read it. See `_read_open_stage`."""
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage: {stage_path}")
    return _read_open_stage(
        stage,
        time=time,
        use_usd_mtlx_plugin=use_usd_mtlx_plugin,
        keep_stage=keep_stage,
        source_label=str(stage_path),
    )


_USD_LIGHT_TYPES = (
    UsdLux.DistantLight, UsdLux.SphereLight, UsdLux.DomeLight,
    UsdLux.RectLight, UsdLux.DiskLight,
)


@dataclass
class AnimationIndex:
    """Which prims carry authored animation, recorded once at load.

    Paths are full Sdf paths (str). The renderer intersects these with its
    baked instances by path to know which TLAS records / light buffers to
    re-evaluate per frame, so per-frame cost scales with the animated set.
    """

    xform_paths: list[str] = field(default_factory=list)
    light_paths: list[str] = field(default_factory=list)
    camera_animated: bool = False
    skinned_mesh_paths: list[str] = field(default_factory=list)

    @property
    def has_animation(self) -> bool:
        return bool(
            self.xform_paths or self.light_paths or self.camera_animated
            or self.skinned_mesh_paths
        )


def _attr_time_varying(prim: "Usd.Prim") -> bool:
    """True if any authored attribute on the prim has >1 time sample."""
    for attr in prim.GetAuthoredAttributes():
        if attr.GetNumTimeSamples() > 1:
            return True
    return False


def build_animation_index(stage: "Usd.Stage") -> AnimationIndex:
    """Scan the stage for prims with authored animation relevant to playback.

    Records geometry prims (UsdGeom.Gprim) whose *world* transform varies over
    time — including those animated only through an ancestor — plus lights and
    the camera whose transform or authored attributes vary. Pure read; safe to
    call on the background load thread.
    """
    index = AnimationIndex()

    # Local-xform time-variability for every xformable prim, so the ancestor
    # walk below is O(depth) instead of recomputing per prim.
    local_varies: dict[object, bool] = {}
    for prim in stage.Traverse():
        xf = UsdGeom.Xformable(prim)
        if xf:
            local_varies[prim.GetPath()] = bool(xf.TransformMightBeTimeVarying())

    def _world_xform_varies(prim: "Usd.Prim") -> bool:
        p = prim
        while p and not p.GetPath().IsAbsoluteRootPath():
            if local_varies.get(p.GetPath(), False):
                return True
            p = p.GetParent()
        return False

    for prim in stage.Traverse():
        if not prim.IsActive() or prim.IsAbstract():
            continue
        path = str(prim.GetPath())
        if any(prim.IsA(t) for t in _USD_LIGHT_TYPES):
            if _world_xform_varies(prim) or _attr_time_varying(prim):
                index.light_paths.append(path)
        elif prim.IsA(UsdGeom.Camera):
            if _world_xform_varies(prim) or _attr_time_varying(prim):
                index.camera_animated = True
        elif prim.IsA(UsdGeom.Gprim):
            if _world_xform_varies(prim):
                index.xform_paths.append(path)

    # Skeletal: a skinned mesh animates via UsdSkel even with a static xform.
    try:
        skel = extract_skeletal_bindings(stage)
        index.skinned_mesh_paths = [m.prim_path for m in skel.meshes]
    except Exception as exc:  # noqa: BLE001
        log.warning("skeletal scan failed: %s", exc)
    return index


def build_playback_clock(
    stage: "Usd.Stage",
    index: AnimationIndex,
    default_fps: float = 24.0,
) -> "PlaybackClock":
    """Build a `PlaybackClock` from stage time metadata + an animation index."""
    from skinny.playback import PlaybackClock

    start = float(stage.GetStartTimeCode())
    end = float(stage.GetEndTimeCode())
    tcps = float(stage.GetTimeCodesPerSecond())
    fps = tcps if tcps > 0.0 else float(default_fps)
    return PlaybackClock(
        start_time_code=start,
        end_time_code=end,
        time_codes_per_second=tcps,
        playback_fps=fps,
        has_animation=index.has_animation,
        current_time_code=start,
    )


@dataclass
class SkinnedMeshBinding:
    """Per-skinned-mesh data for GPU linear-blend skinning.

    `joint_indices`/`joint_weights` are (N, influences) in the mesh's *local*
    joint order (the skinning query's mapper relates that to the skeleton's
    joint order). `skel_query`/`skinning_query` are retained so the renderer can
    evaluate `ComputeSkinningTransforms` per frame; they stay valid while the
    stage (and the owning `SkeletalScene.cache`) is alive.
    """

    prim_path: str
    rest_points: np.ndarray      # (N, 3) f32, bind-pose mesh-local
    rest_normals: np.ndarray     # (N, 3) f32
    joint_indices: np.ndarray    # (N, influences) i32, mesh-local order
    joint_weights: np.ndarray    # (N, influences) f32
    influences: int
    skel_query: object           # UsdSkel.SkeletonQuery
    skinning_query: object       # UsdSkel.SkinningQuery


@dataclass
class SkeletalScene:
    """All UsdSkel skinned meshes in a stage, plus the owning cache + stage.

    The `cache` and `stage` references MUST stay alive for the per-mesh
    `skel_query`/`skinning_query` to remain valid (pxr invalidates skinning
    queries once their cache/stage is dropped). The renderer holds this object
    for the lifetime of the loaded USD scene.
    """

    cache: object                # UsdSkel.Cache
    stage: object = None         # Usd.Stage — kept alive alongside the cache
    meshes: list[SkinnedMeshBinding] = field(default_factory=list)

    @property
    def has_skinning(self) -> bool:
        return bool(self.meshes)


def _skinned_mesh_binding(tq, sq) -> "Optional[SkinnedMeshBinding]":
    """Build a SkinnedMeshBinding from a UsdSkel skinning query + skel query.

    Returns None for malformed or unsupported (faceVarying / rigid) targets so
    the caller skips them (they still bake as static bind-pose geometry).
    """
    m = tq.GetPrim()
    mesh = UsdGeom.Mesh(m)
    pts = mesh.GetPointsAttr().Get()
    if pts is None or len(pts) == 0:
        return None
    rest_points = np.asarray(pts, dtype=np.float32)
    n = rest_points.shape[0]

    infl = int(tq.GetNumInfluencesPerComponent())
    ji_raw = tq.GetJointIndicesPrimvar().Get()
    jw_raw = tq.GetJointWeightsPrimvar().Get()
    if ji_raw is None or jw_raw is None:
        return None
    ji = np.asarray(ji_raw, dtype=np.int32)
    jw = np.asarray(jw_raw, dtype=np.float32)
    # Per-vertex influences expected: N * influences. Rigid (constant) weights
    # would be just `influences` long — unsupported here; skip (static bake).
    if ji.size != n * infl or jw.size != n * infl:
        log.warning(
            "skinned mesh %s has non per-vertex influences (%d for %d verts × %d); "
            "skipping skinning", m.GetPath(), ji.size, n, infl,
        )
        return None
    ji = ji.reshape(n, infl)
    jw = jw.reshape(n, infl)

    nrm = mesh.GetNormalsAttr().Get()
    if (nrm is not None and mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
            and len(nrm) == n):
        rest_normals = np.asarray(nrm, dtype=np.float32)
    else:
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        tri_idx = _triangulate(np.asarray(fvc, np.int32), np.asarray(fvi, np.int32))
        rest_normals = _smooth_normals(rest_points, tri_idx)

    return SkinnedMeshBinding(
        prim_path=str(m.GetPath()),
        rest_points=rest_points,
        rest_normals=rest_normals,
        joint_indices=ji,
        joint_weights=jw,
        influences=infl,
        skel_query=sq,
        skinning_query=tq,
    )


def compute_joint_matrices(
    binding: "SkinnedMeshBinding",
    time: float,
    skel_to_world: "Optional[np.ndarray]" = None,
    up_axis_rt4: "Optional[np.ndarray]" = None,
) -> np.ndarray:
    """Per-joint skinning matrices for a skinned mesh at `time`.

    Returns an (J_local, 4, 4) float32 array in skinny's row-vector "stored"
    convention (a deformed point is `[p, 1] @ M`), in the mesh's *local* joint
    order so `binding.joint_indices` index directly. The geomBindTransform is
    folded in. `skel_to_world` and `up_axis_rt4` (both 4x4 stored) optionally
    place the result in world space.

    Validated to reproduce `UsdSkelSkinningQuery.ComputeSkinnedPoints` (skel
    space) under linear blend skinning — see tests/test_usd_skel.py.
    """
    sq = binding.skel_query
    tq = binding.skinning_query
    xf = sq.ComputeSkinningTransforms(Usd.TimeCode(float(time)))
    mapper = tq.GetJointMapper()
    if mapper is not None:
        xf = mapper.Remap(xf)
    mats = np.array([np.array(m, dtype=np.float32) for m in xf])  # (J,4,4)
    gbt = np.array(tq.GetGeomBindTransform(Usd.TimeCode.Default()), dtype=np.float32)
    # Fold geomBind on the left of points: (p @ G) @ xf  ==  p @ (G @ xf).
    mats = np.einsum("ij,njk->nik", gbt, mats)
    if skel_to_world is not None:
        mats = np.einsum("nij,jk->nik", mats, np.asarray(skel_to_world, np.float32))
    if up_axis_rt4 is not None:
        mats = np.einsum("nij,jk->nik", mats, np.asarray(up_axis_rt4, np.float32))
    return mats.astype(np.float32)


def lbs_points(
    rest_points: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:
    """Linear blend skinning of rest points by per-joint `matrices`.

    `matrices` are row-vector "stored" 4x4s (deformed = `[p,1] @ M`); indices
    are mesh-local. Returns (N, 3) float32 deformed positions.
    """
    n = rest_points.shape[0]
    ph = np.concatenate([rest_points, np.ones((n, 1), np.float32)], axis=1)
    out = np.zeros((n, 3), np.float32)
    for k in range(joint_indices.shape[1]):
        mk = matrices[joint_indices[:, k]]              # (N, 4, 4)
        out += joint_weights[:, k, None] * np.einsum("ni,nij->nj", ph, mk)[:, :3]
    return out.astype(np.float32)


def extract_skeletal_bindings(stage: "Usd.Stage") -> SkeletalScene:
    """Discover all UsdSkel skinned meshes in the stage.

    Walks SkelRoots, populates a UsdSkel cache, and returns one
    SkinnedMeshBinding per (supported) skinning target. The returned
    `SkeletalScene.cache` must be kept alive for the queries to remain valid.
    """
    cache = UsdSkel.Cache()
    scene = SkeletalScene(cache=cache, stage=stage)
    for prim in stage.Traverse():
        if not prim.IsA(UsdSkel.Root):
            continue
        root = UsdSkel.Root(prim)
        cache.Populate(root, Usd.TraverseInstanceProxies())
        try:
            bindings = cache.ComputeSkelBindings(root, Usd.TraverseInstanceProxies())
        except Exception as exc:  # noqa: BLE001
            log.warning("skel binding compute failed for %s: %s", prim.GetPath(), exc)
            continue
        for b in bindings:
            sq = cache.GetSkelQuery(b.GetSkeleton())
            if sq is None:
                continue
            for tq in b.GetSkinningTargets():
                mb = _skinned_mesh_binding(tq, sq)
                if mb is not None:
                    scene.meshes.append(mb)
    return scene


_UI_CONTROL_TYPES = ("slider", "toggle", "combo", "color")


@dataclass
class ControlSpec:
    """One USD-declared UI control (a prim with `skinny:ui:*` attributes).

    `target` is a prefix-typed binding string resolved by the renderer:
    `renderer:<path>`, `mtlx:<field>`, `material:<name>:<input>`, or
    `usd:<primPath>.<attr>`.
    """

    name: str
    label: str
    type: str
    target: str
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.0
    choices: list[str] = field(default_factory=list)
    default: object = None
    order: int = 0


def _ui_attr(prim, name):
    a = prim.GetAttribute(f"skinny:ui:{name}")
    if a and a.IsValid() and a.HasAuthoredValue():
        return a.Get()
    return None


def extract_ui_controls(stage: "Usd.Stage") -> list[ControlSpec]:
    """Discover `skinny:ui:*` control prims and parse them into ControlSpecs.

    A control is any prim with an authored `skinny:ui:type`. Malformed
    declarations (unknown type, missing target) are skipped with a warning.
    Sorted by an authored `skinny:ui:order` then prim path.
    """
    out: list[ControlSpec] = []
    for prim in stage.Traverse():
        if not prim.IsActive() or prim.IsAbstract():
            continue
        ctype = _ui_attr(prim, "type")
        if ctype is None:
            continue
        ctype = str(ctype)
        path = str(prim.GetPath())
        if ctype not in _UI_CONTROL_TYPES:
            log.warning("skinny:ui control %s has unknown type %r; skipping",
                        path, ctype)
            continue
        target = _ui_attr(prim, "target")
        if not target:
            log.warning("skinny:ui control %s has no target; skipping", path)
            continue
        label = _ui_attr(prim, "label")
        choices = _ui_attr(prim, "choices")
        lo = _ui_attr(prim, "min")
        hi = _ui_attr(prim, "max")
        step = _ui_attr(prim, "step")
        order = _ui_attr(prim, "order")
        out.append(ControlSpec(
            name=prim.GetName(),
            label=str(label) if label else prim.GetName(),
            type=ctype,
            target=str(target),
            lo=float(lo) if lo is not None else 0.0,
            hi=float(hi) if hi is not None else 1.0,
            step=float(step) if step is not None else 0.0,
            choices=[str(c) for c in choices] if choices else [],
            default=_ui_attr(prim, "default"),
            order=int(order) if order is not None else 0,
        ))
    out.sort(key=lambda c: (c.order, c.name))
    return out


def _inert_binding(reason: str):
    log.warning("skinny:ui control inert: %s", reason)
    return (lambda: None, lambda _v: None)


def resolve_control_binding(renderer, spec: "ControlSpec"):
    """Resolve a ControlSpec.target into (getter, setter) closures.

    Reuses the renderer's existing live-edit machinery. Unresolvable targets
    return inert closures + a warning rather than raising, so a bad declaration
    leaves the widget present-but-dead instead of breaking the panel.
    """
    from skinny.params import _get_nested, _set_nested

    kind, _, rest = spec.target.partition(":")

    if kind == "renderer":
        return (lambda: _get_nested(renderer, rest),
                lambda v: _set_nested(renderer, rest, v))

    if kind == "mtlx":
        path = "mtlx." + rest
        return (lambda: _get_nested(renderer, path),
                lambda v: _set_nested(renderer, path, v))

    if kind == "material":
        mat_name, _, inp = rest.partition(":")
        if not inp:
            return _inert_binding(f"material target {rest!r} missing input")
        scene = getattr(renderer, "_usd_scene", None)
        mats = getattr(scene, "materials", None) or []
        mat_id = next(
            (i for i, m in enumerate(mats)
             if getattr(m, "name", None) == mat_name
             or getattr(m, "mtlx_target_name", None) == mat_name),
            None,
        )
        if mat_id is None:
            return _inert_binding(f"material {mat_name!r} not found")

        def _get(mid=mat_id, k=inp):
            m = renderer._usd_scene.materials[mid]
            return m.parameter_overrides.get(k)

        def _set(v, mid=mat_id, k=inp):
            renderer.apply_material_override(mid, k, v)

        return (_get, _set)

    if kind == "usd":
        prim_path, _, attr_name = rest.rpartition(".")
        stage = getattr(renderer, "_usd_stage", None)
        if stage is None or not prim_path or not attr_name:
            return _inert_binding(f"usd target {rest!r} unresolvable")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return _inert_binding(f"usd prim {prim_path!r} not found")
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            return _inert_binding(f"usd attr {attr_name!r} not found on {prim_path}")

        def _get(a=attr):
            return a.Get()

        def _set(v, a=attr):
            a.Set(v)
            renderer._usd_live_dirty = True

        return (_get, _set)

    return _inert_binding(f"unknown target prefix in {spec.target!r}")


def bake_usd_prim(
    source: MeshSource,
    transform: np.ndarray,
    material_id: int,
    cache_index: dict,
) -> MeshInstance:
    """Bake one USD prim's mesh (with cache). Thread-safe."""
    cache_key = make_cache_key(
        source.content_hash, None, 0, 0.0, None, 0, 1.0,
    )
    mesh = lookup_cached_mesh(cache_index, cache_key, source)
    if mesh is None:
        mesh = bake_mesh(
            source,
            displacement_bytes=None,
            displacement_res=0,
            displacement_scale_world=0.0,
        )
        save_cached_mesh(cache_index, cache_key, mesh)
    return MeshInstance(
        mesh=mesh,
        transform=transform,
        material_id=material_id,
        name=source.name,
        source=source,
        prim_path=source.name,
    )


def load_scene_from_usd(
    stage_path: Path,
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
) -> Scene:
    """Open a USD stage and return a fully-baked `Scene`.

    Reads prims serially, then bakes meshes in parallel. Blocking.
    """
    scene, prim_data, _ = _read_usd_stage(
        stage_path, time=time, use_usd_mtlx_plugin=use_usd_mtlx_plugin,
    )

    cache_index = load_cache_index()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=_USD_POOL_SIZE) as pool:
        instances = list(pool.map(
            lambda pd: bake_usd_prim(pd[0], pd[1], pd[2], cache_index),
            prim_data,
        ))

    scene.instances.extend(instances)
    return scene


def load_scene_from_stage(
    stage: "Usd.Stage",
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
) -> Scene:
    """Read an already-open USD stage and return a fully-baked `Scene`.

    Same as `load_scene_from_usd` but takes a `Usd.Stage` the caller owns
    (e.g. one they mutate between frames). Blocking; bakes meshes in parallel.
    """
    scene, prim_data, _ = _read_open_stage(
        stage, time=time, use_usd_mtlx_plugin=use_usd_mtlx_plugin,
    )
    cache_index = load_cache_index()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=_USD_POOL_SIZE) as pool:
        instances = list(pool.map(
            lambda pd: bake_usd_prim(pd[0], pd[1], pd[2], cache_index),
            prim_data,
        ))

    scene.instances.extend(instances)
    return scene


def prepare_usd_streaming(
    stage_path: Path,
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
) -> tuple[Scene, list[tuple[MeshSource, np.ndarray, int]]]:
    """Read USD stage metadata (fast), return prim data for background baking.

    The returned Scene has materials, lights, camera, and mm_per_unit
    populated. Its instances list contains only emissive-light quads;
    mesh instances should be baked in the background via `bake_usd_prim`.
    """
    scene, prim_data, _ = _read_usd_stage(
        stage_path, time=time, use_usd_mtlx_plugin=use_usd_mtlx_plugin,
    )
    return scene, prim_data


def summarize(scene: Scene) -> str:
    """Human-readable single-string summary used by scripts/inspect_usd.py."""
    lines = [
        f"Scene: {len(scene.instances)} instance(s), {len(scene.materials)} material(s)",
        f"  mm_per_unit: {scene.mm_per_unit}",
        f"  furnace_mode: {scene.furnace_mode}",
        f"  environment: {scene.environment}",
        f"  lights_dir:  {len(scene.lights_dir)}",
        "",
        "  Materials:",
    ]
    for i, mat in enumerate(scene.materials):
        overrides = (
            ", ".join(f"{k}={v}" for k, v in mat.parameter_overrides.items())
            if mat.parameter_overrides else "(no authored overrides)"
        )
        lines.append(f"    [{i}] {mat.name}: {overrides}")
        if mat.texture_paths:
            for input_name, path in mat.texture_paths.items():
                lines.append(f"         texture[{input_name}] = {path}")
    lines.append("")
    lines.append("  Lights:")
    for i, light in enumerate(scene.lights_dir):
        lines.append(
            f"    distant[{i}]: dir={light.direction.tolist()}  "
            f"radiance={light.radiance.tolist()}"
        )
    if scene.environment is not None:
        env = scene.environment
        lines.append(
            f"    dome: {env.name}  intensity={env.intensity}  "
            f"data_shape={tuple(env.data.shape)}"
        )
    if not scene.lights_dir and scene.environment is None:
        lines.append("    (none)")
    lines.append("")
    lines.append("  Instances:")
    for i, inst in enumerate(scene.instances):
        m = inst.mesh
        translation = inst.transform[3, :3]
        lines.append(
            f"    [{i}] {inst.name} -> mat[{inst.material_id}]: "
            f"{m.num_vertices} verts, {m.num_triangles} tris, "
            f"{m.num_nodes} bvh nodes; translation={translation}"
        )
    return "\n".join(lines)
