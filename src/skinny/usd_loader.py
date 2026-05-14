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
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from skinny.environment import ENV_HEIGHT, ENV_WIDTH, _load_radiance_hdr, _resize_equirect
from skinny.mesh import Mesh, MeshSource, bake_mesh, compute_source_hash
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


def _resolve_texture_input(input_obj: UsdShade.Input) -> Optional[Path]:
    """Walk a connected shader input back to a UsdUVTexture's `file` asset.

    Returns the resolved absolute path or None when the input either isn't
    connected to a texture-loading node or the file path can't be resolved.
    Recognises UsdUVTexture (the UsdPreviewSurface common case) and a
    generic shader with a `file` asset input (covers MaterialX `<image>`
    nodes in the typical wiring).
    """
    if not input_obj.HasConnectedSource():
        return None

    src_info = input_obj.GetConnectedSource()
    if not src_info:
        return None
    src_api, _src_name, _src_type = src_info
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
    # SdfAssetPath: prefer .resolvedPath when the asset resolver succeeded;
    # fall back to the authored path so callers at least see what was asked
    # for. Both cases produce a Path; existence is the caller's problem.
    asset_path = (
        getattr(asset, "resolvedPath", None) or getattr(asset, "path", None) or str(asset)
    )
    if not asset_path:
        return None
    return Path(asset_path)


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
                if inp.HasConnectedSource():
                    tex_path = _resolve_texture_input(inp)
                    if tex_path is not None:
                        textures[inp.GetBaseName()] = tex_path
                    continue
                value = inp.Get()
                if value is None:
                    continue
                overrides[inp.GetBaseName()] = value

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
    cd = shade_mat.GetPrim().GetCustomData()
    if cd:
        hint = cd.get("skinnyMaterialX")
        if isinstance(hint, str) and hint:
            mtlx_target_name = hint
        skinny_overrides = cd.get("skinnyOverrides")
        # USD VtDictionary surfaces as a Python dict (or pxr.Vt.Dictionary).
        if hasattr(skinny_overrides, "items"):
            for k, v in skinny_overrides.items():
                overrides[str(k)] = v

    return Material(
        name=name,
        parameter_overrides=overrides,
        texture_paths=textures,
        mtlx_target_name=mtlx_target_name,
    )


# ─── Direct MaterialX loading (bypass for missing usdMtlx plugin) ────


_STD_SURFACE_TO_FLAT: dict[str, str] = {
    "base_color": "diffuseColor",
    "specular_roughness": "roughness",
    "metalness": "metallic",
    "specular": "specular",
    "specular_IOR": "ior",
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
            transmission_color = overrides.pop("transmission_color", None)
            if isinstance(transmission, (int, float)) and float(transmission) > 0:
                overrides["opacity"] = max(0.0, 1.0 - float(transmission))
                if isinstance(transmission_color, tuple) and len(transmission_color) >= 3:
                    overrides["diffuseColor"] = transmission_color[:3]
                elif overrides.get("diffuseColor") == (0.0, 0.0, 0.0):
                    overrides["diffuseColor"] = (1.0, 1.0, 1.0)

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
    return LightDir(direction=light_dir, radiance=radiance)


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
    radiance = _light_color_radiance(UsdLux.LightAPI(prim))
    return LightSphere(
        position=position, radius=radius, radiance=radiance,
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
        name=source.name, source=source,
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
        focal = float(cam.GetFocalLengthAttr().Get(time) or 50.0)
        v_ap  = float(cam.GetVerticalApertureAttr().Get(time) or 24.0)
        focus_attr = cam.GetFocusDistanceAttr().Get(time)
        focus = float(focus_attr) if focus_attr is not None else None
        fstop_attr = cam.GetFStopAttr().Get(time)
        fstop = float(fstop_attr) if fstop_attr is not None else 0.0
        lens = _extract_lens_system(prim, time)
        return CameraOverride(
            position=position,
            forward=forward,
            focal_length_mm=focal,
            vertical_aperture_mm=v_ap,
            focus_distance=focus,
            fstop=fstop,
            lens=lens,
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


# ─── Public entry point ───────────────────────────────────────────────

_USD_POOL_SIZE = 4


def _read_usd_stage(
    stage_path: Path,
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
    keep_stage: bool = False,
) -> tuple[Scene, list[tuple[MeshSource, np.ndarray, int]], Optional["Usd.Stage"]]:
    """Serial USD stage read: materials, lights, camera, and raw prim data.

    Returns ``(partial_scene, prim_data, stage_or_none)`` where
    *partial_scene* has all metadata (materials, lights, camera,
    mm_per_unit) populated and an ``instances`` list containing only
    emissive-light instances. The mesh prim data is returned separately
    for the caller to bake (blocking or background). When
    ``keep_stage=True`` the open ``Usd.Stage`` handle is returned as the
    third element so the caller can build a scene graph tree from it.
    """
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage: {stage_path}")

    eval_time = time if time is not None else Usd.TimeCode.Default()

    mtlx_materials: dict[str, Material] = {}
    if not use_usd_mtlx_plugin:
        stage_dir = Path(stage.GetRootLayer().realPath).parent
        mtlx_materials = _load_mtlx_materials(stage, stage_dir)

    materials: list[Material] = [Material(name="default")]
    material_index: dict[str, int] = {}

    prim_data: list[tuple[MeshSource, np.ndarray, int]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.IsActive() or prim.IsAbstract():
            continue

        source = _read_mesh_attrs(prim, eval_time)
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
            f"USD stage at {stage_path} contains no usable UsdGeom.Mesh prims"
        )

    lights_dir, lights_sphere, environment, emissive_instances = _extract_lights(
        stage, eval_time, materials, material_index,
    )
    camera_override = _extract_camera(stage, eval_time)

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    mm_per_unit = max(meters_per_unit * 1000.0, 1e-6)

    partial_scene = Scene(
        instances=list(emissive_instances),
        materials=materials,
        lights_dir=lights_dir,
        lights_sphere=lights_sphere,
        environment=environment,
        camera_override=camera_override,
        mm_per_unit=mm_per_unit,
    )
    return partial_scene, prim_data, (stage if keep_stage else None)


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
