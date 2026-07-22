"""Pure-Python GLB → USD conversion for locally generated 3D assets.

Converts a binary glTF (``.glb``) — the output format of image-to-3D models
such as TRELLIS.2 — into a self-contained USD asset the renderer can reference
through ``scene_add_model`` / ``scene_import_glb``. Platform-independent
(pygltflib + pxr): runs identically on macOS, Linux, and Windows, so the asset
pipeline does not depend on Apple's ``usdextract``.

Scope is the shape local generators emit: one or more meshes with
POSITION / NORMAL / TEXCOORD_0, embedded PNG / JPEG / WebP images, and a
``pbrMetallicRoughness`` material. The authored USD is the *canonical simple*
UsdPreviewSurface form — ``file`` values set directly on each ``UsdUVTexture``
(no Material interface indirection), UVs pre-flipped to USD's V convention (no
``UsdTransform2d``), packed metallicRoughness wired ``metallic <- .b`` /
``roughness <- .g``. Out-of-scope glTF features are refused by name rather than
silently mis-imported.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np

# Unsupported glTF features: refused with a naming error rather than
# silently producing a wrong asset.
_UNSUPPORTED_EXTENSIONS = {
    "KHR_draco_mesh_compression": "Draco mesh compression",
    "KHR_mesh_quantization": "mesh quantization",
    "EXT_meshopt_compression": "meshopt compression",
}

# glTF accessor componentType → numpy dtype.
_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GlbImportError(Exception):
    """A GLB cannot be converted (malformed, or uses an unsupported feature)."""


def convert_glb_to_usd(
    glb_path: Path, out_dir: Path, *, overwrite: bool = False
) -> Path:
    """Convert ``glb_path`` into a ``.usdc`` (+ extracted textures) in ``out_dir``.

    Returns the path to the authored ``.usdc``. Raises :class:`GlbImportError`
    for a malformed GLB or one using an unsupported glTF feature, and refuses
    to overwrite an existing conversion unless ``overwrite`` is set.
    """
    import pygltflib
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    glb_path = Path(glb_path)
    out_dir = Path(out_dir)
    usd_path = out_dir / f"{glb_path.stem}.usdc"

    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(out_dir.glob("*.usd*")):
        raise GlbImportError(
            f"conversion already exists in {out_dir} (pass overwrite=true to replace)"
        )

    try:
        gltf = pygltflib.GLTF2().load(str(glb_path))
    except Exception as exc:  # pragma: no cover - pygltflib error shapes vary
        raise GlbImportError(f"could not parse GLB {glb_path.name}: {exc}") from exc
    if gltf is None:
        raise GlbImportError(f"could not parse GLB {glb_path.name}")

    for ext in gltf.extensionsUsed or []:
        if ext in _UNSUPPORTED_EXTENSIONS:
            raise GlbImportError(
                f"GLB uses unsupported feature: {_UNSUPPORTED_EXTENSIONS[ext]} ({ext})"
            )
    if gltf.skins:
        raise GlbImportError("GLB uses unsupported feature: skinning")
    if gltf.animations:
        raise GlbImportError("GLB uses unsupported feature: animation")

    blob = gltf.binary_blob() or b""

    def _accessor(index: int) -> np.ndarray:
        acc = gltf.accessors[index]
        if getattr(acc, "sparse", None) is not None:
            raise GlbImportError("GLB uses unsupported feature: sparse accessors")
        dtype = _COMPONENT_DTYPE.get(acc.componentType)
        if dtype is None:
            raise GlbImportError(f"unsupported accessor componentType {acc.componentType}")
        ncomp = _TYPE_COUNT[acc.type]
        bv = gltf.bufferViews[acc.bufferView]
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        count = acc.count * ncomp
        data = np.frombuffer(
            blob, dtype=dtype, count=count, offset=start
        ).astype(np.float32 if dtype == np.float32 else np.int64)
        return data.reshape(acc.count, ncomp) if ncomp > 1 else data

    textures_by_image = _extract_images(gltf, blob, out_dir)

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())

    usd_materials = _author_materials(
        stage, gltf, textures_by_image, UsdShade, Sdf, Gf
    )

    mesh_num = 0
    for node_mesh in gltf.meshes or []:
        for prim in node_mesh.primitives:
            if prim.mode not in (None, 4):  # 4 = TRIANGLES
                raise GlbImportError(
                    f"unsupported primitive mode {prim.mode} (only triangles)"
                )
            attrs = prim.attributes
            if attrs.POSITION is None:
                continue
            positions = _accessor(attrs.POSITION).astype(np.float32)
            mesh_path = f"/Asset/mesh_{mesh_num}"
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions))

            if prim.indices is not None:
                idx = _accessor(prim.indices).astype(np.int32).reshape(-1)
            else:
                idx = np.arange(len(positions), dtype=np.int32)
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
            mesh.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(np.full(len(idx) // 3, 3, dtype=np.int32))
            )
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

            if attrs.NORMAL is not None:
                normals = _accessor(attrs.NORMAL).astype(np.float32)
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

            if attrs.TEXCOORD_0 is not None:
                st = _accessor(attrs.TEXCOORD_0).astype(np.float32).copy()
                # glTF TEXCOORD_0 is V-down (top-left origin); USD primvars:st
                # is V-up. The loader flips USD→skinny on read, so authoring
                # st = (u, 1 - v_gltf) round-trips back to the raw glTF UV.
                st[:, 1] = 1.0 - st[:, 1]
                primvars = UsdGeom.PrimvarsAPI(mesh)
                pv = primvars.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.vertex,
                )
                pv.Set(Vt.Vec2fArray.FromNumpy(st))

            mat = usd_materials.get(prim.material)
            if mat is not None:
                UsdShade.MaterialBindingAPI(mesh).Bind(mat)
            mesh_num += 1

    if mesh_num == 0:
        stage = None  # release before unlink
        try:
            usd_path.unlink()
        except OSError:
            pass
        raise GlbImportError(f"GLB {glb_path.name} contains no triangle meshes")

    stage.GetRootLayer().Save()
    return usd_path


def _extract_images(gltf, blob: bytes, out_dir: Path) -> dict:
    """Extract embedded glTF images to PNG files beside the USD.

    Returns ``{image_index: relative_png_filename}``. Decodes PNG/JPEG/WebP
    via Pillow and re-encodes as PNG so the renderer's loaders always see a
    format they read.
    """
    from PIL import Image

    result: dict[int, str] = {}
    for i, img in enumerate(gltf.images or []):
        data: Optional[bytes] = None
        if img.bufferView is not None:
            bv = gltf.bufferViews[img.bufferView]
            start = bv.byteOffset or 0
            data = blob[start:start + bv.byteLength]
        elif img.uri and img.uri.startswith("data:"):
            import base64
            data = base64.b64decode(img.uri.split(",", 1)[1])
        if data is None:
            continue
        try:
            decoded = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise GlbImportError(f"could not decode image {i}: {exc}") from exc
        name = f"texture_{i}.png"
        decoded.save(out_dir / name)
        result[i] = name
    return result


def _author_materials(stage, gltf, textures_by_image, UsdShade, Sdf, Gf) -> dict:
    """Author one UsdPreviewSurface material per glTF material.

    Textures reference the extracted PNGs by relative path directly on each
    UsdUVTexture ``file`` input. Packed metallicRoughness is wired with the
    glTF channel convention: roughness ← G, metallic ← B.
    """
    from pxr import UsdGeom

    def _tex_file(tex_index: int) -> Optional[str]:
        if tex_index is None or gltf.textures is None:
            return None
        src = gltf.textures[tex_index].source
        return textures_by_image.get(src)

    materials: dict = {}
    scope = "/Asset/Materials"
    UsdGeom.Scope.Define(stage, scope)
    for i, gmat in enumerate(gltf.materials or []):
        mat_path = f"{scope}/material_{i}"
        material = UsdShade.Material.Define(stage, mat_path)
        surface = UsdShade.Shader.Define(stage, f"{mat_path}/surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(
            surface.ConnectableAPI(), "surface"
        )

        pbr = gmat.pbrMetallicRoughness
        st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        def _texture(name: str, filename: str, colorspace: str):
            tex = UsdShade.Shader.Define(stage, f"{mat_path}/{name}")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{filename}")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            return tex

        base_file = _tex_file(pbr.baseColorTexture.index) if (pbr and pbr.baseColorTexture) else None
        if base_file:
            tex = _texture("diffuseColor", base_file, "sRGB")
            out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(out)
        elif pbr and pbr.baseColorFactor:
            f = pbr.baseColorFactor
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(f[0]), float(f[1]), float(f[2]))
            )

        mr_file = _tex_file(pbr.metallicRoughnessTexture.index) if (pbr and pbr.metallicRoughnessTexture) else None
        if mr_file:
            tex = _texture("metallicRoughness", mr_file, "raw")
            m_out = tex.CreateOutput("b", Sdf.ValueTypeNames.Float)
            r_out = tex.CreateOutput("g", Sdf.ValueTypeNames.Float)
            surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(m_out)
            surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(r_out)
        elif pbr:
            surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                float(pbr.metallicFactor if pbr.metallicFactor is not None else 1.0)
            )
            surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                float(pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0)
            )
        materials[i] = material
    return materials
