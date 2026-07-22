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
# Divisor for glTF `normalized` integer accessors (KHR spec §3.6.2.2).
_NORMALIZE_DIV = {
    np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0,
}


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
    if not overwrite and (any(out_dir.glob("*.usd*"))
                          or any(out_dir.glob("texture_*.png"))):
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

    # The converter emits meshes at identity (single-asset generator output);
    # a node transform or an instanced mesh would be silently misplaced, so
    # refuse loudly rather than drop it (design scope: local generator output).
    mesh_ref_count: dict[int, int] = {}
    for node in gltf.nodes or []:
        if node.mesh is None:
            continue
        mesh_ref_count[node.mesh] = mesh_ref_count.get(node.mesh, 0) + 1
        if node.matrix is not None or node.translation is not None or \
                node.rotation is not None or node.scale is not None:
            raise GlbImportError(
                "GLB uses unsupported feature: node transforms "
                "(mesh is not authored at identity)"
            )
    if any(c > 1 for c in mesh_ref_count.values()):
        raise GlbImportError("GLB uses unsupported feature: mesh instancing")

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
        itemsize = np.dtype(dtype).itemsize
        elem = ncomp * itemsize
        stride = bv.byteStride or elem
        base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        # Validate the read stays inside the bufferView and the buffer, so a
        # malformed accessor errors instead of reading adjacent data.
        span = (acc.count - 1) * stride + elem if acc.count else 0
        bv_end = (bv.byteOffset or 0) + (bv.byteLength or 0)
        if base + span > bv_end or bv_end > len(blob):
            raise GlbImportError(f"accessor {index} reads past its bufferView/buffer")
        if stride == elem:
            arr = np.frombuffer(
                blob, dtype=dtype, count=acc.count * ncomp, offset=base
            ).reshape(acc.count, ncomp)
        else:
            # Interleaved buffer view: gather each element across the stride.
            raw = np.frombuffer(blob, dtype=np.uint8, count=span, offset=base)
            sel = (np.arange(acc.count)[:, None] * stride
                   + np.arange(elem)[None, :]).reshape(-1)
            arr = raw[sel].reshape(acc.count, elem).view(dtype).reshape(acc.count, ncomp)
        if getattr(acc, "normalized", False) and dtype in _NORMALIZE_DIV:
            arr = arr.astype(np.float32) / _NORMALIZE_DIV[dtype]
            if dtype in (np.int8, np.int16):
                arr = np.maximum(arr, -1.0)
        elif dtype == np.float32:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.int64)
        return arr.reshape(-1) if ncomp == 1 else arr

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
            if getattr(prim, "targets", None):
                raise GlbImportError("GLB uses unsupported feature: morph targets")
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
            # External (file-relative) image URI: refuse rather than silently
            # fall the material back to a constant. Generator GLBs embed images.
            raise GlbImportError(
                f"image {i} is an external URI ({img.uri!r}); only embedded "
                "images are supported"
            )
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

        def _texture(name, filename, colorspace, scale=None):
            tex = UsdShade.Shader.Define(stage, f"{mat_path}/{name}")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{filename}")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            # glTF final value = texture × factor; UsdUVTexture `scale` multiplies
            # the sampled RGBA per channel, so the factor rides here.
            if scale is not None and tuple(scale) != (1.0, 1.0, 1.0, 1.0):
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            return tex

        bcf = (pbr.baseColorFactor if pbr and pbr.baseColorFactor else [1.0, 1.0, 1.0, 1.0])
        base_file = _tex_file(pbr.baseColorTexture.index) if (pbr and pbr.baseColorTexture) else None
        if base_file:
            tex = _texture("diffuseColor", base_file, "sRGB",
                           scale=(float(bcf[0]), float(bcf[1]), float(bcf[2]), 1.0))
            out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(out)
        elif pbr and pbr.baseColorFactor:
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(bcf[0]), float(bcf[1]), float(bcf[2]))
            )

        mf = float(pbr.metallicFactor if pbr and pbr.metallicFactor is not None else 1.0)
        rf = float(pbr.roughnessFactor if pbr and pbr.roughnessFactor is not None else 1.0)
        mr_file = _tex_file(pbr.metallicRoughnessTexture.index) if (pbr and pbr.metallicRoughnessTexture) else None
        if mr_file:
            # metallic reads .b, roughness reads .g; scale each by its factor.
            tex = _texture("metallicRoughness", mr_file, "raw", scale=(1.0, rf, mf, 1.0))
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
