"""Hostless coverage for GLB-derived USD asset intake (change glb-asset-import).

Two loader fixes and the in-repo converter, all CPU-only:
  * D1 — a UsdUVTexture `file` authored as a connection to a Material interface
    input resolves (Apple usdextract shape), instead of being silently dropped.
  * D2 — a UsdTransform2d on the `st` chain bakes into mesh UVs in raw USD
    st-space before the loader's convention V-flip; identity is a no-op; the
    post-bake UVs key the mesh content hash.
  * glb_import.convert_glb_to_usd round-trips through the loader with resolvable
    textures, and refuses out-of-scope / malformed GLBs by name.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


# ─── fixtures: author the two texture-network shapes ──────────────────────


def _tiny_png(path: Path, rgb: tuple[int, int, int]) -> None:
    from PIL import Image
    Image.new("RGB", (4, 4), rgb).save(path)


def _write_interface_shape(usd_path: Path, tex_png: Path, transform: bool) -> None:
    """Author the Apple-usdextract shape: UsdUVTexture.file connected to a
    Material interface input, optionally with a UsdTransform2d V-flip on st."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
        np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.array([3, 3], dtype=np.int32)))
    # Asymmetric raw glTF UVs so a V-flip is observable.
    raw = np.array([[0.0, 0.0], [1.0, 0.25], [1.0, 0.75], [0.0, 1.0]], np.float32)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(raw))

    material = UsdShade.Material.Define(stage, "/Asset/Mat")
    # The asset value lives on the Material interface input, not the texture.
    iface = material.CreateInput("baseColorTexture", Sdf.ValueTypeNames.Asset)
    iface.Set(Sdf.AssetPath("./" + tex_png.name))

    surface = UsdShade.Shader.Define(stage, "/Asset/Mat/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    st_reader = UsdShade.Shader.Define(stage, "/Asset/Mat/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, "/Asset/Mat/diffuse")
    tex.CreateIdAttr("UsdUVTexture")
    # file is a CONNECTION to the material interface input (D1 target).
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).ConnectToSource(iface)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")

    if transform:
        xf = UsdShade.Shader.Define(stage, "/Asset/Mat/stXform")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((1.0, -1.0))
        xf.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set((0.0, 1.0))
        xf_out = xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf_out)
    else:
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)

    tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    stage.GetRootLayer().Save()


# ─── D1: interface-connected file resolution ──────────────────────────────


@needs_usd
def test_interface_connected_file_resolves(tmp_path):
    from pxr import Usd, UsdShade
    from skinny import usd_loader as L

    png = tmp_path / "diff.png"
    _tiny_png(png, (200, 100, 50))
    usd = tmp_path / "asset.usda"
    _write_interface_shape(usd, png, transform=False)

    stage = Usd.Stage.Open(str(usd))
    surface = UsdShade.Shader(stage.GetPrimAtPath("/Asset/Mat/surface"))
    binding = L._resolve_texture_binding(surface.GetInput("diffuseColor"))
    assert binding is not None, "interface-connected file must resolve, not drop"
    assert binding.path.name == "diff.png"
    assert binding.source_color_space == "sRGB"


@needs_usd
def test_dangling_interface_connection_skips(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from skinny import usd_loader as L

    usd = tmp_path / "dangling.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.Mesh.Define(stage, "/Asset/mesh")
    material = UsdShade.Material.Define(stage, "/Asset/Mat")
    iface = material.CreateInput("baseColorTexture", Sdf.ValueTypeNames.Asset)  # no value
    surface = UsdShade.Shader.Define(stage, "/Asset/Mat/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    tex = UsdShade.Shader.Define(stage, "/Asset/Mat/diffuse")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).ConnectToSource(iface)
    tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)
    stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(str(usd))
    surface = UsdShade.Shader(stage.GetPrimAtPath("/Asset/Mat/surface"))
    binding = L._resolve_texture_binding(surface.GetInput("diffuseColor"))
    assert binding is None, "a dangling interface connection must skip cleanly"


# ─── D2: UsdTransform2d bake ───────────────────────────────────────────────


@needs_usd
def test_transform2d_capture(tmp_path):
    from pxr import Usd, UsdShade
    from skinny import usd_loader as L

    png = tmp_path / "d.png"
    _tiny_png(png, (10, 20, 30))
    usd = tmp_path / "xform.usda"
    _write_interface_shape(usd, png, transform=True)
    stage = Usd.Stage.Open(str(usd))
    surface = UsdShade.Shader(stage.GetPrimAtPath("/Asset/Mat/surface"))
    binding = L._resolve_texture_binding(surface.GetInput("diffuseColor"))
    assert binding.uv_transform == (1.0, -1.0, 0.0, 1.0, 0.0)


@needs_usd
def test_vflip_bakes_to_raw_uv(tmp_path):
    """glTF V-flip + the loader's convention flip cancel: final uploaded UVs
    equal the raw authored primvars:st."""
    from skinny.usd_loader import load_scene_from_usd

    png = tmp_path / "d.png"
    _tiny_png(png, (10, 20, 30))
    usd = tmp_path / "xform.usda"
    _write_interface_shape(usd, png, transform=True)
    scene = load_scene_from_usd(usd)

    raw = np.array([[0.0, 0.0], [1.0, 0.25], [1.0, 0.75], [0.0, 1.0]], np.float32)
    inst = scene.instances[0]
    got = np.asarray(inst.source.uvs, dtype=np.float32)
    np.testing.assert_allclose(got, raw, atol=1e-6)


@needs_usd
def test_no_transform_uv_bit_identical(tmp_path):
    """Without a UsdTransform2d the loader's UV output is the convention-flipped
    st, unchanged by the new code path."""
    from skinny.usd_loader import load_scene_from_usd

    png = tmp_path / "d.png"
    _tiny_png(png, (10, 20, 30))
    usd = tmp_path / "noxf.usda"
    _write_interface_shape(usd, png, transform=False)
    scene = load_scene_from_usd(usd)

    raw = np.array([[0.0, 0.0], [1.0, 0.25], [1.0, 0.75], [0.0, 1.0]], np.float32)
    expected = raw.copy()
    expected[:, 1] = 1.0 - expected[:, 1]  # the pre-existing convention flip
    got = np.asarray(scene.instances[0].source.uvs, dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-6)


@needs_usd
def test_differing_transforms_distinct_cache_keys(tmp_path):
    """Shared UVs under different transforms must produce different content
    hashes so the mesh disk cache cannot serve one for the other."""
    from skinny.usd_loader import load_scene_from_usd

    png = tmp_path / "d.png"
    _tiny_png(png, (10, 20, 30))
    a = tmp_path / "a.usda"
    b = tmp_path / "b.usda"
    _write_interface_shape(a, png, transform=True)   # V-flip
    _write_interface_shape(b, png, transform=False)  # identity
    ha = load_scene_from_usd(a).instances[0].source.content_hash
    hb = load_scene_from_usd(b).instances[0].source.content_hash
    assert ha != hb


# ─── converter round-trip ──────────────────────────────────────────────────


def _minimal_glb(path: Path) -> None:
    """A single textured triangle GLB (POSITION + TEXCOORD_0, one PNG)."""
    import base64
    import json as _json
    from PIL import Image
    import io as _io

    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint16)
    buf = pos.tobytes() + uv.tobytes() + idx.tobytes()
    while len(buf) % 4:
        buf += b"\x00"

    img_io = _io.BytesIO()
    Image.new("RGB", (2, 2), (200, 50, 50)).save(img_io, format="PNG")
    png_bytes = img_io.getvalue()

    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(buf) + len(png_bytes)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes},
            {"buffer": 0, "byteOffset": pos.nbytes, "byteLength": uv.nbytes},
            {"buffer": 0, "byteOffset": pos.nbytes + uv.nbytes, "byteLength": idx.nbytes},
            {"buffer": 0, "byteOffset": len(buf), "byteLength": len(png_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.8}}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    all_buf = buf + png_bytes
    json_bytes = _json.dumps(gltf).encode()
    while len(json_bytes) % 4:
        json_bytes += b" "
    while len(all_buf) % 4:
        all_buf += b"\x00"
    glb = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes) + 8 + len(all_buf))
    glb += struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    glb += struct.pack("<I4s", len(all_buf), b"BIN\x00") + all_buf
    path.write_bytes(glb)


@needs_usd
def test_converter_roundtrips_through_loader(tmp_path):
    from skinny.glb_import import convert_glb_to_usd
    from skinny.usd_loader import load_scene_from_usd

    glb = tmp_path / "tri.glb"
    _minimal_glb(glb)
    usd = convert_glb_to_usd(glb, tmp_path / "out")
    assert usd.exists()
    assert (tmp_path / "out" / "texture_0.png").exists()

    scene = load_scene_from_usd(usd)
    mats = [m for m in scene.materials if m.texture_paths]
    assert mats, "converted material must carry a resolvable texture"
    diff = mats[0].texture_paths.get("diffuseColor")
    assert diff is not None and diff.exists()


@needs_usd
def test_converter_refuses_unsupported_feature(tmp_path):
    from skinny.glb_import import convert_glb_to_usd, GlbImportError

    glb = tmp_path / "draco.glb"
    _minimal_glb(glb)
    # Re-open and inject a Draco extension marker.
    import pygltflib
    g = pygltflib.GLTF2().load(str(glb))
    g.extensionsUsed = ["KHR_draco_mesh_compression"]
    g.save(str(glb))
    with pytest.raises(GlbImportError, match="Draco"):
        convert_glb_to_usd(glb, tmp_path / "out")


@needs_usd
def test_converter_refuses_overwrite(tmp_path):
    from skinny.glb_import import convert_glb_to_usd, GlbImportError

    glb = tmp_path / "tri.glb"
    _minimal_glb(glb)
    out = tmp_path / "out"
    convert_glb_to_usd(glb, out)
    with pytest.raises(GlbImportError, match="already exists"):
        convert_glb_to_usd(glb, out)
    # overwrite=True succeeds
    assert convert_glb_to_usd(glb, out, overwrite=True).exists()


@needs_usd
def test_converter_rejects_malformed(tmp_path):
    from skinny.glb_import import convert_glb_to_usd, GlbImportError

    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb at all")
    with pytest.raises(GlbImportError):
        convert_glb_to_usd(bad, tmp_path / "out")


# ─── MCP tool: scene_import_glb ────────────────────────────────────────────


class _FakeModelRenderer:
    """Minimal add_model contract against a real in-memory stage."""

    def __init__(self):
        from pxr import Usd
        self._usd_stage = Usd.Stage.CreateInMemory()
        self._usd_edit_layer = self._usd_stage.GetSessionLayer()
        self._usd_stage.SetEditTarget(Usd.EditTarget(self._usd_edit_layer))
        self._usd_scene = object()
        self._scene_graph_version = 1
        self._material_version = 1
        self.model_calls: list[str] = []

    def add_model(self, usd_path, *, parent_prim_path="/World", name=None,
                  transform=None, validate=None):
        from pxr import UsdGeom
        self.model_calls.append(usd_path)
        path = f"{parent_prim_path}/{name or 'Model'}"
        UsdGeom.Xform.Define(self._usd_stage, path)
        self._scene_graph_version += 1
        return path


def _mcp_harness(tmp_path):
    import threading
    import time
    from skinny.mcp_server import SceneTools
    from skinny.render_session import RenderCommandQueue

    class Proxy:
        def __init__(self, q): self._q = q
        def request(self, cb): return self._q.post_with_reply(cb)

    queue = RenderCommandQueue()
    renderer = _FakeModelRenderer()
    tools = SceneTools(Proxy(queue), timeout=5.0, roots=[str(tmp_path)],
                       structural_grace=2.0)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            queue.run_pending(renderer)
            time.sleep(0.001)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return tools, renderer, stop, t


@needs_usd
def test_import_glb_rejects_path_outside_roots(tmp_path):
    from skinny.mcp_server import SceneToolError
    tools, renderer, stop, t = _mcp_harness(tmp_path)
    try:
        with pytest.raises(SceneToolError):
            tools.scene_import_glb("/etc/passwd.glb")
        assert renderer.model_calls == []
    finally:
        stop.set(); t.join(timeout=2)


@needs_usd
def test_import_glb_refuses_unsupported_feature(tmp_path):
    from skinny.mcp_server import SceneToolError
    import pygltflib
    glb = tmp_path / "draco.glb"
    _minimal_glb(glb)
    g = pygltflib.GLTF2().load(str(glb))
    g.extensionsUsed = ["KHR_draco_mesh_compression"]
    g.save(str(glb))
    tools, renderer, stop, t = _mcp_harness(tmp_path)
    try:
        with pytest.raises(SceneToolError, match="Draco"):
            tools.scene_import_glb(str(glb))
        assert renderer.model_calls == []
    finally:
        stop.set(); t.join(timeout=2)


@needs_usd
def test_import_glb_success_delegates_to_add_model(tmp_path):
    glb = tmp_path / "tri.glb"
    _minimal_glb(glb)
    tools, renderer, stop, t = _mcp_harness(tmp_path)
    try:
        res = tools.scene_import_glb(str(glb), name="Tri")
        assert res["path"] == "/World/Tri"
        assert "scene_graph_version" in res
        assert len(renderer.model_calls) == 1
        assert renderer.model_calls[0].endswith(".usdc")
    finally:
        stop.set(); t.join(timeout=2)


@needs_usd
def test_import_glb_overwrite_refused(tmp_path):
    from skinny.mcp_server import SceneToolError
    glb = tmp_path / "tri.glb"
    _minimal_glb(glb)
    tools, renderer, stop, t = _mcp_harness(tmp_path)
    try:
        tools.scene_import_glb(str(glb), name="A")          # first import succeeds
        with pytest.raises(SceneToolError, match="already exists"):
            tools.scene_import_glb(str(glb), name="B")       # second refuses
    finally:
        stop.set(); t.join(timeout=2)
