"""Mesh-UV handling for imagemap textures in the pbrt v4 importer.

Covers: explicit-UV passthrough to ``primvars:st`` (trianglemesh + PLY,
ascii/binary), winding-flip alignment, pbrt-faithful default-UV synthesis for
UV-less *textured* shapes, and sphere parametric UVs.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from skinny.pbrt import emit
from skinny.pbrt.api import import_pbrt
from skinny.pbrt.ply import read_ply

pytest.importorskip("pxr")
from pxr import UsdGeom  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _import(tmp_path, scene_text, assets=None):
    for name, data in (assets or {}).items():
        (tmp_path / name).write_bytes(data)
    src = tmp_path / "scene.pbrt"
    src.write_text(scene_text)
    stage, report = import_pbrt(str(src))
    return stage, report


def _first_mesh(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(prim)
    raise AssertionError("no Mesh in stage")


def _st(mesh):
    """Return (values_as_list_of_tuples | None, interpolation | None)."""
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    if not pv.IsDefined():
        return None, None
    return [tuple(v) for v in pv.Get()], pv.GetInterpolation()


def _ply_ascii(uv_names=("s", "t")):
    u, v = uv_names
    return (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"property float {u}\nproperty float {v}\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0 0 0\n"
        "1 0 0 1 0\n"
        "1 1 0 1 1\n"
        "3 0 1 2\n"
    ).encode("ascii")


def _ply_binary(little, uv_names=("u", "v")):
    u, v = uv_names
    endian = "<" if little else ">"
    fmt = "binary_little_endian" if little else "binary_big_endian"
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"property float {u}\nproperty float {v}\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    verts = [(0, 0, 0, 0, 0), (1, 0, 0, 1, 0), (1, 1, 0, 1, 1)]
    body = b"".join(struct.pack(endian + "5f", *vt) for vt in verts)
    body += struct.pack(endian + "B", 3) + struct.pack(endian + "3i", 0, 1, 2)
    return header + body


_UV_TRI = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


# --------------------------------------------------------------------------- #
# 1. explicit-UV passthrough (characterization — already works)
# --------------------------------------------------------------------------- #
def test_trianglemesh_uv_param_to_st(tmp_path):
    scene = """
WorldBegin
Material "diffuse"
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0 ] \
  "integer indices" [0 1 2] "point2 uv" [ 0 0  1 0  1 1 ]
"""
    stage, _ = _import(tmp_path, scene)
    vals, interp = _st(_first_mesh(stage))
    assert interp == UsdGeom.Tokens.vertex
    assert vals == _UV_TRI


def test_trianglemesh_st_alias_to_st(tmp_path):
    scene = """
WorldBegin
Material "diffuse"
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0 ] \
  "integer indices" [0 1 2] "float st" [ 0 0  1 0  1 1 ]
"""
    stage, _ = _import(tmp_path, scene)
    vals, interp = _st(_first_mesh(stage))
    assert interp == UsdGeom.Tokens.vertex
    assert vals == _UV_TRI


@pytest.mark.parametrize(
    "ply_bytes",
    [
        _ply_ascii(("s", "t")),
        _ply_ascii(("u", "v")),
        _ply_binary(True, ("u", "v")),
        _ply_binary(False, ("s", "t")),
        _ply_binary(True, ("texture_u", "texture_v")),
    ],
)
def test_read_ply_extracts_uv(tmp_path, ply_bytes):
    p = tmp_path / "m.ply"
    p.write_bytes(ply_bytes)
    mesh = read_ply(str(p))
    assert mesh.uvs is not None
    assert np.allclose(mesh.uvs, np.array(_UV_TRI))


def test_plymesh_emits_st(tmp_path):
    scene = """
WorldBegin
Material "diffuse"
Shape "plymesh" "string filename" "m.ply"
"""
    stage, _ = _import(tmp_path, scene, assets={"m.ply": _ply_binary(True, ("u", "v"))})
    vals, interp = _st(_first_mesh(stage))
    assert interp == UsdGeom.Tokens.vertex
    assert vals == _UV_TRI


def test_uv_survives_winding_flip(tmp_path):
    """Orientation-reversing bake (B=diag(1,1,-1,1)) flips winding for an
    identity CTM; per-vertex st stays aligned because st is indexed via
    faceVertexIndices, not reordered."""
    scene = """
WorldBegin
Material "diffuse"
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0 ] \
  "integer indices" [0 1 2] "point2 uv" [ 0 0  1 0  1 1 ]
"""
    stage, _ = _import(tmp_path, scene)
    mesh = _first_mesh(stage)
    vals, interp = _st(mesh)
    assert interp == UsdGeom.Tokens.vertex
    # st is indexed via faceVertexIndices -> per-vertex values unchanged
    assert vals == _UV_TRI
    # winding was reversed by the handedness-flip bake matrix
    fvi = list(mesh.GetFaceVertexIndicesAttr().Get())
    assert fvi == [2, 1, 0]


# --------------------------------------------------------------------------- #
# 2. default-UV synthesis (new behavior)
# --------------------------------------------------------------------------- #
def test_textured_uvless_trianglemesh_gets_facevarying_default(tmp_path):
    """A textured UV-less mesh gets pbrt default faceVarying {(0,0),(1,0),(1,1)}."""
    scene = """
WorldBegin
Texture "kd" "spectrum" "imagemap" "string filename" "tex.png"
Material "diffuse" "texture reflectance" "kd"
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0  -1 1 0 ] \
  "integer indices" [0 1 2 0 2 3]
"""
    stage, _ = _import(tmp_path, scene, assets={"tex.png": b"\x89PNG\r\n"})
    vals, interp = _st(_first_mesh(stage))
    assert interp == UsdGeom.Tokens.faceVarying
    # 2 triangles -> 6 face-vertex UVs, each tri (0,0),(1,0),(1,1)
    assert vals == _UV_TRI * 2


def _surface(stage):
    from pxr import UsdShade

    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            sh = UsdShade.Shader(prim)
            if sh.GetIdAttr().Get() == "UsdPreviewSurface":
                return sh
    raise AssertionError("no UsdPreviewSurface in stage")


def test_textured_roughness_connects_to_roughness_input(tmp_path):
    """A FloatTexture roughness maps to the USD roughness input (scalar .r), not
    diffuseColor, and the UV-less mesh gets synthesized UVs."""
    from pxr import UsdShade  # noqa: F401

    scene = """
WorldBegin
Texture "r" "float" "imagemap" "string filename" "rough.png"
Material "conductor" "texture roughness" "r"
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0 ] "integer indices" [0 1 2]
"""
    stage, _ = _import(tmp_path, scene, assets={"rough.png": b"\x89PNG\r\n"})
    vals, _interp = _st(_first_mesh(stage))
    assert vals is not None  # references_texture includes roughness -> default UVs
    surf = _surface(stage)
    rin = surf.GetInput("roughness")
    assert rin and rin.HasConnectedSource()
    api, out_name, _ = rin.GetConnectedSource()
    assert out_name == "r"  # scalar channel
    from pxr import UsdShade as _US

    assert _US.Shader(api.GetPrim()).GetIdAttr().Get() == "UsdUVTexture"
    din = surf.GetInput("diffuseColor")
    assert not (din and din.HasConnectedSource())  # not assumed diffuse


def test_textured_reflectance_connects_to_diffusecolor(tmp_path):
    """A SpectrumTexture reflectance maps to diffuseColor (color .rgb)."""
    scene = """
WorldBegin
Texture "kd" "spectrum" "imagemap" "string filename" "kd.png"
Material "diffuse" "texture reflectance" "kd"
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0 ] "integer indices" [0 1 2]
"""
    stage, _ = _import(tmp_path, scene, assets={"kd.png": b"\x89PNG\r\n"})
    surf = _surface(stage)
    din = surf.GetInput("diffuseColor")
    assert din and din.HasConnectedSource()
    _api, out_name, _ = din.GetConnectedSource()
    assert out_name == "rgb"  # color channel


def test_untextured_uvless_trianglemesh_has_no_st(tmp_path):
    scene = """
WorldBegin
Material "diffuse" "rgb reflectance" [0.6 0.6 0.6]
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0 ] "integer indices" [0 1 2]
"""
    stage, _ = _import(tmp_path, scene)
    vals, interp = _st(_first_mesh(stage))
    assert vals is None


def test_tessellate_sphere_parametric_uv():
    segments, rings = 8, 4
    out = emit.tessellate_sphere(1.0, segments=segments, rings=rings)
    assert len(out) == 4, "tessellate_sphere must return (pts, idx, nrm, uvs)"
    _pts, _idx, _nrm, uvs = out
    assert uvs is not None
    uvs = np.asarray(uvs)
    row = segments + 1
    # vertex (i,j) -> u = j/segments, v = 1 - i/rings
    for i in (0, 1, rings):
        for j in (0, 1, segments):
            k = i * row + j
            assert np.allclose(uvs[k], [j / segments, 1.0 - i / rings]), (i, j, uvs[k])


def test_sphere_shape_emits_vertex_st(tmp_path):
    scene = """
WorldBegin
Material "diffuse"
Shape "sphere" "float radius" 1.0
"""
    stage, _ = _import(tmp_path, scene)
    vals, interp = _st(_first_mesh(stage))
    assert interp == UsdGeom.Tokens.vertex
    assert vals is not None and len(vals) > 0
