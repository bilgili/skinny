"""Sidecar .mtlx writer + stage-reference authoring (tasks group 2).

Proves the EMPIRICALLY-VERIFIED loader recipe end-to-end:
- ``write_mtlx_document`` authors a ``doc.validate()``-clean MaterialX document
  whose ``surfacematerial`` element names match bound USD material leaf names,
  with the ``standard_surface`` shader at document root (referenced by
  ``nodename``) and texture-bound inputs wired through ``<image>`` nodes in a
  nodegraph (resolvable by ``_find_image_file_in_nodegraph``).
- ``author_mtlx_reference`` authors a stage→.mtlx reference such that
  ``usd_loader._collect_mtlx_asset_paths`` finds it and the bound mesh resolves
  the rich overrides through ``usd_loader._load_mtlx_materials`` (the
  missing-usdMtlx-plugin fallback path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

mx = pytest.importorskip("MaterialX")
usd_pxr = pytest.importorskip("pxr")
usd_loader = pytest.importorskip("skinny.usd_loader")

from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

from skinny.pbrt import mtlx_emit  # noqa: E402


# ── write_mtlx_document ────────────────────────────────────────────────


def _glass_inputs():
    return {
        "inputs": {
            "transmission": 1.0,
            "specular_IOR": 1.5,
            "transmission_color": (0.9, 0.95, 1.0),
        },
        "tex_inputs": {},
    }


def test_write_produces_validate_clean_document(tmp_path):
    out = tmp_path / "scene.mtlx"
    mtlx_emit.write_mtlx_document({"M_Glass": _glass_inputs()}, str(out))
    assert out.exists()
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(out))
    valid, msg = doc.validate()
    assert valid, msg


def test_surfacematerial_name_matches_key(tmp_path):
    out = tmp_path / "scene.mtlx"
    mtlx_emit.write_mtlx_document({"M_Glass": _glass_inputs()}, str(out))
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(out))
    names = {n.getName() for n in doc.getMaterialNodes()}
    assert names == {"M_Glass"}


def test_surfaceshader_referenced_by_nodename_not_nodegraph(tmp_path):
    # The loader (_load_mtlx_materials) reads ss_input.getNodeName(); the
    # standard_surface must be a document-root node referenced by `nodename`.
    out = tmp_path / "scene.mtlx"
    mtlx_emit.write_mtlx_document({"M_Glass": _glass_inputs()}, str(out))
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(out))
    sm = next(n for n in doc.getMaterialNodes() if n.getName() == "M_Glass")
    ss_input = sm.getInput("surfaceshader")
    assert ss_input is not None
    assert ss_input.getNodeName(), "surfaceshader must connect via nodename"
    ss_node = doc.getNode(ss_input.getNodeName())
    assert ss_node is not None
    assert ss_node.getCategory() == "standard_surface"


def test_constants_authored_on_shader(tmp_path):
    out = tmp_path / "scene.mtlx"
    mtlx_emit.write_mtlx_document({"M_Glass": _glass_inputs()}, str(out))
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(out))
    sm = next(n for n in doc.getMaterialNodes() if n.getName() == "M_Glass")
    ss = doc.getNode(sm.getInput("surfaceshader").getNodeName())
    names = {i.getName() for i in ss.getInputs()}
    assert {"transmission", "specular_IOR", "transmission_color"} <= names
    assert ss.getInput("specular_IOR").getValueString() == "1.5"


def test_boolean_input_authored_as_boolean(tmp_path):
    out = tmp_path / "scene.mtlx"
    mtlx_emit.write_mtlx_document(
        {"M_Thin": {"inputs": {"thin_walled": True}, "tex_inputs": {}}},
        str(out),
    )
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(out))
    sm = next(n for n in doc.getMaterialNodes() if n.getName() == "M_Thin")
    ss = doc.getNode(sm.getInput("surfaceshader").getNodeName())
    tw = ss.getInput("thin_walled")
    assert tw is not None
    assert tw.getType() == "boolean"
    assert tw.getValueString().lower() == "true"


def test_accepts_flat_inputs_dict_without_tex_key(tmp_path):
    # A plain {input: value} dict (no "inputs"/"tex_inputs" wrapper) is accepted.
    out = tmp_path / "scene.mtlx"
    mtlx_emit.write_mtlx_document(
        {"M_Diff": {"base_color": (0.6, 0.6, 0.6), "specular_roughness": 1.0}},
        str(out),
    )
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(out))
    sm = next(n for n in doc.getMaterialNodes() if n.getName() == "M_Diff")
    ss = doc.getNode(sm.getInput("surfaceshader").getNodeName())
    names = {i.getName() for i in ss.getInputs()}
    assert {"base_color", "specular_roughness"} <= names


def test_texture_input_wired_through_image_node(tmp_path):
    out = tmp_path / "scene.mtlx"
    mats = {
        "M_Tex": {
            "inputs": {"base_color": (1.0, 1.0, 1.0)},
            "tex_inputs": {
                "base_color": ("textures/diffuse.png", "sRGB", "color3"),
            },
        }
    }
    mtlx_emit.write_mtlx_document(mats, str(out))
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(out))
    sm = next(n for n in doc.getMaterialNodes() if n.getName() == "M_Tex")
    ss = doc.getNode(sm.getInput("surfaceshader").getNodeName())
    bc = ss.getInput("base_color")
    ng_name = bc.getAttribute("nodegraph")
    out_name = bc.getAttribute("output")
    assert ng_name, "textured input must carry a nodegraph attribute"
    assert out_name, "textured input must carry an output attribute"
    # The loader helper must resolve the image path through the nodegraph.
    resolved = usd_loader._find_image_file_in_nodegraph(
        doc, ng_name, out_name, tmp_path
    )
    assert resolved is not None
    assert resolved == (tmp_path / "textures/diffuse.png").resolve()


def test_invalid_document_raises(tmp_path):
    out = tmp_path / "bad.mtlx"
    # A bogus shader input type should make doc.validate() fail -> raise.
    with pytest.raises(Exception):
        mtlx_emit.write_mtlx_document(
            {"M_Bad": {"inputs": {"transmission": object()}, "tex_inputs": {}}},
            str(out),
        )


# ── author_mtlx_reference + full round-trip ────────────────────────────


def _build_stage_with_glass(tmp_path):
    """Author scene.usda (mesh bound to /Materials/M_Glass) + scene.mtlx and
    wire the reference. Returns the .usda path."""
    mtlx_path = tmp_path / "scene.mtlx"
    usd_path = tmp_path / "scene.usda"
    mtlx_emit.write_mtlx_document({"M_Glass": _glass_inputs()}, str(mtlx_path))

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")
    mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    matprim = UsdShade.Material.Define(stage, "/Materials/M_Glass")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(matprim)

    mtlx_emit.author_mtlx_reference(
        stage, "/Materials/M_Glass", "scene.mtlx", "M_Glass"
    )
    stage.GetRootLayer().Save()
    return usd_path


def test_reference_collected_by_loader(tmp_path):
    usd_path = _build_stage_with_glass(tmp_path)
    stage = Usd.Stage.Open(str(usd_path))
    assert usd_loader._collect_mtlx_asset_paths(stage) == {"scene.mtlx"}


def test_no_shadowing_preview_surface(tmp_path):
    # The Material prim must NOT carry an authored UsdPreviewSurface shader
    # (ComputeBoundMaterial succeeding would bypass the .mtlx fallback).
    usd_path = _build_stage_with_glass(tmp_path)
    stage = Usd.Stage.Open(str(usd_path))
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader":
            shader = UsdShade.Shader(prim)
            sid = shader.GetShaderId()
            assert sid != "UsdPreviewSurface", "shadowing preview surface authored"


def test_material_prim_is_typeless_over_for_plugin_absent_fallback(tmp_path):
    # The Material prim must be a typeless `over` so ComputeBoundMaterial fails
    # without the usdMtlx plugin and the rich .mtlx fallback fires. A `def`-typed
    # Material would make ComputeBoundMaterial succeed (empty) and bypass it.
    usd_path = _build_stage_with_glass(tmp_path)
    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath("/Materials/M_Glass")
    assert prim.IsValid()
    assert prim.GetTypeName() == "", "material prim must stay typeless (plugin absent)"
    mesh = stage.GetPrimAtPath("/World/mesh")
    bound, _rel = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert not bound, "ComputeBoundMaterial must fail so the .mtlx fallback fires"


def test_author_reference_downgrades_existing_def(tmp_path):
    # Author a def-typed Material first, then author_mtlx_reference must
    # downgrade it to a typeless over so the fallback path is taken.
    from pxr import Sdf

    mtlx_path = tmp_path / "scene.mtlx"
    usd_path = tmp_path / "scene.usda"
    mtlx_emit.write_mtlx_document({"M_Glass": _glass_inputs()}, str(mtlx_path))
    stage = Usd.Stage.CreateNew(str(usd_path))
    matprim = UsdShade.Material.Define(stage, "/Materials/M_Glass")
    assert matprim.GetPrim().GetTypeName() == "Material"
    mtlx_emit.author_mtlx_reference(
        stage, "/Materials/M_Glass", "scene.mtlx", "M_Glass"
    )
    spec = stage.GetRootLayer().GetPrimAtPath("/Materials/M_Glass")
    assert spec.specifier == Sdf.SpecifierOver
    assert spec.typeName == ""


def test_author_reference_leaf_mismatch_raises(tmp_path):
    usd_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    with pytest.raises(ValueError):
        mtlx_emit.author_mtlx_reference(
            stage, "/Materials/M_Glass", "scene.mtlx", "M_Other"
        )


def test_round_trip_rich_overrides_via_load_mtlx_materials(tmp_path):
    usd_path = _build_stage_with_glass(tmp_path)
    stage = Usd.Stage.Open(str(usd_path))
    mats = usd_loader._load_mtlx_materials(stage, tmp_path)
    assert "M_Glass" in mats
    ovr = mats["M_Glass"].parameter_overrides
    # transmission -> opacity bridge; specular_IOR -> ior; transmission_color
    # -> diffuseColor migration (all per the loader's documented behavior).
    assert ovr["specular_IOR"] == pytest.approx(1.5)
    assert ovr["ior"] == pytest.approx(1.5)
    assert ovr["opacity"] == pytest.approx(0.0)
    assert ovr["diffuseColor"] == pytest.approx((0.9, 0.95, 1.0))


def test_round_trip_via_full_scene_loader(tmp_path):
    # End-to-end: load_scene_from_stage must surface the bound mesh's rich
    # overrides through the _resolve_material_binding -> _load_mtlx_materials
    # fallback (usdMtlx plugin absent in this interpreter).
    usd_path = _build_stage_with_glass(tmp_path)
    stage = Usd.Stage.Open(str(usd_path))
    scene = usd_loader.load_scene_from_stage(stage)
    glass = [
        m for m in scene.materials if m.parameter_overrides.get("ior") == 1.5
    ]
    assert glass, "no material carried the glass ior override"
    assert glass[0].parameter_overrides["opacity"] == pytest.approx(0.0)


def test_texture_round_trip_resolves_same_path(tmp_path):
    # An image texture authored on base_color resolves to the on-disk path via
    # the loader's _load_mtlx_materials texture intake.
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "albedo.png").write_bytes(b"\x89PNG\r\n")
    mtlx_path = tmp_path / "tex.mtlx"
    usd_path = tmp_path / "tex.usda"
    mats = {
        "M_Tex": {
            "inputs": {"base_color": (1.0, 1.0, 1.0)},
            "tex_inputs": {
                "base_color": ("textures/albedo.png", "sRGB", "color3"),
            },
        }
    }
    mtlx_emit.write_mtlx_document(mats, str(mtlx_path))

    stage = Usd.Stage.CreateNew(str(usd_path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")
    mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    matprim = UsdShade.Material.Define(stage, "/Materials/M_Tex")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(matprim)
    mtlx_emit.author_mtlx_reference(stage, "/Materials/M_Tex", "tex.mtlx", "M_Tex")
    stage.GetRootLayer().Save()

    stage2 = Usd.Stage.Open(str(usd_path))
    loaded = usd_loader._load_mtlx_materials(stage2, tmp_path)
    assert "M_Tex" in loaded
    tex = loaded["M_Tex"].texture_paths
    # base_color maps to the flat key "diffuseColor" via _STD_SURFACE_TO_FLAT.
    assert "diffuseColor" in tex
    assert Path(tex["diffuseColor"]) == (tmp_path / "textures/albedo.png").resolve()
