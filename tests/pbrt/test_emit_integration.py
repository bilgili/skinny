"""End-to-end: pbrt scene -> USD -> skinny usd_loader (tasks 4.1, 5, 6)."""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt.api import import_pbrt

usd_loader = pytest.importorskip("skinny.usd_loader")


SCENE = """
LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" 40
Film "rgb" "integer xresolution" 64 "integer yresolution" 64
Integrator "path" "integer maxdepth" 5
WorldBegin
LightSource "distant" "blackbody L" [6500] "point3 from" [0 0 0] "point3 to" [0 0 -1]
AttributeBegin
  Material "diffuse" "rgb reflectance" [0.6 0.6 0.6]
  Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0  -1 1 0 ] "integer indices" [0 1 2 0 2 3]
AttributeEnd
AttributeBegin
  Material "conductor" "float roughness" 0.1
  Translate 0 0 -1
  Shape "sphere" "float radius" 0.5
AttributeEnd
"""


def _import(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    stage, report = import_pbrt(str(src))
    return stage, report


def test_imports_and_loads_via_usd_loader(tmp_path):
    stage, report = _import(tmp_path)
    scene = usd_loader.load_scene_from_stage(stage)
    # two shapes -> two instances
    assert len(scene.instances) == 2
    # one distant light
    assert len(scene.lights_dir) == 1
    # camera authored
    assert scene.camera_override is not None
    assert report.count("skipped") == 0


def test_camera_position_and_forward(tmp_path):
    stage, _ = _import(tmp_path)
    scene = usd_loader.load_scene_from_stage(stage)
    cam = scene.camera_override
    # eye (0,0,5) mirrored by the z-flip -> (0,0,-5); looking back toward +z
    assert np.allclose(cam.position, [0, 0, -5], atol=1e-4)
    assert np.allclose(cam.forward / np.linalg.norm(cam.forward), [0, 0, 1], atol=1e-4)


TEX_SCENE = """
WorldBegin
Texture "kd" "spectrum" "imagemap" "string filename" "tex.png"
Material "diffuse" "texture reflectance" "kd"
Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  0 1 0 ] "integer indices" [0 1 2]
"""


def test_imagemap_reflectance_wires_texture(tmp_path):
    (tmp_path / "tex.png").write_bytes(b"\x89PNG\r\n")  # placeholder asset
    src = tmp_path / "scene.pbrt"
    src.write_text(TEX_SCENE)
    stage, report = import_pbrt(str(src))

    from pxr import UsdShade

    uvtex = [
        p for p in stage.Traverse()
        if p.GetTypeName() == "Shader" and UsdShade.Shader(p).GetShaderId() == "UsdUVTexture"
    ]
    assert uvtex, "expected a UsdUVTexture shader"
    asset = UsdShade.Shader(uvtex[0]).GetInput("file").Get()
    assert str(asset.path).endswith("tex.png")

    # the loader resolves the connection into a diffuseColor texture binding
    scene = usd_loader.load_scene_from_stage(stage)
    assert any("diffuseColor" in (m.texture_bindings or {}) for m in scene.materials)


def test_exported_usda_file_loads(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    out = tmp_path / "scene.usda"
    import_pbrt(str(src), out=str(out))
    assert out.exists()
    from pxr import Usd

    scene = usd_loader.load_scene_from_stage(Usd.Stage.Open(str(out)))
    assert len(scene.instances) == 2


# ── -mtlx end to end (task group 3) ───────────────────────────────────────

# Glass + rough metal: both carry standard_surface slots UsdPreviewSurface drops
# (transmission/specular_IOR, specular_anisotropy), so the .mtlx is load-bearing.
MTLX_SCENE = """
LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" 40
Film "rgb" "integer xresolution" 64 "integer yresolution" 64
WorldBegin
LightSource "distant" "rgb L" [1 1 1]
AttributeBegin
  Material "dielectric" "float eta" 1.5
  Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0  -1 1 0 ] "integer indices" [0 1 2 0 2 3]
AttributeEnd
AttributeBegin
  Material "conductor" "float roughness" 0.1
  Translate 0 0 -1
  Shape "sphere" "float radius" 0.5
AttributeEnd
"""


def test_mtlx_export_writes_validate_clean_sidecar(tmp_path):
    mx = pytest.importorskip("MaterialX")
    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    _stage, report = import_pbrt(str(src), out=str(out), materialx=True)
    mtlx = tmp_path / "scene.mtlx"
    assert out.exists()
    assert mtlx.exists()
    assert report.count("skipped") == 0

    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(mtlx))
    valid, msg = doc.validate()
    assert valid, msg


def test_mtlx_export_stage_references_sidecar_with_matching_names(tmp_path):
    pytest.importorskip("MaterialX")
    from skinny.usd_loader import _collect_mtlx_asset_paths

    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    stage, _ = import_pbrt(str(src), out=str(out), materialx=True)

    # stage references the .mtlx
    asset_paths = _collect_mtlx_asset_paths(stage)
    assert asset_paths
    assert any(p.endswith("scene.mtlx") for p in asset_paths)

    # surfacematerial names in the doc == bound-material-binding leaf names.
    # NB: the Material prims are downgraded to typeless `over`s (so the .mtlx
    # fallback fires without the usdMtlx plugin), hence they are matched by the
    # mesh binding-relationship target leaf, not IsA(Material).
    import MaterialX as mx

    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(tmp_path / "scene.mtlx"))
    sm_names = {n.getName() for n in doc.getMaterialNodes()}

    from pxr import UsdShade

    bound_leaves = set()
    for prim in stage.Traverse():
        binding = UsdShade.MaterialBindingAPI(prim)
        rel = binding.GetDirectBindingRel()
        for target in rel.GetTargets():
            bound_leaves.add(target.name)
    # every authored surfacematerial is referenced by a mesh binding
    assert sm_names <= bound_leaves
    assert {"shape_0_mat", "shape_1_mat"} <= sm_names


def test_mtlx_export_round_trips_rich_overrides_through_loader(tmp_path):
    """The reopened stage carries the rich glass overrides via the .mtlx
    fallback (transmission -> opacity=0 bridge in the loader)."""
    pytest.importorskip("MaterialX")
    from pxr import Usd

    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    import_pbrt(str(src), out=str(out), materialx=True)

    scene = usd_loader.load_scene_from_stage(Usd.Stage.Open(str(out)))
    assert len(scene.instances) == 2
    # glass material: loader bridges transmission -> opacity 0.0
    overrides = [m.parameter_overrides or {} for m in scene.materials]
    assert any(
        ov.get("opacity") == 0.0 or ov.get("specular_IOR") == 1.5 or ov.get("ior") == 1.5
        for ov in overrides
    ), overrides


def test_mtlx_export_does_not_author_preview_surface(tmp_path):
    """Under -mtlx the Material prims must NOT carry a UsdPreviewSurface shader
    (it would shadow the .mtlx fallback)."""
    pytest.importorskip("MaterialX")
    from pxr import UsdShade

    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    stage, _ = import_pbrt(str(src), out=str(out), materialx=True)

    preview = [
        p for p in stage.Traverse()
        if p.GetTypeName() == "Shader"
        and UsdShade.Shader(p).GetShaderId() == "UsdPreviewSurface"
    ]
    assert not preview, "expected no UsdPreviewSurface shaders under -mtlx"


def test_in_memory_materialx_falls_back_to_preview_surface(tmp_path):
    """materialx=True without an output path has nowhere to write the sidecar,
    so it must fall back to UsdPreviewSurface (no dangling reference)."""
    pytest.importorskip("MaterialX")
    from pxr import UsdShade

    from skinny.usd_loader import _collect_mtlx_asset_paths

    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    stage, _ = import_pbrt(str(src), materialx=True)  # no out=
    assert not _collect_mtlx_asset_paths(stage)
    ids = {
        UsdShade.Shader(p).GetShaderId()
        for p in stage.Traverse()
        if p.GetTypeName() == "Shader"
    }
    assert "UsdPreviewSurface" in ids
