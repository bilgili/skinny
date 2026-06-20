"""Tests for the skinny-import-pbrt CLI (task 10.3)."""

from __future__ import annotations

import pytest

from skinny.pbrt.cli import main

SCENE = """
Camera "perspective" "float fov" 45
WorldBegin
LightSource "distant" "rgb L" [1 1 1]
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "sphere" "float radius" 1
"""

# A scene exercising rich standard_surface slots UsdPreviewSurface drops, so the
# .mtlx sidecar is the only faithful carrier (glass transmission/IOR).
MTLX_SCENE = """
Camera "perspective" "float fov" 45
WorldBegin
LightSource "distant" "rgb L" [1 1 1]
AttributeBegin
  Material "dielectric" "float eta" 1.5
  Shape "sphere" "float radius" 1
AttributeEnd
AttributeBegin
  Material "conductor" "float roughness" 0.1
  Translate 3 0 0
  Shape "sphere" "float radius" 1
AttributeEnd
"""


def test_cli_emits_loadable_usd(tmp_path, capsys):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    out = tmp_path / "scene.usda"
    rc = main([str(src), "-o", str(out)])
    assert out.exists()
    assert rc == 0  # nothing unsupported
    captured = capsys.readouterr()
    assert "pbrt import report" in captured.out


def test_cli_reports_unsupported_with_nonzero_exit(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text('WorldBegin\nShape "curve"\n')  # unsupported shape
    out = tmp_path / "scene.usda"
    rc = main([str(src), "-o", str(out), "-q"])
    assert rc == 1


# ── -mtlx flag (task group 3) ─────────────────────────────────────────────


def test_cli_mtlx_writes_validate_clean_sidecar(tmp_path):
    """``-mtlx`` writes a ``doc.validate()``-clean .mtlx next to the .usda."""
    mx = pytest.importorskip("MaterialX")
    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    rc = main([str(src), "-o", str(out), "-q", "-m"])
    assert rc == 0
    mtlx = tmp_path / "scene.mtlx"
    assert out.exists()
    assert mtlx.exists()

    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(mtlx))
    valid, msg = doc.validate()
    assert valid, msg
    # surfacematerial elements for both shapes
    sm_names = {n.getName() for n in doc.getMaterialNodes()}
    assert "shape_0_mat" in sm_names
    assert "shape_1_mat" in sm_names


def test_cli_mtlx_stage_references_sidecar(tmp_path):
    """The exported stage references the .mtlx and surfacematerial names match
    the bound Material leaf names (so the loader resolves them)."""
    pytest.importorskip("MaterialX")
    from pxr import Usd

    from skinny.usd_loader import _collect_mtlx_asset_paths

    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    main([str(src), "-o", str(out), "-q", "-m"])

    stage = Usd.Stage.Open(str(out))
    asset_paths = _collect_mtlx_asset_paths(stage)
    assert asset_paths  # non-empty
    assert any(p.endswith("scene.mtlx") for p in asset_paths)


def test_cli_without_mtlx_writes_no_sidecar(tmp_path):
    """No ``-mtlx`` -> no .mtlx file and the usda authors UsdPreviewSurface."""
    from pxr import Usd, UsdShade

    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    main([str(src), "-o", str(out), "-q"])

    assert out.exists()
    assert not (tmp_path / "scene.mtlx").exists()

    stage = Usd.Stage.Open(str(out))
    ids = {
        UsdShade.Shader(p).GetShaderId()
        for p in stage.Traverse()
        if p.GetTypeName() == "Shader"
    }
    assert "UsdPreviewSurface" in ids


def test_cli_without_mtlx_authors_def_material_with_surface(tmp_path):
    """Without ``-mtlx`` the default path is unchanged: each Material is a
    ``def``-typed prim with a connected surface output (not a typeless over)."""
    from pxr import UsdShade

    src = tmp_path / "scene.pbrt"
    src.write_text(MTLX_SCENE)
    out = tmp_path / "scene.usda"
    main([str(src), "-o", str(out), "-q"])

    from pxr import Usd

    stage = Usd.Stage.Open(str(out))
    mats = [UsdShade.Material(p) for p in stage.Traverse() if p.IsA(UsdShade.Material)]
    assert mats, "expected def-typed Material prims on the default path"
    for mat in mats:
        # def Material with a connected surface output (UsdPreviewSurface)
        assert mat.GetSurfaceOutput().HasConnectedSource()
