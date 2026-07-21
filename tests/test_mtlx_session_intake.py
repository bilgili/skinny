"""Session-layer `.mtlx` material intake (mcp-material-authoring, task 2.2).

`add_material` authors a typed holder + `.mtlx` reference into the stage's
SESSION layer, whose asset paths are absolute. The loader's asset collection and
per-prim reference detection must see those session-layer specs (in addition to
the root layer), and material participation stays binding-driven: an unbound
holder is not loaded; a bound one is. Root-layer scenes must be byte-unchanged.

Hostless — pure `pxr` + `skinny.usd_loader`; never imports `skinny.renderer`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from skinny.usd_loader import (  # noqa: E402
    _collect_mtlx_asset_paths,
    _prim_has_mtlx_reference,
    _read_mtlx_mapping_sidecar,
    load_scene_from_stage,
)

_MTLX = """<?xml version="1.0"?>
<materialx version="1.38">
  <standard_surface name="SR_{name}" type="surfaceshader">
    <input name="base_color" type="color3" value="0.1, 0.8, 0.3" />
  </standard_surface>
  <surfacematerial name="{name}" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_{name}" />
  </surfacematerial>
</materialx>
"""


def _write_mtlx(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.mtlx"
    path.write_text(_MTLX.format(name=name), encoding="utf-8")
    return str(path.resolve())


def _session_holder(stage, name: str, mtlx_abspath: str):
    """Author a typed Material holder referencing `mtlx_abspath` into the SESSION
    layer — exactly what `Renderer.add_material` does under its edit context."""
    from pxr import Usd, UsdGeom, UsdShade
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    UsdGeom.Scope.Define(stage, "/Materials")
    holder = UsdShade.Material.Define(stage, f"/Materials/{name}")
    holder.GetPrim().GetReferences().AddReference(mtlx_abspath)
    return holder


def _base_stage():
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def test_collect_finds_session_layer_reference(tmp_path):
    """A `.mtlx` reference authored only in the session layer is collected."""
    mtlx = _write_mtlx(tmp_path, "SessMat")
    stage = _base_stage()
    _session_holder(stage, "SessMat", mtlx)

    paths = _collect_mtlx_asset_paths(stage)
    assert mtlx in paths
    assert _prim_has_mtlx_reference(stage, "/Materials/SessMat")


def test_unbound_session_holder_not_loaded(tmp_path):
    """Binding-driven participation (design D8): an unbound holder does not enter
    the material table even though its reference is discoverable."""
    from pxr import UsdGeom
    mtlx = _write_mtlx(tmp_path, "SessMat")
    stage = _base_stage()
    UsdGeom.Sphere.Define(stage, "/World/Ball")  # geometry, but binds nothing
    _session_holder(stage, "SessMat", mtlx)

    scene = load_scene_from_stage(stage)
    assert "SessMat" not in {m.name for m in scene.materials}


def test_bound_session_holder_is_loaded(tmp_path):
    """A geometry prim binding the session holder loads it from the session-layer
    reference exactly as a root-layer reference would."""
    from pxr import UsdGeom, UsdShade
    mtlx = _write_mtlx(tmp_path, "SessMat")
    stage = _base_stage()
    ball = UsdGeom.Sphere.Define(stage, "/World/Ball")
    _session_holder(stage, "SessMat", mtlx)
    # Bind in the session layer too (edit target already set there).
    UsdShade.MaterialBindingAPI.Apply(ball.GetPrim())
    UsdShade.MaterialBindingAPI(ball.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/Materials/SessMat"))
    )

    scene = load_scene_from_stage(stage)
    mat = next((m for m in scene.materials if m.name == "SessMat"), None)
    assert mat is not None, "bound session material should be in the table"
    # base_color 0.1,0.8,0.3 folds to the flat diffuseColor override.
    assert "diffuseColor" in mat.parameter_overrides
    assert mat.logical_inputs == {}  # no sidecar → empty mapping


def test_sidecar_mapping_attaches_to_material(tmp_path):
    """A `<stem>.json` sidecar's `logical_inputs` is read onto the Material."""
    from pxr import UsdGeom, UsdShade
    mtlx = _write_mtlx(tmp_path, "SynthMat")
    (tmp_path / "SynthMat.json").write_text(
        json.dumps({"logical_inputs": {"colorA": ["blend_bg"], "scale": ["scaled_in2"]}}),
        encoding="utf-8",
    )
    stage = _base_stage()
    ball = UsdGeom.Sphere.Define(stage, "/World/Ball")
    _session_holder(stage, "SynthMat", mtlx)
    UsdShade.MaterialBindingAPI.Apply(ball.GetPrim())
    UsdShade.MaterialBindingAPI(ball.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/Materials/SynthMat"))
    )

    scene = load_scene_from_stage(stage)
    mat = next((m for m in scene.materials if m.name == "SynthMat"), None)
    assert mat is not None
    assert mat.logical_inputs == {"colorA": ["blend_bg"], "scale": ["scaled_in2"]}


def test_read_sidecar_missing_returns_empty(tmp_path):
    assert _read_mtlx_mapping_sidecar(tmp_path / "nope.mtlx") == {}


def test_root_layer_collection_unchanged(tmp_path):
    """A root-layer-only reference is still collected and no session-layer prim is
    invented — the added session scan must not perturb the root-layer path."""
    from pxr import UsdGeom, UsdShade
    mtlx = _write_mtlx(tmp_path, "RootMat")
    stage = _base_stage()
    UsdGeom.Sphere.Define(stage, "/World/Ball")
    # Author entirely on the root layer (default edit target).
    UsdGeom.Scope.Define(stage, "/Materials")
    holder = UsdShade.Material.Define(stage, "/Materials/RootMat")
    holder.GetPrim().GetReferences().AddReference(mtlx)

    assert _collect_mtlx_asset_paths(stage) == {mtlx}
    assert _prim_has_mtlx_reference(stage, "/Materials/RootMat")
    # Session layer is empty → nothing extra discovered.
    assert stage.GetSessionLayer().empty
