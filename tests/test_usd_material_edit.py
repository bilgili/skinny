"""USD-semantics tests for `skinny.usd_material_edit` (mcp-material-authoring,
task 3.4).

Exercises the GPU-free authoring + save-post-process logic directly against pxr:
explicit-target binding overrides a file-authored binding (and a prepend/AddTarget
would have merged — asserted via the list-op form), the anonymous-branch export
post-process yields a self-contained bundle (relative refs, flatten residue
stripped, documents copied), the file-backed overlay re-anchors, and a rollback
sequence leaves the stage and session dir clean.

Hostless — pure `pxr`; never imports `skinny.renderer`."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pxr")

from skinny import usd_material_edit as ume  # noqa: E402

_MTLX = """<?xml version="1.0"?>
<materialx version="1.38">
  <standard_surface name="SR_{n}" type="surfaceshader">
    <input name="base_color" type="color3" value="0.5, 0.5, 0.5" />
  </standard_surface>
  <surfacematerial name="{n}" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_{n}" />
  </surfacematerial>
</materialx>
"""


def _write_mtlx(directory: Path, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{name}.mtlx"
    p.write_text(_MTLX.format(n=name), encoding="utf-8")
    return str(p.resolve())


# ── Explicit binding replaces a file-authored one ─────────────────────


def test_session_binding_overrides_file_binding_with_explicit_targets():
    from pxr import Usd, UsdGeom, UsdShade
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/M")
    UsdShade.Material.Define(stage, "/Materials/A")
    UsdShade.Material.Define(stage, "/Materials/B")
    # File-authored binding on the root layer → A.
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/Materials/A"))
    )

    # Session binding → B, via the module (edit target = session layer).
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    ume.author_binding(stage, "/World/M", "/Materials/B")

    bound, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
    assert str(bound.GetPath()) == "/Materials/B"  # session wins

    # The session op is EXPLICIT (single-target set), not a prepend/append that
    # would compose alongside the file binding and merge.
    spec = stage.GetSessionLayer().GetPropertyAtPath("/World/M.material:binding")
    tl = spec.targetPathList
    assert tl.isExplicit
    assert [str(p) for p in tl.explicitItems] == ["/Materials/B"]
    assert list(tl.prependedItems) == []
    assert list(tl.appendedItems) == []


def test_binding_untyped_mtlx_holder_uses_explicit_settargets(tmp_path):
    """A holder that is not Material-typed but carries a `.mtlx` reference binds
    via an explicit `material:binding` target list."""
    from pxr import Usd, UsdGeom
    mtlx = _write_mtlx(tmp_path, "Holder")
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/M")
    over = stage.OverridePrim("/Materials/Holder")   # untyped
    over.GetReferences().AddReference(mtlx)

    ume.author_binding(stage, "/World/M", "/Materials/Holder")
    rel = mesh.GetPrim().GetRelationship("material:binding")
    assert [str(t) for t in rel.GetTargets()] == ["/Materials/Holder"]
    spec = stage.GetRootLayer().GetPropertyAtPath("/World/M.material:binding")
    assert spec.targetPathList.isExplicit


# ── Override-preservation identity: same-leaf materials do not collide (D) ──


def _preview_material(stage, mat_path, color):
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def test_same_leaf_materials_get_distinct_source_prim_path():
    """Two materials sharing a leaf name in different scopes (/ScopeA/Foo vs
    /ScopeB/Foo) must load with distinct `source_prim_path` so the override-
    preservation snapshot keys them apart instead of cross-applying (finding
    #7/D). Keyed by leaf name alone they would collide."""
    from pxr import Usd, UsdGeom, UsdShade
    from skinny.usd_loader import load_scene_from_stage

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for scope, color in (("ScopeA", (0.9, 0.1, 0.1)), ("ScopeB", (0.1, 0.1, 0.9))):
        UsdGeom.Scope.Define(stage, f"/World/{scope}")
        _preview_material(stage, f"/World/{scope}/Foo", color)
        gprim = UsdGeom.Sphere.Define(stage, f"/World/{scope}/Ball")
        UsdShade.MaterialBindingAPI.Apply(gprim.GetPrim())
        UsdShade.MaterialBindingAPI(gprim.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath(f"/World/{scope}/Foo"))
        )

    scene = load_scene_from_stage(stage)
    foos = [m for m in scene.materials if getattr(m, "name", None) == "Foo"]
    assert len(foos) == 2  # both same-leaf materials loaded
    paths = {m.source_prim_path for m in foos}
    assert paths == {"/World/ScopeA/Foo", "/World/ScopeB/Foo"}  # distinct identity


# ── Anonymous-root save post-process → self-contained bundle ───────────


def test_anonymous_export_postprocess_bundle(tmp_path):
    from pxr import Sdf, Usd, UsdGeom
    session_dir = tmp_path / "session"
    mtlx = _write_mtlx(session_dir, "SynthMat")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Sphere.Define(stage, "/World/Ball")
    ume.ensure_materials_scope(stage)
    ume.author_material_holder(stage, "/Materials/SynthMat", mtlx)

    holders = ume.collect_material_holders(stage)
    assert holders == {"/Materials/SynthMat": mtlx}

    save_dir = tmp_path / "out"
    save_dir.mkdir()
    target = save_dir / "scene.usda"
    stage.Export(str(target))  # anonymous branch: flattens, strips ref arcs
    saved = Sdf.Layer.FindOrOpen(str(target))
    ume.finalize_saved_materials(
        saved, holders, str(save_dir), str(session_dir), flattened=True,
    )
    saved.Save()

    # Document copied into the bundle.
    assert (save_dir / "materials" / "SynthMat.mtlx").exists()
    # Holder re-authored with a RELATIVE reference and no flatten residue.
    holder_spec = saved.GetPrimAtPath("/Materials/SynthMat")
    refs = list(holder_spec.referenceList.explicitItems)
    assert [r.assetPath for r in refs] == ["materials/SynthMat.mtlx"]
    assert list(holder_spec.nameChildren) == []
    # And the saved bundle recomposes with the holder reference intact.
    reopened = Usd.Stage.Open(str(target))
    assert ume.collect_material_holders(reopened)[
        "/Materials/SynthMat"
    ].endswith("materials/SynthMat.mtlx")


def test_curated_preset_keeps_absolute_reference(tmp_path):
    """Texture carve-out: a curated preset (outside the session dir) is not copied;
    its reference stays absolute into the assets tree."""
    from pxr import Sdf, Usd, UsdGeom
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    assets_dir = tmp_path / "assets"
    curated = _write_mtlx(assets_dir, "Wood")
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Sphere.Define(stage, "/World/Ball")
    ume.ensure_materials_scope(stage)
    ume.author_material_holder(stage, "/Materials/Wood", curated)
    holders = ume.collect_material_holders(stage)

    save_dir = tmp_path / "out"
    save_dir.mkdir()
    target = save_dir / "scene.usda"
    stage.Export(str(target))
    saved = Sdf.Layer.FindOrOpen(str(target))
    ume.finalize_saved_materials(
        saved, holders, str(save_dir), str(session_dir), flattened=True,
    )
    saved.Save()

    assert not (save_dir / "materials" / "Wood.mtlx").exists()  # not copied
    refs = list(saved.GetPrimAtPath("/Materials/Wood").referenceList.explicitItems)
    assert refs[0].assetPath == str(Path(curated).resolve())    # absolute kept


# ── File-backed overlay re-anchor ─────────────────────────────────────


def test_file_backed_overlay_reanchor(tmp_path):
    from pxr import Sdf, Usd, UsdGeom
    session_dir = tmp_path / "session"
    mtlx = _write_mtlx(session_dir, "SynthMat")
    # A real on-disk root layer (file-backed stage).
    root_path = tmp_path / "scene.usda"
    base = Usd.Stage.CreateNew(str(root_path))
    UsdGeom.Xform.Define(base, "/World")
    base.GetRootLayer().Save()

    stage = Usd.Stage.Open(str(root_path))
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    ume.ensure_materials_scope(stage)
    ume.author_material_holder(stage, "/Materials/SynthMat", mtlx)
    holders = ume.collect_material_holders(stage)

    save_dir = tmp_path / "out"
    save_dir.mkdir()
    target = save_dir / "scene.edits.usda"
    stage.GetSessionLayer().Export(str(target))  # overlay: keeps ref arcs
    saved = Sdf.Layer.FindOrOpen(str(target))
    ume.finalize_saved_materials(
        saved, holders, str(save_dir), str(session_dir), flattened=False,
    )
    saved.Save()

    assert (save_dir / "materials" / "SynthMat.mtlx").exists()
    refs = list(saved.GetPrimAtPath("/Materials/SynthMat").referenceList.explicitItems)
    assert refs[0].assetPath == "materials/SynthMat.mtlx"


def test_relative_reference_resolves_against_authoring_layer(tmp_path):
    """A relative `.mtlx` reference resolves against the layer that authored it,
    not the CWD or root layer's assumed dir (finding #6) — so a saved file-backed
    overlay referencing `materials/Foo.mtlx` beside it reloads from a moved dir."""
    from pxr import Usd, UsdGeom, UsdShade
    bundle = tmp_path / "bundle"
    _write_mtlx(bundle / "materials", "Foo")
    root_path = bundle / "scene.usda"
    stage = Usd.Stage.CreateNew(str(root_path))
    UsdGeom.Xform.Define(stage, "/World")
    ume.ensure_materials_scope(stage)
    mat = UsdShade.Material.Define(stage, "/Materials/Foo")
    # A RELATIVE reference authored on the file-backed root layer.
    mat.GetPrim().GetReferences().AddReference("materials/Foo.mtlx")
    stage.GetRootLayer().Save()

    resolved = ume.collect_material_holders(stage)["/Materials/Foo"]
    assert Path(resolved).is_absolute()
    assert Path(resolved) == (bundle / "materials" / "Foo.mtlx").resolve()

    # resolve_layer_asset_path leaves an absolute path untouched, and an
    # anonymous layer (no anchor) passes the path through as-authored.
    abspath = str((bundle / "materials" / "Foo.mtlx").resolve())
    assert ume.resolve_layer_asset_path(stage.GetRootLayer(), abspath) == abspath
    anon_stage = Usd.Stage.CreateInMemory()  # hold the stage so the layer stays live
    anon = anon_stage.GetRootLayer()
    assert ume.resolve_layer_asset_path(anon, "rel/x.mtlx") == "rel/x.mtlx"


# ── Rollback cleanliness ──────────────────────────────────────────────


def test_rollback_leaves_stage_and_session_dir_clean(tmp_path):
    """The add_material rollback sequence — remove holder + auto-created scope,
    delete the session file — leaves no trace on the stage or on disk."""
    from pxr import Usd, UsdGeom
    session_dir = tmp_path / "session"
    mtlx = _write_mtlx(session_dir, "SynthMat")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")

    created_scope = ume.ensure_materials_scope(stage)
    assert created_scope
    ume.author_material_holder(stage, "/Materials/SynthMat", mtlx)
    assert stage.GetPrimAtPath("/Materials/SynthMat").IsValid()

    # Rollback: remove prims (incl. auto-created scope) + delete the file.
    stage.RemovePrim("/Materials/SynthMat")
    if created_scope:
        stage.RemovePrim("/Materials")
    Path(mtlx).unlink()

    assert not stage.GetPrimAtPath("/Materials").IsValid()
    assert not stage.GetPrimAtPath("/Materials/SynthMat").IsValid()
    assert not Path(mtlx).exists()
    assert ume.collect_material_holders(stage) == {}
