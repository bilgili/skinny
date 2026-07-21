"""Pure USD material-authoring helpers shared by the renderer and hostless tests.

The GPU-free half of the `mcp-material-authoring` renderer work (design D2, D6,
D7). Import-light (pxr only, no `vulkan`, no GPU) so the authoring + save
post-process logic is unit-testable without constructing a `Renderer`. The
renderer methods (`add_material`, `bind_material`, `save_edits`) enter the
session `Usd.EditContext` and own rollback/resync; these functions do the actual
`Sdf`/`UsdShade` authoring under whatever edit target is active.

Contract highlights:
- Holder prims are **typed** `UsdShade.Material` (design D2) so they classify as
  materials and give binding a stable target path.
- Bindings are authored with **explicit** targets (`Bind()` on a typed holder is
  explicit — verified — else `SetTargets`), never prepend/append, so a session
  binding *replaces* a file-authored one under LIVRPS rather than merging.
- Save post-process re-anchors `.mtlx` references and copies session-synthesized
  documents into a `materials/` bundle; curated (texture-bearing) presets keep
  absolute references into the assets tree (design D7 texture carve-out).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

# Full UsdPreviewSurface input set authored on an inline preview material so a
# later property edit has a value to override (a post-creation write would not
# persist to a saved edit layer). (name, Sdf type stem, default).
_PREVIEW_INPUTS: tuple[tuple[str, str, object], ...] = (
    ("diffuseColor", "Color3f", (0.8, 0.8, 0.8)),
    ("emissiveColor", "Color3f", (0.0, 0.0, 0.0)),
    ("specularColor", "Color3f", (0.0, 0.0, 0.0)),
    ("roughness", "Float", 0.5),
    ("metallic", "Float", 0.0),
    ("clearcoat", "Float", 0.0),
    ("clearcoatRoughness", "Float", 0.01),
    ("opacity", "Float", 1.0),
    ("ior", "Float", 1.5),
)

_MATERIALS_SCOPE = "/Materials"


def ensure_materials_scope(stage) -> bool:
    """Define the ``/Materials`` scope if absent; return True when it was created.

    The boolean lets the caller roll back an auto-created scope on a later
    failure (design: rollback removes prims *including* the auto-created scope).
    """
    from pxr import UsdGeom
    if stage.GetPrimAtPath(_MATERIALS_SCOPE).IsValid():
        return False
    UsdGeom.Scope.Define(stage, _MATERIALS_SCOPE)
    return True


def author_material_holder(stage, holder_path: str, mtlx_abspath: str) -> None:
    """Author a typed ``UsdShade.Material`` holder carrying a ``.mtlx`` reference.

    The reference uses the **absolute** asset path (the anonymous session layer
    offers no anchor for a relative one — design D2). The holder prim name is the
    caller's responsibility to keep equal to the document's surfacematerial name
    (the naming contract the loader's leaf-name binding resolution relies on).
    """
    from pxr import UsdShade
    material = UsdShade.Material.Define(stage, holder_path)
    material.GetPrim().GetReferences().AddReference(str(mtlx_abspath))


def author_preview_material(stage, holder_path: str, params: dict) -> None:
    """Author an inline ``UsdPreviewSurface`` material holder with a full input set.

    Mirrors `add_primitive`'s inline-material pattern but authors every editable
    UsdPreviewSurface input up front so each can be overridden later. ``params``
    overrides any of the `_PREVIEW_INPUTS` defaults by name.
    """
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, holder_path)
    shader = UsdShade.Shader.Define(stage, f"{holder_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    for name, type_stem, default in _PREVIEW_INPUTS:
        value = params.get(name, default)
        sdf_type = getattr(Sdf.ValueTypeNames, type_stem)
        if type_stem == "Color3f":
            r, g, b = value
            shader.CreateInput(name, sdf_type).Set(Gf.Vec3f(float(r), float(g), float(b)))
        else:
            shader.CreateInput(name, sdf_type).Set(float(value))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")


def author_binding(stage, prim_path: str, material_path: str) -> None:
    """Bind ``material_path`` to ``prim_path`` with explicit (replacing) targets.

    A typed ``UsdShade.Material`` target is bound with the schema `Bind()` — which
    authors an explicit single-target list op (verified), so a session binding
    replaces a file-authored one under LIVRPS. A holder that is not
    Material-typed but carries a ``.mtlx`` reference is bound by authoring the
    ``material:binding`` relationship's targets explicitly (`SetTargets`, never
    `AddTarget`/prepend). `SetTargets` also gives rebind = last-write-wins.
    """
    from pxr import UsdShade
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI.Apply(prim)
    target_prim = stage.GetPrimAtPath(material_path)
    shade_mat = UsdShade.Material(target_prim)
    if shade_mat:
        UsdShade.MaterialBindingAPI(prim).Bind(shade_mat)
        return
    rel = prim.GetRelationship("material:binding")
    if not rel:
        rel = prim.CreateRelationship("material:binding", custom=False)
    rel.SetTargets([material_path])


# ─── Holder discovery + save post-process (design D7) ──────────────────


def collect_material_holders(stage) -> dict[str, str]:
    """`{holder_prim_path: absolute_mtlx_path}` for every `/Materials` holder
    carrying a `.mtlx` reference, read from the composed stage.

    Read off the live stage (root + session layers) *before* an export flattens
    the reference arcs away, so the save post-process knows which holders to
    re-anchor and where their documents live.
    """
    holders: dict[str, str] = {}
    scope = stage.GetPrimAtPath(_MATERIALS_SCOPE)
    if not (scope and scope.IsValid()):
        return holders
    for child in scope.GetChildren():
        asset = _holder_reference_asset(stage, child.GetPath())
        if asset is not None:
            holders[str(child.GetPath())] = asset
    return holders


def _holder_reference_asset(stage, holder_path) -> Optional[str]:
    """Absolute `.mtlx` asset path referenced by a holder, from any layer."""
    for layer in (stage.GetRootLayer(), stage.GetSessionLayer()):
        if layer is None:
            continue
        spec = layer.GetPrimAtPath(holder_path)
        if spec is None:
            continue
        ref_list = spec.referenceList
        for ref in (
            list(ref_list.prependedItems)
            + list(ref_list.appendedItems)
            + list(ref_list.explicitItems)
        ):
            if ref.assetPath.endswith(".mtlx"):
                return ref.assetPath
    return None


def _is_under(path: str, directory: Optional[str]) -> bool:
    """True when ``path`` lives inside ``directory`` (both resolved)."""
    if not directory:
        return False
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except (ValueError, OSError):
        return False


def _copy_session_document(mtlx_abspath: str, materials_dir: Path) -> str:
    """Copy a session `.mtlx` (and its mapping sidecar, if any) into ``materials_dir``.

    Returns the layer-relative reference (`materials/<name>.mtlx`) to author. The
    sidecar copy keeps a reloaded synthesized material editable (its logical→
    uniform mapping survives the round-trip).
    """
    materials_dir.mkdir(parents=True, exist_ok=True)
    src = Path(mtlx_abspath)
    dst = materials_dir / src.name
    shutil.copyfile(src, dst)
    sidecar = src.with_suffix(".json")
    if sidecar.exists():
        shutil.copyfile(sidecar, materials_dir / sidecar.name)
    return f"materials/{src.name}"


def finalize_saved_materials(
    saved_layer,
    holders: dict[str, str],
    save_dir: str,
    session_dir: Optional[str],
    *,
    flattened: bool,
) -> None:
    """Re-anchor `.mtlx` references (and copy session docs) in a saved layer.

    Session-synthesized documents (those living under ``session_dir``) are copied
    into ``<save_dir>/materials/`` and re-authored as a layer-relative reference,
    yielding a self-contained bundle. Curated presets (under the assets tree) keep
    their **absolute** reference — the texture carve-out (design D7): copying a
    textured preset without walking its filename inputs would drop its textures.

    ``flattened`` (the anonymous-root `stage.Export` branch) additionally strips
    the flatten residue — the inlined shader child prims and local properties left
    on the holder — so the re-authored reference is the sole source of the
    material network. The file-backed overlay branch keeps its reference arc, so
    only re-anchoring is needed.
    """
    from pxr import Sdf
    materials_dir = Path(save_dir) / "materials"
    for holder_path, mtlx_abspath in holders.items():
        spec = saved_layer.GetPrimAtPath(holder_path)
        if spec is None:
            continue
        if _is_under(mtlx_abspath, session_dir):
            new_asset = _copy_session_document(mtlx_abspath, materials_dir)
        else:
            new_asset = str(Path(mtlx_abspath).resolve())  # curated: stay absolute
        if flattened:
            for child in list(spec.nameChildren):
                spec.RemoveNameChild(child)
            for prop in list(spec.properties):
                spec.RemoveProperty(prop)
        spec.referenceList.ClearEdits()
        spec.referenceList.explicitItems = [Sdf.Reference(new_asset)]
