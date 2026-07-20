"""load_scene_from_stage(allow_empty=...) on a geometry-less stage.

Backs the scene_create MCP tool: a freshly synthesized stage (a single /World
Xform, no meshes/gprims) must read as a well-formed empty Scene rather than
raising, while the disk-load default still rejects geometry-less input.
"""

from __future__ import annotations

import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


def _bare_world_stage():
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage


@needs_usd
def test_empty_stage_raises_by_default() -> None:
    from skinny.usd_loader import load_scene_from_stage
    with pytest.raises(ValueError, match="no usable mesh or gprim geometry"):
        load_scene_from_stage(_bare_world_stage())


@needs_usd
def test_empty_stage_allow_empty_returns_empty_scene() -> None:
    from skinny.usd_loader import load_scene_from_stage
    scene = load_scene_from_stage(_bare_world_stage(), allow_empty=True)
    assert scene.instances == []
    assert scene.lights_dir == []
    assert scene.lights_sphere == []
    assert scene.has_authored_lighting is False
    # meters_per_unit 1.0 -> mm_per_unit 1000.0 (see _read_open_stage).
    assert scene.mm_per_unit == pytest.approx(1000.0)


# ── Session-layer edit target: real author_local_transform behavior ───────────
# These exercise the REAL `skinny.usd_edit.author_local_transform` (extracted so
# it is importable without `vulkan`) against the failing topology: a transform
# authored in the ROOT/file layer, edited via the session layer (the renderer's
# edit target). They would FAIL on the old broken renderer (weak sublayer: throw
# or silent no-op), so they guard the fix, not just USD semantics.

def _stage_with_file_transform(op_kind: str):
    """Stage whose /World/M prim carries a transform authored in the ROOT layer,
    mimicking a scene loaded from disk, with the session layer as edit target
    (what `_attach_edit_layer` sets). `op_kind`: 'single' (one xformOp:transform),
    'components' (translate+rotateXYZ+scale), or 'inverse' (a sole inverse op)."""
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    xf = UsdGeom.Xform.Define(stage, "/World/M")
    if op_kind == "single":
        xf.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(1, 2, 3)))
    elif op_kind == "components":
        xf.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
        xf.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
    elif op_kind == "inverse":
        # A sole inverse op: op.Set() is illegal, so the reuse branch must skip it.
        xf.AddTransformOp(opSuffix="pivot").Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(1, 2, 3)))
        xf.AddTransformOp(opSuffix="pivot", isInverseOp=True)
        xf.SetXformOpOrder([xf.GetOrderedXformOps()[-1]])  # order = [!invert!...pivot]
    # Edit target = session layer, exactly like Renderer._attach_edit_layer.
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    return stage


def _import_transform():
    from skinny.usd_edit import author_local_transform
    return author_local_transform


@needs_usd
def test_author_transform_single_op_override_wins_and_is_nondestructive() -> None:
    """The op.Set() reuse path overrides a file-authored single transform: the
    override wins, the root spec is untouched, and NO xformOpOrder is authored in
    the session layer (which distinguishes reuse from clear+add)."""
    import numpy as np
    from pxr import Gf, UsdGeom
    author_local_transform = _import_transform()
    stage = _stage_with_file_transform("single")
    session, root = stage.GetSessionLayer(), stage.GetRootLayer()
    before_root = Gf.Matrix4d(root.GetAttributeAtPath("/World/M.xformOp:transform").default)

    m = np.eye(4); m[3, 0] = 9.0
    author_local_transform(UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")), m)

    composed = UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")).GetLocalTransformation()
    assert composed.ExtractTranslation()[0] == pytest.approx(9.0)  # override wins
    after_root = Gf.Matrix4d(root.GetAttributeAtPath("/World/M.xformOp:transform").default)
    assert after_root == before_root  # file untouched (non-destructive)
    assert session.GetAttributeAtPath("/World/M.xformOp:transform") is not None
    # reuse (op.Set) authors only the value, never the order attribute:
    assert session.GetAttributeAtPath("/World/M.xformOpOrder") is None


@needs_usd
def test_author_transform_component_ops_fallback_does_not_throw() -> None:
    """Multi-op file prim -> clear+add fallback: no duplicate-op throw on the
    session target, and the override composes."""
    import numpy as np
    from pxr import UsdGeom
    author_local_transform = _import_transform()
    stage = _stage_with_file_transform("components")
    m = np.eye(4); m[3, 0] = 9.0
    author_local_transform(UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")), m)  # no throw
    composed = UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")).GetLocalTransformation()
    assert composed.ExtractTranslation()[0] == pytest.approx(9.0)


@needs_usd
def test_author_transform_sole_inverse_op_falls_through_to_clear_add() -> None:
    """A sole inverse op must NOT hit op.Set() (USD forbids Set on an inverse op);
    the reuse guard skips it and clear+add composes the override without raising."""
    import numpy as np
    from pxr import UsdGeom
    author_local_transform = _import_transform()
    stage = _stage_with_file_transform("inverse")
    m = np.eye(4); m[3, 0] = 9.0
    author_local_transform(UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")), m)  # no throw
    composed = UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")).GetLocalTransformation()
    assert composed.ExtractTranslation()[0] == pytest.approx(9.0)


@needs_usd
def test_session_override_survives_save_and_reopen() -> None:
    """A file-authored transform overridden via the session layer round-trips
    through an export: reopening the saved layer over the original root composes
    the override (asserted via GetPrimAtPath, not Traverse — the export is an
    over-heavy artifact with no traversal root)."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom
    author_local_transform = _import_transform()
    stage = _stage_with_file_transform("single")
    root = stage.GetRootLayer()
    m = np.eye(4); m[3, 0] = 9.0
    author_local_transform(UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")), m)

    saved = Sdf.Layer.CreateAnonymous("session_edits.usda")
    saved.TransferContent(stage.GetSessionLayer())  # what save_edits exports
    # Reopen: original root + the saved override as a stronger session-style over.
    reopened = Usd.Stage.Open(root)
    reopened.GetSessionLayer().TransferContent(saved)
    composed = UsdGeom.Xformable(reopened.GetPrimAtPath("/World/M")).GetLocalTransformation()
    assert composed.ExtractTranslation()[0] == pytest.approx(9.0)
