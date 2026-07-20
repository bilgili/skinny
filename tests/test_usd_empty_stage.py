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


# ── USD-semantics guards for the session-layer edit target ────────────────────
# These lock the composition facts the renderer's `_attach_edit_layer` /
# `_author_local_transform` fix relies on. The renderer module imports `vulkan`
# at load, so it can't be imported in the hostless sweep — these test the raw
# `pxr` behavior the fix depends on instead (a USD upgrade that changed any of
# them would break editing, so failing here is the early warning).

def _stage_with_file_transform(op_kind: str):
    """Stage whose /World/M prim carries a transform authored in the ROOT layer,
    mimicking a scene loaded from disk. `op_kind` is 'single' (one
    xformOp:transform) or 'components' (translate + rotateXYZ + scale)."""
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    xf = UsdGeom.Xform.Define(stage, "/World/M")
    if op_kind == "single":
        xf.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(1, 2, 3)))
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
        xf.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
    return stage


@needs_usd
def test_session_layer_single_op_override_wins_and_is_nondestructive() -> None:
    """Reusing the single transform op via `op.Set()` with the session layer as
    edit target overrides the file value AND leaves the root spec untouched."""
    from pxr import Gf, Usd, UsdGeom
    stage = _stage_with_file_transform("single")
    session = stage.GetSessionLayer()
    root = stage.GetRootLayer()
    before_root = root.GetAttributeAtPath("/World/M.xformOp:transform").default

    with Usd.EditContext(stage, Usd.EditTarget(session)):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/M"))
        ops = xf.GetOrderedXformOps()
        assert len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform
        ops[0].Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(9, 9, 9)))  # the reuse path

    composed = UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")).GetLocalTransformation()
    assert composed.ExtractTranslation() == Gf.Vec3d(9, 9, 9)  # override wins
    after_root = root.GetAttributeAtPath("/World/M.xformOp:transform").default
    assert Gf.Matrix4d(after_root) == Gf.Matrix4d(before_root)  # file untouched
    assert session.GetAttributeAtPath("/World/M.xformOp:transform") is not None


@needs_usd
def test_session_layer_component_ops_clear_add_does_not_throw() -> None:
    """The clear+add fallback (multi-op prim) does not raise on the session
    target: session's empty xformOpOrder wins over the root's component ops, so
    AddTransformOp sees an empty composed order and composes the override."""
    from pxr import Gf, Usd, UsdGeom
    stage = _stage_with_file_transform("components")
    session = stage.GetSessionLayer()

    with Usd.EditContext(stage, Usd.EditTarget(session)):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/M"))
        xf.ClearXformOpOrder()  # no duplicate-op throw follows
        xf.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(9, 9, 9)))

    composed = UsdGeom.Xformable(stage.GetPrimAtPath("/World/M")).GetLocalTransformation()
    assert composed.ExtractTranslation() == Gf.Vec3d(9, 9, 9)
