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
