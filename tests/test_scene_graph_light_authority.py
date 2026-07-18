"""Scene-graph projection tests for the all-or-nothing fallback pair."""

from __future__ import annotations

from pxr import Usd, UsdGeom, UsdLux

from skinny.scene_graph import (
    RendererRef,
    SceneGraphNode,
    inject_default_lights,
)


def _default_light_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Skinny")
    UsdLux.DistantLight.Define(stage, "/Skinny/DefaultLight")
    UsdLux.DomeLight.Define(stage, "/Skinny/DefaultDome")
    return stage


def _root():
    return SceneGraphNode(path="/", name="/", type_name="PseudoRoot")


def _paths(root):
    paths = {root.path}
    for child in root.children:
        paths.update(_paths(child))
    return paths


def test_fallback_projection_adds_both_synthetic_nodes_as_a_pair():
    root = _root()
    inject_default_lights(root, _default_light_stage(), enabled=True)
    assert {
        "/Skinny/DefaultLight",
        "/Skinny/DefaultDome",
    } <= _paths(root)


def test_authored_authority_removes_both_synthetic_nodes():
    root = _root()
    stage = _default_light_stage()
    inject_default_lights(root, stage, enabled=True)
    inject_default_lights(root, stage, enabled=False)
    assert "/Skinny/DefaultLight" not in _paths(root)
    assert "/Skinny/DefaultDome" not in _paths(root)


def test_real_usd_light_node_survives_projection_transitions():
    root = _root()
    root.children.append(
        SceneGraphNode(
            path="/World/Key",
            name="Key",
            type_name="DistantLight",
            renderer_ref=RendererRef("light_dir", 0),
        ),
    )
    stage = _default_light_stage()
    inject_default_lights(root, stage, enabled=True)
    inject_default_lights(root, stage, enabled=False)
    assert "/World/Key" in _paths(root)


def test_runtime_false_true_false_transition_is_idempotent():
    root = _root()
    stage = _default_light_stage()
    inject_default_lights(root, stage, enabled=False)
    inject_default_lights(root, stage, enabled=True)
    inject_default_lights(root, stage, enabled=True)
    paths = [child.path for child in root.children]
    assert paths.count("/Skinny/DefaultLight") == 1
    assert paths.count("/Skinny/DefaultDome") == 1
    inject_default_lights(root, stage, enabled=False)
    assert "/Skinny/DefaultLight" not in _paths(root)
    assert "/Skinny/DefaultDome" not in _paths(root)
