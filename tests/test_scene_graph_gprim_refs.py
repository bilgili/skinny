"""Regression tests: analytic gprim instances must get an instance
``renderer_ref`` in the scene graph so the transform gizmo can target them.

The loader tessellates analytic gprims (Sphere/Cube/Cylinder/Cone/Capsule/
Plane) into renderable ``MeshInstance``s exactly like ``UsdGeom.Mesh`` prims,
but those prims are *not* ``UsdGeom.Mesh``. Earlier the scene-graph builder and
``populate_instance_refs`` only attached the instance ref to nodes typed
``"Mesh"``, so selecting a gprim in the scene-graph dock left ``renderer_ref``
``None`` and the gizmo never appeared (``set_gizmo_target(-1)``).

Pure Python — uses an in-memory USD stage, no Vulkan / display / baking."""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")

from skinny.scene_graph import (  # noqa: E402
    build_scene_graph,
    find_node_by_path,
    populate_instance_refs,
)


class _Inst:
    """Minimal stand-in for ``MeshInstance`` — the scene-graph builder and
    ``populate_instance_refs`` only read ``name`` and ``enabled``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.prim_path = name
        self.enabled = True


class _Scene:
    """Minimal stand-in for ``Scene`` covering the attributes the builder and
    ``_add_enabled_prop`` touch."""

    def __init__(self, instances: list[_Inst]) -> None:
        self.instances = instances
        self.materials: list = []
        self.lights_dir: list = []
        self.lights_sphere: list = []
        self.camera_override = None


def _stage_with_gprim_and_mesh():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    UsdGeom.Sphere.Define(stage, "/World/Ball")
    UsdGeom.Cube.Define(stage, "/World/Box")
    return stage


def test_populate_instance_refs_targets_gprim_streaming_path():
    """Streaming order: graph built before instances exist, then
    ``populate_instance_refs`` fills the refs in. The Sphere and Cube gprims
    must get instance refs, not just the Mesh."""
    stage = _stage_with_gprim_and_mesh()

    # Phase 1: instances not streamed yet.
    sg = build_scene_graph(stage, _Scene([]))
    assert find_node_by_path(sg, "/World/Ball").renderer_ref is None

    # Phase 2: instances stream in (mesh + both gprims), refs back-filled.
    scene = _Scene([
        _Inst("/World/Mesh"),
        _Inst("/World/Ball"),
        _Inst("/World/Box"),
    ])
    populate_instance_refs(sg, scene)

    for path, idx in (("/World/Mesh", 0), ("/World/Ball", 1), ("/World/Box", 2)):
        node = find_node_by_path(sg, path)
        assert node.renderer_ref is not None, f"{path} should be gizmo-targetable"
        assert node.renderer_ref.kind == "instance"
        assert node.renderer_ref.index == idx


def test_build_scene_graph_targets_gprim_at_build_time():
    """Reload path: ``build_scene_graph`` runs against a fully-baked scene (no
    later ``populate_instance_refs``). The gprim nodes must still get refs."""
    stage = _stage_with_gprim_and_mesh()
    scene = _Scene([
        _Inst("/World/Mesh"),
        _Inst("/World/Ball"),
        _Inst("/World/Box"),
    ])

    sg = build_scene_graph(stage, scene)

    for path, idx in (("/World/Mesh", 0), ("/World/Ball", 1), ("/World/Box", 2)):
        node = find_node_by_path(sg, path)
        assert node.renderer_ref is not None, f"{path} should be gizmo-targetable"
        assert node.renderer_ref.kind == "instance"
        assert node.renderer_ref.index == idx
