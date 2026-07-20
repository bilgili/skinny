"""Headless tests for the runtime scene-graph editing API (usd-scene-editing).

Exercises Renderer.add_model / remove_node / set_transform / save_edits /
list_nodes against a real GPU upload path, mirroring tests/test_headless_api.py
(Vulkan headless + OpenUSD). Run with the repo-root Python 3.13 venv and
VULKAN_SDK / DYLD_LIBRARY_PATH set (see CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


def _have_vulkan() -> bool:
    try:
        import vulkan  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(
    not _have_usd() or not SCENE.exists(),
    reason="OpenUSD (pxr) not installed or cornell_box_sphere.usda missing",
)
needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


@pytest.fixture()
def editor():
    """Fresh renderer with the cornell scene loaded for editing (stage owned)."""
    from pxr import Usd
    from skinny.renderer import Renderer
    from skinny.usd_loader import load_scene_from_stage
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=96, height=96)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR, tattoo_dir=TATTOO_DIR,
    )
    stage = Usd.Stage.Open(str(SCENE))
    scene = load_scene_from_stage(stage)
    renderer.set_usd_scene(scene, stage=stage)
    try:
        yield renderer, stage
    finally:
        renderer.cleanup()
        ctx.destroy()


def _render(renderer, frames: int = 4) -> bytes:
    raw = b""
    for _ in range(frames):
        renderer.update(0.016)
        raw = renderer.render_headless()
    return raw


@pytest.fixture()
def bare_renderer():
    """Renderer with NO scene loaded — the state scene_create starts from."""
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=96, height=96)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR, tattoo_dir=TATTOO_DIR,
    )
    try:
        yield renderer
    finally:
        renderer.cleanup()
        ctx.destroy()


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestEditLayer:
    def test_edit_layer_attached(self, editor):
        renderer, stage = editor
        assert renderer._usd_edit_layer is not None
        # The edit target is the session layer — stronger than the whole root
        # layer stack, so an override wins over a file-authored opinion. A root
        # SUBLAYER (the old design) is weaker than root and could not.
        assert renderer._usd_edit_layer == stage.GetSessionLayer()
        assert stage.GetEditTarget().GetLayer() == renderer._usd_edit_layer
        # No editing sublayer is inserted into the root's sublayer stack.
        assert renderer._usd_edit_layer.identifier not in stage.GetRootLayer().subLayerPaths
        # Non-destructive: the session layer is in-memory, never written by an edit.
        assert renderer._usd_edit_layer.anonymous is True

    def test_original_file_untouched_by_edit(self, editor):
        renderer, stage = editor
        root = stage.GetRootLayer()
        before = Path(root.realPath).stat().st_mtime_ns
        renderer.remove_node("/Cornell/GlassSphere/Sphere")
        # Edits land on the in-memory session edit layer; the root layer and the
        # file on disk are never written by an edit.
        after = Path(root.realPath).stat().st_mtime_ns
        assert before == after, "original USD file must not be written by an edit"


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestAddModel:
    def test_add_grows_instances_and_prim_exists(self, editor):
        renderer, stage = editor
        before = len(renderer._usd_scene.instances)
        ver_before = renderer._material_version
        path = renderer.add_model(str(SCENE), parent_prim_path="/World", name="Added")
        assert path == "/World/Added"
        assert stage.GetPrimAtPath(path).IsValid()
        assert stage.GetPrimAtPath(path).IsActive()
        # cornell defaultPrim /Cornell has 7 mesh instances; referencing it adds 7.
        assert len(renderer._usd_scene.instances) == before + 7
        assert renderer._material_version > ver_before  # accumulation reset

    def test_add_pixels_change(self, editor):
        renderer, _ = editor
        before = _render(renderer)
        renderer.add_model(str(SCENE), parent_prim_path="/World", name="Dup")
        after = _render(renderer)
        assert before != after, "adding visible geometry should change the image"

    def test_add_listed_in_nodes(self, editor):
        renderer, _ = editor
        renderer.add_model(str(SCENE), parent_prim_path="/World", name="Added")
        paths = {n["path"] for n in renderer.list_nodes()}
        assert "/World/Added" in paths

    def test_add_honors_transform(self, editor):
        renderer, _ = editor
        # Translate the added copy far along +X; its instances must sit away from
        # every original instance's world position.
        offset = 1000.0
        m = np.eye(4, dtype=np.float32)
        m[3, 0] = offset
        orig_x = [float(i.transform[3, 0]) for i in renderer._usd_scene.instances]
        renderer.add_model(str(SCENE), parent_prim_path="/World", name="Far", transform=m)
        added = [
            i for i in renderer._usd_scene.instances if i.prim_path.startswith("/World/Far/")
        ]
        assert added, "added reference should produce instances"
        assert all(float(i.transform[3, 0]) > max(orig_x) for i in added)

    def test_add_rejects_obj(self, editor):
        renderer, _ = editor
        with pytest.raises(ValueError):
            renderer.add_model("/some/model.obj")

    def test_add_missing_path_no_mutation(self, editor):
        renderer, stage = editor
        before = len(renderer._usd_scene.instances)
        children_before = len(stage.GetPseudoRoot().GetAllChildren())
        with pytest.raises(ValueError):
            renderer.add_model(str(PROJECT_ROOT / "does_not_exist.usda"))
        assert len(renderer._usd_scene.instances) == before
        assert len(stage.GetPseudoRoot().GetAllChildren()) == children_before


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestRemoveNode:
    def test_remove_drops_instances_and_deactivates(self, editor):
        renderer, stage = editor
        target = "/Cornell/GlassSphere/Sphere"
        assert any(i.prim_path == target for i in renderer._usd_scene.instances)
        renderer.remove_node(target)
        assert not stage.GetPrimAtPath(target).IsActive()
        assert all(i.prim_path != target for i in renderer._usd_scene.instances)

    def test_remove_changes_pixels(self, editor):
        renderer, _ = editor
        before = _render(renderer)
        renderer.remove_node("/Cornell/GlassSphere/Sphere")
        after = _render(renderer)
        assert before != after

    def test_remove_unknown_path_raises(self, editor):
        renderer, _ = editor
        with pytest.raises(ValueError):
            renderer.remove_node("/Cornell/NoSuchPrim")


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestSetTransform:
    def test_set_transform_moves_without_rebake(self, editor):
        renderer, _ = editor
        target = "/Cornell/TallBlock/Block"
        instances = renderer._usd_scene.instances
        idx = next(i for i, inst in enumerate(instances) if inst.prim_path == target)
        before_xf = instances[idx].transform.copy()
        before_mesh = id(instances[idx].mesh)
        ver_before = renderer._material_version

        m = np.eye(4, dtype=np.float32)
        m[3, 1] = 250.0  # lift the block
        renderer.set_transform(target, m)

        moved = renderer._usd_scene.instances[idx]
        assert not np.array_equal(moved.transform, before_xf), "instance should move"
        assert id(moved.mesh) == before_mesh, "transform edit must not re-bake the mesh"
        assert renderer._material_version > ver_before

    def test_set_transform_unknown_path_raises(self, editor):
        renderer, _ = editor
        with pytest.raises(ValueError):
            renderer.set_transform("/Cornell/NoSuchPrim", np.eye(4, dtype=np.float32))

    def test_override_file_authored_transform(self, editor):
        """A prim whose xformOp:transform is authored in the loaded FILE can be
        overridden: no duplicate-op throw, the composed transform follows the
        override, and the file's own opinion is untouched (non-destructive).

        Regression for session-edit-layer: the old weak edit SUBLAYER could not
        override a root opinion — set_transform raised 'xformOp:transform already
        exists in xformOpOrder', and any value it authored was silently ignored.
        """
        from pxr import Gf, Sdf, UsdGeom
        renderer, stage = editor
        target = "/Cornell/TallBlock"  # an Xform with a file-authored transform
        prim = stage.GetPrimAtPath(target)
        assert prim and prim.IsValid()
        root = stage.GetRootLayer()
        before_root = root.GetAttributeAtPath(f"{target}.xformOp:transform").default

        m = np.eye(4, dtype=np.float32)
        m[3, 0] = 123.0  # translate far along +X
        renderer.set_transform(target, m)  # must NOT raise

        composed = UsdGeom.Xformable(prim).GetLocalTransformation()
        assert composed.ExtractTranslation()[0] == pytest.approx(123.0), \
            "session-layer override must win over the file-authored transform"
        # The file's opinion in the root layer is unchanged (non-destructive).
        after_root = root.GetAttributeAtPath(f"{target}.xformOp:transform").default
        assert Gf.Matrix4d(after_root) == Gf.Matrix4d(before_root)
        # The override lives in the session (edit) layer, not the root.
        assert renderer._usd_edit_layer.GetAttributeAtPath(
            f"{target}.xformOp:transform"
        ) is not None
        # op.Set() reuse authors only the value, not the order attribute (this
        # distinguishes the reuse path from an unconditional clear+add).
        assert renderer._usd_edit_layer.GetAttributeAtPath(
            f"{target}.xformOpOrder"
        ) is None


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestSaveEdits:
    def test_save_edits_writes_reopenable_layer(self, editor, tmp_path):
        from pxr import Sdf
        renderer, _ = editor
        renderer.add_model(str(SCENE), parent_prim_path="/World", name="Added")
        out = tmp_path / "scene.edits.usda"
        written = renderer.save_edits(str(out))
        assert Path(written).exists()
        reopened = Sdf.Layer.FindOrOpen(written)
        assert reopened is not None
        # The authored /World/Added prim is present in the saved layer.
        assert reopened.GetPrimAtPath("/World/Added") is not None


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestPrimIndexConsistency:
    def test_index_matches_instances_after_add_remove(self, editor):
        renderer, _ = editor
        renderer.add_model(str(SCENE), parent_prim_path="/World", name="Added")
        renderer.remove_node("/Cornell/GlassSphere/Sphere")
        # Every index in the map points at an instance with that prim path.
        for prim_path, idxs in renderer._prim_to_instances.items():
            for i in idxs:
                assert renderer._usd_scene.instances[i].prim_path == prim_path
        # Every instance with a prim path is represented in the map.
        for i, inst in enumerate(renderer._usd_scene.instances):
            if inst.prim_path:
                assert i in renderer._prim_to_instances[inst.prim_path]


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestCreateEmptyScene:
    def test_creates_editable_stage_with_world(self, bare_renderer):
        r = bare_renderer
        assert r._usd_stage is None  # no scene loaded yet
        r.create_empty_scene()
        # Editable stage now present -> structural edits are admitted.
        assert r._usd_stage is not None
        assert r._usd_edit_layer is not None
        assert r._usd_stage.GetPrimAtPath("/World").IsValid()
        # No authored geometry; the TLAS is empty (no analytic-head ghost).
        assert r._usd_scene.instances == []
        assert r._num_instances == 0
        # Physical scale adopted from the new stage (metersPerUnit 1).
        assert r.mm_per_unit == pytest.approx(1000.0)

    def test_save_created_scene_roundtrips_metadata(self, bare_renderer, tmp_path):
        from pxr import Usd, UsdGeom
        r = bare_renderer
        r.create_empty_scene()
        r.add_primitive("Sphere", parent_prim_path="/World", name="Ball")
        out = tmp_path / "created.usda"
        written = r.save_edits(str(out))
        assert Path(written).exists()
        reopened = Usd.Stage.Open(written)
        # The seed root data must survive the save (P1): without it the file
        # reopens at USD's 0.01 m/unit default.
        assert reopened.GetDefaultPrim().GetName() == "World"
        assert str(UsdGeom.GetStageUpAxis(reopened)) == "Y"
        assert UsdGeom.GetStageMetersPerUnit(reopened) == pytest.approx(1.0)
        assert reopened.GetPrimAtPath("/World").IsValid()
        assert reopened.GetPrimAtPath("/World/Ball").IsValid()

    def test_save_created_empty_preserves_world_and_units(self, bare_renderer, tmp_path):
        from pxr import Usd, UsdGeom
        r = bare_renderer
        r.create_empty_scene()  # nothing added
        out = tmp_path / "empty.usda"
        r.save_edits(str(out))
        reopened = Usd.Stage.Open(str(out))
        assert UsdGeom.GetStageMetersPerUnit(reopened) == pytest.approx(1.0)
        assert reopened.GetPrimAtPath("/World").IsValid()

    def test_created_scene_is_enumerable_and_lit(self, bare_renderer):
        r = bare_renderer
        r.create_empty_scene()
        # list_nodes walks the editable stage -> carries the authored /World.
        assert "/World" in {n["path"] for n in r.list_nodes()}
        # The synthetic default light/dome/camera live in the derived scene
        # graph (the MCP scene_list surface), not on the editable stage.
        graph_paths: set[str] = set()
        stack = [r.scene_graph]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            graph_paths.add(node.path)
            stack.extend(node.children)
        assert "/World" in graph_paths
        assert any(p.startswith("/Skinny/") for p in graph_paths)

    def test_add_primitive_after_create(self, bare_renderer):
        r = bare_renderer
        r.create_empty_scene()
        path = r.add_primitive("Sphere", parent_prim_path="/World", name="Ball")
        assert path == "/World/Ball"
        assert r._usd_stage.GetPrimAtPath(path).IsValid()
        assert len(r._usd_scene.instances) >= 1

    def test_created_scene_renders_without_analytic_head(self, bare_renderer):
        r = bare_renderer
        r.create_empty_scene()
        # Must render the (empty) USD scene, not silently fall back to the
        # analytic head — _usd_model_index stays the active model slot.
        _render(r)
        assert r.model_index == r._usd_model_index
        assert r._num_instances == 0

    def test_force_replace_from_loaded_scene(self, editor):
        renderer, _ = editor
        assert len(renderer._usd_scene.instances) > 0
        # Simulate a session left in USD-camera / mirrored state.
        renderer.camera_mode = "usd"
        renderer._camera_mirror = True
        renderer.create_empty_scene()
        assert renderer._usd_scene.instances == []
        assert renderer._num_instances == 0
        assert renderer._usd_stage.GetPrimAtPath("/World").IsValid()
        # Camera state reset so a drag can't call look() on the OrbitCamera.
        assert renderer.camera_mode != "usd"
        assert renderer._camera_mirror is False
        assert renderer.mm_per_unit == pytest.approx(1000.0)
