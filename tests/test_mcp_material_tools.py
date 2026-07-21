"""Fake-renderer tests for the mcp-material-authoring tool surface:
material_list, scene_add_material, scene_bind_material, and
scene_add_primitive's material= argument (task 4.6).

Same harness pattern as test_mcp_structural_tools.py -- fake renderer, real
command queue, run on a worker thread -- except the fake renderer authors
onto a real in-memory ``pxr`` stage so the preset-dedup path (which reads the
composed stage via ``usd_material_edit.collect_material_holders``) behaves
exactly as it does against the real renderer. Never imports skinny.renderer;
no GPU. The template-form tests exercise the real MaterialX generator dry-run
(``mtlx_synthesis.synthesize``), so this file needs the py3.13 venv with
PyMaterialXGenSlang:

    PYTHONPATH=src ./bin/python3.13 -m pytest tests/test_mcp_material_tools.py -q
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("pxr")
pytest.importorskip("MaterialX")

from pxr import Usd, UsdGeom  # noqa: E402

from skinny import usd_material_edit as ume  # noqa: E402
from skinny.mcp_server import SceneToolError, SceneTools  # noqa: E402
from skinny.render_session import RenderCommandQueue  # noqa: E402
from skinny.scene_graph import SceneGraphNode  # noqa: E402


class FakeRenderer:
    """Mirrors the real add_material/bind_material/add_primitive contracts
    (naming, dedup-raises-on-clash, unique preview names) against a real
    in-memory stage, without any of the renderer's GPU/resync machinery.
    """

    def __init__(self, graph) -> None:
        self.scene_graph = graph
        self._material_version = 7
        self._scene_graph_version = 3
        self._usd_stage = Usd.Stage.CreateInMemory()
        self._usd_edit_layer = self._usd_stage.GetSessionLayer()
        self._usd_stage.SetEditTarget(Usd.EditTarget(self._usd_edit_layer))
        # `has_editable_stage` now also gates on adopted scene metadata (finding #4).
        self._usd_scene = object()
        self.calls: list[tuple] = []
        self.add_material_calls = 0
        self.fail_next_add_primitive = False

    def add_material(self, name, *, mtlx_path=None, preview_params=None,
                      session_dir=None, on_rollback=None):
        self.calls.append(("add_material", name, mtlx_path, preview_params))
        self.add_material_calls += 1
        stage = self._usd_stage
        if (mtlx_path is None) == (preview_params is None):
            raise ValueError("add_material requires exactly one of mtlx_path or preview_params")
        if mtlx_path is not None:
            holder_path = f"/Materials/{name}"
            if stage.GetPrimAtPath(holder_path).IsValid():
                on_rollback and on_rollback()
                raise ValueError(f"add_material: /Materials/{name} already exists")
            ume.ensure_materials_scope(stage)
            ume.author_material_holder(stage, holder_path, mtlx_path)
        else:
            base = f"/Materials/{name}"
            holder_path = base
            i = 1
            ume.ensure_materials_scope(stage)
            while stage.GetPrimAtPath(holder_path).IsValid():
                holder_path = f"{base}_{i}"
                i += 1
            ume.author_preview_material(stage, holder_path, preview_params or {})
        self._material_version += 1
        return holder_path

    def bind_material(self, prim_path, material_path):
        self.calls.append(("bind_material", prim_path, material_path))
        stage = self._usd_stage
        prim = stage.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            raise ValueError(f"bind_material: prim not found: {prim_path}")
        mat_prim = stage.GetPrimAtPath(material_path)
        if not (mat_prim and mat_prim.IsValid()):
            raise ValueError(f"bind_material: material not found: {material_path}")
        ume.author_binding(stage, prim_path, material_path)
        self._material_version += 1

    def add_primitive(self, prim_type, parent_prim_path="/World", name=None,
                       transform=None, color=None, roughness=None, metallic=None,
                       skip_inline_material=False, bind_material_path=None):
        if bind_material_path is not None:
            skip_inline_material = True
        self.calls.append(("add_primitive", prim_type, parent_prim_path, name,
                            color, roughness, metallic, skip_inline_material,
                            bind_material_path))
        path = f"{parent_prim_path}/{name or prim_type}"
        UsdGeom.Sphere.Define(self._usd_stage, path)
        if bind_material_path is not None:
            # Transactional add+bind (finding #8): validate + author the binding
            # in-line so a failure removes the just-created prim.
            mat_prim = self._usd_stage.GetPrimAtPath(bind_material_path)
            if not (mat_prim and mat_prim.IsValid()):
                self._usd_stage.RemovePrim(path)
                raise ValueError(
                    f"add_primitive: material not found: {bind_material_path}"
                )
            ume.author_binding(self._usd_stage, path, bind_material_path)
            if self.fail_next_add_primitive:
                # Simulate a post-bind failure (e.g. resync): mirror the real
                # add_primitive rollback by removing the just-created prim, then
                # raise so the caller's cleanup tears down the fresh material.
                self._usd_stage.RemovePrim(path)
                raise ValueError("add_primitive: simulated post-bind failure")
        elif not skip_inline_material:
            mat_path = f"{path}_material"
            ume.ensure_materials_scope(self._usd_stage)
            # Not under /Materials in the real renderer, but presence is
            # irrelevant here -- these tests never assert on the inline path.
            UsdGeom.Scope.Define(self._usd_stage, mat_path)
        return path

    def remove_node(self, prim_path):
        # Rollback helper: mirror the renderer's non-destructive deactivate.
        self.calls.append(("remove_node", prim_path))
        prim = self._usd_stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            prim.SetActive(False)

    def discard_created_prim(self, prim_path):
        # Hard-remove (finding #8/E): delete the spec so the name is reusable,
        # unlike remove_node's tombstone. Mirrors the real renderer.
        self.calls.append(("discard_created_prim", prim_path))
        if self._usd_stage.GetPrimAtPath(prim_path).IsValid():
            self._usd_stage.RemovePrim(prim_path)


class Proxy:
    def __init__(self, queue) -> None:
        self._commands = queue

    def request(self, callback):
        return self._commands.post_with_reply(callback)


def _scene():
    world = SceneGraphNode(path="/World", name="World", type_name="Xform",
                            children=[], properties=[], renderer_ref=None)
    return SceneGraphNode(path="/", name="/", type_name="Stage",
                           children=[world], properties=[], renderer_ref=None)


class Harness:
    def __init__(self, **tools_kwargs) -> None:
        self.queue = RenderCommandQueue()
        self.renderer = FakeRenderer(_scene())
        self.tools = SceneTools(Proxy(self.queue), timeout=5.0, roots=[], **tools_kwargs)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.queue.run_pending(self.renderer)
            time.sleep(0.001)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


@pytest.fixture()
def h():
    harness = Harness(structural_grace=2.0)
    yield harness
    harness.close()


# ── material_list: renderer-free ───────────────────────────────────────

def test_material_list_is_renderer_free_and_shaped(h) -> None:
    result = h.tools.material_list()
    assert h.renderer.calls == []  # never touched the render thread
    assert {"presets", "models", "graph_nodes", "templates"} <= result.keys()
    names = {p["name"] for p in result["presets"]}
    assert "marble_solid" in names
    assert set(result["models"]) == {"preview", "standard_surface"}
    assert "fractal3d" in result["graph_nodes"]
    assert set(result["templates"]) == {"noise", "marble_veins"}


# ── scene_add_material: arg validation, no side effects on rejection ────

def test_add_material_mixed_forms_rejected(h) -> None:
    with pytest.raises(SceneToolError, match="preset.*model|model.*preset"):
        h.tools.scene_add_material({"preset": "marble_solid", "model": "preview"})
    assert h.renderer.calls == []
    assert h.renderer._material_version == 7


def test_add_material_graph_on_preview_rejected(h) -> None:
    with pytest.raises(SceneToolError, match="standard_surface"):
        h.tools.scene_add_material({"model": "preview", "graph": {"nodes": {}, "connections": []}})
    assert h.renderer.calls == []


def test_add_material_unknown_preset_lists_names(h) -> None:
    with pytest.raises(SceneToolError, match="marble_solid") as exc:
        h.tools.scene_add_material({"preset": "not_a_real_preset"})
    assert "unknown preset" in str(exc.value)
    assert h.renderer.calls == []


def test_add_material_path_shaped_preset_name_is_unknown(h) -> None:
    """A client string is a catalog key, never a filesystem path (design D3):
    a path-shaped name simply misses the dict lookup."""
    with pytest.raises(SceneToolError, match="unknown preset"):
        h.tools.scene_add_material({"preset": "../../etc/passwd"})
    assert h.renderer.calls == []


def test_add_material_name_rejected_for_preset_form(h) -> None:
    with pytest.raises(SceneToolError, match="'name' is not accepted"):
        h.tools.scene_add_material({"preset": "marble_solid"}, name="Custom")
    assert h.renderer.calls == []


def test_add_material_spec_error_leaves_versions_unchanged_and_writes_no_files(h) -> None:
    before_version = h.renderer._material_version
    with pytest.raises(SceneToolError):
        h.tools.scene_add_material({"template": "noise", "params": {"scale": 999.0}})
    assert h.renderer._material_version == before_version
    assert h.renderer.calls == []
    assert list(h.tools._material_store.dir.iterdir()) == []


# ── scene_add_primitive: material= composition rules ────────────────────

def test_add_primitive_material_with_color_is_refused(h) -> None:
    with pytest.raises(SceneToolError, match="cannot be combined with"):
        h.tools.scene_add_primitive("Sphere", material="marble_solid", color=(1, 0, 0))
    assert h.renderer.calls == []


def test_add_primitive_material_with_roughness_is_refused(h) -> None:
    with pytest.raises(SceneToolError, match="cannot be combined with"):
        h.tools.scene_add_primitive("Sphere", material="marble_solid", roughness=0.2)
    assert h.renderer.calls == []


# ── preset dedup ──────────────────────────────────────────────────────

def test_add_material_preset_dedup_returns_existing_path(h) -> None:
    first = h.tools.scene_add_material({"preset": "marble_solid"})
    second = h.tools.scene_add_material({"preset": "marble_solid"})
    assert first["path"] == second["path"]
    assert first["live"] is False
    scope = h.renderer._usd_stage.GetPrimAtPath("/Materials")
    assert len(list(scope.GetChildren())) == 1
    assert h.renderer.add_material_calls == 1  # second call never authored


# ── template salting ─────────────────────────────────────────────────

def test_two_same_template_adds_get_distinct_names(h) -> None:
    first = h.tools.scene_add_material({"template": "noise"})
    second = h.tools.scene_add_material({"template": "noise"})
    assert first["path"] != second["path"]
    assert first["live"] is False and second["live"] is False
    assert h.renderer.add_material_calls == 2


def test_concurrent_same_template_adds_reserve_distinct_names(h) -> None:
    """Two threads adding the same template at once must reserve distinct salted
    names under `_name_lock` — never pick the same name and clobber each other's
    session file (finding #5). Both succeed, both `.mtlx` files survive."""
    results: list = []
    errors: list = []
    barrier = threading.Barrier(2)

    def add() -> None:
        try:
            barrier.wait()
            results.append(h.tools.scene_add_material({"template": "noise"}))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=add) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == []
    assert len({r["path"] for r in results}) == 2  # distinct names
    files = sorted(p.name for p in h.tools._material_store.dir.glob("*.mtlx"))
    assert len(files) == 2  # neither clobbered the other


def test_add_material_preview_form_live_false(h) -> None:
    result = h.tools.scene_add_material({"model": "preview", "params": {"roughness": 0.3}})
    assert result["live"] is False
    assert result["path"] == "/Materials/Material"


# ── scene_bind_material: validation errors ───────────────────────────

def test_bind_material_nonexistent_material_errors(h) -> None:
    # bind_material's own path checks raise ValueError, surfaced by the
    # SceneTools methods as-is; only the transport wrapper (build_app's
    # _wrap, not exercised here) converts it to a client-facing ToolError.
    h.tools.scene_add_primitive("Sphere", name="Ball")
    with pytest.raises(ValueError, match="material not found"):
        h.tools.scene_bind_material("/World/Ball", "/Materials/Ghost")


def test_bind_material_nonexistent_prim_errors(h) -> None:
    h.tools.scene_add_material({"preset": "marble_solid"})
    with pytest.raises(ValueError, match="prim not found"):
        h.tools.scene_bind_material("/World/Ghost", "/Materials/Marble_3D")


def test_bind_material_happy_path_bumps_version(h) -> None:
    add_result = h.tools.scene_add_primitive("Sphere", name="Ball")
    material_result = h.tools.scene_add_material({"preset": "marble_solid"})
    before = h.renderer._material_version
    bind_result = h.tools.scene_bind_material(add_result["path"], material_result["path"])
    assert bind_result["material_path"] == material_result["path"]
    assert h.renderer._material_version > before


# ── scene_add_primitive(material=...) end-to-end composition ─────────

def test_add_primitive_material_by_preset_name_creates_and_binds(h) -> None:
    from pxr import UsdShade
    result = h.tools.scene_add_primitive("Sphere", material="marble_solid", name="Ball")
    assert result["path"] == "/World/Ball"
    assert result["material_path"].startswith("/Materials/")
    kinds = [c[0] for c in h.renderer.calls]
    # The bind is now authored inside add_primitive (transactional, finding #8),
    # not a separate bind_material call: the material is created first, then the
    # primitive add binds it atomically.
    assert "add_material" in kinds
    assert kinds.index("add_material") < kinds.index("add_primitive")
    prim = h.renderer._usd_stage.GetPrimAtPath("/World/Ball")
    bound, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert str(bound.GetPath()) == result["material_path"]


def test_add_primitive_material_by_existing_path_binds_without_creating(h) -> None:
    material_result = h.tools.scene_add_material({"preset": "marble_solid"})
    before_calls = h.renderer.add_material_calls
    result = h.tools.scene_add_primitive(
        "Sphere", material=material_result["path"], name="Ball2",
    )
    assert result["material_path"] == material_result["path"]
    assert h.renderer.add_material_calls == before_calls  # no new material authored


def test_add_primitive_material_by_missing_path_errors(h) -> None:
    with pytest.raises(ValueError, match="material not found"):
        h.tools.scene_add_primitive("Sphere", material="/Materials/DoesNotExist")


def test_add_primitive_material_unknown_name_lists_catalogs(h) -> None:
    with pytest.raises(SceneToolError, match="presets:.*templates:|templates:.*presets:"):
        h.tools.scene_add_primitive("Sphere", material="not_a_thing")


# ── Session-file rollback: no orphan after a post-write failure (finding #5) ──

def test_add_synth_material_deletes_file_on_readiness_failure(h) -> None:
    """A stage-readiness failure raised before add_material's own rollback wiring
    must still delete the just-written session file, not orphan it (finding #5)."""
    store = h.tools._material_store
    written = store.write_document("Orphan", "<materialx/>", None)
    assert store.path_for("Orphan").exists()
    h.renderer._usd_scene = None  # has_editable_stage -> False after the write
    with pytest.raises(SceneToolError, match="editable"):
        h.tools._add_synth_material(h.renderer, "Orphan", str(written))
    assert not store.path_for("Orphan").exists()
    assert not store.sidecar_for("Orphan").exists()


def test_add_synth_material_deletes_file_on_holder_collision(h) -> None:
    """add_material's holder-collision ValueError is raised outside its rollback
    path, so the wrapper must delete the session file on collision (finding #5)."""
    store = h.tools._material_store
    written = store.write_document("Dup", "<materialx/>", None)
    ume.ensure_materials_scope(h.renderer._usd_stage)
    ume.author_material_holder(h.renderer._usd_stage, "/Materials/Dup", str(written))
    with pytest.raises(ValueError, match="already exists"):
        h.tools._add_synth_material(h.renderer, "Dup", str(written))
    assert not store.path_for("Dup").exists()


# ── add+bind rollback hard-removes so the name is reusable (finding #8/E) ──

def test_failed_add_bind_allows_same_name_retry(h) -> None:
    """A failed transactional add+bind must hard-remove the freshly-created
    holder (not leave an active=false tombstone), so a same-name retry succeeds
    instead of colliding forever (finding #8/E)."""
    h.renderer.fail_next_add_primitive = True
    with pytest.raises(ValueError, match="post-bind failure"):
        h.tools.scene_add_primitive("Sphere", material="marble_solid", name="Ball")
    # The rollback hard-removed the holder (discard, not a tombstone deactivate).
    assert any(c[0] == "discard_created_prim" for c in h.renderer.calls)
    assert not any(c[0] == "remove_node" for c in h.renderer.calls)
    holders = ume.collect_material_holders(h.renderer._usd_stage)
    assert holders == {}  # no lingering holder blocks reuse

    # A same-preset retry now re-creates and binds cleanly.
    h.renderer.fail_next_add_primitive = False
    result = h.tools.scene_add_primitive("Sphere", material="marble_solid", name="Ball2")
    assert result["material_path"].startswith("/Materials/")
