"""GPU round-trip tests for MCP material authoring (mcp-material-authoring).

Exercises Renderer.add_material / bind_material / apply_material_overrides and
the branch-aware save_edits against a real GPU upload path on the native Metal
backend (megakernel). Run with the repo-root Python 3.13 venv, one guarded
Metal process at a time (CLAUDE.md Metal dispatch hygiene):

    PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \
        tests/test_material_authoring_gpu.py -m gpu -q

Each nodegraph material add changes the graph-set signature and rebuilds the
megakernel pipeline (design D9), so these tests keep the material count per
fixture minimal and reuse one renderer per class where correctness allows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
BACKEND = "metal"


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


def _have_metal() -> bool:
    try:
        from skinny.backend_select import make_context  # noqa: F401
        import slangpy  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")
needs_metal = pytest.mark.skipif(not _have_metal(), reason="No Metal/slangpy runtime")

pytestmark = [needs_usd, needs_metal, pytest.mark.gpu]


def _make_renderer():
    from skinny.backend_select import make_context
    from skinny.renderer import Renderer

    ctx = make_context(BACKEND, window=None, width=96, height=96)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR, tattoo_dir=TATTOO_DIR,
    )
    return ctx, renderer


def _render(renderer, frames: int = 4) -> np.ndarray:
    raw = b""
    for _ in range(frames):
        renderer.update(0.016)
        raw = renderer.render_headless()
    return np.frombuffer(raw, dtype=np.uint8).reshape(96, 96, 4).astype(np.float32)


def _add_sphere(renderer, name: str, x: float) -> str:
    return renderer.add_primitive(
        "Sphere", name=name, transform=None if x == 0.0 else _translate(x),
    )


def _translate(x: float):
    import numpy as np  # local: keep module import surface identical to peers

    m = np.eye(4)
    m[3, 0] = x
    return m


def _synthesized_material(renderer, store, name: str, spec: dict) -> str:
    """MCP-layer composition mirrored at renderer level: synthesize → write →
    add_material(mtlx) with rollback deleting the session files."""
    from skinny import mtlx_synthesis as ms

    result = ms.synthesize(spec, name)
    path = store.write_document(name, result.document_xml, result.mapping)
    return renderer.add_material(
        name, mtlx_path=str(path), session_dir=str(store.dir),
        on_rollback=lambda: store.delete(name),
    )


@pytest.fixture(scope="class")
def scene():
    """One renderer + empty scene + three spheres, reused across a class.

    Class scope keeps the pipeline-rebuild count bounded (each graph material
    add recompiles the megakernel — design D9).
    """
    ctx, renderer = _make_renderer()
    try:
        renderer.create_empty_scene()
        from skinny.mtlx_synthesis import SessionMaterialStore

        store = SessionMaterialStore()
        yield renderer, store
    finally:
        renderer.cleanup()
        ctx.destroy()


class TestMaterialRoundTrip:
    """5.1 — preset + template + raw graph render distinct, non-fallback."""

    def test_add_bind_render_distinct(self, scene):
        renderer, store = scene
        from skinny import mtlx_synthesis as ms

        s1 = _add_sphere(renderer, "S1", -2.5)
        s2 = _add_sphere(renderer, "S2", 0.0)
        s3 = _add_sphere(renderer, "S3", 2.5)

        marble = renderer.add_material(
            ms.preset_holder_name("marble_solid"),
            mtlx_path=ms.resolve_preset("marble_solid"),
        )
        noise = _synthesized_material(
            renderer, store, "NoiseA",
            {"template": "noise",
             "params": {"colorA": [0.9, 0.2, 0.1], "colorB": [0.1, 0.2, 0.9]}},
        )
        graph = _synthesized_material(
            renderer, store, "RawFractal",
            {"model": "standard_surface",
             "params": {"specular_roughness": 0.4},
             "graph": {
                 "nodes": {
                     "pos": {"type": "position", "output": "vector3"},
                     "noise": {"type": "fractal3d", "output": "color3",
                               "position": {"_connect": "pos"}, "octaves": 4},
                     "tint": {"type": "multiply", "output": "color3",
                              "in1": {"_connect": "noise"},
                              "in2": {"expose": True, "name": "tint_color",
                                      "value": [0.95, 0.85, 0.2]}},
                 },
                 "connections": [["tint.out", "base_color"]],
             }},
        )

        before = _render(renderer)
        renderer.bind_material(s1, marble)
        renderer.bind_material(s2, noise)
        renderer.bind_material(s3, graph)
        after = _render(renderer, frames=6)

        # Bound prims change the image (materials are live, not fallback-gray).
        assert not np.array_equal(before, after)

        # The three spheres are mutually distinct: sample a pixel column per
        # sphere center (camera default looks at origin; spheres offset in x).
        cols = [24, 48, 72]
        samples = [after[40:56, c - 4:c + 4, :3].mean(axis=(0, 1)) for c in cols]
        assert not np.allclose(samples[0], samples[1], atol=2.0)
        assert not np.allclose(samples[1], samples[2], atol=2.0)

    def test_same_template_twice_renders_independently(self, scene):
        """Salting: two same-template materials are separate graph identities."""
        renderer, store = scene

        s4 = _add_sphere(renderer, "S4", -1.2)
        s5 = _add_sphere(renderer, "S5", 1.2)
        red = _synthesized_material(
            renderer, store, "NoiseRed",
            {"template": "noise",
             "params": {"colorA": [1.0, 0.05, 0.05], "colorB": [0.6, 0.0, 0.0]}},
        )
        blue = _synthesized_material(
            renderer, store, "NoiseBlue",
            {"template": "noise",
             "params": {"colorA": [0.05, 0.05, 1.0], "colorB": [0.0, 0.0, 0.6]}},
        )
        renderer.bind_material(s4, red)
        renderer.bind_material(s5, blue)
        img = _render(renderer, frames=6)

        left = img[36:60, 12:44, :3].mean(axis=(0, 1))
        right = img[36:60, 52:84, :3].mean(axis=(0, 1))
        # Red-dominant vs blue-dominant — aliasing would make them equal.
        assert left[0] > left[2], f"left sphere not red-dominant: {left}"
        assert right[2] > right[0], f"right sphere not blue-dominant: {right}"


@pytest.fixture(scope="class")
def edit_scene():
    """Sphere pair with a bound noise template + bound raw graph material.

    Class-scoped fixtures do NOT carry state across classes — TestEditFanout
    builds its own live materials rather than assuming TestMaterialRoundTrip's.
    """
    ctx, renderer = _make_renderer()
    try:
        renderer.create_empty_scene()
        from skinny.mtlx_synthesis import SessionMaterialStore

        store = SessionMaterialStore()
        s1 = _add_sphere(renderer, "E1", -1.5)
        s2 = _add_sphere(renderer, "E2", 1.5)
        noise = _synthesized_material(
            renderer, store, "NoiseA",
            {"template": "noise",
             "params": {"colorA": [0.9, 0.2, 0.1], "colorB": [0.1, 0.2, 0.9]}},
        )
        raw = _synthesized_material(
            renderer, store, "RawFractal",
            {"model": "standard_surface",
             "graph": {
                 "nodes": {
                     "pos": {"type": "position", "output": "vector3"},
                     "noise": {"type": "fractal3d", "output": "color3",
                               "position": {"_connect": "pos"}, "octaves": 4},
                     "tint": {"type": "multiply", "output": "color3",
                              "in1": {"_connect": "noise"},
                              "in2": {"expose": True, "name": "tint_color",
                                      "value": [0.95, 0.85, 0.2]}},
                 },
                 "connections": [["tint.out", "base_color"]],
             }},
        )
        renderer.bind_material(s1, noise)
        renderer.bind_material(s2, raw)
        yield renderer, store
    finally:
        renderer.cleanup()
        ctx.destroy()


class TestEditFanout:
    """5.2 — scene_set-level fan-out edit changes the render; unexposed
    constants surface no property."""

    def test_fanout_edit_changes_pixels(self, edit_scene):
        renderer, store = edit_scene
        from skinny.scene_graph import build_scene_graph, find_node_by_path

        # NoiseA is live (bound in the fixture); find its scene-graph node and
        # edit a promoted input through the fan-out path.
        graph = build_scene_graph(renderer._usd_stage, renderer._usd_scene)
        node = find_node_by_path(graph, "/Materials/NoiseA")
        assert node is not None, "NoiseA holder missing from scene graph"
        prop = next((p for p in node.properties if p.name == "colorA"), None)
        assert prop is not None, (
            "promoted colorA not injected on the live material node; "
            f"has: {[p.name for p in node.properties]}"
        )
        fanout = (prop.metadata or {}).get("fanout")
        assert fanout, "colorA carries no fan-out uniform mapping"

        material_id = node.renderer_ref.index
        before = _render(renderer, frames=4)
        version_before = renderer._material_version
        renderer.apply_material_overrides(
            material_id, {u: (0.05, 1.0, 0.05) for u in fanout},
        )
        assert renderer._material_version == version_before + 1  # single bump
        after = _render(renderer, frames=4)
        assert not np.array_equal(before, after), "fan-out edit had no effect"

    def test_unexposed_constant_absent(self, edit_scene):
        renderer, store = edit_scene
        from skinny.scene_graph import build_scene_graph, find_node_by_path

        graph = build_scene_graph(renderer._usd_stage, renderer._usd_scene)
        node = find_node_by_path(graph, "/Materials/RawFractal")
        assert node is not None
        names = [p.name for p in node.properties]
        # The fractal3d octaves value was authored without expose:true — it must
        # not surface as an editable property (design D5: honest, not maximal).
        assert "octaves" not in names
        # The exposed multiply color DID surface.
        assert "tint_color" in names, names


class TestSaveReload:
    """5.3 — anonymous-branch save is a self-contained bundle that reloads."""

    def test_anonymous_save_reload_matches(self, tmp_path):
        from skinny import mtlx_synthesis as ms
        from skinny.pbrt import metrics

        ctx, renderer = _make_renderer()
        try:
            renderer.create_empty_scene()
            store = ms.SessionMaterialStore()
            sphere = _add_sphere(renderer, "Hero", 0.0)
            mat = _synthesized_material(
                renderer, store, "SaveNoise",
                {"template": "noise",
                 "params": {"colorA": [0.9, 0.6, 0.1], "colorB": [0.1, 0.1, 0.3]}},
            )
            renderer.bind_material(sphere, mat)
            # Reframe to the sphere-bearing scene: the empty-scene framing from
            # create_empty_scene differs from what a standalone reload of the
            # bundle computes (set_usd_scene frames to scene bounds on first
            # load) — align the two camera states so the images are comparable.
            renderer._frame_camera_to_scene(renderer._usd_scene)
            before = _render(renderer, frames=6)

            saved = tmp_path / "bundle" / "scene.usda"
            saved.parent.mkdir(parents=True)
            written = renderer.save_edits(str(saved))

            # Bundle is self-contained: session doc + sidecar copied beside it.
            copied = Path(written).parent / "materials" / "SaveNoise.mtlx"
            assert copied.exists(), "session .mtlx not copied into the bundle"
            assert copied.with_suffix(".json").exists(), "mapping sidecar not copied"
        finally:
            renderer.cleanup()
            ctx.destroy()

        # Reload STANDALONE from the bundle location in a fresh renderer.
        from pxr import Usd
        from skinny.usd_loader import load_scene_from_stage

        ctx2, renderer2 = _make_renderer()
        try:
            stage = Usd.Stage.Open(str(saved))
            scene2 = load_scene_from_stage(stage)
            names = [m.name for m in scene2.materials]
            assert any("SaveNoise" in n for n in names), (
                f"reloaded scene lost the synthesized material: {names}"
            )
            renderer2.set_usd_scene(scene2, stage=stage)
            after = _render(renderer2, frames=6)
            m = metrics.compute_metrics(
                (after[..., :3] / 255.0).astype(np.float32),
                (before[..., :3] / 255.0).astype(np.float32),
            )
            assert m.relmse < 0.05, f"reloaded render diverges: relMSE={m.relmse}"

            # Editability survives the reload via the copied sidecar.
            mat2 = next(m2 for m2 in scene2.materials if "SaveNoise" in m2.name)
            assert mat2.logical_inputs, "sidecar mapping lost on reload"
        finally:
            renderer2.cleanup()
            ctx2.destroy()

    def test_wood_preset_textures_survive_save(self, tmp_path):
        """Texture carve-out: curated wood keeps its absolute assets reference."""
        from skinny import mtlx_synthesis as ms

        ctx, renderer = _make_renderer()
        try:
            renderer.create_empty_scene()
            sphere = _add_sphere(renderer, "WoodBall", 0.0)
            wood = renderer.add_material(
                ms.preset_holder_name("wood_tiled"),
                mtlx_path=ms.resolve_preset("wood_tiled"),
            )
            renderer.bind_material(sphere, wood)
            before = _render(renderer, frames=6)

            saved = tmp_path / "wood" / "scene.usda"
            saved.parent.mkdir(parents=True)
            written = renderer.save_edits(str(saved))
            # Curated preset must NOT be copied (texture carve-out, design D7).
            assert not (Path(written).parent / "materials").exists() or not list(
                (Path(written).parent / "materials").glob("*wood*")
            )
        finally:
            renderer.cleanup()
            ctx.destroy()

        from pxr import Usd
        from skinny.usd_loader import load_scene_from_stage

        ctx2, renderer2 = _make_renderer()
        try:
            stage = Usd.Stage.Open(str(saved))
            scene2 = load_scene_from_stage(stage)
            renderer2.set_usd_scene(scene2, stage=stage)
            after = _render(renderer2, frames=6)
            # A dropped texture collapses the sphere to flat gray; the wood
            # pattern has spatial variance well above that.
            center = after[36:60, 36:60, :3]
            assert float(center.std()) > 2.0, (
                f"wood texture appears lost on reload (std={center.std():.2f})"
            )
            assert not np.array_equal(before, np.zeros_like(before))
        finally:
            renderer2.cleanup()
            ctx2.destroy()


class TestPrimitiveWithMaterial:
    """5.4 — the one-call composition path at renderer level."""

    def test_add_primitive_skip_inline_then_bind(self):
        from skinny import mtlx_synthesis as ms

        ctx, renderer = _make_renderer()
        try:
            renderer.create_empty_scene()
            marble = renderer.add_material(
                ms.preset_holder_name("marble_solid"),
                mtlx_path=ms.resolve_preset("marble_solid"),
            )
            path = renderer.add_primitive(
                "Sphere", name="OneCall", skip_inline_material=True,
            )
            renderer.bind_material(path, marble)

            # No orphan inline preview material was authored for the prim.
            stage = renderer._usd_stage
            assert not stage.GetPrimAtPath(f"{path}_material").IsValid()

            img = _render(renderer, frames=6)
            center = img[36:60, 36:60, :3]
            # Marble is a light procedural — a fallback-gray or black sphere
            # fails both the brightness and the variance floor.
            assert float(center.mean()) > 10.0
            assert float(center.std()) > 1.0
        finally:
            renderer.cleanup()
            ctx.destroy()
