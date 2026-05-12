"""Tests for MaterialX nodegraph extraction + per-graph SSBO packing.

Covers `MaterialLibrary.generate_for_compute` and `pack_uniform_block` —
the two functions the renderer relies on to translate a MaterialX
surfacematerial into Slang code + GPU-ready bytes. Pure Python, no GPU
required.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

mx = pytest.importorskip("MaterialX")

from skinny.materialx_runtime import (
    GRAPH_ID_FIRST,
    MaterialLibrary,
    assign_graph_ids,
    pack_uniform_block,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "Usd-Mtlx-Example" / "materials"


@pytest.fixture(scope="module")
def lib():
    library = MaterialLibrary.from_install()
    library.load()
    return library


def _import_asset(library: MaterialLibrary, fname: str) -> None:
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(ASSETS / fname))
    library.import_document(doc)


# ─── generate_for_compute ─────────────────────────────────────────────


def test_marble_extracts_graph(lib):
    """Marble_3D's nodegraph emits a non-trivial evaluator + uniform set."""
    _import_asset(lib, "standard_surface_marble_solid.mtlx")
    gf = lib.generate_for_compute("Marble_3D", write_to_disk=False)
    assert gf is not None, "marble must produce a graph fragment"
    assert gf.func_name == "evalGraph_Marble_3D"
    assert gf.struct_name == "GraphParams_Marble_3D"
    assert gf.outputs_struct == "GraphOutputs_Marble_3D"
    names = {u.name for u in gf.uniform_block}
    # Marble's nodegraph wires these inputs through to the math body.
    for required in ("add_xyz_in2", "noise_octaves", "noise_amplitude",
                     "color_mix_fg", "color_mix_bg"):
        assert required in names, f"marble missing uniform {required!r}"
    # The extracted function body must invoke fractal3d.
    assert "mx_fractal3d_float(" in gf.slang_source
    # base_color must be one of the graph-driven std_surface inputs.
    output_names = {name for name, _ in gf.outputs}
    assert "base_color" in output_names


@pytest.mark.parametrize("asset,target", [
    ("standard_surface_glass.mtlx",   "Glass"),
    ("standard_surface_jade.mtlx",    "Jade"),
    ("standard_surface_chrome.mtlx",  "Chrome"),
    ("standard_surface_velvet.mtlx",  "Velvet"),
])
def test_constant_input_materials_return_none(lib, asset, target):
    """Constant-input materials produce no graph-driven std_surface inputs
    — `generate_for_compute` returns None and the renderer routes them
    through the flat / std_surface SSBO path."""
    _import_asset(lib, asset)
    gf = lib.generate_for_compute(target, write_to_disk=False)
    assert gf is None, f"{target} should fall back to flat path"


def test_brass_multi_output_graph(lib):
    """Tiled_Brass drives specular_roughness + coat_color + coat_roughness
    via its nodegraph (base_color stays constant). The extractor must
    surface all three as outputs even though base_color is not driven."""
    _import_asset(lib, "standard_surface_brass_tiled.mtlx")
    gf = lib.generate_for_compute("Tiled_Brass", write_to_disk=False)
    assert gf is not None
    output_names = {name for name, _ in gf.outputs}
    assert output_names == {"specular_roughness", "coat_color", "coat_roughness"}
    # GraphOutputs struct exposes all three with public fields.
    for name in output_names:
        assert f"public float" in gf.slang_source or \
               f"public float3" in gf.slang_source
        assert name in gf.slang_source


def test_texture_bound_graph_supported(lib):
    """Texture-using graphs (NG_tiledimage_*) compile via the helper-block
    inclusion path; mtlx_gen_shim resolves SamplerTexture2D against the
    bindless flatMaterialTextures array."""
    _import_asset(lib, "standard_surface_wood_tiled.mtlx")
    gf = lib.generate_for_compute("Tiled_Wood", write_to_disk=False)
    assert gf is not None, "Tiled_Wood must produce a graph fragment now"
    names = {u.name: u.type_name for u in gf.uniform_block}
    assert names.get("image_color_file") == "filename"
    assert names.get("image_roughness_file") == "filename"
    # Slang struct emission maps `filename` → SamplerTexture2D shim.
    assert "SamplerTexture2D image_color_file" in gf.slang_source
    # NG_tiledimage_* helpers are inlined as `internal` decls.
    assert "internal void NG_tiledimage_color3" in gf.slang_source


# ─── pack_uniform_block ────────────────────────────────────────────────


def test_pack_uniform_block_round_trip(lib):
    _import_asset(lib, "standard_surface_marble_solid.mtlx")
    gf = lib.generate_for_compute("Marble_3D", write_to_disk=False)
    assert gf is not None

    # Default-packed bytes must match offsets+sizes from the uniform block.
    buf = pack_uniform_block(gf.uniform_block)
    expected_size = max(f.offset + f.size for f in gf.uniform_block)
    expected_size = (expected_size + 3) & ~3
    assert len(buf) == expected_size

    # Override packs at the correct offset + reads back as int32.
    buf2 = pack_uniform_block(gf.uniform_block, {"noise_octaves": 7})
    oct_field = next(f for f in gf.uniform_block if f.name == "noise_octaves")
    val = struct.unpack_from("<i", buf2, oct_field.offset)[0]
    assert val == 7, "noise_octaves override must round-trip"


def test_pack_uniform_block_color3_override(lib):
    _import_asset(lib, "standard_surface_marble_solid.mtlx")
    gf = lib.generate_for_compute("Marble_3D", write_to_disk=False)
    assert gf is not None
    fg_field = next(f for f in gf.uniform_block if f.name == "color_mix_fg")

    buf = pack_uniform_block(gf.uniform_block, {"color_mix_fg": (0.25, 0.5, 0.75)})
    r, g, b = struct.unpack_from("<3f", buf, fg_field.offset)
    assert (r, g, b) == pytest.approx((0.25, 0.5, 0.75))


# ─── assign_graph_ids ──────────────────────────────────────────────────


def test_assign_graph_ids_starts_at_first(lib):
    _import_asset(lib, "standard_surface_marble_solid.mtlx")
    gf = lib.generate_for_compute("Marble_3D", write_to_disk=False)
    ids = assign_graph_ids([gf])
    assert ids == {"Marble_3D": GRAPH_ID_FIRST}
    # Empty list maps to empty dict.
    assert assign_graph_ids([]) == {}


# ─── aggregator emission ──────────────────────────────────────────────


def test_aggregator_emits_all_graphs(lib):
    """ComputePipeline._emit_generated_materials wires Marble + Wood +
    Brass into one generated_materials.slang with one binding per
    graph, the right dispatch switch, and per-graph apply functions
    covering each driven std_surface input."""
    from skinny.vk_compute import ComputePipeline, GRAPH_BINDING_BASE

    # Write into the real shader tree — the file is .gitignored and gets
    # rewritten on every renderer scene-load, so this just exercises the
    # same path the renderer uses.
    shader_dir = (
        Path(__file__).resolve().parent.parent
        / "src" / "skinny" / "shaders"
    )

    frags = []
    for asset, target in [
        ("standard_surface_marble_solid.mtlx", "Marble_3D"),
        ("standard_surface_wood_tiled.mtlx",   "Tiled_Wood"),
        ("standard_surface_brass_tiled.mtlx",  "Tiled_Brass"),
    ]:
        _import_asset(lib, asset)
        gf = lib.generate_for_compute(target, write_to_disk=False)
        assert gf is not None
        frags.append(gf)

    class Stub:
        pass
    stub = Stub()
    stub.shader_dir = shader_dir
    stub.graph_fragments = frags
    ComputePipeline._emit_generated_materials(stub)

    agg = (shader_dir / "generated_materials.slang").read_text(encoding="utf-8")
    # One import per fragment.
    for gf in frags:
        assert f"import generated.{gf.sanitized_name}_graph;" in agg
        # SSBO declaration at the right binding.
        assert f"binding({GRAPH_BINDING_BASE + frags.index(gf)}, 0)" in agg
        assert f"graphParams_{gf.sanitized_name}" in agg
        # Per-graph apply function references each driven std_surface input.
        for input_name, _ in gf.outputs:
            assert f"sp.{input_name} = g.{input_name};" in agg
    # Mapping graph_bindings exposed for the renderer to write descriptors.
    assert stub.graph_bindings == {
        gf.target_name: GRAPH_BINDING_BASE + idx
        for idx, gf in enumerate(frags)
    }
