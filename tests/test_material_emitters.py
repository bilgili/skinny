"""GPU-free tests for the two material codegen emitters.

Both emitters consume the same GraphFragment metadata. The megakernel emitter
stitches all graphs into one `evalSceneGraph` switch (current behaviour); the
wavefront emitter produces a per-graph surface-evaluation function that a
wavefront shade kernel calls. No MaterialX / Vulkan needed — a fake fragment
exposing the attributes the emitters read is enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skinny.vk_compute import (
    GRAPH_BINDING_BASE,
    emit_megakernel_aggregator,
    emit_wavefront_material_modules,
    graph_param_combined_stride,
)


@dataclass
class _FakeField:
    name: str
    offset: int
    size: int
    type_name: str = "float"


@dataclass
class _FakeGF:
    sanitized_name: str
    struct_name: str
    func_name: str
    outputs_struct: str
    target_name: str
    slang_source: str = "// graph source"
    # list of (std_surface input name, slang type) the graph drives
    outputs: list = field(default_factory=list)
    # scalar-layout param fields (offset/size bytes) — drives the combined stride
    uniform_block: list = field(default_factory=list)


def _marble() -> _FakeGF:
    return _FakeGF(
        sanitized_name="Marble_3D",
        struct_name="GraphParams_Marble_3D",
        func_name="evalGraph_Marble_3D",
        outputs_struct="GraphOutputs_Marble_3D",
        target_name="Marble_3D",
        outputs=[("base_color", "float3"), ("specular_roughness", "float")],
        # color3 base @0 + float rough @12 → scalar size 16
        uniform_block=[_FakeField("base", 0, 12), _FakeField("rough", 12, 4)],
    )


def _brass() -> _FakeGF:
    # drives only non-base_color inputs → no base_color override case
    return _FakeGF(
        sanitized_name="Tiled_Brass",
        struct_name="GraphParams_Tiled_Brass",
        func_name="evalGraph_Tiled_Brass",
        outputs_struct="GraphOutputs_Tiled_Brass",
        target_name="Tiled_Brass",
        outputs=[("specular_roughness", "float"), ("coat_color", "float3")],
        # bigger struct: two color3 + float → scalar size 28
        uniform_block=[_FakeField("a", 0, 12), _FakeField("b", 16, 12),
                       _FakeField("c", 28, 4)],
    )


# ── megakernel emitter ─────────────────────────────────────────────


def test_megakernel_empty_emits_noop_switch():
    agg = emit_megakernel_aggregator([], GRAPH_BINDING_BASE)
    assert "import mtlx_std_surface;" in agg
    assert "void evalSceneGraph(" in agg
    assert "bool evalSceneGraphBaseColor(" in agg
    # No per-graph cases; default paints magenta.
    assert "case 2u" not in agg
    assert "sp.base_color = float3(1.0, 0.0, 1.0);" in agg


def test_megakernel_emits_graph_import_combined_buffer_and_case():
    gf = _marble()
    agg = emit_megakernel_aggregator([gf], GRAPH_BINDING_BASE)
    assert "import generated.Marble_3D_graph;" in agg
    # One combined byte-addressed buffer at the single binding.
    assert f"binding({GRAPH_BINDING_BASE}, 0)" in agg
    assert "ByteAddressBuffer graphParamsCombined;" in agg
    # Accessor loads the struct by matId * stride, not StructuredBuffer indexing.
    assert "graphParamsCombined.Load<GraphParams_Marble_3D>(" in agg
    assert "matId * GRAPH_PARAM_STRIDE" in agg
    assert "StructuredBuffer<GraphParams_Marble_3D>" not in agg
    assert "case 2u:" in agg
    assert "evalGraph_Marble_3D(P, N, T, UV, " in agg
    assert "applyGraphOutputs_Marble_3D(sp, g);" in agg
    # apply helper assigns each driven input
    assert "sp.base_color = g.base_color;" in agg
    assert "sp.specular_roughness = g.specular_roughness;" in agg


def test_megakernel_uses_one_binding_for_any_graph_count():
    agg = emit_megakernel_aggregator([_marble(), _brass()], GRAPH_BINDING_BASE)
    # Exactly one graph-param binding regardless of graph count — the slot-cap
    # fix (no per-graph binding 26, 27, …).
    assert agg.count("ByteAddressBuffer graphParamsCombined;") == 1
    assert f"binding({GRAPH_BINDING_BASE}, 0)" in agg
    assert f"binding({GRAPH_BINDING_BASE + 1}, 0)" not in agg
    # Both graphs still get their accessor + dispatch case.
    assert "graphParamsCombined.Load<GraphParams_Marble_3D>(" in agg
    assert "graphParamsCombined.Load<GraphParams_Tiled_Brass>(" in agg
    assert "case 2u:" in agg
    assert "case 3u:" in agg


def test_combined_stride_is_max_scalar_16_aligned():
    # marble scalar size 16, brass scalar size 32 (max field end 32) → max 32,
    # already 16-aligned.
    assert graph_param_combined_stride([_marble()]) == 16
    assert graph_param_combined_stride([_brass()]) == 32
    assert graph_param_combined_stride([_marble(), _brass()]) == 32
    # Empty scene → a non-zero stub stride (never indexed).
    assert graph_param_combined_stride([]) == 16
    # The aggregator bakes the combined (max) stride, shared by all accessors.
    agg = emit_megakernel_aggregator([_marble(), _brass()], GRAPH_BINDING_BASE)
    assert "static const uint GRAPH_PARAM_STRIDE = 32u;" in agg


def test_megakernel_empty_emits_no_graph_buffer():
    agg = emit_megakernel_aggregator([], GRAPH_BINDING_BASE)
    assert "graphParamsCombined" not in agg
    assert "GRAPH_PARAM_STRIDE" not in agg


def test_megakernel_base_color_case_only_for_base_color_graphs():
    agg = emit_megakernel_aggregator([_marble(), _brass()], GRAPH_BINDING_BASE)
    # Marble drives base_color → participates in the override switch.
    assert "outColor = g.base_color;" in agg
    # Brass drives no base_color → its evaluator must not appear in the
    # base-color override switch.
    assert agg.count("evalGraph_Tiled_Brass") == 1  # only the main switch case


# ── wavefront emitter ──────────────────────────────────────────────


def test_wavefront_emits_per_graph_surface_eval():
    gf = _marble()
    src = emit_wavefront_material_modules([gf])
    # One self-contained surface-eval entry per graph, built from the SAME
    # fragment func + params struct the megakernel uses.
    assert "evalGraphSurface_Marble_3D(" in src
    assert "evalGraph_Marble_3D(" in src
    assert "GraphParams_Marble_3D" in src
    assert "applyGraphOutputs_Marble_3D" in src


def test_wavefront_emits_one_entry_per_graph():
    src = emit_wavefront_material_modules([_marble(), _brass()])
    assert "evalGraphSurface_Marble_3D(" in src
    assert "evalGraphSurface_Tiled_Brass(" in src


def test_wavefront_empty_is_valid_noop():
    src = emit_wavefront_material_modules([])
    # Still imports the surface-params struct so the module compiles standalone.
    assert "import mtlx_std_surface;" in src
    assert "evalGraphSurface_" not in src
