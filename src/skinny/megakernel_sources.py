"""Backend-agnostic emission of the megakernel's runtime-generated Slang.

`main_pass.slang` does **not** compile standalone: it `import`s three Slang
artifacts the renderer generates at scene-load and writes into the shader
include path — `generated_materials.slang` (the `evalSceneGraph` aggregator),
`generated/<graph>_graph.slang` (one per scene MaterialX nodegraph), and
`python_materials_dispatcher.slang` (the `PythonMaterial`/`loadPythonMaterial`
bridge). Both GPU backends must run this emission before compiling the kernel:
the Vulkan `slangc` path and the in-process Slang→Metal path alike.

This module is the single, **Vulkan-free** home for that emission so the Metal
`ComputePipeline` can call it without importing `skinny.vk_compute` (which pulls
in the `vulkan` extension and needs the Vulkan SDK on the dynamic-library path).
The text generators are pure — no file I/O beyond the explicit `write_text`
sinks, no GPU handles. `skinny.vk_compute` re-exports the public names from here
for backward compatibility with existing importers.
"""

from __future__ import annotations

from pathlib import Path

# First descriptor binding for MaterialX nodegraph parameter SSBOs. Each
# loaded graph gets its own StructuredBuffer<GraphParams_X> at
# GRAPH_BINDING_BASE + graphIdx; idx 0 == graphId 2 in the dispatch (0=skin,
# 1=flat are reserved). Keep clear of bindings 0..24 used by the renderer.
GRAPH_BINDING_BASE = 25


def _class_has_imaterial_conformance(cls_node) -> bool:
    """True when the class is decorated `@sp.struct(conforms_to="IMaterial")`."""
    import ast

    for dec in cls_node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        for kw in dec.keywords:
            if kw.arg == "conforms_to" and isinstance(kw.value, ast.Constant) \
                    and kw.value.value == "IMaterial":
                return True
    return False


_PY_TYPE_TO_SLANG: dict[str, str] = {
    "float32":    "float",
    "float32x2":  "float2",
    "float32x3":  "float3",
    "float32x4":  "float4",
    "int32":      "int",
}

# FlatHitData → inputs-struct field mapping. Each entry is the Slang
# expression that supplies the value when the inputs struct declares a
# field with that name. Unknown fields fall back to a zero-default per
# type. Single source of truth for the v1 adapter; later versions may
# add per-material params SSBOs to lift this constraint.
_PY_INPUT_FROM_FHD: dict[str, str] = {
    "diffuseColor":  "h.mat.albedo",
    "baseColor":     "h.mat.albedo",
    "albedo":        "h.mat.albedo",
    "color":         "h.mat.albedo",
    "roughness":     "h.mat.roughness",
    "metallic":      "h.mat.metallic",
    "metalness":     "h.mat.metallic",
    "specular":      "h.mat.specular",
    "emissiveColor": "h.mat.emission",
    "emission":      "h.mat.emission",
    "opacity":       "h.mat.opacity",
    "ior":           "h.mat.ior",
    "normalScale":   "1.0",
    "coat":          "h.mat.coat",
    "coatRoughness": "h.mat.coatRoughness",
    "coatColor":     "h.mat.coatColor",
    "coatIOR":       "h.mat.coatIOR",
}

_PY_TYPE_ZERO: dict[str, str] = {
    "float32":    "0.0",
    "float32x2":  "float2(0.0)",
    "float32x3":  "float3(0.0)",
    "float32x4":  "float4(0.0)",
    "int32":      "0",
}


def scan_python_materials() -> list[dict]:
    """Walk `python_materials/*.py` and return one record per
    `IMaterial`-conforming class.

    Single source of truth for the python_id assignment shared between
    `emit_python_dispatcher` (dispatcher switch order) and the renderer's
    `materialTypes[matId]` packing. Order is the `sorted(glob("*.py"))` of
    files containing such classes.

    Each entry:
        {
            "module":          "python_materials.<stem>",
            "struct":          IMaterial-conforming struct name,
            "inputs_struct":   type annotation of its `params` field,
            "inputs_fields":   [(name, slang_type_alias), ...],
        }
    """
    import ast

    materials_dir = Path(__file__).resolve().parent.parent.parent / "python_materials"
    if not materials_dir.is_dir():
        return []
    py_files = [f for f in sorted(materials_dir.glob("*.py"))
                if f.name != "__init__.py"]
    out: list[dict] = []

    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError:
            continue
        classes_by_name: dict[str, dict[str, str]] = {}
        material_classes: list[tuple[str, str]] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            annotations: dict[str, str] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    annotations[stmt.target.id] = ast.unparse(stmt.annotation)
            classes_by_name[node.name] = annotations
            if _class_has_imaterial_conformance(node):
                params_anno = annotations.get("params")
                if params_anno:
                    material_classes.append((node.name, params_anno))

        for cls_name, inputs_struct_anno in material_classes:
            inputs_struct = inputs_struct_anno.split(".")[-1]
            inputs_fields = []
            for fname, ftype in classes_by_name.get(inputs_struct, {}).items():
                tail = ftype.split(".")[-1]
                if tail not in _PY_TYPE_TO_SLANG:
                    continue
                inputs_fields.append((fname, tail))
            out.append({
                "module": f"python_materials.{f.stem}",
                "struct": cls_name,
                "inputs_struct": inputs_struct,
                "inputs_fields": inputs_fields,
            })

    return out


def python_material_ids() -> dict[str, int]:
    """Module-name → python_id mapping, sourced from `scan_python_materials`.
    The dispatcher emits `case <id>u:` in the same order.
    """
    return {e["module"]: i for i, e in enumerate(scan_python_materials())}


def emit_megakernel_aggregator(graph_fragments, graph_binding_base: int) -> str:
    """Build the `generated_materials.slang` aggregator text for the megakernel.

    Stitches every scene MaterialX nodegraph into one `evalSceneGraph` switch
    plus the `evalSceneGraphBaseColor` override switch, with one SSBO binding,
    param helper, and apply helper per graph. Pure: no file I/O, no Vulkan —
    the megakernel half of the two-emitter split. The wavefront half is
    `vk_compute.emit_wavefront_material_modules`; both consume the same
    GraphFragment list.
    """
    imports: list[str] = []
    ssbo_decls: list[str] = []
    param_helpers: list[str] = []
    apply_helpers: list[str] = []
    cases: list[str] = []
    for idx, gf in enumerate(graph_fragments):
        module_name = f"{gf.sanitized_name}_graph"
        imports.append(f"import generated.{module_name};")
        binding = graph_binding_base + idx
        ssbo_decls.append(
            f"[[vk::binding({binding}, 0)]]\n"
            f"StructuredBuffer<{gf.struct_name}> graphParams_{gf.sanitized_name};\n"
        )
        param_helpers.append(
            f"{gf.struct_name} _graphParams_{gf.sanitized_name}(uint matId)\n"
            f"{{\n"
            f"    return graphParams_{gf.sanitized_name}[matId];\n"
            f"}}\n"
        )

        # Per-graph apply: copy each output field onto the matching
        # StdSurfaceParams slot. Built from the fragment's `outputs`
        # metadata so multi-output graphs (brass = specular_roughness +
        # coat_color + coat_roughness) drive several inputs at once.
        assignments = "\n".join(
            f"    sp.{input_name} = g.{input_name};"
            for input_name, _ in gf.outputs
        )
        apply_helpers.append(
            f"void applyGraphOutputs_{gf.sanitized_name}("
            f"inout StdSurfaceParams sp, in {gf.outputs_struct} g)\n"
            f"{{\n"
            f"{assignments}\n"
            f"}}\n"
        )

        cases.append(
            f"        case {idx + 2}u:  // graphId 0=skin, 1=flat reserved\n"
            f"        {{\n"
            f"            {gf.outputs_struct} g = {gf.func_name}(P, N, T, UV, "
            f"_graphParams_{gf.sanitized_name}(matId));\n"
            f"            applyGraphOutputs_{gf.sanitized_name}(sp, g);\n"
            f"            return;\n"
            f"        }}\n"
        )

    # Per-hit base_color override path (FlatMaterial.albedo). Only
    # graphs whose outputs include `base_color` participate; for
    # graphs that drive other inputs (brass: specular_roughness +
    # coat_*) the caller must NOT override albedo, so the case
    # simply returns false and the caller keeps the SSBO constant.
    base_color_cases = ""
    for idx, gf in enumerate(graph_fragments):
        has_base = any(i == "base_color" for i, _ in gf.outputs)
        if not has_base:
            continue
        base_color_cases += (
            f"        case {idx + 2}u:\n"
            f"        {{\n"
            f"            {gf.outputs_struct} g = {gf.func_name}(P, N, T, UV, "
            f"_graphParams_{gf.sanitized_name}(matId));\n"
            f"            outColor = g.base_color;\n"
            f"            return true;\n"
            f"        }}\n"
        )

    switch_body = "".join(cases) if cases else ""

    return (
        "// Auto-generated. Do not edit — written by "
        "megakernel_sources.emit_generated_materials().\n"
        "// Imports each scene MaterialX nodegraph as a Slang module.\n"
        "// Per-graph modules expose only `evalGraph_<target>` + the\n"
        "// matching `GraphParams_<target>` / `GraphOutputs_<target>`\n"
        "// structs; their `internal` helpers stay module-private, so\n"
        "// duplicate symbol names across graphs do not collide.\n\n"
        "import mtlx_std_surface;  // StdSurfaceParams\n\n"
        + "\n".join(imports)
        + ("\n\n" if imports else "\n")
        + "\n".join(ssbo_decls)
        + ("\n" if ssbo_decls else "")
        + "\n".join(param_helpers)
        + ("\n" if param_helpers else "")
        + "\n".join(apply_helpers)
        + ("\n" if apply_helpers else "")
        + "// Evaluate the per-hit nodegraph and overlay each driven\n"
        "// std_surface input on `sp`. graphId 0 / 1 reserved (skin /\n"
        "// flat) — callers gate the call by `materialGraphId(mid) >= 2`.\n"
        "void evalSceneGraph(uint graphId, uint matId,\n"
        "                    float3 P, float3 N, float3 T, float2 UV,\n"
        "                    inout StdSurfaceParams sp)\n"
        "{\n"
        "    switch (graphId)\n"
        "    {\n"
        f"{switch_body}"
        "        default:\n"
        "            sp.base_color = float3(1.0, 0.0, 1.0);\n"
        "            return;\n"
        "    }\n"
        "}\n\n"
        "// Returns true and fills `outColor` only when the active graph\n"
        "// drives std_surface.base_color (marble, wood). Graphs that\n"
        "// drive only other inputs (brass: specular_roughness + coat_*)\n"
        "// return false; the caller keeps the SSBO-uploaded constant\n"
        "// for FlatMaterial.albedo.\n"
        "bool evalSceneGraphBaseColor(uint graphId, uint matId,\n"
        "                              float3 P, float3 N, float3 T, float2 UV,\n"
        "                              out float3 outColor)\n"
        "{\n"
        "    outColor = float3(0.0);\n"
        "    switch (graphId)\n"
        "    {\n"
        f"{base_color_cases}"
        "        default:\n"
        "            return false;\n"
        "    }\n"
        "}\n"
    )


def run_codegen(shader_dir: Path) -> None:
    """Regenerate .slang from python_materials/ via embedded SlangPile."""
    import logging
    import sys

    from skinny.slangpile import build_module

    log = logging.getLogger("skinny.codegen")
    materials_dir = Path(__file__).resolve().parent.parent.parent / "python_materials"
    if not materials_dir.is_dir():
        return
    py_files = [f for f in sorted(materials_dir.glob("*.py")) if f.name != "__init__.py"]
    if not py_files:
        return
    out_dir = shader_dir.parent / "mtlx" / "genslang"
    sys.path.insert(0, str(materials_dir.parent))
    try:
        for f in py_files:
            mod_name = f"python_materials.{f.stem}"
            log.debug("codegen: %s", mod_name)
            build_module(mod_name, out_dir)
    except Exception as exc:
        log.debug("codegen failed (non-fatal): %s", exc)
    finally:
        sys.path.pop(0)


def emit_generated_materials(shader_dir: Path, graph_fragments) -> dict[str, int]:
    """Materialise shaders/generated_materials.slang + per-graph files.

    `main_pass.slang` `import generated_materials;` always, even when the scene
    carries no MaterialX graphs. Empty list ⇒ aggregator emits only the
    macro-alias prelude and a no-op `evalSceneGraph` switch that returns magenta
    for any graphId (caller never invokes it when no graphs are bound).

    Per-graph files are written under `shaders/generated/` so the existing
    `-I shaders/` include path (Vulkan slangc) and the Metal `include_paths`
    resolve them. Returns the per-graph descriptor-binding map (`graph_bindings`)
    so the renderer can update descriptor sets into the right slot.
    """
    gen_dir = shader_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale per-graph files: a scene reload may drop materials, and
    # stale Slang in the include dir can mask missing wiring or collide on
    # struct names.
    for old in gen_dir.glob("*_graph.slang"):
        old.unlink()

    # Per-graph module files (shared by both emitters): the `-I shaders/`
    # include path resolves `generated.<module>`.
    for gf in graph_fragments:
        module_name = f"{gf.sanitized_name}_graph"
        (gen_dir / f"{module_name}.slang").write_text(
            gf.slang_source, encoding="utf-8"
        )

    # Megakernel aggregator: one switch over all graphs.
    aggregator = emit_megakernel_aggregator(graph_fragments, GRAPH_BINDING_BASE)
    (shader_dir / "generated_materials.slang").write_text(
        aggregator, encoding="utf-8"
    )

    return {
        gf.target_name: GRAPH_BINDING_BASE + idx
        for idx, gf in enumerate(graph_fragments)
    }


def emit_python_dispatcher(shader_dir: Path) -> list[str]:
    """Materialise `shaders/python_materials_dispatcher.slang`.

    Builds dispatch helpers + a `PythonMaterial` wrapper conforming to
    `IMaterial`. Always emitted (even with zero detected materials) so
    `import python_materials_dispatcher` in `main_pass` / `path` stays valid
    across reloads. Returns the assignment order (`python_material_modules`) so
    the renderer can mirror it when packing `materialTypes[matId]`.

    Python material IDs are the index in the sorted-file order; the renderer
    mirrors this when packing `materialTypes[matId]`.
    """
    entries = scan_python_materials()

    imports = [f"import {e['module']};" for e in entries]
    adapters: list[str] = []
    sample_cases: list[str] = []
    eval_cases: list[str] = []
    for idx, e in enumerate(entries):
        assignments: list[str] = []
        for fname, ftype in e["inputs_fields"]:
            expr = _PY_INPUT_FROM_FHD.get(
                fname, _PY_TYPE_ZERO.get(ftype, "0.0"),
            )
            assignments.append(f"    ins.{fname} = {expr};")
        adapter_body = "\n".join(assignments) if assignments else "    /* no mappable fields */"
        adapters.append(
            f"{e['inputs_struct']} _pyMatInputs_{idx}(FlatHitData h)\n"
            f"{{\n"
            f"    {e['inputs_struct']} ins;\n"
            f"{adapter_body}\n"
            f"    return ins;\n"
            f"}}\n"
        )
        sample_cases.append(
            f"        case {idx}u:\n"
            f"        {{\n"
            f"            {e['struct']} m;\n"
            f"            m.params = _pyMatInputs_{idx}(fhd);\n"
            f"            return m.sample(wo, rng);\n"
            f"        }}\n"
        )
        eval_cases.append(
            f"        case {idx}u:\n"
            f"        {{\n"
            f"            {e['struct']} m;\n"
            f"            m.params = _pyMatInputs_{idx}(fhd);\n"
            f"            return m.evaluate(wo, wi);\n"
            f"        }}\n"
        )

    sample_default = (
        "        default:\n"
        "        {\n"
        "            BSDFSample s;\n"
        "            s.valid = false; s.transmitted = false; s.pdf = 0.0;\n"
        "            s.emission = float3(0.0); s.wi = float3(0.0); s.weight = float3(0.0);\n"
        "            s.response = float3(0.0);\n"
        "            return s;\n"
        "        }\n"
    )
    eval_default = (
        "        default:\n"
        "        {\n"
        "            BSDFSample e;\n"
        "            e.valid = false; e.transmitted = false; e.pdf = 0.0;\n"
        "            e.emission = float3(0.0); e.wi = float3(0.0); e.weight = float3(0.0);\n"
        "            e.response = float3(0.0);\n"
        "            return e;\n"
        "        }\n"
    )

    dispatcher = (
        "// Auto-generated. Do not edit — written by "
        "megakernel_sources.emit_python_dispatcher().\n"
        "// Bridges each python_materials/*.py IMaterial-conforming struct\n"
        "// into a `PythonMaterial` wrapper consumed by the integrators.\n\n"
        "import common;\n"
        "import bindings;\n"
        "import interfaces;\n"
        "import materials.flat.flat_shading;\n"
        + ("\n".join(imports) + "\n\n" if imports else "\n")
        + "".join(adapters)
        + "\n"
        + "BSDFSample samplePythonMaterial(uint pyId, FlatHitData fhd, float3 wo, inout RNG rng)\n"
        "{\n"
        "    switch (pyId)\n"
        "    {\n"
        + "".join(sample_cases)
        + sample_default
        + "    }\n"
        "}\n\n"
        "BSDFSample evalPythonMaterial(uint pyId, FlatHitData fhd, float3 wo, float3 wi)\n"
        "{\n"
        "    switch (pyId)\n"
        "    {\n"
        + "".join(eval_cases)
        + eval_default
        + "    }\n"
        "}\n\n"
        "// IMaterial-conforming wrapper. Monomorphised by allLightsNEE so\n"
        "// each Python material gets its own NEE code path with no\n"
        "// indirect-call cost on the GPU.\n"
        "struct PythonMaterial : IMaterial\n"
        "{\n"
        "    uint pyId;\n"
        "    FlatHitData data;\n"
        "    BSDFSample sample(float3 wo, inout RNG rng)\n"
        "    {\n"
        "        return samplePythonMaterial(pyId, data, wo, rng);\n"
        "    }\n"
        "    BSDFSample evaluate(float3 wo, float3 wi)\n"
        "    {\n"
        "        return evalPythonMaterial(pyId, data, wo, wi);\n"
        "    }\n"
        "}\n\n"
        "PythonMaterial loadPythonMaterial(HitInfo h, uint pyId)\n"
        "{\n"
        "    PythonMaterial m;\n"
        "    m.pyId = pyId;\n"
        "    m.data = fetchFlatHitData(h);\n"
        "    return m;\n"
        "}\n"
    )

    (shader_dir / "python_materials_dispatcher.slang").write_text(
        dispatcher, encoding="utf-8",
    )

    return [e["module"] for e in entries]


def emit_megakernel_sources(shader_dir: Path, graph_fragments) -> tuple[dict[str, int], list[str]]:
    """Run the full backend-agnostic emission that `main_pass.slang` needs
    before it can compile, on either GPU backend.

    Regenerates the python-material genslang, emits `generated_materials.slang`
    + per-graph modules + the python dispatcher, and returns
    `(graph_bindings, python_material_modules)`. Both the Vulkan
    `ComputePipeline` (slangc → SPIR-V) and the Metal `ComputePipeline`
    (in-process Slang → Metal) call this before linking.
    """
    run_codegen(shader_dir)
    graph_bindings = emit_generated_materials(shader_dir, graph_fragments)
    python_material_modules = emit_python_dispatcher(shader_dir)
    return graph_bindings, python_material_modules
