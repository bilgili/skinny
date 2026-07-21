# Proposal: mcp-material-authoring

## Why

The MCP scene tools can author primitives only with a fixed three-parameter
preview material (color/roughness/metallic) and cannot create, enrich, or
rebind materials at all — an agent cannot produce a marble or wood look even
though the renderer already evaluates exactly those MaterialX nodegraph
materials when a scene references them (as `assets/three_materials_demo.usda`
does). Material generation and binding is the missing half of scene authoring.

## What Changes

- New MCP tool `material_list`: one discovery call returning the curated
  `.mtlx` preset catalog (editable inputs via generator reflection), the
  parametric model schemas (UsdPreviewSurface and MaterialX
  standard_surface), the supported nodegraph node types, and the procedural
  template schemas.
- New MCP tool `scene_add_material(spec, name?)`: creates a typed
  `UsdShade.Material` holder under `/Materials` from one of four spec forms
  — curated preset reference (deduped per preset), parametric
  UsdPreviewSurface, parametric standard_surface with an optional procedural
  nodegraph (Perlin/fractal noise etc.), or a server-owned procedural
  template. Synthesized documents are gated by a GPU-free Slang-generator
  dry-run before anything is authored. Returns the material prim path,
  flagged not-live until bound.
- New MCP tool `scene_bind_material(prim_path, material_path)`: binds (or
  rebinds) a material with explicit binding targets in the session edit
  layer, overriding file-authored bindings.
- `scene_add_primitive` gains an optional `material` argument accepting a
  preset/template name or an existing `/Materials/...` path, replacing the
  inline color/roughness/metallic material when given.
- New GPU-free synthesis module: validated material-spec → MaterialX
  document builder (node-type whitelist, dangling-connection checks,
  generator dry-run gate, element-name salting) plus the preset catalog,
  template expansion, and the logical-input → generated-uniform-name
  mapping that defines each material's editability contract.
- Loader intake extension: session-layer `.mtlx` reference discovery
  (asset-path collection and per-prim reference detection today scan only
  the root layer, so session-authored materials would otherwise be
  invisible on the default no-plugin path).
- Scene-graph surfacing: live materials' promoted logical inputs (and
  constant-shader override keys) injected as editable properties on
  material nodes, with `scene_set` writes fanned out through the uniform
  mapping via the existing material-override path.
- Renderer-side authoring: `add_material` / `bind_material` methods with
  the same edit-layer + rollback + resync discipline as `add_primitive`
  (rollback also deletes session `.mtlx` files), and a branch-aware
  `save_edits` extension (flattened-export post-process for created scenes;
  overlay re-anchor for file-backed scenes; texture-bearing presets keep
  absolute asset references).

## Capabilities

### New Capabilities

- `mtlx-material-synthesis`: turning a JSON material spec (preset name,
  flat parameters, nodegraph, or template) into a validated, gen-proven,
  name-salted MaterialX document plus its editable-input mapping — preset
  catalog, node-type whitelist, template expansion, session-file lifecycle,
  and rejection semantics. GPU-free and independently testable.

### Modified Capabilities

- `mcp-scene-control`: adds the three material tools and the
  `scene_add_primitive` `material` argument to the advertised tool surface,
  with self-describing results (including not-live-until-bound and
  job-degradation expectations for graph adds), server-side preset
  resolution, and logical-name property editing on live materials.
- `usd-scene-editing`: adds material authoring and binding as edit-layer
  structural edits — typed `/Materials` holders, session-layer `.mtlx`
  intake, binding-driven participation, explicit-target bind/rebind
  semantics, scene-graph property surfacing for live materials, rollback,
  and branch-aware persistence.

## Impact

- `src/skinny/mcp_server.py` — three new tools + `scene_add_primitive`
  extension and arg validation.
- New `src/skinny/mtlx_synthesis.py` (name indicative) — spec validation,
  preset catalog, template expansion, MaterialX doc build, generator
  dry-run gate, uniform-name mapping, session-file lifecycle.
- `src/skinny/usd_loader.py` — `_collect_mtlx_asset_paths` /
  `_prim_has_mtlx_reference` extended to scan session-layer prim specs
  (root-layer behavior unchanged).
- `src/skinny/scene_graph.py` — material nodes gain injected editable
  properties from the persisted mapping / parameter overrides.
- `src/skinny/renderer.py` — `add_material` / `bind_material` edit-layer
  authoring beside `add_primitive`; `save_edits` branch-aware material
  persistence; no shader or descriptor-binding changes (the MaterialX →
  Slang generation path is consumed as-is).
- `assets/Usd-Mtlx-Example/materials/` becomes the curated preset source
  (read-only; server-side resolution).
- Tests: hostless synthesis + whitelist gen dry-run + intake + tool
  suites; guarded Metal GPU round-trip (add → bind → render → edit →
  save/reload), budgeting one pipeline rebuild per graph-set change.
- Docs: MCP tool documentation and `docs/PythonAPI.md` for any new public
  symbols.
