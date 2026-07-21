# Design: mcp-material-authoring

Revision 2 — folds in the pre-implementation design review (B1–B3, M1–M6,
m1–m5 findings), which ground-truthed revision 1 against the loader,
generator, and scene-graph code.

## Context

The MCP structural tools (`scene_add_primitive`, `scene_add_model`,
`scene_add_light`, `scene_remove`, `scene_save`) author into the USD session
edit layer through renderer methods that follow one discipline: build under
`Usd.EditContext(stage, EditTarget(edit_layer))`, roll back created prims on
any failure, then `_resync_geometry_from_stage()`. Materials today exist in
exactly one authored form — the inline `UsdPreviewSurface` that
`add_primitive` creates with three seeded inputs (renderer.py:6285).

The renderer's material pipeline is richer than the tool surface, but the
review established precisely how it works — several revision-1 assumptions
were wrong:

- **`.mtlx` intake is root-layer-only and binding-driven.** With the default
  `use_usd_mtlx_plugin=False`, `.mtlx` materials are ingested by
  `_load_mtlx_materials`, fed by `_collect_mtlx_asset_paths`, which walks
  only `stage.GetRootLayer().rootPrims` (usd_loader.py:804-823) — a
  reference authored in the session layer is invisible. Materials enter
  `scene.materials` only via `_resolve_material_binding` during geometry
  traversal (usd_loader.py:1094-1182) — an unbound material is not loaded
  at all.
- **Editability is panel-dock plumbing, not scene-graph plumbing.** Graph
  uniforms surface through `renderer.iter_graph_uniforms`
  (renderer.py:7191), consumed only by the Qt panel dock. Scene-graph
  material nodes get editable properties only from stage-authored
  `UsdShade.Shader` prims (`_add_shader_props`); fallback-loaded `.mtlx`
  materials have none, so `scene_set` cannot reach them today.
- **The generator mangles names.** `generate_for_compute` emits
  PublicUniforms named after interior node inputs (`color_mix_fg`,
  `power_in2`), not nodegraph interface inputs, and one interface input
  feeding N node inputs shatters into N uniforms. Override packing matches
  strictly by uniform-field name (materialx_runtime.py:906) filtered to
  `used_uniforms`.
- **`save_edits` has two branches** (renderer.py:6439-6462): anonymous root
  → `stage.Export()` (flattened; reference arcs resolved/removed);
  file-backed root → `edit_layer.Export()` (an overlay layer, not
  self-contained).
- **Element names are a global namespace.** `importLibrary` skips
  already-present element names (materialx_runtime.py:261-271); material
  tables key by surfacematerial element name and binding-target leaf name —
  same-named documents alias or shadow each other.

The curated `.mtlx` corpus (`assets/Usd-Mtlx-Example/materials/`, 12
standard_surface materials including the `fractal3d`-driven procedural
marble) remains the proven composition pattern, and MaterialX ≥1.39 with
Python bindings is a hard dependency, so synthesis may use the MaterialX
Python API freely — including running the Slang generator itself as a
GPU-free validation and reflection step.

## Goals / Non-Goals

**Goals:**

- One discovery tool that tells an agent everything it can build: presets,
  parametric schemas, supported graph nodes, templates — with editable-input
  names that are the *actual writable keys* (gen reflection, not `.mtlx`
  parsing).
- Create materials in four spec forms: curated preset, parametric
  UsdPreviewSurface, parametric standard_surface with optional procedural
  nodegraph, server-owned template.
- Bind/rebind any `/Materials` material to any geometry prim.
- Everything authored lands in the session edit layer, is rolled back on
  failure (including session `.mtlx` files), survives `scene_save` per the
  branch-specific save plan (D7), and is editable via `scene_set` through a
  new scene-graph surfacing of graph uniforms (D5).
- Preset names resolve server-side by dict lookup into the enumerated
  catalog — never by joining a client string onto a directory path.

**Non-Goals:**

- No new shader or descriptor-binding work — the MaterialX → Slang path is
  consumed as-is. Node types the generator cannot compile are refused at
  validation.
- No client-supplied texture/image inputs in spec forms (`image`/
  `tiledimage` excluded from the v1 whitelist). Texture-bearing curated
  presets keep absolute references into `assets/` on save (D7 carve-out).
- No editing of curated `.mtlx` files; presets are read-only sources.
- No promise that unexposed graph constants are editable — they are absent
  from the property set and a `scene_set` on them is the existing
  "no property" error.
- No material deletion tool (`scene_remove` deactivates; bound geometry
  falls to the fallback slot on next resync — accepted).
- No mitigation of the per-add pipeline rebuild beyond documenting it (D9);
  batching is a possible v1.1.

## Decisions

### D1 — All authoring goes through USD in the session edit layer

Unchanged from revision 1. Materials are prims; renderer-side direct pushes
into `scene.materials` are rejected (invisible to `scene_save`, second code
path beside the proven stage-resync one).

### D2 — Synthesized documents are session `.mtlx` files referenced from
typed `UsdShade.Material` holder prims; the loader learns to see them

One composition mechanism (`references = @file.mtlx@`) covers curated and
generated materials. Two corrections from review:

- **Holder prims are typed.** `add_material` authors
  `UsdShade.Material.Define(stage, "/Materials/<Name>")` and adds the
  `.mtlx` file reference on that prim — not an untyped `def` like the demo
  file. Typed holders classify as materials in the scene graph
  (`_is_shade_material`) and give `bind_material` a stable, existing target
  path (fixes review M4: the demo's binding targets a child path that never
  composes on the default no-plugin path).
- **Loader intake extension (new scoped work, was review B1).**
  `_collect_mtlx_asset_paths` and `_prim_has_mtlx_reference` additionally
  scan the session layer's prim specs. Session-layer references are
  authored with **absolute** asset paths (the session layer is anonymous —
  there is nothing to anchor relative paths to; `_read_open_stage` falls
  back to `Path.cwd()` for anonymous roots). Root-layer behavior is
  byte-unchanged.

Session `.mtlx` files live in a server-owned session directory (tempdir
based). The directory is server configuration in the same trust domain as
the preset catalog — it is *not* required to fall inside the configured
allowed roots when `SKINNY_MCP_ROOTS` is customized (review m3); clients
never supply paths into it. Files are flushed to disk **before**
`_resync_geometry_from_stage` runs (resync re-reads `.mtlx` from disk), and
the rollback path deletes the written file along with the created prims.

Rejected alternatives unchanged (bare standard_surface shader prims in USD:
unproven loader path; anonymous in-memory layers: unpersistable).

### D3 — Preset names are a server-side catalog, not client paths

Unchanged, with one sharpening (review NOTE): resolution is a dict lookup
into the catalog enumerated from `assets/Usd-Mtlx-Example/materials/*.mtlx`
(names = filename minus `standard_surface_` prefix); the client string is
never joined onto a filesystem path. The allowed-roots check is not
consulted for catalog resolution.

### D4 — One spec schema, four forms, validated by a generator dry-run

Validation order: shape → parameter bounds (reusing `_coerce`/
`_check_bounds` + `_MATERIAL_FLOAT_RANGES`, finite-check on unknown
numerics) → for graphs, node-type whitelist, dangling-connection check,
MaterialX document build — then a **GPU-free generator dry-run**
(`MaterialLibrary.generate` + `generate_for_compute` on the synthesized
document). The dry-run replaces bare `MaterialX.validate()` as the gate
(resolves revision 1's open question, per review m1: the real failure
surface includes generator bailouts that document validation cannot see,
and the dry-run measured in seconds on marble-sized graphs). Its reflection
output is also what feeds D5's name mapping — one step, two purposes. Any
failure is a `SceneToolError` before any prim or file exists.

The node-type whitelist is a data tuple: `fractal3d`, `noise2d`, `noise3d`,
`position`, `texcoord`, `mix`, `multiply`, `add`, `subtract`, `sin`,
`power`, `dotproduct`, `ramplr`, `ramptb`. Provenance corrected
(review m1): the curated corpus proves only a subset (`fractal3d`,
`position`, `dotproduct`, `multiply`, `add`, `sin`, `power`, `mix`); the
rest are established by a per-node generator dry-run test in the hostless
suite, which is the standing gate for any future whitelist extension. A
node that fails its dry-run test is removed from the tuple, and any
template depending on it is dropped, before merge — this fired once
already: `checker` (review m2) failed the gate because this MaterialX build
ships the node as `checkerboard`, so the node and its template were
dropped.

Templates (`noise`, `marble_veins`) expand server-side into
form-3 specs before validation; unchanged.

### D5 — Editability = gen-reflection mapping + fan-out writes + scene-graph
surfacing

Three review findings (M1, M2, m4) replace revision 1's "rides existing
rails" claim:

- **Name mapping (M1).** The generator names uniforms after interior node
  inputs and may shatter one interface input into N uniforms. Synthesis
  therefore records, from the dry-run reflection, a mapping
  `spec-param-name → {gen uniform names}` for every promoted input
  (template params always promoted; raw-graph node params promoted only
  with `expose: true`). The mapping is persisted with the material
  (renderer-side, keyed by material) — it is the editability contract.
- **Fan-out writes.** A logical edit (`scene_set colorA`) applies
  `apply_material_override` once per mapped gen uniform, then relies on the
  existing re-upload + `_material_version` bump. Partial application is not
  possible (all targets are entries in the same overrides dict).
- **Scene-graph surfacing (M2, new scoped work).** `scene_set` reaches only
  properties on scene-graph nodes, and fallback-loaded `.mtlx` materials
  currently surface none. `build_scene_graph` (via the resync path) gains
  injection of editable properties onto material nodes: the promoted
  logical inputs (from the persisted mapping) for graph materials, and
  `parameter_overrides` keys for constant-shader `.mtlx` materials
  (chrome/glass/jade fall back to `GraphFragment=None`). Property values
  round-trip from the current override state. The Qt dock's panel path
  (`iter_graph_uniforms`) is untouched.
- **Honest errors (m4).** A non-promoted constant is *absent* from the
  property set, so `scene_set` on it yields the existing
  "no property X on path" error — the spec scenario says exactly that, not
  "not editable".

`material_list`'s per-preset editable inputs come from the same gen
reflection, run per preset file and cached by file mtime (review m5) —
never from parsing `.mtlx` inputs directly, which reports names that are
not writable keys.

### D6 — Element-name salting, a naming contract, and preset dedup

Review M3: MaterialX element names are a global namespace across the
library and material tables; same-named documents alias, and stale library
entries shadow re-imports.

- **Salting.** Every synthesized document's surfacematerial, shader, and
  nodegraph element names are salted with the (unique) material prim name.
- **Naming contract.** Holder prim name == surfacematerial element name;
  binding target == holder prim path. This is what the loader's
  leaf-name-keyed binding resolution requires to be collision-free.
- **Preset dedup.** `scene_add_material({preset})` returns the existing
  `/Materials` holder when that preset was already added (one holder per
  preset per scene) — curated files have fixed element names, so a second
  holder could never resolve anyway. This *changes* revision 1's
  per-call-entry stance. Synthesized/template materials are never deduped
  (each call = fresh salted document).

`scene_bind_material` binds with explicit binding-rel targets
(`material:binding` set, not prepended/appended — review NOTE: explicit
list-op is what makes the session binding *replace* a file-authored one
under LIVRPS rather than merge). Validation: target path exists AND (is
`Material`-typed OR carries a `.mtlx` reference) — review M4's corrected
form. `scene_add_primitive` composition rules unchanged: `material=` name →
create (or dedup-reuse preset) + bind; path → bind existing, error if
absent; refuse alongside inline `color`/`roughness`/`metallic`.

### D7 — Save plan per `save_edits` branch (was review B3)

`save_edits` behaves differently by stage origin, and the save plan must
too:

- **Anonymous root (`scene_create` scenes)** — `stage.Export()` flattens
  and strips reference arcs, so after export the save path post-processes
  the exported `Sdf.Layer`: for each `/Materials/<Name>` holder, re-author
  the `.mtlx` reference (relative to the saved file), remove flatten
  residue under the holder, and copy the referenced `.mtlx` into a
  `materials/` subdirectory beside the stage. Result: a self-contained
  bundle in the demo-asset shape.
- **File-backed root** — the exported edit layer is an *overlay*; it keeps
  reference arcs, which are re-anchored relative to the export target, and
  session `.mtlx` files are copied the same way. "Reload" for this branch
  is defined as: re-open the original scene and re-attach the exported
  layer as the edit layer — the overlay is not claimed to be a standalone
  scene.
- **Curated-preset carve-out (review M5, widened at codex review #11).**
  ALL curated presets — not only the texture-bearing ones — keep absolute
  references into `assets/` and are never copied. The original motive was
  texture safety (copying a doc without walking its filename-typed inputs
  silently drops textures to the fallback slot); the widened rule is
  deliberate: curated presets are renderer configuration in the same trust
  domain as the assets directory, and a saved scene referencing them is
  exactly as repository-dependent as any scene referencing renderer HDRs.
  The saved scene is self-contained *except* for curated-preset assets.
  Synthesized documents (textureless by the v1 whitelist) are always
  copied.

### D8 — Participation is binding-driven; `scene_add_material` says so

Review B2: `scene.materials` ingestion happens only through
`_resolve_material_binding` during geometry traversal — an unbound material
is not loaded, generated, or listed in the material table. Rather than
teaching the loader to ingest unbound `/Materials` prims (larger loader
change, no rendering benefit), the design accepts binding-driven
participation: `scene_add_material` creates the prim and session file,
validates the document via the dry-run, and returns
`{path, live: false}` — honest that the material renders only once bound.
`scene_bind_material` (or `scene_add_primitive(material=…)`) is the moment
of participation. Scene-graph presence of the unbound holder is whatever
stage enumeration already provides; its editable properties appear once the
material is loaded (bound).

### D9 — First binds of graph materials are jobs, not fast calls

Review M6, reconciled with D8 at codex review #12: a nodegraph enters
`_graph_set_signature()` only when the material becomes live — participation
is binding-driven (D8), so an *unbound* `scene_add_material` performs no
compile and stays fast; it is the **first bind** (or an add-plus-bind via
`scene_add_primitive(material=…)`) that changes the signature and rebuilds
the megakernel pipeline on the render thread — a full compile. Structural
tools already degrade to pollable jobs past a grace period; first binds of
graph materials will essentially always take that path, and the tool docs
say so. GPU tests budget one compile per distinct graph
set change and respect the one-guarded-Metal-process rule. Batching
(N materials, one resync) is deliberately deferred to a follow-up.

## Risks / Trade-offs

- [Loader session-layer scan misses an intake corner (sublayer-of-session,
  nested refs)] → Scope is exactly "prim specs of the session layer with
  absolute asset paths"; hostless intake tests cover authored-in-session
  reference discovery, and root-layer scan is untouched.
- [Gen dry-run cost per add] → Measured in seconds for marble-sized graphs;
  it runs on the MCP thread *before* the structural closure is posted, so
  the render thread pays only the real rebuild (D9).
- [Fan-out mapping drifts from a regenerated graph (e.g. after reload)] →
  The mapping is regenerated by the same dry-run wherever the document is
  re-ingested; it is derived state, never hand-authored.
- [Whitelisted node fails in some composition the per-node test misses] →
  Same degradation as any broken `.mtlx` today (fallback material +
  renderer log); the add already returned honestly. Accepted v1 risk.
- [Session `.mtlx` files leak on abnormal exit] → Tempdir hygiene bounds
  the leak; no cleanup daemon v1.
- [Save post-process drifts from demo-asset conventions] → Round-trip tests
  per branch: anonymous → self-contained bundle reloads standalone;
  file-backed → overlay re-attach reproduces the render.
- [Preset catalog / docs drift] → `material_list` is generated from the
  directory listing + cached gen reflection at call time.

## Migration Plan

Purely additive: three new tools, one new optional argument, one new
module, two scoped extensions (loader session-layer intake, scene-graph
material-property injection). No persisted-format or settings changes.
Rollback = removing the tools; saved anonymous-branch scenes are plain
USD + `.mtlx` bundles loadable by any current build.

## Open Questions

None blocking. Resolved this revision: validation gate is the generator
dry-run (D4); promotion is explicit `expose: true` for raw graphs, always
for template params (D5); preset dedup replaces per-call entries (D6).
