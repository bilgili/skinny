# Tasks: mcp-material-authoring

## 1. Synthesis module (GPU-free)

- [x] 1.1 Create `src/skinny/mtlx_synthesis.py`: spec-form dispatch
      (`preset` | `template` | `model`), mixed-form/unknown-form rejection,
      `graph` only with `standard_surface`
- [x] 1.2 Preset catalog: directory scan of
      `assets/Usd-Mtlx-Example/materials/`, `standard_surface_` prefix
      strip, dict-lookup resolution (client string never joined to a
      path), unknown-preset error listing names
- [x] 1.3 Parameter validation: reuse `_coerce`/`_check_bounds` discipline +
      `_MATERIAL_FLOAT_RANGES`; finite-check unknown numerics
- [x] 1.4 Node-type whitelist (data tuple incl. `checker`) + graph
      validation: node types, dangling-connection detection, error message
      with supported set
- [x] 1.5 standard_surface document builder via MaterialX Python API: flat
      params, optional nodegraph, element-name salting with the material
      prim name (holder name == surfacematerial name contract),
      `expose: true` promotion to interface inputs
- [x] 1.6 Generator dry-run gate: `MaterialLibrary.generate` +
      `generate_for_compute` on the synthesized document; rejection on any
      bailout; derive the logical-input → gen-uniform-name mapping from
      reflection (shattered inputs map to all their uniforms)
- [x] 1.7 Templates `noise`, `checker`, `marble_veins`: param schemas with
      bounds, expansion to standard_surface graph specs, all declared
      params promoted; drop any template whose node fails the whitelist
      gen test
- [x] 1.8 Session `.mtlx` file lifecycle: server-owned tempdir session dir
      (not constrained by allowed roots), one file per material named after
      the prim, flush-before-resync ordering, delete-on-rollback hook
- [x] 1.9 Hostless tests: form dispatch, catalog + path-shaped-name
      refusal, whitelist refusal, connection validation, template
      expansion + bounds, salting/naming contract, mapping derivation
      (incl. shattered input), **per-node gen dry-run over every
      whitelisted type** (standing gate), preset editable-input reflection
      with mtime cache

## 2. Loader + scene-graph extensions

- [x] 2.1 `usd_loader.py`: extend `_collect_mtlx_asset_paths` and
      `_prim_has_mtlx_reference` to scan session-layer prim specs
      (absolute asset paths); root-layer behavior byte-unchanged
- [x] 2.2 Hostless intake tests: session-authored reference discovered and
      loaded once bound; unbound holder not loaded (binding-driven
      participation); root-layer scenes (demo file) unchanged
- [x] 2.3 `scene_graph.py`: inject editable properties on live material
      nodes from the persisted mapping (graph materials) and
      `parameter_overrides` keys (constant-shader `.mtlx` materials);
      values round-trip current override state
- [x] 2.4 Fan-out write path: `scene_set` on a logical input applies
      `apply_material_override` per mapped gen uniform, single material
      version bump; absent (non-promoted) names give the existing
      no-such-property error
- [x] 2.5 Hostless scene-graph tests: property injection, fan-out
      multiplicity, no-such-property on unexposed constants

## 3. Renderer authoring (edit layer)

- [x] 3.1 `Renderer.add_material`: typed `UsdShade.Material.Define` holder
      under `/Materials` (+scope auto-create) carrying the `.mtlx`
      reference (absolute path) or inline preview shader, in session edit
      layer; rollback removes prims incl. auto-created scope AND the
      session file; resync; persist the mapping keyed by material
- [x] 3.2 `Renderer.bind_material`: explicit binding-rel targets (set, not
      prepend) in session edit layer, resync; validation: path exists AND
      (Material-typed OR carries `.mtlx` reference); geometry prim
      bindable check
- [x] 3.3 `save_edits` branches: anonymous-root — post-process exported
      flattened layer (re-author `/Materials` references relative to the
      saved file, strip flatten residue, copy synthesized docs to
      `materials/`); file-backed — re-anchor overlay references + copy
      docs; all curated presets keep absolute assets
      references
- [x] 3.4 Hostless USD-semantics tests (pxr direct, no renderer import):
      session explicit-target binding overrides file-authored binding
      (and a prepend would merge — assert explicit); anonymous-branch
      export post-process yields self-contained bundle; overlay re-anchor;
      rollback leaves stage + session dir clean

## 4. MCP tool surface

- [x] 4.1 `material_list` tool: catalog + model schemas + whitelist +
      template schemas; per-preset editable inputs via gen reflection
      cached by file mtime
- [x] 4.2 `scene_add_material` tool: validate + dry-run on MCP thread
      BEFORE posting the structural closure; renderer `add_material` on
      render thread; result carries path + `live: false` + version
      counters; preset dedup returns existing holder
- [x] 4.3 `scene_bind_material` tool: path checks, renderer
      `bind_material`, version counters
- [x] 4.4 `scene_add_primitive` `material=` argument: name →
      create/dedup + bind, path → bind-existing (error if absent), refuse
      alongside color/roughness/metallic
- [x] 4.5 Tool docs: first binds of graph materials degrade to pollable jobs
      (pipeline rebuild) — document in docstrings/spec text
- [x] 4.6 Hostless tool tests: arg validation, error texts, spec-error
      leaves versions unchanged, dedup, ambiguous-composition refusal
      (mock renderer marshalling per existing mcp_server test pattern)

## 5. GPU validation (Metal, guarded, one process at a time, one compile per graph-set change)

- [x] 5.1 Round-trip test: fresh scene → add marble preset + noise template
      + raw fractal3d graph → bind to spheres → render: bound prims produce
      non-fallback, mutually distinct pixels; two same-template materials
      render independently (salting)
- [x] 5.2 Edit test: `scene_set` on promoted template input changes
      accumulation (fan-out observed); non-promoted constant returns
      no-such-property error
- [x] 5.3 Save/reload tests per branch: anonymous-root bundle reloads
      standalone and matches pre-save (metrics via
      `metrics.compute_metrics`); file-backed overlay re-attach matches;
      `wood_tiled` textures survive save
- [x] 5.4 `scene_add_primitive(material=…)` one-call path renders bound
      material; job-degradation path exercised via `scene_job_status`

## 6. Docs + closure

- [x] 6.1 Update MCP tool documentation (module docstring in
      `mcp_server.py` + any MCP section in docs/) with the three tools,
      the `material` argument, not-live-until-bound, and job expectations
- [x] 6.2 Update `docs/PythonAPI.md` if `add_material`/`bind_material`
      become public Python symbols
- [x] 6.3 CHANGELOG entry
- [x] 6.4 `openspec validate mcp-material-authoring`; run
      `.venv/bin/ruff check src/` + hostless pytest suites
