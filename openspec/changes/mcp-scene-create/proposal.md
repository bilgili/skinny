## Why

Every structural MCP tool (`scene_add_model`, `scene_add_primitive`,
`scene_add_light`, `scene_remove`, `scene_save`) refuses with *"no editable USD
stage is loaded"* unless a scene was already loaded from disk (`--usd`, or File →
Open). A client that wants to author a scene from scratch has no way to obtain an
editable stage over MCP — it must first hand the renderer a file out-of-band.
`scene_create` closes that gap: it synthesizes a bare editable stage in-process,
so a client can start adding geometry and lights immediately and `scene_save`
the result.

## What Changes

- Add MCP tool **`scene_create(force: bool = False)`** to the in-process scene
  server, wrapped in the existing `_structural` sync-or-job path.
- Add renderer method **`create_empty_scene()`** that synthesizes an in-memory
  `Usd.Stage` (single `/World` Xform as `defaultPrim`, `upAxis = Y`,
  `metersPerUnit = 1`), makes it the active USD scene, attaches the
  non-destructive edit sublayer, and rebuilds the derived scene graph with the
  synthetic default light/dome/camera re-injected — the same finalize path a
  structural edit already runs.
- `scene_create` **refuses** when an editable stage is already present (returns
  an error) so an accidental call cannot silently discard unsaved structural
  edits; `force = true` overrides and replaces the stage.
- The tool returns **version counters only** (no `path`) — the synthesized stage
  is anonymous until `scene_save` writes it to an allowed root.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `mcp-scene-control`: adds a requirement for creating an editable stage without
  a pre-loaded file, and for the refuse-unless-`force` guard that protects
  unsaved edits.

## Impact

- **Code**: `src/skinny/renderer.py` (new `create_empty_scene()`, reusing
  `_attach_edit_layer` + `_resync_geometry_from_stage`); `src/skinny/mcp_server.py`
  (new `scene_create` tool + FastMCP registration, `__signature__` set
  explicitly for the `force` param).
- **No new dependency, no asset file, no shader change, no path-allowlist
  interaction** (nothing is written to disk).
- **Tests**: hostless `create_empty_scene` unit (asserts `_usd_stage` +
  `_usd_edit_layer` set and the graph contains `/World`); MCP tool-schema test
  gains a `scene_create` entry; a refuse-unless-`force` behavior test.
- **Docs**: README MCP tool list gains `scene_create`.
