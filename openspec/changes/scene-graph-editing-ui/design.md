## Context

Spec 1 (`usd-scene-editing`, archived) made the loaded `Usd.Stage` authoritative
and added `Renderer.add_model/remove_node/set_transform/save_edits/list_nodes`,
authoring to an in-memory edit sublayer. The app does not yet call any of it.

Two scene-graph panels render `renderer.scene_graph` (a `SceneGraphNode` tree
built by `build_scene_graph`):
- **Qt** `SceneGraphDock` (`ui/qt/windows/scene_graph.py`): a `QTreeWidget` +
  per-node properties panel; refreshes on a 200 ms `_tick` that compares
  `id(graph)` + `_scene_graph_version`. No add/delete/save controls.
- **Panel** (`ui/panel/windows.py`): a flat `Select` dropdown + properties;
  refreshes on a 1000 ms poll comparing `id(scene_graph)`.

Both already toggle nodes (`apply_node_enabled`) and edit TRS
(`apply_instance_transform`, runtime-only). The file-picker pattern
(`build_app_ui._add_scene_loader` → `ui.file_picker(category="model")` with
`get_last_dir`/`record_last_dir`) is reused for "Add model".

`_resync_geometry_from_stage` (the add/remove resync) updates only the flat
`Scene` (instances + materials); it neither rebuilds `_scene_graph` nor re-reads
lights/camera. That is the central gap this change closes.

## Goals / Non-Goals

**Goals:**
- After `add_model`/`remove_node`, both panels repaint automatically.
- `remove_node` works for any prim, including lights and the camera.
- "Add model", "Delete node", and "Save edits" controls in both front-ends.
- Per-node TRS edits persist (authored to the edit layer, savable).
- Identical behavior across Qt and Panel (GUI-consistency).

**Non-Goals:**
- Reparent / drag-move, OBJ adds, file-watch reload (Spec 3), undo/redo.
- No new rendering features; no change to the Spec 1 authoring model.

## Decisions

### D1: Resync rebuilds the scene graph + bumps the version
`_resync_geometry_from_stage` calls `build_scene_graph(stage, scene)` (reusing
the synthesized default-light injection the load path uses), assigns a new
`_scene_graph`, and increments `_scene_graph_version`. Qt polls the version;
Panel polls `id(scene_graph)`. A new object + bumped counter triggers both.
- *Alternative*: emit a Qt signal / Panel event instead of polling. Rejected —
  the pollers already exist and are cheap; signals would diverge the two paths.

### D2: Resync re-reads lights + camera; lights gain prim-path identity
`_resync_geometry_from_stage` additionally re-extracts distant/sphere lights
(reuse `_reextract_animated_lights`) and the camera override from the stage, so a
deactivated light/camera prim drops out. `LightDir`/`LightSphere` gain
`prim_path`; before resync the renderer snapshots `{prim_path: enabled}` for
lights (as it already does for instances) and re-applies it after, so a runtime
enable-toggle survives a geometry edit. Synthesized `/Skinny/DefaultLight` /
`/Skinny/DefaultDome` are renderer-owned and re-injected by `build_scene_graph`,
not dropped. Environment stays session state (unchanged).
- *Alternative*: only refresh lights when the edited prim is a light. Rejected —
  always re-reading is simpler and matches the animation path's whole-list
  re-extract; the cost is a handful of prims.

### D3: Front-end wiring lives behind pure helpers
The GUI cannot be driven headless, so the testable logic is extracted into small
pure functions (shared in `build_app_ui` or a sibling), each front-end calling
them:
- `add_parent_for_node(node) -> str`: selected node's path if it is a
  group/Xform/Scope, else `/World`.
- `is_deletable(node) -> bool`: False for the pseudo-root and `/Skinny/*`
  synthesized nodes, else True.
- `trs_to_matrix(translate, rotate_deg, scale) -> np.ndarray`: compose the panel
  TRS triple into the 4×4 `set_transform` expects (reuse
  `scene_graph.compose_trs_matrix`).
- *Why*: keeps Qt/Panel widget code thin and identical, and puts the decision
  logic under unit test.

### D4: TRS editing routes through `set_transform`
Per-node TRS commit calls `renderer.set_transform(prim_path, trs_to_matrix(...))`
instead of `apply_instance_transform`. The panel TRS fields are local-space
(decomposed from the prim's local xform by `_decompose_matrix`/
`_add_transform_props`); `set_transform` authors local `xformOp:transform` and
recomposes world — consistent. The gizmo drag keeps using
`apply_instance_transform` (interactive, high-frequency, not persisted) to avoid
authoring a stage opinion on every mouse-move; a future change can add a
commit-on-release.
- *Integration check*: confirm during implementation that the TRS fields are
  local, not world. If world, convert before authoring (compose with the inverse
  parent-to-world). The headless seam test pins the expected matrix.

### D5: Controls and guards
- **Add model**: toolbar button → `category="model"` picker → `add_model(path,
  parent_prim_path=add_parent_for_node(selected))`. USD-only; an OBJ pick (or
  `add_model` `ValueError`) surfaces as a non-fatal toast/status, no mutation.
- **Delete**: Qt context-menu item + `Del` shortcut on the tree; Panel "Delete"
  button in properties. Enabled only when `is_deletable(node)`. → `remove_node`.
- **Save edits**: toolbar button → `save_edits()` (default sibling path) or a
  save dialog; disabled when `renderer._usd_edit_layer is None`. Result surfaced.

## Risks / Trade-offs

- **Re-reading lights resets a runtime enable-toggle** → snapshot/restore
  `enabled` by light `prim_path` across resync; cover with a test (toggle a
  light off, add a model, assert it is still off).
- **Synthesized default lights dropped by a light re-read** → they are re-injected
  by `build_scene_graph`; assert a default-light scene still shows them after an
  edit.
- **Stale selection after delete** → the Qt/Panel selection may point at a
  removed node; clear/var-guard the properties panel when the selected path is
  gone after a refresh.
- **TRS space mismatch** → D4 integration check + a seam test pin the matrix.
- **Adding under a non-group prim** → `add_parent_for_node` only returns a
  selected path for group-like prims, else `/World`; `add_model` already creates
  the parent on demand.

## Migration Plan

1. Renderer: extend `_resync_geometry_from_stage` (scene-graph rebuild + version
   bump + light/camera re-read + light enabled preservation); add `prim_path` to
   lights and populate it in `usd_loader` extraction.
2. Add the pure wiring helpers.
3. Wire Qt: toolbar (Add/Save) + context-menu/Del (Delete) + TRS→`set_transform`.
4. Wire Panel: buttons (Add/Save/Delete) + TRS→`set_transform`.
5. No persisted-settings or on-disk format change; rollback is reverting the
   branch.

## Open Questions

- None blocking. Save-edits UX (silent default-path save vs save dialog) defaults
  to a dialog when a path is needed and silent default otherwise.
