## Why

Spec 1 added a runtime scene-graph editing API (`add_model`, `remove_node`,
`set_transform`, `save_edits`, `list_nodes`) but nothing in the app calls it: the
scene-graph panels (Qt tree, Panel dropdown) only display and toggle nodes. They
also do not refresh after a non-load mutation, because `_resync_geometry_from_stage`
never rebuilds the scene graph. This change makes the panels a live editor and
fixes that gap, so users can add, delete, move, and persist models without a
script.

## What Changes

- **Refresh-gap fix**: `_resync_geometry_from_stage` rebuilds `_scene_graph` and
  bumps `_scene_graph_version`, so both front-ends repaint after add/remove.
- **Resync covers lights + camera**: geometry resync re-reads distant/sphere
  lights and the camera override from the stage so deleting a light/camera prim
  drops it. `LightDir`/`LightSphere` gain a `prim_path`; runtime `enabled` flags
  are carried across resync by prim path (mirroring `MeshInstance`). Synthesized
  `/Skinny/*` default lights stay renderer-owned; environment stays session state.
  **BREAKING** (internal): `LightDir`/`LightSphere` constructors gain `prim_path`.
- **Add model**: an "Add model…" toolbar control in both front-ends (reusing the
  `category="model"` file picker + last-dir memory) → `add_model(path,
  parent_prim_path=<selected group or "/World">)`.
- **Delete node**: Qt context-menu + `Del`, Panel delete button → `remove_node`;
  guarded against the pseudo-root and synthesized `/Skinny/*` nodes.
- **Save edits**: a "Save edits…" control → `save_edits(path)`; disabled with no
  edit layer.
- **Transform edits persist**: per-node TRS editing routes through
  `set_transform` (authored to the edit layer) instead of
  `apply_instance_transform` (runtime-only).

## Capabilities

### New Capabilities
- `usd-scene-editing-ui`: in-app controls that drive the scene-graph editing API
  from both front-ends — add a model, delete a node, save edits, and edit
  transforms that persist — with consistent behavior across the Qt and Panel
  scene-graph panels.

### Modified Capabilities
- `usd-scene-editing`: geometry resync now refreshes the derived scene graph
  (rebuild + version bump) and re-reads lights/camera so `remove_node` works for
  any prim; lights gain prim-path identity for enabled-flag preservation.

## Impact

- **Code**: `src/skinny/renderer.py` (`_resync_geometry_from_stage` rebuilds
  scene graph + re-reads lights/camera; light enabled preservation),
  `src/skinny/scene.py` (`LightDir`/`LightSphere` `prim_path`),
  `src/skinny/usd_loader.py` (populate light `prim_path` in extraction),
  `src/skinny/ui/qt/windows/scene_graph.py`, `src/skinny/ui/panel/windows.py`,
  `src/skinny/ui/build_app_ui.py` (toolbar/controls + wiring helpers).
- **Dependencies**: none new. Reuses the existing file-picker + settings
  (`get_last_dir`/`record_last_dir`) and the Spec 1 editing API.
- **Tests**: pure-function wiring-seam tests (selected-node → add parent;
  node → deletable?; TRS triple → matrix) + a headless renderer test that
  add/remove rebuilds the scene graph and bumps the version, and that removing a
  light drops it. GUI itself smoke-tested manually in both front-ends.
- **Out of scope** (later changes): reparent / drag-move, OBJ adds, file-watch
  reload (Spec 3), undo/redo.
