## 1. Renderer: resync refresh + lights/camera

- [x] 1.1 Add `prim_path: str = ""` to `LightDir` and `LightSphere` in `scene.py`; populate it in `usd_loader.py` light extraction (`_extract_distant_light`, `_extract_sphere_light`, area-light paths) from the source prim path
- [x] 1.2 In `_resync_geometry_from_stage`: rebuild `_scene_graph` via `build_scene_graph(stage, scene)` (with synthesized default-light injection) and bump `_scene_graph_version`
- [x] 1.3 In `_resync_geometry_from_stage`: re-read distant/sphere lights (reuse `_reextract_animated_lights`) and the camera override from the stage
- [x] 1.4 Preserve light `enabled` across resync by snapshotting `{prim_path: enabled}` before re-read and re-applying after (mirror the instance pattern)
- [x] 1.5 Headless test: `add_model`/`remove_node` rebuild the scene graph and bump `_scene_graph_version`; removing a light prim drops it; a light toggled off stays off after an unrelated add; default lights survive a resync

## 2. Pure wiring helpers (testable seams)

- [x] 2.1 `add_parent_for_node(node) -> str`: selected node path if group-like (`Xform`/`Scope`/has children & no renderer_ref), else `/World`
- [x] 2.2 `is_deletable(node) -> bool`: False for pseudo-root and `/Skinny/*` synthesized nodes, else True
- [x] 2.3 `trs_to_matrix(translate, rotate_deg, scale) -> np.ndarray`: compose via `scene_graph.compose_trs_matrix`
- [x] 2.4 Verify panel TRS fields are local-space (`_decompose_matrix`/`_add_transform_props`); document/convert if world-space
- [x] 2.5 Unit tests for 2.1–2.3 (plain functions, no GUI)

## 3. Qt scene-graph panel controls

- [x] 3.1 Toolbar/header in `SceneGraphDock`: "Add model…" button → `category="model"` picker → `add_model(path, add_parent_for_node(selected))`; report invalid pick
- [x] 3.2 "Save edits…" button → `save_edits()`; disabled when `_usd_edit_layer is None`; surface result
- [x] 3.3 Delete: tree context-menu item + `Delete` shortcut → `remove_node`, gated by `is_deletable`
- [x] 3.4 Route per-node TRS commit through `set_transform(prim_path, trs_to_matrix(...))`
- [x] 3.5 Clear/guard the properties panel when the selected node is gone after a refresh

## 4. Panel scene-graph controls

- [x] 4.1 "Add model" button → file picker → `add_model(path, add_parent_for_node(selected))`; report invalid pick
- [x] 4.2 "Save edits" button → `save_edits()`; disabled when no edit layer; surface result
- [x] 4.3 "Delete" button in node properties → `remove_node`, gated by `is_deletable`
- [x] 4.4 Route per-node TRS commit through `set_transform`
- [x] 4.5 Guard the properties column when the selected node is gone after a refresh

## 5. Verification

- [x] 5.1 `ruff check src/` and `pytest` (renderer headless + helper unit tests via the 3.13 venv with `VULKAN_SDK`/`DYLD_LIBRARY_PATH`); fix failures
- [ ] 5.2 Manual smoke via `/run` in Qt: add a model, move it, delete a node, delete a light, save edits, reload to confirm persistence
- [ ] 5.3 Manual smoke via `/run` in Panel: same flow
- [x] 5.4 Update `Architecture.md` / `CHANGELOG.md` with the editing controls
