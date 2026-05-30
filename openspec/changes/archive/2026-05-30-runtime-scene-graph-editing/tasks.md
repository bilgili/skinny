## 1. Prim-path identity

- [x] 1.1 Add `prim_path: str` to `MeshInstance` in `scene.py`; populate it in the USD load/bake path (`usd_loader.py` `bake_usd_prim` + `_area_light_to_instance`) from the source prim path
- [x] 1.2 Build and maintain `_prim_to_instances: dict[str, list[int]]` in `renderer.py`, rebuilt inside `_upload_usd_scene` (every geometry upload)
- [x] 1.3 Migrate `apply_instance_transform` and `apply_node_enabled` from index-keyed to prim-path-keyed; resolve via `_prim_to_instances` (instances) + `find_node_by_path`→`RendererRef` (lights/camera); update callers: gizmo (`renderer.py`), ImGui panel (`ui/panel/windows.py`), Qt panel (`ui/qt/windows/scene_graph.py`)
- [x] 1.4 Headless test: prim-path index matches the live instance list after a (re)build

## 2. Authoritative stage + edit layer

- [x] 2.1 Add `_attach_edit_layer()` (create/open `<scene>.edits.usda`, insert as strongest sublayer in memory, set as stage edit target); call it in `_load_usd_model` where `self._usd_stage` is set, and from `set_usd_scene(stage=...)`
- [x] 2.2 Add optional `stage=None` to `set_usd_scene`; when given, set `self._usd_stage` + attach edit layer (headless editing entry, D9)
- [x] 2.3 Guarantee no editing path writes the root layer; restore the prior edit target in a `finally` around each authoring op
- [x] 2.4 Headless test: edit layer is strongest sublayer + edit target; original file unchanged after an edit

## 3. Dirty-typed resync

- [x] 3.1 Transform-only resync: author `xformOp:transform`, recompute affected instances' world transforms from the stage (`_world_transform` + up-axis), reuse `_reupload_instance_transforms()` (no re-bake)
- [x] 3.2 Geometry resync: re-read the stage via `load_scene_from_stage`, carry runtime `enabled` flags by prim path, swap instances/materials, run `_gen_scene_materials()` + `_upload_usd_scene()` (content-hash cache avoids re-baking unchanged meshes)
- [x] 3.3 Ensure every successful edit resets progressive accumulation via `self._material_version += 1`

## 4. Editing API

- [x] 4.1 `add_model(usd_path, parent_prim_path="/World", name=None, transform=None) -> str`: validate path opens as a USD layer before authoring; define Xform + AddReference + author transform; create parent Xform if absent; geometry resync; return prim path; roll back the prim define if the resync/bake fails
- [x] 4.2 `remove_node(prim_path)`: author `active=false` in edit layer; drop instances under the path; geometry resync; `ValueError` on unknown path
- [x] 4.3 `set_transform(prim_path, matrix)`: author `xformOp:transform`; transform-only resync; `ValueError` on unknown path
- [x] 4.4 `save_edits(path=None)`: write the edit layer to disk (default sibling path); reopenable
- [x] 4.5 `list_nodes()`: return editable prims (prim path + active flag at minimum)
- [x] 4.6 Reject OBJ paths in `add_model` with a clear error (deferred capability)

## 5. Tests & verification

- [x] 5.1 Headless test: `add_model` grows instance count, prim exists, rendered pixels change, transform honored
- [x] 5.2 Headless test: `set_transform` moves instances and triggers the fast path (no re-bake)
- [x] 5.3 Headless test: `remove_node` deactivates the prim and drops its instances; survivors re-uploaded without re-bake
- [x] 5.4 Headless test: `save_edits` writes a reopenable layer containing the edits
- [x] 5.5 Headless test: invalid `add_model` path and unknown prim paths raise `ValueError` with no mutation
- [x] 5.6 Run `.venv/bin/ruff check src/` and `.venv/bin/pytest` (headless tests via the 3.13 venv with `VULKAN_SDK`/`DYLD_LIBRARY_PATH`); fix failures
- [x] 5.7 Update `Architecture.md` / `CHANGELOG.md` with the new editing API
