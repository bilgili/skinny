## Why

Today a USD scene is baked once at load into a flat `Scene` and the live stage
is effectively read-only: `load_model_from_path` clears and replaces the whole
scene, there is no way to add, remove, or move a model while the app runs, and
no programmatic entry point for scripting or testing scene composition. Making
the loaded stage the authoritative model unlocks runtime scene-graph editing —
the foundation for an in-app scene tree (later) and live USD workflows — without
requiring NVIDIA Omniverse.

## What Changes

- The loaded `Usd.Stage` becomes the authoritative scene model for runtime
  mutation. The flat `Scene`/GPU buffers become a derived cache kept in sync by
  an explicit, dirty-typed resync (Architecture C — no `Tf` notice listener).
- A dedicated non-destructive edit sublayer (`<scene>.edits.usda`) is attached
  in-memory as the strongest sublayer and set as the stage edit target. The
  original file is never written unless `save_edits()` is called.
- New public `Renderer` API: `add_model()`, `remove_node()`, `set_transform()`,
  `save_edits()`, `list_nodes()`. `add_model` accepts USD references only.
- `MeshInstance` gains a `prim_path`; a `_prim_to_instances` index is added.
  **BREAKING** (internal API): `apply_instance_transform` / `apply_node_enabled`
  migrate from name-keyed to prim-path-keyed lookup.
- Every successful edit resets progressive accumulation.

## Capabilities

### New Capabilities
- `usd-scene-editing`: runtime mutation of the authoritative USD stage —
  add/remove/transform models via a Python API, authored to a non-destructive
  edit sublayer, with explicit dirty-tracked resync of the derived scene and GPU
  buffers, and optional persistence of edits.

### Modified Capabilities
<!-- None. apply_instance_transform / apply_node_enabled are not spec-governed
     capabilities; their name→prim-path migration is an implementation detail
     captured in design.md and tasks.md. -->

## Impact

- **Code**: `src/skinny/renderer.py` (stage ownership, edit-layer setup, new API,
  dirty-tracked resync, `_prim_to_instances`, migration of
  `apply_instance_transform`/`apply_node_enabled`); `src/skinny/scene.py`
  (`MeshInstance.prim_path`); `src/skinny/usd_loader.py` (subtree bake reuse for
  added references).
- **Dependencies**: USD (`pxr`) edit-layer / composition APIs — already a runtime
  dependency. No new packages.
- **Tests**: new headless tests under `tests/` reusing `tests/test_headless.py`
  patterns (Python 3.13 venv, `VULKAN_SDK`/`DYLD_LIBRARY_PATH`).
- **Out of scope** (later changes): in-app scene-graph tree UI; live USD
  layer / file-watch reload; reparent; OBJ adds; `Tf`-notice auto-sync;
  remote/network control.
