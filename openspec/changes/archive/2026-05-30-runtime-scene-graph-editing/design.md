## Context

`renderer.py` loads a USD scene once into a flat `Scene` (a list of
`MeshInstance`, plus materials/lights/environment) and uploads concatenated
vertex/index/BVH buffers with a TLAS over the instances. The live `Usd.Stage`
(`_usd_stage`) is retained only to re-evaluate animation at the current time
code; the file is otherwise read-only. `load_model_from_path` performs a
clear-and-replace (`_clear_model_state`). There is no API to add/remove/move a
single model at runtime, and existing instance edits
(`apply_instance_transform`, `apply_node_enabled`) key off the instance `name`.

This change (Spec 1 of a larger effort) makes the stage the authoritative model
and adds a Python editing API. A later change adds an in-app scene-graph tree
UI; another adds live file-watch reload. Reparent, OBJ adds, `Tf`-notice
auto-sync, and remote control are explicitly deferred.

The renderer is a progressive path tracer: `_current_state_hash()` gates
accumulation, and any scene change must reset `accum_frame` to 0. Geometry
upload already supports N instances; the gap is mutation after load, not
multi-mesh rendering.

## Goals / Non-Goals

**Goals:**
- Make `_usd_stage` the authoritative scene model; the flat `Scene` + GPU
  buffers become a derived cache refreshed by an explicit, dirty-typed resync.
- Author all runtime edits to a dedicated non-destructive edit sublayer; never
  modify the original file unless `save_edits()` is called.
- Provide `add_model`, `remove_node`, `set_transform`, `save_edits`,
  `list_nodes` on `Renderer`, headless-testable.
- Identify instances by USD prim path; migrate the two existing instance-edit
  methods to prim-path keying.
- Reset progressive accumulation on every successful edit.

**Non-Goals:**
- No in-app scene-graph tree UI (later change).
- No live USD layer / file-watch reload, no `Tf`-notice listener (later change).
- No reparent, no OBJ `add_model`, no remote/network control.
- No change to initial-load behavior or to the OBJ/analytic-head paths.

## Decisions

### D1: Stage is authoritative, resync is explicit (Architecture C)
Edits mutate the stage, mark a typed dirty flag, then drive a targeted resync.
We do **not** install a `Usd.Notice.ObjectsChanged` listener.
- *Why*: explicit resync is deterministic and far less plumbing than mapping
  USD notices to incremental GPU buffer updates; it covers the in-app API and
  Python-API drivers. Notice-driven auto-sync (needed only for 3rd-party live
  edits) is deferred to the file-watch change.
- *Alternatives*: (A) full notice-driven incremental sync — most "live" but the
  largest refactor; (B) flat `Scene` stays primary, USD authored on the side —
  smaller but two sources of truth drift and reparent has no hierarchy to act
  on. Rejected for this foundation.

### D2: Edits land in a dedicated in-memory edit sublayer
On load, create/open `<scene>.edits.usda` and insert it as the **strongest**
sublayer of the stage root layer in memory, then set the stage edit target to
it. The root layer is never saved. `save_edits(path=None)` saves only the edit
layer (default sibling path).
- *Why*: non-destructive (original file untouched), edits are isolated and
  persistable, survives a future reload by re-inserting the sublayer.
- *Alternatives*: session layer (cannot persist without flatten); write-through
  to the root layer (destructive). Rejected.

### D3: Dirty-typed resync — transform-only vs geometry
Edits classify their dirty effect:
- *transform-only* (`set_transform`) → author the prim's local
  `xformOp:transform`, recompute the affected instances' world transforms from
  the stage (the same `_world_transform` + up-axis path used at load and during
  animation), then reuse the existing `_reupload_instance_transforms()` fast
  path. No re-bake, no buffer re-concatenation.
- *geometry* (`add_model`, `remove_node`) → re-read the authoritative stage via
  `load_scene_from_stage`, swap in the resulting instances/materials, and run
  the existing `_gen_scene_materials()` + `_upload_usd_scene()` upload path
  (same per-call work as `set_usd_scene`). Meshes are cached by content hash, so
  unchanged prims are not re-baked — only genuinely new geometry bakes. Runtime
  `enabled` flags (which are not authored to the stage) are carried across the
  re-read by prim path; deactivated prims simply drop out (USD does not traverse
  inactive prims).
- *Why*: re-reading the stage with the content-hash cache costs about the same
  as hand-baking a subtree but reuses the proven load path, keeping the edit API
  small and correct under stage-as-truth. Transform edits stay on the cheap
  fast path.

### D4: Prim-path identity
Add `prim_path: str` to `MeshInstance` (populated from the source prim;
mesh prims already carry the full path in `MeshInstance.name`) and maintain
`_prim_to_instances: dict[str, list[int]]` (one prim may expand to several
instances), rebuilt on every geometry upload. The existing
`apply_instance_transform(instance_index, …)` and
`apply_node_enabled(kind, index, …)` are currently **index-keyed** (not
name-keyed); they migrate to a `prim_path` signature. They resolve
`prim_path → RendererRef(kind, index)` via the scene-graph node
(`find_node_by_path`), preserving the existing per-kind dispatch; instances also
resolve through `_prim_to_instances` so the methods work headless (no scene
graph). All in-repo callers update: the gizmo (`renderer.py`), the ImGui panel
(`ui/panel/windows.py`), and the Qt scene-graph panel
(`ui/qt/windows/scene_graph.py`) already have `node.path` / the instance in
scope. This is an internal-API break, acceptable as no external consumers exist.
- *Why*: removal and transform must target stage prims unambiguously; integer
  indices shift when instances are added/removed.

### D5: add_model accepts USD references only
`add_model` defines an `Xform` prim in the edit layer under `parent_prim_path`
(default `/World`), calls `AddReference(usd_path)`, and authors an
`xformOp:transform`. OBJ is rejected with a clear error (deferred).
- *Why*: references are USD-native composition, smallest and idiomatic; OBJ
  would require synthesizing in-memory `UsdGeom.Mesh` prims, deferred.

### D6: Edits are synchronous
`add_model`, `remove_node`, and `set_transform` run synchronously. The geometry
edits re-read the stage (D3); the content-hash mesh cache keeps the re-read
cheap, so no background thread is needed for correctness or for the headless
tests. The interactive front-ends may later wrap `add_model` in a thread if a
very large reference proves slow — out of scope here.
- *Why*: synchronous edits are deterministic and trivially testable headless;
  the cache removes the latency that motivated the original async load path.

### D7: Validate-then-author with rollback
`add_model` validates that `usd_path` exists and opens as a USD layer **before**
mutating the stage; on failure raises `ValueError` with zero mutation. After
authoring the prim, if the bake fails, the authored prim define is removed so no
half-authored prim remains. `remove_node` / `set_transform` raise `ValueError`
for an unknown prim path.
- *Why*: edits must be atomic from the caller's view; a failed edit leaves the
  scene and stage unchanged.

### D8: Accumulation reset
Accumulation reset is indirect: `_current_state_hash()` is sampled each frame
and `accum_frame` resets to 0 when it changes. `_material_version` feeds that
hash, so every successful edit bumps `self._material_version += 1` (the same
mechanism `apply_instance_transform` already uses). No direct `accum_frame = 0`
poke from the edit methods.

### D9: Headless editing entry
The editing API operates on `self._usd_stage`, but `set_usd_scene(scene)` (the
headless entry) does not own a stage. `set_usd_scene` gains an optional
`stage=None`; when supplied it stores `self._usd_stage = stage` and attaches the
edit layer, so a headless test can `Usd.Stage.Open` a scene, build it via
`load_scene_from_stage`, call `set_usd_scene(scene, stage=stage)`, then exercise
`add_model` / `remove_node` / `set_transform`. The interactive load path
(`_load_usd_model`) attaches the edit layer where it already sets
`self._usd_stage`.

## Risks / Trade-offs

- **Stale `_prim_to_instances` after a geometry edit** → rebuild the index as
  part of every geometry resync; cover with a test that adds then removes and
  asserts the index matches the live instance list.
- **Edit-layer ordering / edit target leaks** → set the edit target only while
  authoring an edit and assert the strongest-sublayer invariant in a test;
  restore the prior edit target in a `finally`.
- **Full re-concatenation on each geometry edit is O(total instances)** →
  acceptable for skinny's head-sized scenes; meshes are cached so no re-bake.
  Incremental buffer updates are a future optimization, not needed now.
- **Reference target moves/breaks on disk** → `add_model` stores the path as
  authored; a missing reference renders nothing for that prim (USD default) and
  is logged. We validate readability at add time, not continuously.
- **Internal-API break (name→prim-path keying)** → update all in-repo callers in
  the same change; lint + tests guard regressions.

## Migration Plan

1. Add `prim_path` to `MeshInstance` (default derived from the source prim);
   populate it in the loader/bake path.
2. Build `_prim_to_instances` whenever the instance list is (re)built.
3. Migrate `apply_instance_transform` / `apply_node_enabled` to prim-path keys
   and update callers.
4. Add edit-layer setup at USD load; add the editing API; wire dirty-typed
   resync.
5. No persisted-settings or on-disk format changes; rollback is reverting the
   branch. The edit sublayer is only written on explicit `save_edits()`.

## Open Questions

- None blocking. Default `parent_prim_path` is `/World`; if a loaded stage has
  no `/World`, `add_model` creates it as an `Xform` (decided: create on demand).
