# Proposal: session-edit-layer

## Why

The non-destructive edit layer is attached as the **strongest sublayer of the
root layer** (`_attach_edit_layer` inserts it into `root.subLayerPaths[0]`). In
USD a sublayer is **weaker than the root layer's own opinions**, so this layer
can never override an attribute authored in the root/file layer. Two failures
follow, both reproduced:

1. `set_transform` (and any `_author_local_transform` re-author) on a prim that
   already carries an authored `xformOp:transform` **throws**:
   `UsdGeomXformable::AddXformOp ... 'xformOp:transform' already exists in
   xformOpOrder`. `ClearXformOpOrder()` only clears the opinion in the current
   (weak) edit target; the stronger root op survives in the composed order, so
   `AddTransformOp()` sees a duplicate. This is what breaks `scene_set scale`
   (MCP) / a dock transform edit on any light or prim whose transform came from
   the loaded file.
2. Even without the throw, a value authored into the weak sublayer is **silently
   ignored** — the composed value stays at the root/file value. So transform
   overrides of file-authored prims never take effect.

Every scene loaded from disk has its prim transforms in the root/file layer, so
transform editing of authored prims is effectively broken. It appeared to work
only for prims added at runtime (whose ops live in the edit layer) and for prims
with no authored transform (where `AddTransformOp` adds a fresh op).

## What Changes

- `_attach_edit_layer` authors non-destructive edits into the stage's **session
  layer** (`stage.GetSessionLayer()`) — the canonical USD override layer that is
  **always stronger than the whole root layer stack** — instead of inserting a
  root sublayer. Overrides now win over file opinions; the original root layer
  and file on disk stay untouched (session layer is in-memory and never written
  except by an explicit `save_edits`/`scene_save`).
- `_author_local_transform` reuses an existing single `xformOp:transform` op via
  `op.Set(matrix)` instead of `ClearXformOpOrder()` + `AddTransformOp()`. With
  the session layer this authors a value-over that wins; it also removes the
  duplicate-op throw entirely. The clear+add path stays as the fallback for the
  no-op / multi-op cases.
- `save_edits` / `scene_save` export the session layer (or, for an anonymous
  synthesized root, the composed stage as today) — behavior is unchanged for the
  anonymous-root case and now captures session-layer overrides for loaded scenes.
- Spec `usd-scene-editing` requirement "Stage is the authoritative model with a
  non-destructive edit layer" is amended: the edit target is the session layer,
  not the strongest sublayer.

## Impact

- Affected specs: `usd-scene-editing` (MODIFIED requirement + new scenario).
- Affected code: `renderer.py` (`_attach_edit_layer`, `_author_local_transform`,
  `save_edits`, `_default_edit_path`), no MCP/tool signature changes.
- Affected tests: `test_scene_editing.py::TestEditLayer::test_edit_layer_attached`
  (assert session-layer edit target instead of sublayer) plus a new regression
  that a file-authored transform can be overridden. No behavior change for
  `add_model`/`remove_node`/`add_light`/`add_primitive` callers.
- Non-breaking for MCP clients: `scene_set` transforms that previously threw now
  succeed; no signature changes.
