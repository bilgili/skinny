# Design: session-edit-layer

## Root cause (verified)

USD layer strength: within a root layer stack, the root layer's own opinions are
**stronger** than any of its sublayers. `_attach_edit_layer` inserts the edit
layer at `root.subLayerPaths[0]` — "strongest sublayer", but still weaker than
root. Consequences, both reproduced against `assets/first_mcp_scene.usda`:

- `ClearXformOpOrder()` on the weak edit target does not remove the root's
  `xformOpOrder` opinion; the composed order still lists `xformOp:transform`, so
  `AddTransformOp()` raises `'xformOp:transform' already exists in xformOpOrder`.
- Authoring a new `xformOp:transform` value into the weak sublayer is ignored in
  composition (root value wins) — the move silently does nothing.

The session layer (`stage.GetSessionLayer()`) sits **above** the entire root
layer stack in strength and is the USD-canonical home for non-destructive,
in-memory overrides. Verified: authoring the override into the session layer
makes the composed value follow the override (`WORKS`), and reusing the existing
op via `op.Set()` avoids the duplicate-op throw.

## Decision 1 — session layer as the edit target

`_attach_edit_layer`:
- `edit_layer = stage.GetSessionLayer()` (already exists per stage; anonymous,
  never written to disk implicitly).
- `stage.SetEditTarget(Usd.EditTarget(edit_layer))`.
- No `subLayerPaths` mutation.
- `self._usd_edit_layer = edit_layer`; `_edit_layer_default_path` unchanged
  (still derived from the root file's realPath for the default `.edits.usda`).

Non-destructive guarantee is preserved and strengthened: the session layer is
in-memory, is not part of the root layer stack that `Export`/save writes, and the
original file is never touched by an edit. `save_edits(path)` still explicitly
exports the overlay when the user asks.

Idempotence: the session layer is a stable per-stage object; the guard stays
`if self._usd_edit_layer is not None: return`. Re-entry is a no-op.

## Decision 2 — reuse the existing transform op

`_author_local_transform(xformable, matrix)`:

```python
ops = xformable.GetOrderedXformOps()
if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
    ops[0].Set(gm)          # value-over in the session layer; wins over root
    return
xformable.ClearXformOpOrder()
xformable.AddTransformOp().Set(gm)
```

skinny always authors transforms as a single `xformOp:transform` (see
`add_light`/`add_model`/`add_primitive`), so the single-op branch is the common
path for every prim this code has authored, and also for any file prim that uses
the same convention (the pbrt/USD importer emits exactly one transform op). The
clear+add fallback covers the fresh-prim (no ops) and unusual multi-op cases;
with the session-layer edit target it no longer throws because clearing the
session-layer order and re-adding composes cleanly above root.

Rationale for reusing rather than always clear+add: `op.Set()` is the minimal,
idempotent, in-place value update — it authors only the value, never touches the
order attribute, and is immune to any stronger/weaker order opinion elsewhere in
the layer stack.

## Decision 3 — save/reload contract

There is no auto-reload consumer of a saved `.edits.usda` in the codebase;
`save_edits` is a one-way Export. So the format of the saved overlay is
observationally identical (an over-heavy USD layer), and switching the source of
the export from a root sublayer to the session layer changes nothing a consumer
can see. For the anonymous synthesized-root case (`scene_create`), `save_edits`
already exports the **composed stage** (to preserve `/World` + stage metadata on
the anonymous root); that branch is unchanged and now also folds in session-layer
overrides via composition.

## Risks

- **Session layer is now the DEFAULT edit target.** Any renderer write that
  authors to the stage without an explicit `EditContext` lands on the session
  layer and now composes **above** the root/file layer. Two such sites exist —
  the env-mirror dome writes at `renderer.py:~7521` (`CreateIntensityAttr`) and
  `~8321` (`CreateTextureFileAttr`/`Format`). Under the old weak sublayer these
  were silently ignored when the file authored a dome value; under the session
  layer they win. This is benign (a latent fix — nothing re-reads those composed
  values on that path, and they already folded into `save_edits`), but it is a
  real strength change. Verified none of these writes relied on being overridden
  by the file. The synthetic `/Skinny/DefaultLight`/`DefaultDome` writes live on
  a separate `_default_light_stage`, so they do not interact.
- **`op.Set()` reuse is optional for correctness.** Design review verified that
  clear+add on the session target already composes cleanly above root (session's
  empty `xformOpOrder` wins → no duplicate-op throw), for both single-`transform`
  and component-op (`translate`+`rotateXYZ`+`scale`) file prims. The reuse branch
  is kept as the smaller-diff in-place update, not as the thing that avoids the
  throw — session-layer strength is what avoids it.
- **Session layer used elsewhere.** No other subsystem sets the session layer as
  a persistent edit target; the only `SetEditTarget` is `_attach_edit_layer`.
  Mitigation: keep all editing routed through the edit target.
- **Instance-proxy descendants** (`/Model/Geo` under an instanceable prim) remain
  non-authorable — a fundamental USD constraint, identical under the old sublayer.
  `list_nodes` uses `TraverseAll()` which does not surface instance-proxy paths,
  so no MCP caller can hand such a path to `set_transform`/`remove_node`. Not a
  regression.
- **Saved loaded-scene edit layer is `over`-rooted with no `defaultPrim`.** The
  session-layer export of a loaded scene is a one-way over-heavy artifact:
  reopening it standalone yields an empty `Traverse()` (the added prim resolves
  via `GetPrimAtPath` but is not a traversal root). Round-trip tests MUST assert
  via `GetPrimAtPath`/composed values, never `Traverse()`. Same behavior as the
  old sublayer path.
- **`Export` of a session layer with references.** `add_model` authors a `def`
  + reference into the edit target. A session-layer `def` composes and exports
  correctly (verified for over/def specs). Regression test covers save→reopen.
- **GPU test drift.** `test_edit_layer_attached` asserts the sublayer topology;
  it is updated to assert the session-layer edit target. This is the intended
  behavior change, not a regression.
