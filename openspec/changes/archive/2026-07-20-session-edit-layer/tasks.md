# Tasks: session-edit-layer

## 1. Renderer edit-target change
- [x] 1.1 `_attach_edit_layer`: target `stage.GetSessionLayer()` as the edit
      target; drop the `root.subLayerPaths` insert; keep the `_usd_edit_layer`
      / `_edit_layer_default_path` assignments and the idempotence guard.
- [x] 1.2 Update the `_attach_edit_layer` docstring + inline comments (no more
      "strongest sublayer"; describe session-layer strength + non-destructive).

## 2. Robust transform authoring
- [x] 2.1 `_author_local_transform`: reuse a single existing `xformOp:transform`
      via `op.Set(matrix)`; keep clear+add as the fallback for no-op / multi-op.

## 3. Save path
- [x] 3.1 Confirm `save_edits` exports the session layer for a real (on-disk)
      root and the composed stage for an anonymous root; adjust the anonymous-
      root check/comment if the session layer changes `root.anonymous` reasoning.

## 4. Tests (hostless first)
- [x] 4.1 New hostless regression: load `assets/first_mcp_scene.usda` (or a
      built stage) with a file-authored `xformOp:transform`, attach edit layer,
      `set_transform`/`_author_local_transform`, assert (a) no throw and (b) the
      composed transform equals the override.
- [x] 4.2 Hostless: save→reopen round-trips a session-layer add + transform
      override.
- [x] 4.3 Update `test_scene_editing.py::TestEditLayer::test_edit_layer_attached`
      to assert the session-layer edit target (not a sublayer).

## 5. GPU verification
- [x] 5.1 `test_scene_editing.py` GPU suite (`-m gpu`) green on Metal:
      set_transform / add / remove / save_edits + the new override case.
- [ ] 5.2 Live MCP re-check: `scene_set scale` on a light with an authored
      transform succeeds and the light visibly resizes.

## 6. Docs
- [x] 6.1 Update `docs/Architecture.md` edit-layer description if it names the
      sublayer topology.

## 7. Review + merge
- [x] 7.1 xhigh design review subagent on this design; fold findings.
- [x] 7.2 codex pre-merge review; fold findings.
- [ ] 7.3 `openspec validate session-edit-layer --strict`; archive after merge.
