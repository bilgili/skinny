## 1. Proposal and renderer editing API

- [x] 1.1 Add hostless failing tests for supported light types, explicit
  defaults, selected parents, unique names, invalid types, and no-stage errors.
- [x] 1.2 Implement edit-layer-backed `Renderer.add_light`, including rollback
  and the existing full scene resync.
- [x] 1.3 Verify that adding the first authored light disables both synthesized
  fallback lights and that the new prim appears in the rebuilt scene graph.

## 2. Scene-graph GUI controls

- [x] 2.1 Add shared supported-light metadata and reuse the selected group
  parent decision in both front-ends.
- [x] 2.2 Add a render-worker-safe Qt proxy method and an "Add light" menu
  control with success/error reporting and loaded-stage enablement.
- [x] 2.3 Add the equivalent Panel menu control under the session lock, with
  matching parent resolution, enablement, and status behavior.
- [x] 2.4 Add focused source/widget tests proving all five actions route to
  `add_light` without blocking the Qt GUI thread.

## 3. Validation and documentation

- [x] 3.1 Run focused scene-editing, scene-graph, light-authority, Qt, and Panel
  tests, then Ruff on all touched Python files.
- [x] 3.2 Update `docs/Architecture.md`, `docs/PythonAPI.md`, README,
  `CHANGELOG.md`, and the worktree/venv workflow rule in `CLAUDE.md`.
- [x] 3.3 Run `openspec validate scene-graph-light-controls --strict`, review
  proposal/spec/task alignment, and mark the change implementation-complete.

## 4. Review remediation

- [x] 4.1 Add regression tests for edit-layer-based control enablement, nested
  missing-parent rollback, and non-`ValueError` Panel failures.
- [x] 4.2 Require an active edit layer before Qt or Panel enables or dispatches
  Add light.
- [x] 4.3 Roll back every ancestor prim introduced while defining a missing
  nested parent.
- [x] 4.4 Surface every Panel creation/resync exception as a non-fatal status
  message.
- [x] 4.5 Add DiskLight to the existing README and Architecture supported-light
  summaries, then rerun focused tests, Ruff, and strict OpenSpec validation.
