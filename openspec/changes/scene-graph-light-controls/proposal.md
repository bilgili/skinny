## Why

The scene-graph GUI can reference models, delete prims, edit transforms, and
save an edit layer, but it cannot create lights. Lighting a scene therefore
requires leaving Skinny and editing the USD in another application or by hand,
even though the renderer already loads and displays all supported USD light
schemas.

## What Changes

- Add a renderer editing API that authors DistantLight, SphereLight, DomeLight,
  RectLight, and DiskLight prims into the active USD edit layer.
- Give newly authored lights unique prim paths and usable schema defaults, place
  them below the selected group-like prim (or `/World`), and resync the loaded
  scene immediately.
- Add an "Add light" control to both the Qt and Panel scene-graph GUIs, with an
  action for each supported light type.
- Keep failures non-fatal and disable light creation when no editable USD stage
  is loaded.
- Ensure the first authored light atomically replaces the renderer fallback
  DistantLight/IBL pair under the existing USD light-authority policy.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `usd-scene-editing-ui`: add light creation to the shared scene-graph editing
  workflow.

## Impact

- `src/skinny/renderer.py`: add the edit-layer-backed `add_light` operation and
  reuse the existing full stage resync.
- `src/skinny/ui/scene_edit_actions.py`: share supported light-type labels and
  selected-parent resolution between front-ends.
- `src/skinny/ui/qt/render_session.py`,
  `src/skinny/ui/qt/windows/scene_graph.py`, and
  `src/skinny/ui/panel/windows.py`: expose asynchronous/thread-safe light
  creation controls.
- Tests cover USD authorship, unique paths, parent selection, authority
  transitions, UI wiring, and failure behavior.
- `docs/Architecture.md`, `docs/PythonAPI.md`, README, `CHANGELOG.md`, and the
  repository workflow guidance are updated where touched.
- No shader, descriptor, GPU dispatch, or dependency changes are expected.
