# Qt render-thread separation

## Why

`skinny-gui` renders frames on a `QThread`, but the GUI still constructs, owns,
and directly mutates the `Renderer`. Startup scene load, renderer creation, menu
actions, dock refreshes, sidebar edits, resize requests, camera/gizmo input, and
session snapshots can all execute renderer work from the Qt GUI thread or block
behind the render lock. When rendering is busy, these synchronous calls make the
desktop UI feel frozen even though the frame loop itself is nominally off-thread.

The GUI should instead behave like a thin controller: it posts commands to a
render session that owns the GPU context and renderer on the render thread, then
receives immutable frame/status/state snapshots back on the GUI thread.

## What changes

- **Render-thread ownership.** Move the Qt renderer lifecycle behind a
  render-thread session object. The render thread constructs the GPU context and
  `Renderer`, runs `update()`/`render_headless()`, applies queued commands, and
  performs cleanup. The GUI thread does not call heavy renderer methods directly.
- **Command queue.** Replace direct GUI-thread mutations with posted commands for
  scene loading, parameter edits, camera/gizmo input, zoom, resize, online
  training enablement, material edits, and session snapshot requests. Commands
  are drained on the render thread between frames so state changes reset
  accumulation predictably.
- **Snapshots for UI.** The render thread emits frame bytes, accumulation count,
  backend/GPU/status text, and cheap immutable state snapshots. GUI docks read
  snapshots for display and request actions through commands rather than sharing
  the live renderer object.
- **Graceful shutdown.** Closing the Qt app posts a stop command, drains any
  requested settings snapshot, disables online training, cleans up the renderer,
  and destroys the GPU context on the owning render thread.

## Out of scope

- Changing renderer math, integrators, shaders, or backend parity.
- Moving the GLFW or web front-ends to the same session abstraction.
- Rewriting every inspector dock in one step; this change establishes the
  render-thread boundary and ports the blocking `skinny-gui` paths needed for a
  responsive viewport and common scene/control interactions.

## Impact

- Affected specs: new `qt-render-threading`.
- Affected code: `src/skinny/ui/qt/app.py`, `src/skinny/ui/qt/viewport.py`, Qt
  dock/controller modules that currently take a live renderer, and focused tests
  for the new command/session boundary.
- Affected docs: `README.md` Qt desktop notes and `docs/Architecture.md` UI
  architecture section.
