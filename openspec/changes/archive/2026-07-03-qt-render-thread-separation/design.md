# Design — Qt render-thread separation

## Current shape

`MainWindow` constructs `ctx = make_context(...)` and `renderer = Renderer(...)`
on the GUI thread, then passes that renderer to `RenderViewport` and every dock.
`RenderViewport` starts `_RenderWorker` on a `QThread`; that worker calls
`renderer.update()`, online-training ticks, and `renderer.render_headless()`
under a Python `Lock`. Many GUI handlers also acquire the same lock and call
renderer methods directly. Some paths, such as session restore/snapshot and the
File → Open scene action, bypass the viewport lock entirely.

This leaves two problems:

1. The expensive lifecycle path is still GUI-thread work.
2. Any GUI action that needs the renderer either races the render worker or waits
   for the render lock while the GPU frame is in flight.

## Proposed shape

Introduce a Qt render-session boundary:

- `QtRenderSession` lives on the GUI thread as a controller/proxy.
- `_RenderWorker` lives on a `QThread` and owns the actual GPU context and
  renderer after startup.
- GUI code calls `session.post(command, *args)` or typed helpers such as
  `session.load_scene(path)`, `session.resize_render_target(w, h)`,
  `session.camera_drag(...)`, and `session.set_param(path, value)`.
- The worker drains queued commands before rendering each frame. Long commands
  can emit progress/error signals without blocking Qt event processing.
- The worker emits immutable dataclasses/dicts: `FrameSnapshot` for pixels and
  accumulation, `RendererSnapshot` for status/sidebar/dock data, and one-shot
  replies for operations that need a result.

## Threading rules

- The GUI thread SHALL NOT invoke renderer methods that can upload resources,
  rebuild scene data, dispatch GPU work, or wait for backend fences.
- The render thread SHALL be the only owner of `ctx`, `Renderer`, online-training
  lifecycle, and GPU resource cleanup.
- GUI widgets MAY cache last-known snapshots and paint from them.
- GUI commands SHALL be idempotent where practical and applied in FIFO order.
- Resize commands MAY coalesce so a resize drag does not enqueue unbounded GPU
  reallocations.

## Migration strategy

1. Add the session/command abstraction while keeping the existing viewport frame
   blit behavior.
2. Move context/renderer construction into the worker and make startup emit a
   ready signal with backend/GPU info.
3. Port viewport input and resize paths to commands.
4. Port File → Open scene and settings snapshot/restore to command/reply flow.
5. Port the shared sidebar parameter callbacks so slider drags post parameter
   commands rather than writing directly to the renderer.
6. Update docks that need live inspection to consume snapshots or request
   one-shot operations through the session.

The first useful landing point is a responsive GUI under continuous rendering:
mouse/keyboard input, menu open, render-target resize, and common sidebar edits
must not synchronously wait for frame rendering on the GUI thread.
