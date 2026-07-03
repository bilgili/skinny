# Tasks — Qt render-thread separation

## 1. Render-session boundary

- [ ] 1.1 Add a Qt render-session/worker abstraction that owns `make_context`,
  `Renderer`, frame rendering, online-training ticks, and cleanup on the worker
  thread.
- [x] 1.2 Add a FIFO command queue with typed commands for renderer mutations and
  one-shot replies for commands that need a result.
- [ ] 1.3 Emit immutable frame/status snapshots from the worker; keep GUI painting
  on the main thread.

## 2. Move blocking lifecycle off the GUI thread

- [ ] 2.1 Move GPU context and `Renderer` construction from `MainWindow` into the
  render worker; show startup errors via a Qt signal instead of raising from the
  GUI constructor.
- [ ] 2.2 Move renderer cleanup, online-training disable, and context destroy into
  worker-thread shutdown.
- [ ] 2.3 Keep backend selection/persisted CLI precedence in `main()` while the
  actual backend resources are created by the worker.

## 3. Port interactive commands

- [ ] 3.1 Port camera, gizmo, zoom, focus, HUD, free-cam movement, and viewport
  resize actions to render-session commands.
- [ ] 3.2 Port File → Open scene and scene-pick requests to render-session
  commands with completion/error signals.
- [ ] 3.3 Port sidebar parameter edits to render-session commands, coalescing
  high-rate slider changes where needed.

## 4. Port snapshots and docks

- [ ] 4.1 Replace direct GUI-thread session restore/snapshot calls with
  worker-thread apply/snapshot commands.
- [ ] 4.2 Update status bar and online-training state to use emitted snapshots.
- [ ] 4.3 Update inspector/editor docks that need renderer data to use snapshots
  or explicit commands rather than sharing the live renderer object.

## 5. Tests, validation, and docs

- [ ] 5.1 Add no-GPU unit coverage for FIFO command draining, resize coalescing,
  shutdown ordering, and snapshot delivery.
- [ ] 5.2 Add a focused Qt smoke/integration test or harness proving a long render
  command does not block GUI event processing.
- [ ] 5.3 Run `ruff check src/`, focused pytest coverage, and `openspec validate
  qt-render-thread-separation`.
- [ ] 5.4 Update `README.md` and `docs/Architecture.md` for the Qt render-session
  architecture.
