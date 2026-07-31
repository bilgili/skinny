# Tasks: renderer-command-interface

## 1. Baseline

- [x] 1.1 Enumerate every mutation path into a renderer per front-end, and mark
      which thread it runs on and whether it is synchronised. Record the web
      session's split (control/autofocus/resize/screenshot take the lock;
      `set_param` and the six panel setters do not).
- [x] 1.2 Capture reference renders from `skinny-render` for the identity gate
      in 4.2.
- [x] 1.3 List the proxy verbs the web front-end will need that the Qt proxy
      does not already have.

## 2. Web session posts

- [x] 2.1 Give the session a `RenderCommandQueue`; drain it in the render loop
      before `update()`. *(landed ahead of this change in 9e6322d.)*
- [x] 2.2 Bind the shared control tree to a proxy instead of the live renderer
      (`web_app.py:548`) — this fixes all six `ui/panel/backend.py` setters
      without editing them. *(landed ahead of this change in 9e6322d.)*
- [x] 2.3 Route `set_param`, `handle_camera`, `handle_control`,
      `handle_autofocus`, `resize`, `screenshot` through the queue. Keep the
      lock for the render+encode span only. Also the nine lock-mediated
      mutations in `ui/panel/windows.py` and the two sidebar load callbacks,
      which are the same off-thread write in a different file.
- [x] 2.4 Delete the index-based button synthesis (`web_app.py:516-533`);
      Camera Debug actions become posted commands.
- [x] 2.5 Add the proxy verbs from 1.3.

## 3. Headless posts

- [x] 3.1 `HeadlessRenderer` uses the queue internally; its public method
      surface in `docs/PythonAPI.md` is unchanged.
- [x] 3.2 Synchronous drain, no ordering change.

## 4. Gates

- [x] 4.1 Hostless tests for all four front-ends' command paths against a stub
      renderer, including the web paths that have no coverage today.
      `tests/test_renderer_command_interface.py`, 35 tests. Negative control:
      23 of the 35 fail against the pre-change tree.
- [x] 4.2 `skinny-render` images identical to 1.2 — identical, not close.
      `tools/rci_identity_gate.py`: all six SHA-256 hashes unchanged.
- [x] 4.3 Manual web smoke: drag a slider during an active render; no torn
      state, no stall. Scripted as `tools/rci_web_smoke.py` — a real session,
      real GPU, real render thread, 921 slider writes + 618 camera gestures +
      12 control toggles from two other threads. Frames kept flowing, the
      awaited screenshot settled in 0.06 s, accumulation resumed to 46 in the
      quiet phase, and the last slider value landed. The slider writes go
      through `MarshalledRenderer.set_path`, which is the path a browser
      slider really takes — **not** `SkinnySession.set_param`, which the UI
      never calls. The browser client itself was not driven.
- [x] 4.4 Source gate: no direct renderer attribute write from a non-owning
      thread.
- [x] 4.5 `ruff check src/`; full hostless `pytest`.
- [x] 4.6 Docs: `docs/Architecture.md` threading section.
- [x] 4.7 `openspec validate renderer-command-interface --strict`.
