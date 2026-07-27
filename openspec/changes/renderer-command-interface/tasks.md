# Tasks: renderer-command-interface

## 1. Baseline

- [ ] 1.1 Enumerate every mutation path into a renderer per front-end, and mark
      which thread it runs on and whether it is synchronised. Record the web
      session's split (control/autofocus/resize/screenshot take the lock;
      `set_param` and the six panel setters do not).
- [ ] 1.2 Capture reference renders from `skinny-render` for the identity gate
      in 4.2.
- [ ] 1.3 List the proxy verbs the web front-end will need that the Qt proxy
      does not already have.

## 2. Web session posts

- [ ] 2.1 Give the session a `RenderCommandQueue`; drain it in the render loop
      before `update()`.
- [ ] 2.2 Bind the shared control tree to a proxy instead of the live renderer
      (`web_app.py:548`) — this fixes all six `ui/panel/backend.py` setters
      without editing them.
- [ ] 2.3 Route `set_param`, `handle_camera`, `handle_control`,
      `handle_autofocus`, `resize`, `screenshot` through the queue. Keep the
      lock for the render+encode span only.
- [ ] 2.4 Delete the index-based button synthesis (`web_app.py:516-533`);
      Camera Debug actions become posted commands.
- [ ] 2.5 Add the proxy verbs from 1.3.

## 3. Headless posts

- [ ] 3.1 `HeadlessRenderer` uses the queue internally; its public method
      surface in `docs/PythonAPI.md` is unchanged.
- [ ] 3.2 Synchronous drain, no ordering change.

## 4. Gates

- [ ] 4.1 Hostless tests for all four front-ends' command paths against a stub
      renderer, including the web paths that have no coverage today.
- [ ] 4.2 `skinny-render` images identical to 1.2 — identical, not close.
- [ ] 4.3 Manual web smoke: drag a slider during an active render; no torn
      state, no stall.
- [ ] 4.4 Source gate: no direct renderer attribute write from a non-owning
      thread.
- [ ] 4.5 `ruff check src/`; full hostless `pytest`.
- [ ] 4.6 Docs: `docs/Architecture.md` threading section.
- [ ] 4.7 `openspec validate renderer-command-interface --strict`.
