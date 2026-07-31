# Change: renderer-command-interface

## Why

"Drive the renderer" has three different answers depending on the entry point,
and one of them writes to a live renderer from another thread with no lock.

- **Queue-mediated** (`skinny-gui`, `skinny`, MCP): `RenderCommandQueue`
  (`render_session.py:574-652`) with `post` / `post_with_reply` and
  last-write-wins coalescing. `mcp_server.py` states the invariant explicitly
  at `:36-49` — "no `renderer.` outside a posted closure" — and is, not
  coincidentally, the best-tested interactive surface in the repo.
- **Lock-mediated, partially** (`skinny-web`): `SkinnySession` holds a
  `threading.Lock` (`web_app.py:84`) across `update()` + `render_headless()` +
  encode on the render thread. `handle_control`, `handle_autofocus`, `resize`
  and `screenshot` take that lock. **`set_param` (`web_app.py:204`) does not** —
  it calls `_set_nested` on the live renderer directly. Nor do the sidebar
  widget setters: `ui/panel/backend.py:239,263,286,304,324,356` call
  `node.setter(...)` on the Bokeh worker thread, and `web_app.py:548` binds the
  shared control tree to `session.renderer` — the **live object** — where
  `ui/qt/app.py:182` binds the same tree to the marshalling proxy.
- **Direct, synchronous** (`skinny-render`): `HeadlessRenderer` constructs and
  calls, no queue, no thread — and uses a different scene-ingest path from the
  other three.

So the same shared control tree carries two opposite thread-safety contracts
depending on which front-end mounts it. Every parameter change from a browser
races the render thread.

Secondary consequences of the split: only the queue path has a reply/error
contract, so the web front-end reports no edit outcome; the web sidebar reaches
Camera-Debug controls by **synthesising button clicks into the pane's widget
tree by index** (`web_app.py:516-533`, `btn_row.objects[idx].clicks += 1`);
and `web_app.SkinnySession` is untestable without a real device, so none of
this is covered.

## What Changes

- Make the command queue the interface for driving a renderer, on every
  front-end. `skinny-web` binds the shared control tree to a proxy, as
  `skinny-gui` already does, and its session posts instead of mutating.
- `skinny-render` drives through the same interface, posting and draining
  synchronously — one call shape for interactive and non-interactive callers.
- Every mutation carries a reply, so the web front-end can report an edit
  outcome the way MCP already does.
- Delete the index-based button synthesis in `web_app.py:516-533`; the debug
  camera actions become posted commands like every other action.
- Extend the queue's stub-based tests to cover all four front-ends' command
  paths — they need no GPU.

## Capabilities

### New Capabilities

- `renderer-command-interface`: one interface for driving a renderer from any
  front-end or tool — post, post-with-reply, coalescing, drain on the owning
  thread — with no front-end mutating a live renderer off-thread.

### Modified Capabilities

- `qt-render-threading`: "Single-threaded front-end owns and drains a command
  queue" generalises from the Qt/GLFW pair to every front-end, including the
  web session's background render thread and the headless driver.

## Impact

- Modified: `src/skinny/web_app.py` (session mutation path, sidebar binding,
  control dispatch, button synthesis), `src/skinny/ui/panel/windows.py` (the
  scene-graph, material-graph and Camera-Debug edits post instead of taking the
  session lock), `src/skinny/usd_controls.py` (the `usd` setter posts its stage
  write), `src/skinny/ui/build_app_ui.py` (the scene-loader and capture controls
  refuse a missing owning-thread callback), `src/skinny/headless.py` (drives
  through the queue), `src/skinny/render_session.py` (proxy verbs the web
  needs). **Not** `src/skinny/ui/panel/backend.py` — binding the tree to a proxy
  fixes its six setters without editing it, which is what D3 predicted.
- Unchanged: the queue's semantics, MCP tools, Qt behaviour, rendered output.
- **User-visible**: web parameter changes stop racing the render thread. Under
  load the current behaviour is a torn read at best; there is no evidence it
  has been benign, only that it has been unreported.
- Depends on nothing; interacts with `scene-intake-interface`, which must agree
  on the same off-thread-build / owning-thread-apply discipline.
- Docs: `docs/Architecture.md` threading section; `docs/PythonAPI.md` if the
  headless driving surface changes.
