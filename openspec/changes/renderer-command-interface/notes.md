# Notes: renderer-command-interface

## 1.1 Mutation paths into a renderer, per front-end

The renderer's **owning thread** is the thread that calls `renderer.update()`
and `render_headless()`. A mutation is safe only if it runs on that thread.

| Front-end | Path | Thread | Synchronised | State at 2dd4f4f |
|---|---|---|---|---|
| `skinny` (GLFW) | `ALL_PARAMS` setters → `_set_nested` | main loop | owning thread | safe |
| `skinny` (GLFW) | in-process MCP server | MCP worker | `RenderCommandQueue` | safe |
| `skinny-gui` (Qt) | shared control tree → `QtRendererProxy` | GUI thread | queue | safe |
| MCP tools | `post` / `post_with_reply` only | any | queue | safe |
| `skinny-web` | `SkinnySession.set_param` | — | queue | safe, but **no caller** |
| `skinny-web` | `SkinnySession.resize` | IOLoop/Bokeh | queue (reply) | safe (9e6322d) |
| `skinny-web` | shared control tree → `MarshalledRenderer` | Bokeh | queue for writes | **partly** — see 1.3 |
| `skinny-web` | `SkinnySession.handle_camera` | IOLoop | **none** | **RACY** |
| `skinny-web` | `SkinnySession.handle_control` | IOLoop | `_lock` | off-thread mutation |
| `skinny-web` | `SkinnySession.handle_autofocus` | IOLoop | `_lock` | off-thread mutation |
| `skinny-web` | `SkinnySession.screenshot` | Bokeh/IOLoop | `_lock` | off-thread GPU work |
| `skinny-web` | `_load_model` / `_load_hdr` sidebar callbacks | Bokeh | `_lock` | off-thread mutation |
| `skinny-web` | debug pane `view_*` / `reset` buttons | Bokeh | `_lock` | off-thread mutation |
| `skinny-web` | debug pane `_tick` render | Bokeh | `_lock` | off-thread GPU work |
| `skinny-web` | sidebar Camera-Debug shortcut | Bokeh | `_lock` (indirect) | widget-tree poke by index |
| `skinny-render` / `HeadlessRenderer` | direct attribute writes + calls | caller | n/a (single-threaded) | separate path |

`handle_camera` is the sharpest case: it takes no lock at all, and it mutates
the camera object the render thread reads in the same instant.

**The proposal's headline example is wrong.** It opens with `set_param`
(`web_app.py:204`) and says "every parameter change from a browser races the
render thread". No browser reaches that method. `VideoStreamHandler.on_message`
routes only `camera`, `control` and `autofocus`; a sidebar slider goes through
`set_param_value` → `MarshalledRenderer.set_path`, which posts, and has posted
since 9e6322d. `set_param` is correct and unused — the natural home if a
parameter WebSocket message is ever added, but not a defect this change fixed.
The real off-thread mutations are the rest of this table.

`_lock` is the render+encode lock. Holding it serialises a mutation against the
render, but it does not move the mutation onto the owning thread — that
distinction is the whole change (design D1).

## 1.3 Proxy verbs the web front-end needs

`MarshalledRenderer.__getattr__` refuses only verbs that match `apply_*`. The
prefix is a naming convention, not a boundary: a renderer mutation named
anything else passes straight through to the live object on the caller's
thread. The shared control tree reaches exactly one such verb:

- `toggle_material_furnace(material_id, enabled)` — `build_app_ui.py:422`,
  the per-material Furnace checkbox. Runs inline on the Bokeh thread today.

Verbs the tree reaches that already have a marshalled path: `set_path`,
`set_clock_state`, `apply_material_override`, plain attribute writes
(`camera_mode`, …).

Verbs the tree reaches that the **web host overrides** with its own callback,
so they never reach the proxy: `load_model_from_path` (`AppCallbacks.load_model`)
and `save_screenshot` (`AppCallbacks.capture_screenshot`). Both host callbacks
take `_lock` and are therefore off-thread mutations in their own right — fixed
at the session, not at the proxy.

Conclusion: one new marshalled verb (`toggle_material_furnace`), plus a
refusal rule that names mutations explicitly instead of deriving them from a
prefix.

## 1.2 Identity-gate baseline

`tools/rci_identity_gate.py` renders three suite scenes through
`HeadlessRenderer` at 128×128 / 32 spp, path + megakernel + flat — the
deterministic corner of the envelope. Each scene is rendered twice on the same
renderer (second with `exposure=1.5`) so the between-render mutation path is
covered, not only first-render setup.

Pre-change hashes are in `identity_before.json` (beside these notes), captured at 2dd4f4f on
native Metal, and reproduced byte-for-byte by a second run before being
recorded.
