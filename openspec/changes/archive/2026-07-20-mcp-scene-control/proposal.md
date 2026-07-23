## Why

Driving a skinny scene today means clicking docks or hand-editing USD. There is
no way for an agent to inspect what is in the scene and change it while the
render is live — the only remote-control surface in the codebase is the video
WebSocket in `web_app.py`, which accepts exactly three fixed verbs (`camera`,
`control`, `autofocus`) and exposes no scene state at all.

Everything needed for agent-driven scene editing already exists in-tree and is
already load-bearing for the Qt UI: a path-addressed scene tree with typed,
range-annotated, editable properties (`scene_graph.py`), a Qt-free cross-thread
command queue (`RenderCommandQueue`), and ~20 settled mutation verbs on
`QtRendererProxy`. What is missing is a wire. MCP is that wire, and adopting it
means writing no protocol of our own.

## What Changes

- **New opt-in MCP server hosted inside the running renderer process.** A daemon
  thread serves MCP over streamable HTTP. It attaches to the renderer that is
  already running; it never constructs a `Renderer` or a GPU context of its own.
- **Both interactive front-ends can host it** — `skinny-gui` (Qt) and `skinny`
  (GLFW). Qt already has the marshalling seam; GLFW gains a `queue.drain()` call
  in its main loop.
- **`render_session.py` moves to a front-end-neutral module** — the whole module,
  queue *and* `QtRendererProxy`, which already has zero Qt imports. The queue also
  gains the execute-and-reply loop currently inlined in the Qt worker, so a second
  front-end cannot get it subtly wrong.
- **Three path-addressed MCP tools**, not a thin mirror of proxy methods and not a
  Python escape hatch: `scene_list`, `scene_get`, `scene_set`. Everything is
  addressed by USD prim path plus property name.
- **One shared dispatch function.** The scene-graph dock's property-apply logic is
  extracted into a GUI-free `apply_scene_property`, called by both the dock and the
  server — so an MCP edit and a dock edit execute the same code rather than two
  tables that drift.
- **Tool responses are self-describing.** `scene_get` returns each property's
  `editable` flag and `metadata` ranges, so an agent can tell what it may set and
  to what, without out-of-band documentation.
- **Security posture is opt-in and layered**: off by default behind `--mcp`
  (env `SKINNY_MCP`); binds loopback only; rejects any request carrying an
  `Origin` header; requires a persistent bearer token. Startup prints the
  ready-to-paste `claude mcp add --transport http` line.
- **The MCP thread never touches the `Renderer` or the GPU.** Every read and
  every mutation crosses to the render thread through the command queue. This is
  a hard invariant, not a convention — `Renderer` has no internal lock.

### Deliberately out of scope for v1

- **No `save_edits` tool.** Material, light, and instance edits mutate the flat
  in-memory `Scene` and bypass USD entirely, so a `save_edits` tool would export
  a layer containing almost none of what the agent changed — a silent, invisible
  data loss. Persistence is a separate change with its own gate.
- **No `scene_add` / `scene_remove`.** They author solely into the USD edit layer,
  which without `save_edits` is discarded at exit — so their entire output would be
  thrown away — and they are inoperative on scenes with no USD stage. They also
  carry the widest attack surface (`add_model` takes an arbitrary filesystem path).
  They belong in the same change as export.
- **No image/render tool.** The operator watches the live window. This is a real
  limitation (the agent edits without seeing the result) accepted on purpose: it
  also removes the need to wait on progressive accumulation before every readback.
- **No new renderer capability.** The MCP exposes what the Qt docks can already do.

## Capabilities

### New Capabilities

- `mcp-scene-control`: An opt-in, in-process MCP server that exposes the live
  renderer's scene graph to an MCP client for read and mutation — the tool
  surface, its path addressing and dispatch rules, the cross-thread invariant,
  the transport and its authentication, and the front-end lifecycle (enable
  flag, port selection, bind collision behavior).

### Modified Capabilities

- `qt-render-threading`: The render-thread command queue is no longer Qt-specific.
  Its requirement is generalized so that any front-end — and any in-process
  server thread — marshals renderer mutations through it, and so that the GLFW
  front-end drains it in its main loop. The existing Qt ownership and
  snapshot-reading requirements are unchanged in substance.
- `render-cli`: Adds the `--mcp` / `--mcp-port` flags and their env equivalents to
  the interactive front-ends, including startup rejection rules and the
  bind-collision warning.

## Impact

**Code**

- New: MCP server module (transport, auth, tool handlers, path→verb dispatch).
- New: neutral `RenderCommandQueue` module; `ui/qt/render_session.py` imports from
  it instead of defining it.
- Modified: `app.py` (GLFW) — drain the queue in the main loop; construct and
  own the queue.
- Modified: `ui/qt/app.py` — hand its existing queue to the server when enabled.
- Modified: `cli_common.py` — new flags.
- Modified: `settings.py` — token file under the existing settings dir.

**Dependencies**

- Adds an MCP server library and its HTTP/ASGI transport. This is the first
  network-server dependency in the interactive path; it must be an optional
  extra so a default install is unchanged and the flag errors clearly when the
  extra is absent.

**Security**

- Introduces a listening socket in a process that can load asset paths from disk.
  Loopback bind, `Origin` rejection, and bearer-token auth are all requirements,
  not defaults — each covers a different attacker, and the token alone does not
  substitute for the bind.

**Docs**

- `docs/PythonAPI.md` (public surface), `docs/Architecture.md` (module map and
  the threading seam), `README.md` (flags, setup line, compatibility), and
  `CHANGELOG.md`.

**Not affected**: shaders, integrators, GPU pipelines, the parity matrix. No
`.spv` changes.

## Open Questions

None blocking. Both questions raised in the first draft are resolved in `design.md`
(D4 read-surface shape, D5 concurrency), and two further questions it left open were
answered by adversarial design review: the transform verb is whatever the dock picks
at runtime, and the filesystem-path question is moot now that node authoring is cut
from v1.

That review also corrected two factual errors in the first draft — the dispatch rule
(which dead-ended on material parameters, the most common edit) and the version-counter
claim (which named the wrong counter and inverted D5's rationale). Both corrections are
recorded inline in `design.md` so the record shows what was wrong, not just what is now
right.
