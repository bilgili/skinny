## Context

skinny has no agent-facing control surface. The only remote surface is the video
WebSocket in `web_app.py:434-453`, which accepts three fixed verbs and exposes no
scene state.

Four facts about the existing code shape this design.

**1. `Renderer` is single-owner and has no internal lock.** The Qt worker thread is
"the single owner of the renderer + GPU context, so no lock is needed"
(`ui/qt/viewport.py:139`); GLFW runs everything on one thread; `web_app.py` wraps a
`threading.Lock`. Any new caller must marshal, not lock.

Reads must be marshalled too, not just writes: `build_scene_graph` runs on the
background streaming load thread (`renderer.py:3495-3502`) and the result is
swapped in at `:5373`, so an off-thread read races a swap.

**2. The seam already exists, and it is already GUI-free.**
`src/skinny/ui/qt/render_session.py` contains **zero** Qt imports, and the package
`__init__` files it sits under are docstrings only — it imports today with no GUI
toolkit present. It holds both `RenderCommandQueue` (`:555`) and `QtRendererProxy`
(`:247`), and the proxy already exposes every verb this change needs, already
queue-backed and already correctly coalesced: `apply_material_override` (`:399`),
`apply_light_override` (`:408`), `apply_instance_transform` (`:415`),
`set_transform` (`:421`), `apply_camera_param` (`:393`), `apply_node_enabled` /
`apply_subtree_enabled` (`:385`/`:388`), and `refresh_scene_state()` returning a
detached `copy_scene_graph` snapshot (`:180-181`).

**3. The scene model is already self-describing.** `SceneGraphProperty`
(`scene_graph.py:27`) carries `editable` and `metadata`; `scene_graph_to_dict`
(`:764-786`) already emits both per property. The "self-describing tools"
requirement needs no new code.

**4. Mutation dispatch is *not* a function of `(path, property_name)`.** This is the
central technical constraint and it is covered in D3.

The Qt Scene Graph dock (`ui/qt/windows/scene_graph.py`) is a generic property
editor over this model. This change adds a second client of it, and introduces no
renderer capability.

## Goals / Non-Goals

**Goals:**

- An MCP client can enumerate the live scene, read a node's properties with their
  editable flags and bounds, and mutate them — against a renderer already running
  and visible.
- Zero new protocol; MCP is the wire.
- Works from both interactive front-ends through one shared seam.
- Off by default, loopback-only, authenticated.
- The server thread never touches `Renderer` or the GPU.
- An MCP edit and the equivalent dock edit execute **the same dispatch code**, not
  two tables that drift.

**Non-Goals:**

- Persistence of agent edits (D6).
- Returning rendered images (D7).
- Adding or removing scene nodes (D8 — cut from v1).
- Driving or spawning a headless renderer.
- `skinny-web`. It has a live renderer but guards it with a `Lock`, not a command
  queue (`web_app.py:85`), so it has no seam to attach to. Out of scope.
- Multi-client coordination. One agent, one operator.

## Decisions

### D1 — Server lives inside the renderer process, not in a bridge

A daemon thread inside `skinny` / `skinny-gui` serves MCP over streamable HTTP.

*Alternative rejected: a stdio bridge process.* Requires inventing, versioning and
debugging a bespoke protocol whose only consumer is a translator into MCP — every
tool defined twice, in two processes.

*Accepted cost:* with no skinny running, the client gets connection-refused rather
than "no renderer attached".

### D2 — Move `render_session.py` whole; give the queue an executor; the server holds a proxy

Three parts, all corrections to an earlier draft of this design.

**(a) Move the whole module, not two classes.** `src/skinny/ui/qt/render_session.py`
moves to `src/skinny/render_session.py`, with `ui/qt/render_session.py` re-exporting
so existing imports and tests are unaffected. Moving only `RenderCommandQueue` would
strand `QtRendererProxy` behind `ui/qt/` and guarantee the server re-implements it.
The module has no Qt imports, so this is a relocation, not a port.

**(b) `drain()` does not execute — add `run_pending`.** `RenderCommandQueue.drain()`
(`render_session.py:591-596`) only removes and returns the pending list. The
executor — the try/except that calls the callback and sets `reply.set_result` /
`reply.set_exception` — is inlined in the Qt worker (`ui/qt/viewport.py:200-209`)
and exists nowhere else in the repo. A second front-end calling `drain()` and
looping the callbacks would, if it omitted the reply calls, hang every awaited read
until timeout — a silent, timeout-shaped bug.

So the queue gains `run_pending(target, on_error=None)` holding that loop, and both
`viewport.py` and the GLFW loop call it. Qt passes `on_error=self.error.emit` to
keep its current behavior. This is less total code than duplicating the executor,
and it is what makes the relocation earn its place.

**(c) The server holds a `QtRendererProxy` (Qt) or a bare queue (GLFW).** Reads
resolve nodes on the render thread; writes dispatch through the shared
`apply_scene_property`.

**Correction after implementation review:** an earlier draft of this design said
MCP writes would be posted, not awaited, so they could coalesce. The implemented
behavior is the opposite and deliberately so — node resolution, type/bounds
validation, and routing all have to run on the render thread, and the client must
be told whether its edit applied or why it did not. A fire-and-forget write would
report success for an edit a validation check then discarded. The cost is that MCP
writes do not coalesce (`post_with_reply` takes no `coalesce_key`); the operator's
own slider drags still do, through the dock's proxy verbs. A timed-out request is
cancelled and `run_pending` skips cancelled commands, so a write cannot land after
the client was told it failed.

**Hard invariant, unchanged:** the server thread performs no attribute access on
`Renderer`, no GPU call, and no direct scene-graph read. Enforceable in review by
grepping the server module for `renderer.` outside a posted closure.

GLFW gains a queue: `app.py` constructs one and calls `run_pending` at `app.py:689`,
after `glfw.poll_events()` (`:688`) and before `renderer.update(dt)` (`:695`). Note
`input_handler.update(dt)` (`:694`) sits between and mutates the camera on the same
thread — commands land before it, which matches Qt's ordering.

### D3 — Dispatch on the resolved *property*, via one shared function

**`(prim_path, property_name)` alone does not determine the verb.** Material
parameters live on Shader prims, and `_add_shader_props` (`scene_graph.py:445-493`)
attaches **no** `renderer_ref` — so `_resolve_renderer_ref` (`renderer.py:7385`)
returns `None` for the single most common edit in the feature. The dock handles this
with an ancestor walk to the enclosing Material (`windows/scene_graph.py:698-702`,
`_find_shader_material_ref` `:756-770`).

The full dispatch actually requires, per case:

| Case | What it needs | Dock site |
|---|---|---|
| Shader property | ancestor walk to Material ref | `:698-702`, `:756-770` |
| vec3 on camera | `metadata["camera_axis"]`, then fan one vec3 into three scalar `apply_camera_param` calls | `:720-730` |
| vec3 on instance/light | node `type_name`, **plus sibling translate/rotate/scale read off the node** to recompose TRS | `:731-746` |
| transform verb | runtime `_usd_stage is not None` → `set_transform`, else `apply_instance_transform` | `:750-754` |
| bool | `metadata["toggle"]` → `apply_subtree_enabled` vs `apply_node_enabled` | `:455-470` |
| texture_file | `ref.kind == "light_env"` → `apply_dome_light_texture` | `:665-690` |

Two traps worth naming: `RendererRef.kind` has **two** camera kinds — `camera`
(`scene_graph.py:219`) and `renderer_camera` (`:895`) — and `_apply_property` routes
only `renderer_camera`, so a `camera`-kind node silently no-ops. And a transform
write is not a per-property write at all: setting `translate` must read the sibling
`rotate`/`scale` and recompose.

**Decision:** extract the dock's `_apply_property` / `_apply_vec3_property` / bool
branch into one GUI-free function — `apply_scene_property(renderer, node, prop, value)`
in `scene_graph.py` (or a sibling module, next to the existing
`ui/scene_edit_actions.py`). The dock and the MCP server both call it.

This is a *smaller* diff than a second dispatch table, and it is the only way to
honor the spec's own requirement that an MCP edit and a dock edit take the same code
path — a parallel table violates it by construction on day one.

Addressing stays `(path, property)` at the tool boundary; the server resolves the
node once, then dispatches on the resolved `SceneGraphProperty` plus node context.

*Alternatives rejected:* one MCP tool per proxy method (~20) — forces the agent to
speak in material/light/instance index integers, and ~20 near-identical tools is the
shape LLM tool-routing does worst on. A single `eval_python` tool — unbounded blast
radius in a process holding a GPU context, reachable over a socket.

### D4 — `scene_list` returns structure; `scene_get` drills down

`scene_list(path="/", depth=2, kind=None)` returns path, name, `type_name`, child
count — **no properties**. `scene_get(path)` returns full properties for one node.

`build_scene_graph` walks the whole stage (`scene_graph.py:156`) and
`scene_graph_to_dict` serializes every node *with all properties*; on a
pbrt-imported scene that is far past what should be handed to an agent, and wasted
on the ~99% of nodes it will not edit. The agent's real workflow is `ls` then `cat`.

`kind` filters on `RendererRef.kind`, already stamped by `populate_instance_refs`
(`:1004`) — a predicate over existing data, not a new index.

Default `depth=2`: USD nests hard enough that `depth=1` at root often shows only
`/World`.

**Resolve the node once per tool call.** `find_node_by_path` (`:1051-1058`) is an
unindexed full-tree DFS, and the shader ancestor walk calls it once per level —
O(n·depth) per write, executed on the render thread between frames. Fine at click
rates; an agent sweeping a value is a different load.

*Alternatives rejected:* full tree depth-capped (still ships every property);
cursor pagination (most machinery, least benefit).

### D5 — Concurrency is last-write-wins; echo **both** version counters

No locking, no rejection, docks stay enabled. Every tool result reports the current
version so the agent can observe the scene moved beneath it.

**Correction to an earlier draft:** that draft claimed `_scene_graph_version`
(`renderer.py:2017`) is bumped at ~19 sites on essentially any edit, and rejected
optimistic concurrency on that basis. That is false. `_scene_graph_version` is
written at exactly **five** sites — `:5438`, `:5969`, `:8077`, `:8231`, `:8250` — all
**structural**. The ~19 figure is `_material_version`, a different counter. The code
is explicit at `renderer.py:8327`: *do not* bump `_scene_graph_version` on property
edits, because the dock's widgets are bound to the live `SceneGraphProperty` objects.

So a property edit moves `_material_version` and leaves `_scene_graph_version`
still. Tool results therefore carry **both**; `SceneStateSnapshot` already exposes
each (`render_session.py:180`, `:188`), so this costs nothing.

Last-write-wins remains the decision, on its real merits: there is no invariant
spanning two properties that a torn write corrupts, and the worst case — a value
lands somewhere unintended — is immediately visible in the viewport the operator is
already watching, and undone by dragging it back. Per-node versioning does not
exist and is not worth inventing here.

*Alternative rejected: freeze the docks during an agent session.* The premise is a
live window the operator is watching; disabling their controls defeats it.

### D6 — No `save_edits` tool in v1

Two mutation paths with different persistence. Structural edits (`add_model`
`renderer.py:6034`, `add_light` `:6085`, `remove_node` `:6173`, `set_transform`
`:6189`) author into the anonymous edit layer (`_attach_edit_layer` `:5826`) and are
exported by `save_edits` (`:6219`). Everything else — `apply_material_override`
(`:6991`), `apply_light_override` (`:7271`), `apply_instance_transform` (`:7414`) —
mutates the flat in-memory `Scene` and re-uploads GPU buffers, touching USD not at
all. Only dome intensity (`:7286-7292`) and dome texture (`:8014`) write back.

A `save_edits` tool would export a layer containing almost none of what the agent
changed, invisibly, until someone reloads. Shipping a tool named "save" with that
behavior is worse than shipping no save.

Follow-up change: write flat-path edits back to the edit layer. Touches methods
every dock calls, changes what a `.usda` export contains, needs the parity harness —
its own proposal, its own gate.

### D7 — No image tool in v1

The operator watches the live window. Accepted deliberately, and it removes
otherwise-mandatory machinery: any edit resets `accum_frame` to 0
(`renderer.py:10518`), so an immediate readback is a 1-spp noise field. A naive
image tool would show the agent mush, from which it would conclude the edit was
wrong and "fix" it — worse than no image. Doing it properly needs a `min_samples`
wait and downscaling, and belongs to a change that wants the visual loop.

### D8 — Three tools in v1: `scene_list`, `scene_get`, `scene_set`

`scene_add` / `scene_remove` are **cut**.

They author solely into the edit layer, and D6 removes the only way to export it —
so an agent adds a light and the sole artifact is discarded at exit. `remove_node`
additionally raises `RuntimeError` with no USD stage (`renderer.py:6173-6180`), so
both are inoperative on non-stage scenes. They are also the widest attack surface,
`add_model` taking an arbitrary filesystem path.

Inspect-and-mutate-a-live-scene is the demonstrated value and it is fully delivered
by three tools. Add/remove returns alongside `save_edits`, where it is coherent.

### D9 — Transport: MCP SDK, own socket, no signal handlers

Add the MCP Python SDK under a new `[mcp]` extra. `--mcp` without it exits with an
install hint, in the style of the existing `reject_*` paths
(`cli_common.py:194/212/277`).

**The runtime must not install signal handlers.** `uvicorn.Server.run()` installs
`SIGINT`/`SIGTERM` handlers by default. From a non-main thread `signal.signal()`
raises `ValueError: signal only works in main thread of the main interpreter`; and
were it to succeed it would overwrite `MetalContext`'s chained SIGINT/SIGTERM
teardown (`metal_context.py:53`, `:77-85`) — which `CLAUDE.md` designates as the
backstop preventing an abandoned kernel from wedging the GPU until reboot. That is
the project's highest-consequence invariant and this change must not weaken it.

So: **construct the listening socket ourselves** (which is also where D10's loopback
assertion belongs — assert on the socket, not on a config string), hand it over via
`uvicorn.Config(..., fd=sock.fileno())`, and run `Server.serve()` on the thread's own
event loop with `install_signal_handlers=False`. A test asserts the process's SIGINT
handler is unchanged after the server starts.

Owning the socket also makes D11's bind-collision `OSError` catchable *before* the
registration line is printed.

*Alternative considered and rejected: hand-roll MCP over tornado.* Tornado is already
installed (transitive via `panel`) and already used for this shape of thing
(`web_app.py:28-30`, router injection `:775-777`), so it would add **zero**
dependencies — normally decisive. Rejected because tornado covers HTTP but not MCP:
hand-rolling means owning JSON-RPC framing, initialize/capability negotiation,
session handling and SSE against a moving spec. That is the class of thing that
looks like 150 lines and then drifts.

### D10 — Security: opt-in, loopback, Host+Origin checks, bearer token

Layers, each covering a different attacker; none substitutes for another.

1. **Off by default.** `--mcp` (env `SKINNY_MCP`).
2. **Bind literally `127.0.0.1`,** asserted on the socket at creation. `--mcp-port`
   takes a port number only, never a host component. One wrong default turns a local
   tool into a remote scene-and-filesystem control plane.
3. **Validate `Host`** against the loopback host and bound port. A DNS-rebound
   request carrying **no** `Origin` — `<script src>`, `<img>`, other no-CORS GETs
   send none — otherwise reaches the handler. The token still stops it, so this is
   defense in depth, but it is one line.
4. **Refuse any request carrying an `Origin` header.** *Known cost, accepted:* this
   also blocks the MCP Inspector and any browser-hosted MCP client. An earlier draft
   asserted "a real MCP client sends no `Origin`" — true for stdio and CLI clients,
   false for browser-based ones. Recorded as a cost, not asserted away.
5. **Bearer token.** `secrets.token_urlsafe(32)`, stored at `SETTINGS_DIR /
   "mcp_token"` (`settings.py:26`, `ensure_dirs()` `:41`) with mode `0600` applied at
   creation; env override `SKINNY_MCP_TOKEN`; compared with `hmac.compare_digest`
   (`==` short-circuits on the first differing byte and leaks the token to a local
   attacker who can time requests). Persistent across restarts so the client config
   stays valid.

The token defends against other local processes and against browser reach; it does
**not** defend against a process that can read the operator's home directory —
which is why layers 1-4 stay.

This deliberately does not inherit the existing web front-end's posture, where
`check_origin` returns `True` unconditionally (`web_app.py:359-360`) under
`allow_websocket_origin=["*"]` (`:766-770`).

**The startup line must not print the token.** It prints the substitution form, so
the secret does not land in terminal scrollback or CI logs:

```
claude mcp add --transport http skinny http://127.0.0.1:<port>/mcp \
  --header "Authorization: Bearer $(cat ~/.skinny/mcp_token)"
```

### D11 — Fixed default port, `--mcp-port` override, collision warns and continues

Both front-ends may host, so two `--mcp` instances collide. The second logs a
warning and **continues running without MCP** — it does not exit, and does not
silently pick another port.

*Alternative rejected: OS-assigned port written to `~/.skinny/mcp.json`.* Avoids
collisions, but a static client config cannot name a port it does not know, which
defeats copy-paste setup for the common single-instance case.

### D12 — Writes are rejected out of bounds, never clamped

`scene_set` validates against the published `metadata` bounds and **rejects** a
violation with an error quoting them. It does not clamp, and it skips the check
entirely when `metadata.get("growable")`.

The range tables are widget affordances, not legal bounds. `_add_float`
(`windows/scene_graph.py:484-500`) reads `metadata["growable"]` and sets the spin-box
max to `1e9`, **not** `hi` — the dock deliberately allows exceeding the published
max. Clamping would make the agent strictly less capable than the operator.
`_MATERIAL_FLOAT_RANGES["roughness"]` is `(0.04, 1.0)` (`scene_graph.py:114`), so a
legitimate `roughness=0.0` would be silently raised to `0.04` and quietly change the
render; `ior` caps at 3.0 (`:118`) though real materials exceed it.

Rejecting tells the agent; clamping is a silent lie.

## Risks / Trade-offs

- **Bind widens to `0.0.0.0` through a later refactor** → Assert loopback on the
  socket at creation; test fails on any non-loopback bind. Highest-consequence
  failure in the change.
- **Server thread's signal handlers clobber `MetalContext` teardown** → D9:
  `install_signal_handlers=False`, own socket, plus a test asserting the process
  SIGINT handler is unchanged after start.
- **A handler touches `Renderer` off-thread** → Non-deterministic GPU corruption
  that will not reproduce. Mitigation: all bodies posted; review rule "no
  `renderer.` outside a posted closure"; handler tests use a fake renderer with a
  manually run queue, as `tests/test_qt_render_session.py` already does.
- **Dock and MCP dispatch drift** → D3's single shared `apply_scene_property`. If
  the extraction is skipped, this risk is certain, not possible.
- **`scene_set` silently no-ops** on an unrouted `(path, property)` — the `camera`
  vs `renderer_camera` case does exactly this today → Return an explicit error
  naming path and property; never report success for an edit that did not happen.
  Silent no-ops are what send an agent into a retry loop.
- **Queue flooding under an agent value-sweep** → Not mitigated by coalescing: MCP
  writes are awaited (D2c), and `post_with_reply` (`render_session.py:583-590`) takes
  no `coalesce_key`. The round-trip itself paces the client — each write waits for
  the render thread before the next is issued — so the queue cannot grow unbounded
  from one client. The operator's own slider drags still coalesce through the dock's
  proxy verbs, keyed path-only for transforms (`:415`/`:421`), since a per-property
  key would lose updates when a TRS write recomposes from siblings.
- **Deadlock or hang on an awaited read** → Every wait carries a timeout returning
  an MCP error; the server thread is a daemon so it cannot hold up exit.
- **Renderer exceptions surface as transport 500s** → `remove_node` and friends
  raise `ValueError`/`RuntimeError` (`renderer.py:6176-6180`) which re-raise through
  `Future.result()` on the server thread. Map to MCP errors explicitly.
- **Token file world-readable** if mode is applied after creation → Create with the
  restrictive mode; verify mode on read and refuse a group/world-readable file.
- **Metal thermal rule** — no GPU work, no kernel added, dispatch-hygiene harness
  unaffected. The server holds no reference to the renderer or the context, only the
  proxy, so it cannot extend a `MetalContext` lifetime.

## Migration Plan

Additive and opt-in; nothing changes when `--mcp` is absent, which is the default.

The `render_session.py` relocation and the `run_pending` extraction are the only
touches to existing paths; `ui/qt/render_session.py` re-exports, so imports and the
Qt suite are unaffected. Extracting `apply_scene_property` out of the dock is
behavior-preserving and covered by the existing dock tests. Rollback is dropping the
flag; the relocation and extraction are inert on their own.

Docs on completion: `docs/PythonAPI.md`, `docs/Architecture.md`, `README.md`,
`CHANGELOG.md`.

## Open Questions

None blocking. Two earlier open questions are resolved:

- *Which transform verb* — answered by the code, not by a decision: do what the dock
  does, `set_transform` when `renderer._usd_stage is not None`, else
  `apply_instance_transform` (`windows/scene_graph.py:750-754`). Extracting the
  shared dispatch (D3) inherits this automatically.
- *Whether `scene_add` exposes a filesystem path* — moot; add/remove are cut from v1
  (D8), along with that threat surface.
