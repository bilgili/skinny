# Design review — renderer-command-interface

Adversarial review, 2026-07-27, against the tree at `8247148`. Recorded rather
than folded. **Fold before implementing.**

**Verdict: survives, must be re-anchored.** The thesis holds and the bugs are
worse than the proposal claims — but the headline exhibit is dead code, and one
of the two "already locked" methods is uncalled too.

## MAJOR

**M1 — The headline exhibit has no production caller.** `SkinnySession.set_param`
(`web_app.py:204-205`) is real, but its only callers are
`tests/test_web.py:115,124,135`. The browser never reaches it: the WebSocket
handler dispatches only `camera`, `control`, `autofocus`
(`web_app.py:428-435`), and the client sends only `autofocus` and `control`
(`web_templates/video_player.html:258,325`). Parameter edits arrive through the
Panel sidebar instead. Re-anchor the Why on `web_app.py:548` binding the shared
tree to the live renderer → `ui/panel/backend.py:239,263,286,304,324,356` →
`build_app_ui.py:148,174-179` → `_set_nested`. Then delete `set_param` or mark
it test-only.

**M2 — `SkinnySession.resize` is also uncalled, and the path production uses has
no lock at all.** The proposal cites `resize` as evidence the lock exists; it
does take the lock (`web_app.py:271-304`) and has no caller. The live path is the
sidebar resolution picker: `ui/panel/backend.py:469-516` → `node.on_apply` →
`build_app_ui.py:279-281`, which falls through to `renderer.resize(w, h)`
because `web_app.py:534-546` never sets `resize_render_target` on
`AppCallbacks` (the field exists, `build_app_ui.py:82`; Qt passes it,
`ui/qt/app.py:179`). `Renderer.resize` destroys and recreates the offscreen
image, readback buffer, accumulation image and HUD overlay
(`renderer.py:10989-11009`) — from the Tornado IOLoop thread, while
`_render_loop` may be inside `render_headless()` on those exact objects. **This
is the worst unsynchronised path in the file and the proposal does not mention
it.** One-line fix available today, independent of the queue.

**M3 — "Torn read" is the wrong failure mode; the two real ones are worse.**
- *Safe:* the store itself. `params.py:419` `setattr` and `:408,413`
  `overrides[field] = v` are single bytecodes. Scalar param writes cannot tear.
- *Hard crash:* `params.py:247-249`, the `mtlx_overrides` accumulation-state
  provider, iterates the live dict in Python on the render thread via
  `_current_state_hash()` (`renderer.py:10541`). An `mtlx.*` slider inserts a new
  key — plus 1–3 more through `_GANGED_MTLX_FIELDS` (`params.py:409-415,
  276-280`) — raising `RuntimeError: dictionary changed size during iteration`.
  `_render_loop` has no `try` (`web_app.py:149-181`), so the render thread dies,
  `_running` stays `True`, and **the video stream freezes permanently**. The
  whole Skin section is exposed.
- *Silent corruption:* a write landing between the hash at `:10541` and the
  uniform pack later in the frame renders the new value while `accum_frame`
  increments instead of resetting (`:10542-10551`) — blended into an
  accumulation built from the old value, wrong until an unrelated state change
  resets it.

The risk framing is not overstated; it is **understated and mis-mechanised**,
which is worse — a reviewer who knows the GIL would dismiss it.

**M4 — D3 is wrong: binding the tree to a proxy does not fix the six setters,
and it regresses working sections.**
- *(a) Sub-object writes are silently swallowed.* `build_app_ui.py:196,201,207`
  do `setattr(renderer.clock, …)` / `renderer.clock.set_normalized(…)`.
  `QtRendererProxy.clock` is a local `PlaybackClock` installed with
  `object.__setattr__` (`render_session.py:290`); those calls mutate the mirror
  and post nothing. **The Qt Animation transport is already dead for this
  reason**; D3 would extend the bug to the web.
- *(b) Missing verbs turn sections into error alerts.* `build_app_ui.py:404,410`
  call `toggle_material_furnace` and `iter_graph_uniforms`, which exist only on
  `Renderer` (`renderer.py:7070`, `:6985`); the proxy's `__getattr__` raises
  (`render_session.py:540-543`), and the dynamic-section build is
  `try/except`-wrapped in both backends — so the web Materials section degrades
  to "Build failed: …".
- *(c) No snapshot feeder exists on the web.* The proxy's reads come from
  `apply_snapshot` (`render_session.py:323-337`) and `apply_scene_state`
  (`:363-377`). Qt emits `state_ready` once before the loop
  (`ui/qt/viewport.py:211`); the web has neither. Bound to a bare proxy,
  `_usd_scene` stays `None` and `_usd_controls` stays `[]`, so Materials,
  Animation and Scene Controls never populate after the async USD load. Today
  they do.

Drop "without touching `backend.py` at all — which is the lazy and correct fix".
Add tasks for a per-frame snapshot producer in `_render_loop`, a scene-state
refresh, the missing proxy verbs, and a rule that no shared-tree setter may write
through a sub-object. (a) is a live Qt bug deserving its own change.

**M5 — D4 and D1 contradict: `post_with_reply` cannot coalesce.**
It takes no `coalesce_key` and unconditionally appends
(`render_session.py:597-604`); `mcp_server.py:44-47` states the consequence —
"writes cannot coalesce … a client value-sweep is paced by the round-trip". A
slider drag through reply-carrying commands is one round-trip per pixel. Say
replies are *available* for operations whose outcome a caller consumes (load,
resize, screenshot, structural edits) while streaming setters keep coalescing
`post` — which is what Qt and MCP already do jointly.

**M6 — Making `resize` a posted command inverts an ordering the current code
deliberately guarantees.** `web_app.py:277-303` holds the lock across
`renderer.resize` → encoder rebuild → stale-frame drain → `send_resize` /
`send_codec_config`, with a comment at `:293-296` explaining that the WS writes
are scheduled *before* the render thread is unblocked, so the browser never
decodes a new-dimension H264 packet with the old decoder config. Post-with-reply
moves only `renderer.resize` onto the render thread, which can then push a
new-resolution frame while the poster is still rebuilding the encoder.
General rule: any operation whose correctness depends on nothing being rendered
between its steps must be **one** command, not a sequence of awaited ones.

**M7 — Reads race too, and the gate only covers writes.** The Panel pull loop
reads the live renderer every 200 ms on the IOLoop thread
(`ui/panel/backend.py:78-84,105-110`), and `mcp_server.py:33-35` already records
why that is unsafe: "*Renderer* has no internal lock, and its scene graph is
rebuilt and swapped by the streaming load thread, so *reads* race too." Qt solved
reads with `RendererStateSnapshot`; the generalisation drops that half. Extend
the requirement and gate 4.4 to off-thread reads, and make the snapshot the
sanctioned read path — which M4(c) requires anyway.

**M8 — `ui/panel/windows.py` submits GPU work from the IOLoop thread and is not
in Impact.** `windows.py:950-965` constructs a `DebugViewport` against the
session's GPU context and `:971-976` calls `dv.render_embedded(renderer)` from a
`pn.state.add_periodic_callback` — GPU submission from a non-owning thread,
serialised *only* by `session._lock`, which D1 demotes to the render/encode
span. Twenty-plus other `with session._lock:` sites in that file are in the same
position. Add the file to Impact plus a conversion task, or state that the lock
retains its serialising role for these panes until a follow-on.

**M9 — D2 (headless through the queue) rests on a false premise; cut it.**
"The same setter code runs under all four front-ends" — `HeadlessRenderer` never
calls `build_main_ui`, has no widget tree, and sets attributes directly in
`_prepare` (`headless.py:203-224`). There is no shared setter to co-verify, and
`headless.py` creates no thread at all.
*For:* a future async headless would need no second path. *Against:* it adds a
queue, a drain and a post-per-attribute to a nine-line `_prepare` whose caller
cannot be off-thread by construction — and the gate that proves it correct
(identical images) proves only that the indirection changed nothing.
**Verdict: over-engineering.** Replace with a one-line invariant in
`headless.py`'s docstring: the caller thread owns the renderer and mutates
directly; a future off-thread caller uses the queue.

**M10 — "Untestable, so none of this is covered" is false.**
`tests/test_web.py` has 47 tests including `TestSkinnySession`, and `set_param`
is covered three times (`:111,120,131`). The real problem is
`pytestmark = pytest.mark.gpu` at `:35` plus `needs_vulkan` at `:34`, which
exclude the module from the hostless sweep. Restate as: the paths that need
coverage are stub-drivable and should be split into a non-GPU module.

## MINOR

- `handle_camera` (`web_app.py:207-233`) is omitted from the Why's
  unsynchronised set although tasks include it. It takes no lock and mutates the
  live camera on every mouse drag — far more frequent than any parameter edit.
- `handle_camera`, `handle_control` and `handle_autofocus` call
  `self.encoder.force_keyframe()` **outside** the lock (`:232-233`, `:244-245`,
  `:268-269`) while `_render_loop` uses the encoder inside it (`:161-168`) — a
  real race on a non-renderer object that gate 4.4 as worded would not catch.
- D5 mis-diagnoses the button synthesis. `web_app.py:516-533` targets the
  pane-local `DebugViewport`, not the renderer, and the pane's own handlers
  already take the lock (`windows.py:987-999`). It exists because the
  `_debug_view` closures are built at `:540-542`, before the pane and its
  `dv_holder` exist. Four-line fix, no queue involved.
- There is no Bokeh "worker thread": `pn.serve` starts without threading
  (`web_app.py:738-751`), so widget callbacks, periodic pulls and WS handlers all
  run on the single Tornado IOLoop. It matters — the setters never race *each
  other*, only `_render_loop`.
- "Uses a different scene-ingest path" is correct but belongs entirely to
  `scene-intake-interface`.
- `AppCallbacks.load_hdr` (`build_app_ui.py:81`) is passed by web
  (`web_app.py:545`) and consumed by nothing. Dead field; delete while in the
  area.
