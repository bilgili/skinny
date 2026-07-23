## 1. Shared seam: relocate the module and give the queue an executor

- [x] 1.1 Move `src/skinny/ui/qt/render_session.py` (whole module — queue **and**
      `QtRendererProxy`) to `src/skinny/render_session.py`. It has zero Qt imports, so
      this is a relocation, not a port.
- [x] 1.2 Re-export from `src/skinny/ui/qt/render_session.py` so existing imports and
      `tests/test_qt_*.py` keep working unchanged.
- [x] 1.3 Add `RenderCommandQueue.run_pending(target, on_error=None)` holding the
      execute-and-reply loop currently inlined at `ui/qt/viewport.py:200-209`
      (callback → `reply.set_result` / `reply.set_exception`).
- [x] 1.4 Switch `viewport.py:200-209` to `run_pending(self.renderer, on_error=self.error.emit)`.
- [x] 1.5 Test: an awaited command receives its result, and a raising callback delivers
      its exception, without waiting for a timeout.
- [x] 1.6 Test: `skinny.render_session` imports with no GUI toolkit loaded (assert
      `PySide6` absent from `sys.modules` after import).
- [x] 1.7 Run `tests/test_qt_render_session.py tests/test_qt_scene_state_proxy.py` to
      confirm the relocation is inert.

## 2. Extract the shared property-dispatch function

- [x] 2.1 Extract `SceneGraphDock._apply_property` (`windows/scene_graph.py:695-714`),
      `_apply_vec3_property` (`:716-746`), `_find_shader_material_ref` (`:756-770`),
      the transform-verb choice (`:750-754`) and the bool-toggle branch (`:455-470`)
      into one GUI-free `apply_scene_property(renderer, node, prop, value)` in
      `scene_graph.py` (or a sibling module beside `ui/scene_edit_actions.py`).
- [x] 2.2 Preserve every case verbatim: shader ancestor walk; vec3-on-camera fanning
      one vec3 into three scalar `apply_camera_param` calls via
      `metadata["camera_axis"]`; vec3-on-instance/light recomposing TRS from sibling
      properties; `_usd_stage is not None` → `set_transform` else
      `apply_instance_transform`; `metadata["toggle"]` → subtree vs node enable;
      `light_env` → `apply_dome_light_texture`.
- [x] 2.3 Make the unrouted case explicit — return a typed "no route" result instead of
      the current silent `return`, so callers can report it. Covers the `camera` vs
      `renderer_camera` kind gap (`scene_graph.py:219` vs `:895`), which no-ops today.
- [x] 2.4 Point the dock at the extracted function; confirm behavior is unchanged.
- [x] 2.5 Run the dock tests (`tests/test_qt_scene_graph_dock.py`) — the extraction must
      be behavior-preserving.
- [x] 2.6 Test the function directly against a fake renderer: material-on-shader routes
      via ancestor walk; one transform component recomposes from siblings; unrouted
      pair returns "no route" rather than silently succeeding.

## 3. GLFW front-end owns and runs a queue

- [x] 3.1 Construct a `RenderCommandQueue` in `app.py:main()` near the renderer
      (`app.py:586-620`) and expose it for the server to attach to.
- [x] 3.2 Call `run_pending(renderer)` at `app.py:689` — after `glfw.poll_events()`
      (`:688`), before `input_handler.update(dt)` (`:694`) and `renderer.update(dt)`
      (`:695`).
- [x] 3.3 Confirm the call is unconditional, not gated on MCP being enabled.
- [x] 3.4 Test: a mutation posted from another thread applies before the frame advances.

## 4. CLI flags and startup rejection

- [x] 4.1 Add `--mcp` (env `SKINNY_MCP`) and `--mcp-port` (env `SKINNY_MCP_PORT`) to
      `add_render_flags` (`cli_common.py:358`) behind a keyword-only `mcp: bool = True`
      group, following the `_env_flag` / `_env_int` pattern (`:43` / `:63`).
- [x] 4.2 `--mcp-port` accepts a bare port number; reject any value carrying a host or
      bind-address component.
- [x] 4.3 Add `reject_mcp_unsupported(...)` beside the existing rejections
      (`cli_common.py:194/212/277`): fail at startup with an install hint when MCP is
      enabled and the optional dependency is not importable.
- [x] 4.4 Pass `mcp=False` at **both** non-hosting call sites — `headless.py:364` and
      `web_app.py:717`. (`skinny-web` has a live renderer but guards it with a `Lock`,
      not a queue, so it cannot host the server.)
- [x] 4.5 Wire the flags into `app.py:main()` (`:509`) and `ui/qt/app.py:main()` (`:661`).
- [x] 4.6 Tests: default disabled; flag beats env; host-bearing port rejected; headless
      and web front-ends reject the flag as unrecognized.

## 5. Packaging

- [x] 5.1 Add an `mcp` extra to `[project.optional-dependencies]` (`pyproject.toml:62`)
      carrying the MCP server SDK and its transport.
- [x] 5.2 Confirm a default install pulls in nothing new and no startup-path module
      imports the SDK unconditionally.

## 6. Auth, bind, and process safety

- [x] 6.1 Token load-or-create at `SETTINGS_DIR / "mcp_token"` (`settings.py:26`,
      `ensure_dirs()` `:41`): `secrets.token_urlsafe(32)`, mode `0600` applied at
      creation, `SKINNY_MCP_TOKEN` env override.
- [x] 6.2 Verify stored mode on read; refuse a group- or world-readable token file.
- [x] 6.3 Auth with `hmac.compare_digest` against `Authorization: Bearer <token>`;
      unauthorized requests enqueue no renderer work.
- [x] 6.4 Refuse any request carrying an `Origin` header.
- [x] 6.5 Validate `Host` against the loopback host and bound port.
- [x] 6.6 Create the listening socket in the front-end; assert the bound address is
      loopback at creation; refuse to start otherwise.
- [x] 6.7 Hand the socket to the server runtime by descriptor, with signal-handler
      installation disabled, running the server on the thread's own event loop.
- [x] 6.8 Tests: missing/wrong token rejected; `Origin` present rejected even with a
      valid token; mismatched `Host` rejected with no `Origin`; token file `0600`;
      token survives a simulated restart; non-loopback bind refused.
- [x] 6.9 Test: the process's `SIGINT` and `SIGTERM` handlers are the same objects
      after the server starts — the `MetalContext` teardown chain
      (`metal_context.py:53`, `:77-85`) must survive.

## 7. Server lifecycle

- [x] 7.1 Start the server on a daemon thread when enabled, holding a
      `QtRendererProxy` — never the renderer or the GPU context.
- [x] 7.2 On bind collision (detected at 6.6, before any output), log a warning and
      continue with MCP disabled; do not exit, do not pick another port.
- [x] 7.3 Print the registration command with the bound port, referencing the token
      **file** — never the token value.
- [x] 7.4 Confirm clean process exit with the server running.
- [x] 7.5 Tests: collision leaves the front-end running with the server absent;
      registration line carries the bound port and no token value.

## 8. Tool handlers

- [x] 8.1 Handler contract: reads **and writes** go through `proxy.request` /
      `post_with_reply` with a timeout, because resolution + validation + dispatch must
      run on the render thread and the client needs a definitive applied-or-rejected
      answer. Writes therefore do not coalesce (`post_with_reply` takes no
      `coalesce_key`) — an accepted trade, documented in the module docstring. No
      `renderer.` access outside a posted closure.
- [x] 8.2 `scene_list(path, depth=2, kind=None)` — structure only (path, name,
      `type_name`, child count), depth-bounded, `kind` filtered on `RendererRef.kind`
      (`scene_graph.py:37`, populated by `populate_instance_refs` `:1004`).
- [x] 8.3 `scene_get(path)` — full properties for one node. `scene_graph_to_dict`
      (`:764-786`) already emits `editable` and `meta` per property; reuse it rather
      than re-deriving.
- [x] 8.4 `scene_set(path, property, value)` — resolve the node **once**, then call the
      shared `apply_scene_property` from task 2. Do not re-walk;
      `find_node_by_path` (`:1051-1058`) is an unindexed DFS running on the render
      thread between frames.
- [x] 8.5 Reject out-of-bounds writes with an error quoting the published bounds; do
      **not** clamp; skip the check when `metadata.get("growable")` — the dock itself
      raises the spin-box max to `1e9` for growable properties
      (`windows/scene_graph.py:484-500`).
- [x] 8.6 Explicit errors for an unresolved path and for a "no route" dispatch result;
      never report success for an edit that did not happen.
- [x] 8.7 Map renderer exceptions (`ValueError` / `RuntimeError`, e.g.
      `renderer.py:6176-6180`) arriving via `Future.result()` to MCP tool errors, not
      transport-level failures.
- [x] 8.8 Include **both** `scene_graph_version` and `material_version` in every tool
      result — property edits move only the latter (`renderer.py:8327`);
      `SceneStateSnapshot` already carries each (`render_session.py:180`, `:188`).
- [x] 8.9 Confirm only three tools are registered: no save/export, no add/remove, no
      image.

## 9. Handler tests

- [x] 9.1 Harness: fake renderer + real `RenderCommandQueue` run manually — the pattern
      `tests/test_qt_render_session.py` already uses. No GPU, no Qt, no network.
- [x] 9.2 `scene_list` carries no property payload; respects depth; `kind` filter
      returns only matching nodes.
- [x] 9.3 `scene_get` surfaces `editable` and range metadata.
- [x] 9.4 `scene_set`: material-on-shader routes via ancestor walk; transform component
      recomposes from siblings; out-of-bounds rejected with bounds quoted; growable
      accepted above max; unknown property and unknown path both error explicitly.
- [x] 9.5 Both version counters present in every result; material version changes after
      a property edit; structural version changes after a structural change.
- [x] 9.6 Timeout path: queue never run → read returns an error, server still serves the
      next request.
- [x] 9.7 Static check: no direct renderer access in the server module outside posted
      closures.

## 10. End-to-end verification

- [ ] 10.1 Live check on Metal (`--backend metal`): start `skinny-gui --mcp`, register
      with the printed command, enumerate, read a light, change its intensity, confirm
      the viewport updates and accumulation resets.
- [ ] 10.2 Repeat on `skinny --mcp` to confirm the shared seam.
- [ ] 10.3 Edit a material parameter on a shader prim end-to-end — the case the original
      `kind`-based dispatch would have failed.
- [ ] 10.4 Concurrency: drag a dock slider while the client writes; both apply, docks
      stay enabled, no rejection, no corruption.
- [ ] 10.5 Confirm no GPU regression with MCP disabled.

## 11. Docs and review gate

- [x] 11.1 `docs/Architecture.md` — module map for `skinny/render_session.py` and the
      server; extend the threading-seam section to cover the server thread.
- [x] 11.2 `docs/PythonAPI.md` — new public surface (`skinny.render_session`, the
      shared `apply_scene_property`, server entry point).
- [x] 11.3 `README.md` — `--mcp` / `--mcp-port`, the `[mcp]` extra, the registration
      command, and the v1 limits (no save, no add/remove, no image).
- [x] 11.4 `CHANGELOG.md` entry.
- [x] 11.5 `.venv/bin/ruff check src/` clean.
- [x] 11.6 Hostless suite green: `.venv/bin/pytest -m "not gpu"`.
- [ ] 11.7 Pre-merge codex review; fold findings in or consciously dismiss each.
