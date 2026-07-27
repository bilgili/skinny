# Tasks: review-surfaced-defects

Each numbered group is independently landable. 1–4 are worth doing soon; 5–7 are
hygiene.

## 1. Web render thread cannot be killed by a parameter edit

- [ ] 1.1 Reproduce: drive an `mtlx.*` slider against a running web session and
      capture the `RuntimeError: dictionary changed size during iteration` from
      the accumulation-state provider. Record the traceback in the change.
- [ ] 1.2 Marshal the write — the parameter path from the web front-end posts to
      the session's render-thread queue instead of calling `_set_nested` on the
      live renderer.
- [ ] 1.3 Guard `_render_loop`'s body so no exception retires the thread while
      the session stays marked running: report it, and either continue or tear
      the session down visibly.
- [ ] 1.4 Hostless test against a stub renderer: a key-inserting mutation posted
      from another thread while the loop computes state does not raise, and a
      command that does raise leaves the loop alive and the failure reported.
- [ ] 1.5 Measure a slider drag before and after; coalescing should make the
      posted path no worse.

## 2. Resolution changes are marshalled on the web

- [ ] 2.1 Set `resize_render_target` on the web `AppCallbacks`, routed to the
      session's existing `resize` method — **not** to `renderer.resize` — so the
      lock-held ordering of resize → encoder rebuild → frame drain → WebSocket
      notify is preserved.
- [ ] 2.2 Decide and record: should `_add_resolution`'s direct-renderer fallback
      raise when no callback is supplied? Sweep the call sites first.
- [ ] 2.3 Manual: change resolution mid-render from a browser; no torn frame, no
      decoder-config mismatch.

## 3. Qt animation transport works

- [ ] 3.1 Make the Play / Time / FPS setters reach the owning thread — either
      the proxy gains clock verbs that post, or the setters post directly.
      Whichever, a write through a renderer-owned sub-object must not be
      absorbed by a proxy-local copy.
- [ ] 3.2 Audit the proxy's other locally-held objects (`film`, `scene_graph`)
      for the same hazard; record findings.
- [ ] 3.3 Hostless test: setting playback state through the proxy posts a
      command.
- [ ] 3.4 Manual: press Play in `skinny-gui` and confirm the animation advances.

## 4. Dome-light edits work on web

- [ ] 4.1 Diff the Panel dispatcher copy against the shared
      `apply_scene_property` per reference kind; record every difference as
      intended or accidental before deleting.
- [ ] 4.2 Delete `_apply_prop_value`, `_apply_vec3_value` and
      `_find_material_ancestor` from the Panel window; call the shared functions
      from the six call sites; surface the returned reason string in the status
      line. The two re-inlined fan-out guards go with them.
- [ ] 4.3 Hostless test: a `light_env` property edit routes to the environment
      light override, and an unroutable edit yields a reason rather than `None`.
- [ ] 4.4 Manual: edit a dome-light property in `skinny-web` and see it apply.

## 5. Camera Debug key maps

- [ ] 5.1 Tabulate the GLFW, Qt-viewport, Qt-debug-dock and web key maps.
- [ ] 5.2 Bind `Key_D` → depth-of-field planes in the Qt debug dock, serving the
      movement and toggle uses from the separate channels that already exist
      there.
- [ ] 5.3 For every remaining divergence: fix, or record with its reason and pin
      the recorded set by test.

## 6. HUD is current in offscreen output

- [ ] 6.1 Fill the HUD staging buffer on the offscreen path, as the windowed
      path does before copying.
- [ ] 6.2 Test: an offscreen frame taken after the HUD text changes shows the
      new text; a path that copies without filling fails.
- [ ] 6.3 Confirm the empty-HUD early-return still means headless renders that
      set no text are byte-unchanged.

## 7. Stale claims

- [ ] 7.1 Delete the vestigial binding-1 rewrite in the offscreen path and the
      two comments at the rewrite and at descriptor creation that describe the
      windowed path as rebinding rather than blitting.
- [ ] 7.2 Source `REC_VERTEX_STRIDE` from the derived wavefront layout instead
      of the hand-typed literal; confirm the value is unchanged.
- [ ] 7.3 `prepare_usd_streaming`: delete it and remove it from
      `docs/PythonAPI.md`, or make the streaming path call it. Record which and
      why.
- [ ] 7.4 Correct the `backend_select` module docstring to match
      `select_backend` and CLAUDE.md.

## 8. Gates

- [ ] 8.1 `ruff check src/`; full hostless `pytest`.
- [ ] 8.2 GPU smoke: one windowed and one offscreen render per backend; images
      unchanged except for the HUD fix in 6.
- [ ] 8.3 Parity matrix dual gate unchanged.
- [ ] 8.4 `CHANGELOG.md`: the web freeze, the dome-light no-op, the dead Qt
      transport, and the stale HUD are all user-visible fixes.
- [ ] 8.5 `openspec validate review-surfaced-defects --strict`.
