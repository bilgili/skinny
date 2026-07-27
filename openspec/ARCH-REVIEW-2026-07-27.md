# Architecture review — 2026-07-27

Ten deepening proposals, each adversarially design-reviewed against the tree at
`8247148`. Proposals are **as originally written**; each change directory
carries a `review.md` with the objections. Fold before implementing.

The earlier round (`arch-deepening-proposals`, 2026-07-23) produced 8 changes,
7 of which have landed. These are what those left behind.

## Cross-change findings

### `src/skinny/gfx/` is an abandoned backend abstraction

2,501 lines — `Device` / `Queue` / `CommandList` / `ComputePipeline` / `Buffer`
ABCs, a `BackendCaps` capability record, a full Vulkan implementation
(`gfx/vulkan/`, 1,159 lines), a `NotImplementedError` Metal stub. **Zero
importers** in `src/` or `tests/`. It defines its own `select_backend`
(`gfx/__init__.py:57`), colliding by name with `backend_select.select_backend`,
and its stated premise is false (`gfx/__init__.py:9`: "default is Vulkan
everywhere").

This is `gpu-backend-adapter`'s undisclosed predecessor and evidence that
"declare the interface first" has already stalled here once, at zero consumers.
**Resolve it before that change starts.**

### Live bugs the reviews surfaced, independent of any refactor

| bug | evidence |
|---|---|
| Web: an `mtlx.*` slider mutates `mtlx_overrides` while the render thread iterates it → `RuntimeError: dictionary changed size during iteration`; `_render_loop` has no `try`, so the render thread dies and the stream freezes permanently | `params.py:247-249` vs `renderer.py:10541`; `web_app.py:149-181` |
| Web: dome-light property edits are a silent no-op — Panel's copied dispatcher handles only `light_dir`/`light_sphere`, dropping `light_env` | `ui/panel/windows.py:434` vs `scene_edit_actions.py:91-95` |
| Web: `renderer.resize` runs from the IOLoop thread with no lock, destroying the offscreen image / readback / accum while the render thread may be inside `render_headless()` — `resize_render_target` is never set on the web `AppCallbacks` | `build_app_ui.py:279-281`, `web_app.py:534-546` |
| Qt: Animation transport (Play / Time / FPS) is already dead — those setters write through `renderer.clock`, and the proxy's `clock` is a local mirror, so nothing posts | `build_app_ui.py:196-207`, `render_session.py:290` |
| Settings: two front-ends erase five of each other's keys on exit; a top-level merge would still lose the `params` and `last_dirs` sub-dict entries | `settings.py:60-65`, `params.py:304-306`, `settings.py:70-83` |
| Headless screenshots composite the **previous** frame's HUD — `render_headless` never calls `_build_hud_bytes` and copies stale staging | `renderer.py:10876` vs `:10609` |
| Qt debug dock: `Key_D` → `show_dof_planes` missing; GLFW has the identical WASD conflict and binds it anyway | `debug_viewport.py:2333` vs `qt/windows/debug_viewport.py:344-347` |
| `render_headless`'s binding-1 rewrite is vestigial; two comments describe behaviour that does not exist | `renderer.py:10833-10846`, `:10830-10832`, `:4307-4309` |
| `vk_wavefront.py:597 REC_VERTEX_STRIDE = 76` is a hand-typed copy of a shader-derived value | `wavefront_layout.py:107` |
| `prepare_usd_streaming` has zero call sites but is published in the Python API docs | `usd_loader.py:2853`, `docs/PythonAPI.md:541` |
| `backend_select.py:16-19` docstring still says `auto` resolves to Vulkan everywhere, contradicting its own `select_backend` docstring and CLAUDE.md | — |

### Two structural lessons

**Derived beats captured.** `flat-material-field-table` proposed pinning lane
assignment from the current packer; the lanes are already declared in 24
`property` accessors in `common.slang:110-141`, which `slang_layout.py:105`
deliberately discards. Pinning one side from the other makes the gate vacuous.

**Check for the cheap version first.** `renderer-pure-core-extraction` proposed
moving 1,330 lines to escape a module-scope `import vulkan`; a lazy proxy plus
moving one import into `TYPE_CHECKING` achieves more, in ~2 lines, because it
also unskips the tests that need the `Renderer` class itself.

## Execution order

```
1  web-panel-dispatcher-fix     dome-light no-op, ~90 lines deleted   (from ui-spec review)
2  session-settings-owner       folded in place, ready
3  renderer-vk-lazy-import      ~2 lines                              (from pure-core review)
4  renderer-command-interface   dict-crash + resize race + dead Qt clock
5  gfx-disposition              revive or delete 2,501 dead lines
6  renderer-gpu-resource-set    shrunk
7  gpu-backend-adapter          split into 3-4; only after 5
8  choice-table-owners          shrunk; must not collapse the meta-gates
9  flat-material-field-table    rewrite on property-accessor derivation
10 scene-intake-interface       reshape: 3 trigger fns over one _adopt
–  frame-plan-split             fold into 7, or drop
```

Items 1–4 are the payload: small, near-hostless, and each fixes a live bug.

## Per-change verdicts

| change | verdict |
|---|---|
| `session-settings-owner` | survives; scope halved; **findings folded in place** — its `review.md` records the delta |
| `renderer-gpu-resource-set` | survives, shrinks. Not zero external callers — one test slices `renderer.py` source text using `def _build_metal_binds` as terminator |
| `renderer-command-interface` | survives, re-anchor. Headline exhibit is dead code; the real races are worse. Cut headless-through-queue |
| `scene-intake-interface` | survives, reshape. Five adoption paths, not three; one `SceneUpdate` needs ~12 trigger-conditioned fields |
| `choice-table-owners` | survives, shrinks. 11 divergences not 6; consolidation would make three build gates tautological |
| `flat-material-field-table` | premise wrong in three places — rewrite before scheduling |
| `gpu-backend-adapter` | blocked on `gfx/`; split into 3-4; capability record largely already exists as `ctx.supports_*` |
| `renderer-pure-core-extraction` | collapses to ~2 lines |
| `frame-plan-split` | duplication is 27 lines, not the block; fold into `gpu-backend-adapter` stage 4 or drop |
| `ui-spec-scene-properties` | drop as scoped — net **+100** lines, and its gate test cannot exist. Replaced by a ~90-line deletion |

## Report

The HTML candidate report this set came from was written to the session
scratchpad and is ephemeral: `architecture-review-20260727-0830.html`.
