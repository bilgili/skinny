# Baseline: frame-plan-split

This document records tasks 1.1, 1.2 and 1.3. It states what the per-frame path
does today, which orderings are load-bearing, and where the windowed and
headless paths genuinely differ.

Line numbers refer to `src/skinny/renderer.py` at commit `b58ee49`.

## 1.1 — The steps and their ordering dependencies

### `update(dt)` (`:8523`-`8619`)

| # | Step | Ordering dependency |
|---|------|---------------------|
| 1 | `time_elapsed += dt`, `frame_index += 1` | none |
| 2 | `_emit_config_matrix()` | none (dedup-guarded) |
| 3 | FPS smoothing | none |
| 4 | `_poll_usd_streaming()` | **LOAD-BEARING.** Applies new USD metadata before the snapshot. Makes the authority transition atomic in one frame. |
| 5 | `clock.advance(dt)` | **LOAD-BEARING.** Must precede step 6. |
| 6 | `_apply_animation_frame()` | **LOAD-BEARING.** Must precede every read of `_usd_scene` below. |
| 7 | `_refresh_usd_live_state()` when `_usd_live_dirty` | **LOAD-BEARING.** A `usd:` control edit must land before the snapshot. |
| 8 | `_update_light()` | **LOAD-BEARING.** Must precede step 9, which reads the light state. |
| 9 | `scene = _build_scene_from_state()` | **LOAD-BEARING.** Consumes steps 4-8. Produces the snapshot steps 10-13 read. |
| 10 | `_ensure_default_scene_graph()` when `_scene_graph is None` | after step 9 |
| 11 | `_inject_default_lights_into_scene_graph()` on authority flip | after step 10 (needs the graph) |
| 12 | `_upload_distant_lights(...)`, gated on `uses_default_lights` | **LOAD-BEARING.** After step 9: the gate reads the rebuilt authority. |
| 13 | `_sync_auxiliary_light_authority()` | after step 12 |
| 14 | `_ensure_env_uploaded()` | none |
| 15 | `_rebake_if_needed(time.monotonic())` | **LOAD-BEARING.** Wall-clock, not frame time — this is what debounces a slider drag. |
| 16 | `_ensure_tattoo_uploaded()` | none |
| 17 | `_upload_material_types()` on scatter change | none |
| 18 | state hash → `accum_frame` reset or increment, plus the BDPT splat zero | **LOAD-BEARING.** Last. Must observe every mutation of steps 1-17. |
| 19 | `_refresh_gizmo_segments()` | after step 9 (reads camera and instances) |

Step 18 owns the accumulation decision. Its hash comes from the `params.py`
registry (change `param-registry-accumulation-reset`). This change does not
re-derive it.

### `render()` (`:8621`-`8861`), Vulkan arm

1. `_backend_render_ready` guard — early return.
2. `is_metal` branch — early return through `_render_windowed_metal()`.
3. `f = current_frame`.
4. `vkWaitForFences`.
5. `poll_pick_result()` — **LOAD-BEARING.** Must precede step 7 so a satisfied
   pick disarms in this frame's uniform buffer.
6. `vkAcquireNextImageKHR`.
7. `_pack_uniforms()` → `uniform_buffer.upload`.
8. `_pack_mtlx_skin_array()` → `mtlx_skin_buffer.upload_sync` when non-empty.
9. Read swapchain extent and image.
10. Reset and begin the command buffer.
11. Cross-frame accumulation memory barrier.
12. `_sync_hud_overlay(cmd)`.
13. Execution-mode gate → wavefront record or megakernel bind plus dispatch.
14. Barrier: offscreen `GENERAL`→`TRANSFER_SRC` **and** swapchain
    `UNDEFINED`→`TRANSFER_DST`, in one call.
15. `vkCmdBlitImage` offscreen → swapchain.
16. Barrier: offscreen `TRANSFER_SRC`→`GENERAL`.
17. Barrier: swapchain `TRANSFER_DST`→`PRESENT_SRC`.
18. `vkEndCommandBuffer`.
19. `vkResetFences` — **LOAD-BEARING.** Immediately before the submit that
    signals the fence, never earlier. An early reset leaves the fence
    unsignalled across every step that can raise, so a caller that catches and
    retries blocks forever (codex pre-merge review, finding 1).
20. `vkQueueSubmit` with wait `image_available`, signal `render_finished`.
21. `vkQueuePresentKHR`.
22. `_online_frame_end_swap()` when `_online_training`.
23. `current_frame = (f + 1) % MAX_FRAMES_IN_FLIGHT`.

## 1.2 — `render()` against `render_headless()`

`render_headless` (`:8863`-`9019`) runs the same 23 steps with these
differences. Everything not listed is character-identical.

| Region | `render()` | `render_headless()` | Verdict |
|--------|-----------|--------------------|---------|
| Not-ready guard | returns `None` | returns a zeroed RGBA8 frame | genuine target difference |
| Pick drain position | after the fence wait | **before** the fence wait | **not** a target difference — see below |
| Swapchain acquire | yes | none | genuine target difference |
| Post-dispatch barrier | 2 image barriers in one call (offscreen→`TRANSFER_SRC`, swapchain→`TRANSFER_DST`) | 1 image barrier (offscreen→`TRANSFER_SRC`) | the offscreen barrier is shared; the swapchain barrier is the target's |
| Output record | `vkCmdBlitImage` to the swapchain | `_readback.record_copy_from` | genuine target difference |
| Restore barrier | offscreen→`GENERAL` | offscreen→`GENERAL` | identical |
| Extra tail barrier | swapchain→`PRESENT_SRC` | none | genuine target difference |
| Submit | waits `image_available`, signals `render_finished` | no semaphores | genuine target difference |
| After submit | present | wait the fence, then read back | genuine target difference |
| Return | `None` | `_readback.read()` | genuine target difference |

**The pick-drain position is safe to unify on the windowed order.**
`poll_pick_result` performs no GPU synchronisation of its own. It counts frames
and reads `tool_buffer` only after `MAX_FRAMES_IN_FLIGHT + 1` calls. In the
headless path the previous call already waited the same fence at its tail
(`:9008`), so the fence is signalled on entry and `vkWaitForFences` returns
immediately. Moving the drain after that wait therefore reads the identical
buffer contents. The reverse move — draining before the wait in the windowed
path — would read the tool buffer while frame `f` is still in flight. This
change unifies on **fence wait, then pick drain**: identical behaviour for
headless, unchanged behaviour for windowed.

### The four Metal entry points

`_render_headless_metal` (`:8003`) and `_render_windowed_metal` (`:8012`) both
call `_render_scene_metal` (`:7904`), which calls `_render_wavefront_metal`
(`:7926`) or `_render_megakernel_metal` (`:7855`). The barrier and dispatch
block is **not** duplicated on Metal — resources bind at dispatch and there is
no command-buffer machinery. The Metal duplication is a duplication of
*decisions*, not of recording:

- `_render_scene_metal` re-derives the execution mode and the integrator name
  from `integrator_index`, exactly as `_record_wavefront_dispatch` does for
  Vulkan.
- `_render_wavefront_metal` re-derives the MLT bootstrap condition
  (`accum_frame == 0 or not staged.seeded`) and the SPPM first-frame flag,
  exactly as `_record_wavefront_dispatch` does.
- `_metal_megakernel_bands` and `_mlt_metal_chain_batch` branch on `is_metal`.

The plan removes those re-derivations. The pick drain sits in
`_render_windowed_metal` for the windowed path and in `render_headless` for the
headless path — the same asymmetry the Vulkan side has.

## Merge-time collision — `docs-split-large-docs`

`docs-split-large-docs` merged to `main` while this change was in development and
split `docs/Architecture.md` from 2639 lines into 364 plus 23 subject documents.
This change had added its per-frame section to the monolithic file, so the merge
conflicted over almost the whole document. Resolution: take `main`'s hub verbatim
and re-home the section in **`docs/HostModules.md`**, whose stated subject —
host module map and ownership, already holding `bringup.py`, the renderer
carve-out pattern and the device-free pure core — is exactly where a carve-out
stage that produces a device-free module belongs. `openspec/config.yaml` is the
authoritative ownership map; read it rather than guessing.

Lesson, and it is the same one `scene-intake-interface` recorded: run
`git log <branch-point>..main` **before** merging, not after. A doc reorganisation
on `main` invalidates a doc edit the same way a spun-off code fix invalidates a
fixture.

## 1.4 — The identity target, and what "identical" can mean here

The full matrix ran before and after on the same host (Metal, 128²-256²). Every
printed per-combo metric was compared digit for digit by a comparator whose
negative control (one mutated digit in 400 lines) is checked.

| Comparison | Shared metric lines | Digit-identical | Differ |
|---|---|---|---|
| Vulkan, before vs after (×3 runs) | 73 | **73** | **0** |
| Vulkan, final branch vs main | 73 | 70 | 3 |
| Vulkan, **main vs itself** (control) | 73 | 70 | **the same 3** |
| Metal, before vs after | 1250 | 1207 | 43 |
| Metal, heavy scenes before vs after | 20 | **20** | **0** |
| Metal, after vs after — **same tree, run twice** | 256 | 242 | 14 |

**Vulkan `bdpt|wavefront|spectral` is nondeterministic too — measured, not
assumed.** Three of 73 lines differed on the final branch-vs-main comparison, so
`main` was rendered **twice against itself** as a control: the same three keys
differed, in the same combo family, at the same one-digit magnitude
(`noiseσ 0.006049→0.00605`, `FLIP 0.008708→0.008709`, `PSNR 32.87→32.86`, with
relMSE / MSE / MAE / var identical). Three earlier Vulkan comparisons had come
back 73/73, so the family is *rarely* divergent under Vulkan and *reliably*
divergent under Metal — the reason the control matters is that "identical three
times running" is not the same as "deterministic".

**The Vulkan result is the one that gates this change.** `_execute_vulkan_frame`
is what the split rewrote, and on this Metal host every Metal render takes the
`is_metal` arm and never enters it. The Vulkan subset covers 18 combos —
path/bdpt/sppm/mlt × megakernel/wavefront × spectral × env/neural/restir-di —
and is exactly identical.

**Three Metal combo families are not reproducible run to run**, on an unchanged
tree: `bdpt|wavefront|spectral`, `path|wavefront|restir-di` and
`path|wavefront|neural`. Two runs of the *same* post-change tree differ at a
**higher** rate (14/256) than before-vs-after does (43/1250), and the
run-to-run set includes a family that before-vs-after does not flag at all. The
differences are fourth-significant-digit only (`PSNR 32.87→32.86`,
`noiseσ 0.006052→0.006047`); relMSE agrees to four significant figures in every
case. `samp_many_lights` and `samp_many_lights_mtlx` hold the same content, and
their `var` values *swapped* between runs — a systematic change would move both
the same way. The same families are exactly identical under Vulkan, which
places the cause on the Metal side (relaxed-order atomics and the CPU
slot-count-readback fallback for indirect dispatch), not in this change.

So the identity assertion this change can honestly make is: **identical
everywhere the backend is reproducible, and within the tree's own measured
run-to-run noise everywhere it is not.** Quoting an unqualified "identical" for
those three Metal families would be quoting luck.

Two controls worth keeping:

- The one pre-existing failure, `conductor_infinite_mtlx`, reproduces
  bit-identically before and after — `relmse=0.1126902596622648`,
  `flip=0.09440565172684953` to sixteen digits. It is a `-mtlx` export mismatch
  with no connection to the frame path.
- The first after-run showed 20 missing metric lines and four extra skips. That
  was **not** the code: `dragon`, `clouds`, `bunny_cloud` and `disney_cloud`
  resolve to `usd:` assets under `assets/`, which is gitignored, so a fresh
  worktree has 19 of the primary checkout's 79 entries and those four scenes
  failed to load. The gates swallow a load failure into
  `pytest.skip("render backend unavailable")`, so the hole read as a skip count
  rather than an error. Symlinking the assets and re-running gave the 20
  identical lines above. **A fresh worktree needs `assets/` linked before the
  parity matrix means anything** — the same scene-data hole recorded in change
  `parity-scene-asset-integrity`.

## Codex pre-merge review — six findings, all fixed

The review returned "not fully behaviour-preserving", and it was right. The
common cause of the two HIGH findings is one design error: **a snapshot taken
before the state it describes stops changing.**

1. **HIGH — the plan was derived before the pick drain.** `poll_pick_result`
   runs pick callbacks, and `_on_autofocus_hit` sets `accum_frame = 0`. A plan
   derived first carried a stale `first_frame` while `_pack_uniforms` packed the
   mutated live `fc.accumFrame` — two readings of one value, on a frame where
   SPPM would then skip its photon-accumulator clear. The pre-split code had no
   snapshot, so it could not disagree with itself. Fixed: every path drains, then
   derives. Gated by `test_the_plan_is_derived_after_the_pick_drain`.
2. **HIGH — `target.submit_info()` was evaluated after `vkResetFences`.** The
   hook sits inside the reset/submit critical region, so a raise there leaves the
   fence unsignalled and a retrying caller blocks forever — the exact freeze
   `review-surfaced-defects` finding 1 exists to prevent, and my code satisfied
   its test textually while violating its intent. Fixed: build the submit
   descriptor first. `FENCE_RESET` is now a plan step with two invariants
   (`END_CMD → FENCE_RESET → SUBMIT`).
3. **HIGH — `plan.steps` was descriptive, not enforced.** Nothing replayed it, so
   the plan and the executor were two unreconciled authorities on order, and
   `check_invariants` silently skips a rule when a step is absent. Replaying it
   for real is task 4.3's recording adapter, which has not landed; until then
   `test_plan_step_order_matches_the_executor_source_order` pins the plan's order
   to the executor's actual source order, and the missing `FENCE_RESET` step is
   added.
4. **MEDIUM — the weight swap became a start-of-frame snapshot**, deferring an
   OFF→ON transition by a frame. Arming online training is a frame-END decision,
   so `online_swap` is **removed from the plan** and the swap sites read
   `_online_training` live, as before. A field that cannot be authoritative does
   not belong in the plan.
5. **MEDIUM — `plan.target` and the concrete target object were unchecked
   against each other.** `_execute_vulkan_frame` now asserts they agree.
6. **MEDIUM — the Metal MLT bootstrap re-derived `_mlt_metal_chain_batch()`**
   while the mutation dispatch used `plan.mlt_chain_batch`, so the phases could
   disagree if the environment changed mid-frame. The plan now owns the batch for
   all three phases.

The review also noted (LOW) that the headless pick-drain move is a **correctness
fix, not a preservation**: `render_headless` is documented to work in a windowed
session, where the previous call may have been `render()`, which does not wait the
fence at its tail — so the old pre-wait drain could read the tool buffer while a
frame was in flight. §1.2 above claims equivalence on the assumption that the
previous call was also headless; that assumption holds only in a headless
session. Waiting first is strictly safer in both.

After the fixes: Vulkan parity is **still 73/73 identical** to `main`, hostless
is 2877 passed with the same 7 pre-existing failures, and the Metal kill harness
plus the Metal frame-path smoke are green.

## 1.3 — The per-call binding rewrite in `render_headless`

**Verdict: already removed. It was dead compensation, not a target difference
and not a live bug.**

`render_headless` used to rewrite descriptor binding 1 to `_offscreen_output`
on every call. The comment said it restored a binding that windowed `render()`
had pointed at the acquired swapchain image. `render()` stopped doing that
rebind when it moved to an offscreen-plus-blit design, so the restore
compensated for a write that no longer happened.

Commit `9e6322d` (change `review-surfaced-defects`) deleted the rewrite and
corrected the docstring. Binding 1 now points at `_offscreen_output` for the
whole session — written at init and again on resize, and owned by
`gpu_resources.SceneResourceSet` since change `renderer-gpu-resource-set`.

There is therefore nothing to move and no separate fix to announce. This
change inherits a headless path that already differs from the windowed path
only in its target.
