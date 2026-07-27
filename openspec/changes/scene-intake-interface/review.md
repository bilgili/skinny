# Design review — scene-intake-interface

Adversarial review, 2026-07-27, against the tree at `8247148`. Recorded rather
than folded. **Fold before implementing.**

**Verdict: survives, reshaped.** The two things that pay — deleting the lazy
imports and inverting the back-reference — need no `SceneUpdate` at all. The
value type as designed does not fit the streaming trigger.

## MAJOR

**M1 — D3 counts 3 change-token sites; there are 8 across 5 files, and the Qt
path is not one of them.** `ui/build_app_ui.py:361` is a *docstring*; the real
comparisons there are `:552` and `:565`. Full set: `build_app_ui.py:552,565`;
`ui/panel/windows.py:583,765`; `render_session.py:188` produces
`usd_scene_id=id(usd_scene)`, mirrored at `:379` and read by
`ui/qt/windows/bxdf.py:538` and `ui/qt/windows/material_graph.py:528`; plus two
**renderer-internal memo keys** — `renderer.py:6481` and `:10349,:10351`.

Cheaper option: the four UI sites fail **safe** under a swap (token changes →
rebuild). The unsafe ones are the two renderer memos, which memoize *skips*.
Re-key only those: 2 sites in 1 file instead of 8 in 5, and `SceneStateSnapshot`
need not grow a field. UI migration becomes its own change.

**M2 — The spec fixes an adoption order that two existing paths disagree on,
before the task meant to decide it runs.** `set_usd_scene` does
`_sync_volume_grid` (`:5453`) **before** `_gen_scene_materials` (`:5454`), with
a comment at `:5449-5452` explaining why. `_poll_usd_streaming` does the
opposite (`:4926` then `:4951`). Either streaming has a latent volume-σ bug or
the order is not load-bearing — resolve before it becomes a requirement.

The step list is also incomplete: the spec names 7, the three paths perform ≥14.
Missing: model label + `_usd_model_index` (`:2997-2999`, `:5434`), stage
adoption + `_attach_edit_layer` (`:5449`), `mm_per_unit` gated on the `!= 120.0`
sentinel (`:4943-4944` — absent from `set_usd_scene`, unconditional in
`create_empty_scene:5567`), `_override_to_orbit` seeding (`:4931`),
`_upload_usd_scene` + `_usd_uploaded_count` (`:4953`, `:4981`), scene-graph
attach vs build (`:4925` vs `:5612`), `populate_instance_refs` (`:4986`),
`_build_skinning_passes` at completion (`:5008`), and the version bumps
(`:5607`, `:5619`).

**M3 — Five adoption paths, not three.** `_refresh_usd_live_state` (`:5332`)
re-extracts all lights, every instance transform and the camera whenever
`_usd_live_dirty` is set (`:10489`); `_resync_instance_transforms` (`:5638`)
re-derives a subtree via `_world_transform`; `create_empty_scene` (`:5461`) is a
sixth entry composing resync. The requirement leaves these unowned.

**M4 — D2 applied uniformly silently changes animation behaviour.**
`_reextract_animated_lights` (`:5119`) replaces `scene.lights_dir` /
`lights_sphere` with freshly constructed lights (`usd_loader.py:1466`), and
`LightDir.enabled` defaults to `True` (`scene.py:231`). Today: disable a light,
press play, it comes back. Making runtime-state preservation a property of
applying an update fixes that — a user-visible change absent from "Unchanged".
Decide it in the spec.

**M5 — D2's runtime-state list is incomplete; mutate-in-place preserves fields
the 8-field copy never touches.** `Scene` has 11 fields (`scene.py:462-495`);
the resync copies 8 (`:5595-5602`). `mm_per_unit`, `film_max_component` and
`furnace_mode` survive *only* because the object is mutated — a swap takes all
three from the re-read. Needs a field-level replace/preserve policy.

Also: "keyed by source prim path with a fallback to name" is true only for
materials (`:5568-5569`); instances and lights key on `prim_path` alone with no
fallback (`:5545-5553`), so an empty `prim_path` loses its flag. Decide whether
that is a defect.

**M6 — D4 misses `resolve_control_binding`'s second caller and the
resolution-time question.** It is also called from `ui/build_app_ui.py:242-245`,
which needs a live **getter** per widget, and directly by
`tests/test_usd_controls.py:129-174` and `tests/test_headless_controls.py:101-103`.
Resolution time is semantic: `material:` resolves `mat_id` against the *current*
`_usd_scene.materials` at bind time (`usd_loader.py:2720-2726`), `usd:` captures
a live `Usd.Attribute` (`:2745-2747`). Resolve once at load and a later resync
that reorders materials leaves the widget editing the wrong material. The spec
must state when a description is resolved and re-resolved.

**M7 — "Off-thread build / on-thread apply" is not achievable as stated.**
`_bg_usd_stream` assigns `_usd_stage` (`:3016`), calls `_attach_edit_layer`
(`:3019` — sets the stage's edit target), then `_anim_index` (`:3031`), `clock`
(`:3032`), `_usd_up_axis_rt` (`:3033`), `_skeletal` (`:3038`), `_usd_controls`
(`:3039`), `_last_eval_time_code` (`:3040`), and builds the whole scene graph
(`:3053-3055`). Only `(scene, sg)` crosses the queue (`:3059`).

On the `SkeletalScene` open question: the stream thread builds a `UsdSkel.Cache`
and the render thread calls `compute_joint_matrices` on it every frame
(`:5235-5245`) with no synchronization beyond the GIL, while the scene-edit API
concurrently mutates the same stage. Close the question as: the skel handle
**and** stage ownership publish through the same single hand-off, and stage
mutation must not overlap a skel query.

**D1 answered — one `SceneUpdate` does not fit.** It would need ~12
trigger-conditioned fields: `trigger`, `instances_mode`,
`prebuilt_scene_graph`, `preserve_runtime_flags`, `frame_camera`,
`seed_usd_camera`, `adopt_mm_per_unit`, `model_label`,
`apply_control_defaults`, `inject_default_lights`, `refresh_camera_node`,
`upload`, `build_skinning_passes` — an `if trigger == …` chain wearing a
dataclass. And streaming cannot be one value at all: metadata → N append batches
→ completion, spanning seconds with the graph already on screen. Close the open
question as "a sequence", and prefer the cheaper shape: **three thin trigger
functions over one ordered `_adopt`**, which still states the order once.

**D6 confirmed** — deferring the customData merge-ordering fix to
`flat-material-field-table` is correct and cross-referenced both ways. But both
changes edit `usd_loader.py:631-745` and `:1224-1254`; add an explicit landing
order.

## MINOR

- "9 privates across 15 sites": 15 sites is right, but 2 (`:5479`, `:5551`)
  import the **public** `load_scene_from_stage`. Distinct privates = **10**;
  restate as "15 sites, 19 symbols, 10 private".
- `_usd_live_dirty` is `usd_loader.py:2757`; the `_usd_stage` read is `:2742`.
- `_reextract_animated_lights` is `renderer.py:5119`.
- `:5342` is inside `_refresh_usd_live_state` (def `:5332`), the live-edit
  refresh — **not** per-frame animation re-extraction, which is
  `_apply_animation_frame` (`:5155`).
- "95 module-level defs" → actual **73** (85 including nested).
- "three separate readers with their own merge orderings" → material readers are
  **two** (`:728`, `:1240`); the third (`:1481`) reads a *light* prim's spectral
  payload and performs no merge.
- "the renderer uses none of them": two of three adoption paths call the public
  `load_scene_from_stage`, and `headless.py:120` calls `load_scene_from_usd`.
  Soften to "the interactive path bypasses them".
- **`prepare_usd_streaming` (`usd_loader.py:2853`) has zero call sites** in
  `src/`, `tests/`, `scripts/` — streaming re-implements it inline — yet it is
  published in `docs/PythonAPI.md:541`. Delete it or make streaming call it; do
  not carry it into the new interface unexamined.
- Task 1.1's baseline diff is not runnable as written: `set_usd_scene` builds no
  scene graph (`:5424-5427`) and `_resync_geometry_from_stage` early-returns
  without an adopted scene + stage (`:5551-5554`). Needs a stated harness.
- Impact is short by four files: `render_session.py`, `ui/panel/windows.py`,
  `ui/qt/windows/{bxdf,material_graph}.py`, and the two control tests.
  `ui/build_app_ui.py` is listed only under "Watch" though D3 and D4 both
  modify it.
