# Tasks: scene-intake-interface

## 1. Baseline: what the three adoption paths actually do

- [x] 1.1 Run the same scene through `set_usd_scene`, the streaming poll, and
      `_resync_geometry_from_stage`; diff the resulting `Scene`, scene graph,
      and renderer state. Record every difference. → `baseline.md` step table.
- [x] 1.2 For each difference, decide and record: bug to fix, or deliberate
      per-trigger step that must survive as a field of `SceneUpdate`.
      → `baseline.md` verdict table.
- [x] 1.3 Capture per-frame animated light/camera values across a time-code
      sweep as a fixture — the identity target for the re-read call.
      → `tests/fixtures/anim_reread.json`, captured from a real headless Metal
      renderer by `tests/fixtures/_capture_anim_reread.py` (Z-up stage, so the
      up-axis rotation is exercised).

## 2. `SceneUpdate` value

- [x] 2.1 Define the update type covering instances, materials, lights,
      camera, volume, controls, skeletal bindings, film clamp, plus the
      per-trigger fields from 1.2. → `scene_intake.SceneUpdate`, with one
      constructor per trigger (`streamed` / `adopted` / `resynced` /
      `replacing`) so nothing fills the per-trigger flags by hand.
- [x] 2.2 Decide the `SkeletalScene` handle question (live pxr object inside a
      value vs a separate handle) and record it. → design D7: the update
      carries it, because it already carries the live stage.
- [x] 2.3 Hostless tests building updates from synthetic stages.
      → `tests/test_scene_intake.py`.

## 3. Invert the back-reference

- [x] 3.1 `resolve_control_binding` returns a description; the renderer
      applies it. Remove `usd_loader`'s import of `skinny.params` and its
      three writes into renderer state. → `scene_intake.ControlBinding` +
      `usd_controls.control_accessors`; the loader's version is deleted.
- [x] 3.2 Hostless test: USD-driven control behaviour unchanged.
      → `tests/test_usd_controls.py` (unchanged assertions, new seam) plus
      `TestIntakeHoldsNoRendererReference`, whose AST gate fails if the
      loader takes a renderer argument or reads a renderer attribute again.

## 4. Promote the privates, delete the lazy imports

- [x] 4.1 Promote the per-frame needs (camera/light re-read at time *t*, joint
      matrices, LBS, smooth normals) into the interface; verify against 1.3.
      → `read_at_time` + `deform_skinned_mesh`; verified against the captured
      fixture by `TestReadAtTimeMatchesPreChangeExtraction`.
- [x] 4.2 Fold the incidental ones (`_prim_has_mtlx_reference`, `_up_axis_rt`)
      into the update. → `_up_axis_rt` folds in as `SceneUpdate.up_axis_rt`;
      `_prim_has_mtlx_reference` cannot (design D9) and is promoted to public
      `prim_has_mtlx_reference` instead.
- [x] 4.3 Delete all 15 function-local imports; source gate asserts none
      return. → gate
      `test_no_function_local_imports_of_the_loader_anywhere_in_src`.

## 5. One application path

- [x] 5.1 Implement apply-update with the adoption order stated once.
      → `Renderer.apply_scene_update`; order pinned by
      `TestAdoptionOrder.test_full_load_runs_every_step_in_the_stated_order`.
- [x] 5.2 Move the runtime-state carry-over ("finding #7") into apply; test it
      directly. → `Renderer._carry_runtime_state_into` +
      `TestRuntimeStateSurvivesAStageReread`.
- [x] 5.3 Resolve the `id(_usd_scene)` change token: explicit version counter,
      and update the three `ui/build_app_ui.py` sites. → `scene_version`;
      there were **six** sites, not three (`build_app_ui.py` ×2,
      `ui/panel/windows.py` ×2, and two renderer-internal caches). Gate:
      `test_ui_sites_read_the_counter_not_the_object_id`.
- [x] 5.4 Route the streaming drain through updates; confirm off-thread build
      / render-thread apply is safe. → the background thread is now a pure
      producer: it calls `read_stage` + `bake_pending` and writes nothing to
      the renderer. Every renderer write happens in `apply_scene_update` on
      the render thread (design D8).
- [x] 5.5 Delete the three old adoption paths. → `set_usd_scene` and
      `_resync_geometry_from_stage` are now three-line callers of
      `apply_scene_update`; the streaming metadata phase is one call.
      `_reextract_animated_lights` is absorbed by `read_at_time`.

## 6. Gates

- [x] 6.1 `ruff check src/`; full hostless `pytest`. → ruff clean; 2693 passed,
      7 failed — all seven reproduce on `main` (6 pbrt-mtlx corpus imports +
      `test_all_ten_tools_are_advertised`), so the branch adds none.
- [x] 6.2 GPU: load a pbrt-imported scene, an animated scene, and a
      UsdSkel scene; images unchanged vs pre-change. → `tests/fixtures/`
      `_ab_scene_intake.py` renders all four adoption paths (synchronous
      pbrt import, streamed Z-up animated at three time codes, CPU skeletal at
      two, force-replace + two post-edit resyncs) on `main` and on the branch:
      **9/9 bit-identical, maxdiff 0.0**. Every case has lit pixels and the
      animated/skeletal pairs differ across time codes, so none is a vacuous
      black pass. The repo's UsdSkel tests all gate on an
      ElephantWithMonochord asset that is not checked in, so the harness
      authors its own minimal skinned stage rather than skipping.
- [x] 6.3 Scene-editing GPU tests (`test_scene_editing.py`,
      `test_usd_controls.py`, `test_scene_property_dispatch.py`) green.
      → 77 passed, with `test_usd_empty_stage`, `test_scene_add_primitive`,
      `test_headless_animation` and `test_headless_controls` added because the
      apply path owns what they exercise.
- [x] 6.4 Parity matrix dual gate unchanged. → 20 passed, 1 skipped,
      1 xfailed.
- [x] 6.5 Docs: `docs/Architecture.md` intake section; `docs/PythonAPI.md` if
      the public loader surface changed. → new **Scene intake** section in
      `Architecture.md` plus its animation/skinning/controls/headless
      cross-references, a new **6a** section in `PythonAPI.md`, the module
      tables in both and in `README.md`, and a `CHANGELOG.md` entry.
- [x] 6.6 `openspec validate scene-intake-interface --strict`. → valid.
