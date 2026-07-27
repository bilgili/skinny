# Tasks: scene-intake-interface

## 1. Baseline: what the three adoption paths actually do

- [ ] 1.1 Run the same scene through `set_usd_scene`, the streaming poll, and
      `_resync_geometry_from_stage`; diff the resulting `Scene`, scene graph,
      and renderer state. Record every difference.
- [ ] 1.2 For each difference, decide and record: bug to fix, or deliberate
      per-trigger step that must survive as a field of `SceneUpdate`.
- [ ] 1.3 Capture per-frame animated light/camera values across a time-code
      sweep as a fixture — the identity target for the re-read call.

## 2. `SceneUpdate` value

- [ ] 2.1 Define the update type covering instances, materials, lights,
      camera, volume, controls, skeletal bindings, film clamp, plus the
      per-trigger fields from 1.2.
- [ ] 2.2 Decide the `SkeletalScene` handle question (live pxr object inside a
      value vs a separate handle) and record it.
- [ ] 2.3 Hostless tests building updates from synthetic stages.

## 3. Invert the back-reference

- [ ] 3.1 `resolve_control_binding` returns a description; the renderer
      applies it. Remove `usd_loader`'s import of `skinny.params` and its
      three writes into renderer state.
- [ ] 3.2 Hostless test: USD-driven control behaviour unchanged.

## 4. Promote the privates, delete the lazy imports

- [ ] 4.1 Promote the per-frame needs (camera/light re-read at time *t*, joint
      matrices, LBS, smooth normals) into the interface; verify against 1.3.
- [ ] 4.2 Fold the incidental ones (`_prim_has_mtlx_reference`, `_up_axis_rt`)
      into the update.
- [ ] 4.3 Delete all 15 function-local imports; source gate asserts none
      return.

## 5. One application path

- [ ] 5.1 Implement apply-update with the adoption order stated once.
- [ ] 5.2 Move the runtime-state carry-over ("finding #7") into apply; test it
      directly.
- [ ] 5.3 Resolve the `id(_usd_scene)` change token: explicit version counter,
      and update the three `ui/build_app_ui.py` sites.
- [ ] 5.4 Route the streaming drain through updates; confirm off-thread build
      / render-thread apply is safe.
- [ ] 5.5 Delete the three old adoption paths.

## 6. Gates

- [ ] 6.1 `ruff check src/`; full hostless `pytest`.
- [ ] 6.2 GPU: load a pbrt-imported scene, an animated scene, and a
      UsdSkel scene; images unchanged vs pre-change.
- [ ] 6.3 Scene-editing GPU tests (`test_scene_editing.py`,
      `test_usd_controls.py`, `test_scene_property_dispatch.py`) green.
- [ ] 6.4 Parity matrix dual gate unchanged.
- [ ] 6.5 Docs: `docs/Architecture.md` intake section; `docs/PythonAPI.md` if
      the public loader surface changed.
- [ ] 6.6 `openspec validate scene-intake-interface --strict`.
