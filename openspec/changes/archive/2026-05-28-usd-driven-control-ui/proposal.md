## Why

A USD asset author has no way to surface a curated control panel with their
scene. Today every knob in skinny is hard-coded in `params.py` / `build_app_ui`.
Letting a stage declare its own controls (e.g. a "skin tone" slider, a "light
warmth" knob) makes assets self-describing and lets authors expose exactly the
parameters that matter for their scene — driving either skinny's own parameters
or USD attributes directly.

## What Changes

- Read **USD-declared controls** at load: each control is a child prim under a
  scope (e.g. `/SkinnyControls`) carrying `skinny:ui:*` attributes (`type`,
  `target`, `label`, `min`/`max`/`step`, `choices`, `default`, `order`).
- Add a **binding resolver** that maps a control's prefix-typed `target` string
  to live get/set closures:
  - `renderer:<path>` → renderer attribute via `_get_nested`/`_set_nested`
  - `mtlx:<field>` → MaterialX skin override via `mtlx.<field>`
  - `material:<matName>:<input>` → `apply_material_override`
  - `usd:<primPath>.<attr>` → USD attribute `.Get()` / `.Set()` + live refresh
- Add a **"Scene Controls" section** to the shared UI spec tree (Qt + web panel
  + debug viewport) that builds slider/toggle/combo/color widgets from the
  declared controls, shown only when the stage declares any.
- For `usd:` targets, on edit write the attribute to the stage and **refresh the
  live-applicable state** (lights, instance transforms, camera) so common knobs
  update in place; other attributes apply on reload.

Out of scope: vector/file/button widget types; live updating arbitrary USD
attributes beyond the lights/transforms/camera set; watching the stage for
external edits (controls drive USD, not vice-versa); any shader/compute work.

## Capabilities

### New Capabilities
- `usd-driven-control-ui`: discovery + parsing of `skinny:ui:*` control prims,
  the prefix-typed binding resolver (renderer / mtlx / material / usd targets),
  and the data-driven "Scene Controls" section in the shared spec tree.

### Modified Capabilities
<!-- None: no existing archived spec's requirements change. -->

## Impact

- **Code**: `src/skinny/usd_loader.py` (`ControlSpec` + `extract_ui_controls`);
  `src/skinny/renderer.py` (store `_usd_controls` at load; binding resolver;
  `usd:` live-refresh hook reusing the existing light/transform/camera re-eval);
  `src/skinny/ui/build_app_ui.py` (Scene Controls `DynamicSection`). Both UI
  backends pick it up via the shared tree.
- **Dependencies**: none new — pxr already available; reuses existing param /
  material-override / scene re-eval machinery.
- **Shaders**: none.
