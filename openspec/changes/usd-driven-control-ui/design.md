## Context

skinny's controls are hard-coded in `params.py` (`ParamSpec`/`STATIC_PARAMS`)
and assembled into a backend-agnostic spec tree in `ui/build_app_ui.py`
(`build_main_ui`), which both the Qt and Panel backends walk. Live edits route
through `params._get_nested`/`_set_nested` (renderer attrs + `mtlx.*` overrides),
`Renderer.apply_material_override` (MaterialX/std-surface inputs), and the
scene-graph editor's USD attribute writeback (lights, lens, transforms). The
`skinny:lens:*` custom attributes on camera child prims are an existing
precedent for reading `skinny:*`-namespaced declarations from USD. The
animation work added a per-frame re-eval (`_apply_animation_frame`) plus helpers
to re-extract lights and re-evaluate instance transforms from the retained
`_usd_stage`.

## Goals / Non-Goals

**Goals:**
- A USD stage can declare a curated control panel via `skinny:ui:*` prims.
- Controls bind to either skinny parameters (renderer / mtlx / material) or raw
  USD attributes, through one prefix-typed `target` scheme.
- Controls appear in every front-end (Qt, web panel, debug) via the shared spec
  tree, only when the stage declares them.
- Reuse existing get/set + re-eval machinery; no new GPU/shader code.

**Non-Goals:**
- Widget types beyond slider / toggle / combo / color.
- Live updates for raw USD attributes outside the lights/transforms/camera set.
- Watching the stage for external edits (one-way: controls → USD/params).
- A formal USD schema (`.usda` codeless schema) — plain `skinny:ui:*` custom
  attributes suffice, matching the `skinny:lens:*` precedent.

## Decisions

### D1: Prim-per-control with `skinny:ui:*` attributes
Each control is a prim carrying authored `skinny:ui:type`. Discovery is "any
prim with an authored `skinny:ui:type`" (not a fixed scope path) so authors can
place/group controls freely; order by an optional `skinny:ui:order` then prim
path. Attributes: `type` (`slider`/`toggle`/`combo`/`color`), `target`,
`label` (default: prim name), `min`/`max`/`step` (slider), `choices` (token[],
combo), `default` (optional), `order` (optional).

- *Alternative (fixed `/SkinnyControls` scope)*: simpler discovery but brittle if
  the author names it differently; rejected for v1. Single "Scene Controls"
  section for now (group-by-scope is a later nicety).

### D2: Prefix-typed `target` string + binding resolver
`target` encodes both the binding kind and the path:
`renderer:<path>`, `mtlx:<field>`, `material:<matName>:<input>`,
`usd:<primPath>.<attr>`. A resolver returns `(getter, setter)` closures:
- `renderer:`/`mtlx:` → `_get_nested`/`_set_nested` (mtlx prefixes `mtlx.`)
- `material:` → resolve `mat_id` by name in `_usd_scene.materials`; read
  `parameter_overrides`, write via `apply_material_override`
- `usd:` → resolve the attribute on `_usd_stage`; `Get()` / `Set()`

Explicit and extensible; inferring kind from path shape would be fragile.

### D3: `usd:` live refresh reuses the animation re-eval
A `usd:` setter writes the attribute then sets a `_usd_live_dirty` flag; the next
`update()` re-extracts lights, re-evaluates instance transforms, and re-extracts
the camera from the stage (the existing cheap, live-applicable path). This covers
the knobs authors realistically expose (light intensity/color, prim transforms,
camera focal). Geometry/topology/material-graph attributes are written but only
take effect on reload — documented, not silently broken.

### D4: Data-driven "Scene Controls" section in the shared tree
A `DynamicSection` (rebuild token = `(id(_usd_scene), len(_usd_controls))`)
emits one widget per `ControlSpec` via the resolver closures. Built in
`build_main_ui`, so Qt + Panel + debug all render it. Gated on a non-empty
control list, so non-declaring scenes show nothing — matching the Animation
section pattern from `usd-animation-playback`.

### D5: Defaults
If `skinny:ui:default` is authored, apply it to the target at load (so the scene
opens in the authored state); otherwise the widget reflects the target's current
value. Combo `default`/value is an integer index into `choices`.

## Risks / Trade-offs

- **Malformed declarations** (bad `target`, missing `min`/`max`, unknown `type`)
  → the parser validates and skips bad controls with a logged warning rather
  than failing the load.
- **`usd:` target on a non-live attribute** → written but not visibly applied
  until reload. Mitigation: document the live set; consider a one-line "reload to
  apply" hint later. Not a correctness bug.
- **Name collisions / stale material names** → `material:` resolution by name can
  miss if the scene's materials changed; resolver returns a no-op + warning,
  leaving the widget inert rather than crashing.
- **Async load race** → controls discovered at load like the animation index;
  the `DynamicSection` token picks them up when `_usd_scene` is set. No widget
  bounds depend on async-discovered ranges (min/max are authored constants).

## Open Questions (leaning)
- Group controls by parent scope into sub-sections? Lean: single section for v1.
- Allow `vector`/`button` widget types? Lean: defer; the four cover the asks.
