# Coverage notes (Task 1) — investigation results

Recorded before any code moved. Three proposal/design premises are **factually
wrong** and are corrected here; the change is still valid — the real duplication
is the per-toolkit property→widget *type switch*, and the real drift is a set of
Panel feature gaps.

## 1.2 Qt vs Panel scene-property widget diff

Both switch on `prop.type_name`, read `prop.value`, read `prop.metadata`
(`min`, `max`, `growable`), and **both already commit through the shared
`scene_edit_actions.apply_scene_property`**. The type switch is the duplicate.

| prop type | Qt | Panel | difference |
|---|---|---|---|
| bool | QCheckBox + camera live-pull | Checkbox | Panel lacks camera pull (accidental) |
| float | QSlider+SpinBox, growable range, camera live-pull | Float/EditableFloatSlider, step-based | Qt growable + camera pull (Panel missing — accidental) |
| color3f (edit) | swatch button + QColorDialog | inline ColorPicker | idiom (intended) |
| color3f (readonly) | swatch + text | **falls through to fallback** | **Panel missing (accidental)** |
| vec3f / vec2f (edit) | N× QDoubleSpinBox (−1e6..1e6, 4dp, .05) | N× FloatInput (same) | same |
| vec2f/vec/int/float (readonly) | styled labels | Markdown fallback | idiom |
| int | QSpinBox (min/max or ±1e6) | IntInput (min/max or None) | default range (minor) |
| rel / asset | styled QLabel | Markdown | idiom |
| lens_file (edit) | "Load…" dialog + async load | **not handled** | **Panel missing (accidental)** |
| texture_file (edit) | "Load…" dialog, `light_env` guard, async | **not handled** | **Panel missing (accidental)** |

**CORRECTION to proposal/spec:** Panel does **not** re-inline the fan-out-first
guard. `windows.py:256-261` and `:342-345` are **docstrings acknowledging the
shared guard**, not re-inlined copies — Panel routes every edit through
`apply_scene_property`. Task 3.1's "delete the two Panel re-inlinings" is a
**no-op**; there is nothing to delete. (The historical local dispatcher that
handled only `light_dir`/`light_sphere` was already removed; the docstrings
explain why.)

## 1.3 Qt vs Panel material-graph input diff

Input descriptor is `PortView(name, type_name, value, connected_from)` —
**no constraint metadata**. Qt handles float/color3/vector3/vector2/integer/
boolean/filename; **Panel is missing vector2 and filename**. Both share the
commit logic (`apply_material_override` / graph-regen pair).

## 1.4 Four key maps

| action | GLFW | Qt viewport | Qt debug dock | Web/Panel debug |
|---|---|---|---|---|
| WASD/QE move | ✓ | ✓ | ✓ | — |
| toggle camera (C) | ✓ | ✓ | ✓ | — |
| reset (F) | ✓ | ✓ | ✓ | Button |
| HUD | F1 | F1 | Space | — |
| focus overlay (L) | ✓ | ✓ | — | — |
| vignette (V) | ✓ | ✓ | — | — |
| wires/grid/focus-plane/render-area/ortho/**DOF** | — | — | M/G/P/I/O/**D** | — |
| top/left/back view | — | — | T/L/B | Buttons |
| gizmo cycle (Space) | ✓ | ✓ | — | — |
| zoom Z/X | ✓ | ✓ | — | — |
| Escape close | ✓ | default | **absent** | close pane |

**CORRECTION to proposal/design:** the Qt debug dock is **not** missing
`Key_D → show_dof_planes` — it is bound (`debug_viewport.py:137`) and correctly
guarded from WASD (movement keys return before PRESS_ACTIONS dispatch). The
genuine divergences to reconcile-or-record: Qt debug dock has **no Escape**;
web/Panel debug has **no keyboard and no mouse** (four buttons only).
`test_gizmo_mode_parity.py` pins Space/F1 across GLFW+Qt only, excluding web by
construction — the pattern to extend.

## 2.2 decision — one node family, two source adapters

Graph inputs and scene properties map onto the **same** widget vocabulary
(float→slider, color→picker, vec→vector, bool→checkbox, file→picker,
readonly→label). Their metadata differs (scene props carry `min/max`
constraints; `PortView` carries `connected_from` connectivity), but that changes
how a node is *sourced* (what range/setter), not *which node type* it is.

Decision: **one node family = the existing `ui/spec.py` leaf node types**, and
**two thin adapter functions** that both emit those nodes —
`scene_property_to_node(...)` and `graph_input_to_node(...)`. Not two node
families, not a metadata-tagged union. This is the minimum that removes the
duplicate switch without inventing a framework (design D2/D1).
