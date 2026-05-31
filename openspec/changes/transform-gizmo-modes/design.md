## Context

The viewport gizmo (`src/skinny/gizmo.py`, class `RotateGizmo`) is rotate-only
and world-axis aligned. It draws three rings as 2D screen-space line segments
into the binding-22 `GizmoSegment` buffer; `main_pass.slang` blends each segment
as an AA line over the tonemapped image. Drag math computes a screen-space angle
delta and adds it to one euler component, then re-applies the instance transform
via `apply_instance_transform(translate, rotate_deg, scale)`.

Front-ends share a GUI-agnostic layer: `src/skinny/ui/gizmo_input.py`
(`gizmo_press_action` precedence + `GizmoMouseController` drag lifecycle). GLFW
wires it in `app.py`; Qt in `ui/qt/viewport.py`. Space currently toggles the HUD
in both, redundantly with F1.

This change turns the rotate gizmo into a four-mode transform gizmo (rotate /
translate × world / local) cycled by space, with a W/L glyph hint and persisted
mode.

## Goals / Non-Goals

**Goals:**
- Axis-only translation gizmo (3 arrows, single-axis drag).
- World and local coordinate space for both rotate and translate.
- One enum of four modes, cycled by space (`(mode+1)%4`, grouped by type).
- W/L glyph hint billboarded a fixed pixel offset above the projected pivot.
- Persist the active mode across sessions.
- Keybinding parity (space cycle) across every render-surface front-end.
- No shader change; reuse the existing segment buffer and line draw.

**Non-Goals:**
- Plane handles (XY/YZ/XZ), center free-move, scale gizmo — future change.
- A combined all-axes manipulator shown at once (modes are mutually exclusive).
- Fixing the screen-angle drag heuristic's view-dependent sign quirk (inherited).
- Reworking how instance transforms are stored (stays TRS / euler via
  `apply_instance_transform`).

## Decisions

**1. One `TransformGizmo` class, internal dispatch on mode.**
Rename `RotateGizmo` → `TransformGizmo`. Keep the renderer-facing API shape
(`set_gizmo_target`, `hit_test`→axis string, `begin_drag`, `update_drag`,
`end_drag`, `build_segments`). `build_segments`, `hit_test`, and the drag methods
branch on `self.mode`. *Alternative considered:* separate `RotateGizmo` +
`TranslateGizmo` + a manager. Rejected — the shared spine (pivot, projection,
px-per-unit, segment buffer, hit tolerance, hover) is large; one class with a
mode is less duplication and keeps `gizmo_input.py` untouched.

**2. Mode enum, grouped by type, cycled by space.**
`GizmoMode` ∈ {`ROT_WORLD`, `ROT_LOCAL`, `TRA_WORLD`, `TRA_LOCAL`}; space →
`(mode+1)%4`. Type = rotate for the first two, translate for the last two; space
= world for the even-indexed, local for the odd. Grouped by type so two presses
flip only world↔local (glyph changes, shape stays), then the shape changes.
*Alternative:* group by space, or split into two keys (type vs space). Rejected
— the user wants a single space cycle through all four.

**3. World vs local = which axis basis feeds geometry + drag.**
World: canonical X/Y/Z. Local: `R_inst · axis`, where `R_inst` is the rotation
part of the instance transform. `set_target` extends to receive/stash `R_inst`
(renderer already holds the transform). Build/hit/drag read the stashed basis.

**4. Unified rotation drag via world-space axis-angle.**
Replace the per-euler-component add with: delta rotation about a world-space axis
vector `n` (`n` = canonical axis for world, `n = R_inst·axis` for local);
`R_new = axisAngle(n, Δθ) @ R_start` (premultiply, since `n` is already
world-space); decompose `R_new` back to euler for `apply_instance_transform`.
Keep the existing screen-angle heuristic to produce `Δθ`. This unifies world and
local (only `n` differs). *Alternative:* keep euler-add for world, matrix only
for local. Rejected — two code paths, and world-rotate would stay approximate.

**5. Translation drag = screen-projected axis pixel mapping.**
Project pivot and `pivot + axis_dir` to screen → screen-space axis direction
(unit). `pixels_along = (mouse − start) · screenDir`. Convert px→world with the
existing px-per-unit measurement (`_ring_radius_world` style). `t_new = t_start +
axis_dir_world · pixels_along / px_per_unit`. Single-axis constrained; rotate and
scale pass through unchanged.

**6. W/L glyph as billboarded segments in the same buffer.**
Stroke `W` or `L` with ~6 line segments at a fixed pixel offset above the
projected pivot, in screen space (so it auto-billboards), drawn on top like the
rings. Rings use 3×64 = 192 segments; arrows far fewer; glyph ~6 — all within the
256-segment cap. *Alternative:* HUD text (CPU PIL→R8). Rejected — heavier and
couples the gizmo to the HUD; segments need no shader change and no new binding.

**7. Persist mode in `settings.json`.**
Add a `gizmo_mode` field to the persisted snapshot, restored at startup like
camera/params. Mode is global (not per-instance).

**8. Keybinding parity, F1 keeps HUD.**
Bind space → `cycle_mode` in `app.py`, `ui/qt/viewport.py` (under the render
lock), and the debug viewport if it can target the gizmo. Leave F1 → HUD toggle.
A parity test asserts each render-surface front-end reaches the cycle.

## Risks / Trade-offs

- **Matrix→euler round-trip is gimbal-ambiguous, and unifying rotate changes the
  existing world-rotate feel.** → The applied matrix is the one we decompose, so
  the visible result is consistent frame-to-frame; verify with a headless A/B
  (rotate before/after) plus a manual drag check. Document the behavior change.
- **Space was muscle-memory for HUD toggle.** → F1 retains it; note in
  CHANGELOG/help text.
- **Local basis with non-uniform / negative scale.** → Take only the rotation
  part from `decompose_trs_matrix`; normalize basis vectors so skew/scale don't
  leak into ring/arrow orientation.
- **Glyph clutter at small gizmo size or near screen edges.** → Fixed pixel
  offset and small fixed glyph size keep it legible; it may clip off-screen near
  edges (acceptable — it is a hint, not a control).
- **Segment-cap overflow if future handles are added.** → Current worst case
  (rings + glyph) is ~198 < 256; assert/clamp on upload remains.
