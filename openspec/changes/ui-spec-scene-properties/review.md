# Design review — ui-spec-scene-properties

Adversarial review, 2026-07-27, against the tree at `8247148`. Recorded rather
than folded. **Fold before implementing.**

**Verdict: not worth doing as scoped.** Three of the four named wins are already
true, cheaper without the seam, or arithmetically impossible. The one real defect
the proposal found is a ~90-line deletion needing no spec-node work — and it
carries a live bug the proposal never noticed.

## MAJOR

**M1 — The Panel duplication is ~5× larger than described, and it contains a
live bug.** Panel does not call `scene_edit_actions.apply_scene_property` **at
all**; it reimplements the whole dispatcher:

| shared | Panel duplicate |
|---|---|
| `apply_scene_property` `scene_edit_actions.py:121-231` | `_apply_prop_value` `panel/windows.py:423-437` |
| `_apply_vec3` `:234-267` | `_apply_vec3_value` `panel/windows.py:441-475` (line-for-line, incl. the "Author to the stage" comment) |
| `find_material_ref` `:103-118` | `_find_material_ancestor` `panel/windows.py:477-492` |

The two "re-inlined guards" at `:262` and `:350` are a *consequence*: the shared
function checks fan-out first (`scene_edit_actions.py:159-167`); Panel's copy
does not, so each call site re-adds it.

**The copy has diverged into a bug.** Shared `_LIGHT_KIND_TO_TYPE`
(`scene_edit_actions.py:91-95`) maps `light_env → "env"`; Panel's copy handles
only `("light_dir", "light_sphere")` (`panel/windows.py:434`). **A dome-light
property edit in `skinny-web` is a silent no-op** — `light_env` refs are real
(`scene_graph.py:208`) and `apply_light_override("env", …)` handles them
(`renderer.py:7328`). Panel's copy also returns `None` on every unrouted case
where the shared one returns a reason string (`:231`), so Panel cannot surface
the failure either.

**M2 — The ~530 → ~200 reduction is not credible; the property widgets *are* the
adapter code.** `_add_float` (`qt/windows/scene_graph.py:480-536`, 57 lines) is
`QSignalBlocker` on the slider↔spin pair (`:502`, `:513`, `:541-545`), int↔float
range mapping over a *mutable* `rng` dict so a growable max can rescale mid-drag
(`:493-500`), and a camera live-pull registered into `self._pulls` (`:520-536`).
Panel's float (`panel/windows.py:279-310`, 32 lines) has none of it. The shared
residue is one line: `float + editable → (lo, hi, growable)`.

Measured against the existing seam, the sidebar's per-node-type adapter cost is
~31 lines in `qt/backend.py` and ~27 in `panel/backend.py`. Eight new
scene-property node types therefore cost roughly **450-550 new adapter lines**.

| | lines |
|---|---|
| deleted: one of two type switches | −45 |
| deleted: Panel's duplicate dispatcher `:413-494` | −82 |
| relocated Qt `_add_*` bodies `:453-744` | ±0 |
| relocated Panel widget bodies `:255-408` | ±0 |
| new: spec dataclasses + builders, 8 types | +130 |
| new: two backends' adapter dispatch + plumbing | +100 |
| **net** | **+100** |

Task 4.4's own gate — "materially reduced, **not merely relocated**" — is the
gate this change fails.

**M3 — D4 cannot be an extension of `test_every_param_bound_exactly_once`.**
That test (`tests/test_ui_spec.py:117-140`) works by enumerating a **static
universe** (`build_all_params(stub_renderer)`), recovering each bound path from
`setter.__defaults__` (`:99-113`), and asserting set equality. Scene properties
have no static universe — they are derived per-prim at runtime. The most you can
assert is *type-level* coverage, a different test with a different failure mode,
needing a hand-maintained type list — the artifact the change exists to remove.

It is also blocked structurally: `spec.walk` **does not descend into
`DynamicSection`** (`spec.py:348-351`, "DynamicSection bodies are NOT walked —
their contents are dynamic by definition"), which is the only existing mechanism
for runtime-varying children. A `walk()`-based test would see nothing.

**M4 — D1 understates the structural change: the two refresh models are
incompatible.** The sidebar builds once; leaves hold closures over renderer
attribute paths; refresh re-runs one flat list every tick (`qt/backend.py:56`,
`:111-119`). The scene-graph dock rebuilds props on **selection change**
(`scene_graph.py:360-381`) and tears down + rebuilds on graph identity/version
change (`:791-812`, keyed on `_scene_graph_id`/`_scene_graph_version` because
`graph` is a fresh `copy_scene_graph` each poll); widget callbacks write back
into the snapshot (`prop.value = value`, `:461`, `:507`), legal only because that
object survives until the next repopulate; `_pulls` is cleared on every rebuild.

So a `spec.Node` holding a closure over a `SceneGraphProperty` is a closure over
an object replaced wholesale. D1 needs a per-node value-source rebind or a node
identity + versioned re-resolve — precisely the structural change it claims to
avoid.

**M5 — The bind-exactly-once scenario forces new Panel features the proposal
declares out of scope.** Panel has **zero** `lens_file` / `texture_file`
handling (`grep` returns nothing); Qt has `_add_lens_file` (`:669-703`) and
`_add_texture_file` (`:704-744`), both with an OS file dialog plus async `_await`
error reporting. Panel's graph-input widget (`panel/windows.py:782-856`) handles
float / color3+vector3 / boolean / integer only; Qt also handles `vector2`
(`material_graph.py:761`) and `filename` (`:810`, with a browse dialog). The
spec scenario fails on day one for four types, and satisfying it means building
browser file-pick UX in Panel — new feature work in the change that promises
none.

**M6 — D2's line is not drawable where D2 draws it.** Per-toolkit residue that a
toolkit-free node cannot carry, in the scene-graph dock specifically:
- **Commit semantics differ.** Qt vec3/vec2 commit on `editingFinished`
  (asserted at `tests/test_qt_scene_graph_dock.py:185`); Panel commits on every
  `value` change (`panel/windows.py:353-354`, `:388-389`). Per-drag-end vs
  per-keystroke. A shared node either picks one — changing behaviour in one
  front-end — or carries a toolkit hint, which breaks D2's line already.
- **Signal blocking.** Every Qt pull writes under `QSignalBlocker` (`:472`,
  `:530`, `:534`) to avoid a pull→setter→apply loop; Panel has no pull path.
- **Focus/teardown.** `_clear_props` walks the layout calling `setParent(None)` +
  `deleteLater()` (`:383-389`); the `_pulls` loop swallows `RuntimeError` from
  deleted C++ objects (`:813-816`). Qt-only lifecycle with no node-level
  expression.

D2 is defensible only restated as: *the shared layer is the type→node-kind
switch and nothing else, ~50 lines* — which is M2's arithmetic.

**M7 — D5's ordering includes two docks with nothing to migrate.** BXDF has no
property switch on either side (`grep -n "type_name\|_add_"
qt/windows/bxdf.py` → nothing; Panel's pane at `windows.py:495-680` is a
numpy → PIL → image pipeline). Camera Debug likewise: a key table
(`qt/windows/debug_viewport.py:347-358`) and four Panel buttons. Tasks 3.3 and
3.4 have no defined work under this change's scope.

**On the ordering question: Scene Graph first is wrong.** It has the most
toolkit-entangled behaviour (live pulls, async file dialogs, worker marshalling,
snapshot write-back) *and* the most existing coverage — highest risk, least
learning per line. **Material Graph inputs are the right first cut**: 6 flat
types, no pulls, no async, no snapshot write-back, and both sides already reduce
to a plain `_apply_*` call (`material_graph.py:924`, `panel/windows.py:858`).
If the seam does not pay for itself there, it will not pay in the scene graph.

## MINOR

- **Nine `_add_*` helpers, not eight**: `_add_bool` `:453`, `_add_float` `:480`,
  `_add_color` `:547`, `_add_color_readonly` `:578`, `_add_vec3` `:591`,
  `_add_vec2` `:616`, `_add_int` `:645`, `_add_lens_file` `:669`,
  `_add_texture_file` `:704`.
- **The 363-vs-170 comparison is apples-to-oranges.** Qt's cited range
  **includes** its apply helpers (`:745`, `:752`); Panel's **excludes** its
  82-line dispatcher (`:413-494`). Like-for-like: Qt 364, Panel 252 — a 1.4×
  difference carrying 82 lines of copied dispatcher, not a lean 2.1×. The "~530"
  is really ~616.
- **D6's justification for the missing `Key_D` is false.** The GLFW viewport has
  the identical WASD conflict and binds it anyway — `KEY_D` is polled for
  right-strafe at `debug_viewport.py:805` *and* toggles `show_dof_planes` at
  `:2333-2335`. Qt already has the same two-channel structure (`_poll_wasd` reads
  `self._wasd` at `qt/windows/debug_viewport.py:328-333`; `keyPressEvent`
  returns early for WASD at `:344-347`). Three-line fix: record the WASD state,
  then fall through to the toggle table. Move it from "record" to "fix".
- **"The Qt dock tests are getsource assertions" is overstated for the dock
  migrated first.** True of `test_qt_debug_viewport_dock.py` and
  `test_qt_bxdf_dock.py` — the two docks with nothing to migrate. **Not** true of
  `tests/test_qt_scene_graph_dock.py`, which has four real behavioural tests at
  `:78-190`, two constructing a `QApplication` and calling
  `_build_property_widget` directly (`:141-164`, `:167-190`). Panel has four
  matching ones (`tests/test_panel_scene_graph_lights.py:123-198`). So task 1.1
  is already done for bool, vec3, int, vec2 on both sides — restate as "extend to
  float, color3f, and the Qt-only file types".
- Divergences the 1.2 diff will find, pre-listed: Qt has read-only renderers for
  `color3f` (`:429`, `:578`) and `vec2f` (`:424-427`); Panel has read-only
  `vec3f` only (`:394-398`) and falls through to generic Markdown. Qt registers
  camera live-pulls for `bool` and `float`; Panel has no pull path, so a camera
  property edited elsewhere never updates in the browser. Panel takes
  `session._lock` synchronously on the server thread for every apply (`:271`,
  `:296`, `:327`, `:346`, `:369`, `:387`); Qt posts to the render worker and
  never blocks.

## Recommended replacement

One small change, no new seam:

1. Delete `panel/windows.py:413-494`; call
   `scene_edit_actions.apply_scene_property(renderer, node, prop, value,
   graph=renderer.scene_graph)` from the six Panel call sites; surface its return
   string in `_set_status`. The two re-inlined guards disappear for free. Fixes
   the `light_env` silent no-op. ~90 lines deleted.
2. Fix `Key_D` in `qt/windows/debug_viewport.py:344-347`; record the rest of the
   key-map table.
3. Extend the existing behavioural prop tests to `float` and `color3f` on both
   sides.

The spec-node extension buys a ~45-line switch dedup for ~230 new lines of
plumbing, a test that cannot be the test it claims to be, and forced feature work
in Panel. Revisit if a *third* front-end appears — the only condition under which
the adapter cost amortizes.
