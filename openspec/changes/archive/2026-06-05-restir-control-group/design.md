## Context

ReSTIR DI's runtime controls are defined in `src/skinny/params.py::STATIC_PARAMS`
— the `Reuse` selector (`reuse_index`) and seven `restir_*` tuning controls. The
interactive control panel is built once by
`src/skinny/ui/build_app_ui.py::build_main_ui`, which returns a `Section` tree
consumed verbatim by all front-ends: the Qt GUI (`ui/qt/app.py`), the web/Panel
server (`web_app.py` → `ui/panel/`), and the debug viewport. Params are bucketed
into sections by `_classify(p) -> str`; `_group_params` seeds the bucket dict and
`build_main_ui` renders them in a fixed `section_order`.

Today `_classify` has no rule for the ReSTIR or Reuse paths, so they fall through
to the default `"Render"` bucket and render among the integrator, proposal,
per-lobe sampler, tonemap, and exposure controls.

## Goals / Non-Goals

**Goals:**
- A dedicated **ReSTIR** control group holding the `Reuse` selector plus the
  seven `restir_*` tuning controls, defined once and inherited by every
  front-end and the debug viewport.
- The group is self-contained: it carries the enabler (`Reuse`) so the user can
  switch into ReSTIR and tune it in one place, and stays visible regardless of
  the active reuse mode.

**Non-Goals:**
- No change to parameter definitions, paths, value ranges, CLI flags,
  persistence, accumulation-reset triggers, or any rendering/shader behaviour.
- No new spec capability; no `scene-sampling` requirement change.
- No per-backend UI code (the shared tree makes that unnecessary).

## Decisions

**D1 — Edit only the shared control tree (`build_app_ui.py`).**
`_classify`, `_group_params`, and `build_main_ui` are the single source of truth
for Qt, Panel/web, and the debug viewport. Three edits there propagate
everywhere.
_Alternative rejected:_ touching each backend's panel builder — guarantees drift
and violates the project's GUI-consistency rule.

**D2 — Classify by `path == "reuse_index" or path.startswith("restir_")`.**
The seven tuning params share the `restir_` prefix, so the prefix rule absorbs
any future `restir_*` control for free; `reuse_index` is matched explicitly
because it does not share the prefix.
_Alternative rejected:_ an explicit 8-path set — more verbose and needs editing
whenever a tuning control is added.

**D3 — The `Reuse` selector lives in the ReSTIR group (chosen with the user).**
Makes the group a one-stop enable-and-tune panel; the seven tuning controls are
only effective when `Reuse = ReSTIR DI`, so co-locating the enabler is the
coherent UX.
_Trade-off:_ `Reuse` is the general scene-sampling seam selector (it will host
non-ReSTIR reuse modes such as neural reuse later), so naming its host group
`ReSTIR` is slightly broad. Accepted; see Risks for the rename trigger.

**D4 — Place `ReSTIR` immediately after `Render` in `section_order`, expanded.**
Render holds the adjacent integrator/proposal selectors; ReSTIR is a reuse
strategy layered on NEE, so it reads naturally right after. Expanded (like
Render) so the controls are visible; only `Skin` stays collapsed.

**D5 — Always-visible group.** `build_main_ui` drops empty groups (except
IBL/Direct Light), but the ReSTIR group always contains at least `reuse_index`,
so it always renders — which is required, since the group owns the enabler. No
special-casing of the drop logic is needed.

**D6 — Unit-test `_classify` directly.** `_classify(ParamSpec) -> str` is pure
and import-light (no Vulkan), so the new grouping is covered by a fast,
environment-independent test rather than a renderer/front-end test.

## Risks / Trade-offs

- **Stale grouping test hides regressions** → `tests/test_web.py::TestParamGrouping`
  imports `web_app._group_params`, which no longer exists (grouping moved to
  `ui/build_app_ui` with a renderer-based signature); it currently fails on
  Vulkan import under `.venv`. _Mitigation:_ repoint/rewrite it against the
  current API (or replace it with the new `_classify` test) so the suite passes
  Vulkan-free and actually guards the grouping.
- **Group-name breadth** → a `ReSTIR` group hosting the general `Reuse` selector.
  _Mitigation:_ documented here; the rename trigger is the first non-ReSTIR reuse
  mode shipping (then consider `Reuse / ReSTIR`). No code coupling to the name
  beyond the section string.
- **Docs drift** → if `docs/Architecture.md` enumerates the UI control groups it
  must gain `ReSTIR`. _Mitigation:_ doc task in the task list.
- **Back-compat** → none. Paths and the settings snapshot are unchanged; only the
  display group/order moves.

## Migration Plan

Pure additive UI grouping. No data, settings, or format migration. Rollback is
reverting the three-line `build_app_ui.py` change; nothing persisted depends on
the grouping.

## Open Questions

None blocking. Group membership (Reuse + seven tuning controls) and group name
(`ReSTIR`) are resolved.
