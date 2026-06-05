## Why

ReSTIR DI exposes eight runtime controls — the **Reuse** selector that turns it
on, plus seven tuning controls (regime, combine, M light, M bsdf, neighbours,
radius, M cap). They currently render inline in the catch-all **Render** group,
interleaved with the integrator, proposal, per-lobe sampler, tonemap, and
exposure controls. The ReSTIR controls are hard to find, and the relationship
that makes them meaningful — they only take effect when *Reuse = ReSTIR DI* — is
invisible because the enabler and its tuning sit among unrelated sliders. A
dedicated, self-contained **ReSTIR** group makes the feature discoverable and
keeps enable-and-tune in one place.

## What Changes

- Add a new **ReSTIR** control group to the shared control tree
  (`build_main_ui`), so it appears on every front-end — the windowed `skinny`
  app, the Qt `skinny-gui`, the web `skinny-web`, and the debug viewport — from a
  single definition.
- Move the **Reuse** selector (`reuse_index`) and the seven ReSTIR tuning
  controls (`restir_regime_index`, `restir_biased`, `restir_m_light`,
  `restir_m_bsdf`, `restir_spatial_k`, `restir_spatial_radius`, `restir_m_cap`)
  out of the **Render** group into **ReSTIR**.
- Place **ReSTIR** immediately after **Render** in section order, expanded by
  default. The group is always visible (it owns the Reuse enabler), so the user
  can switch *into* ReSTIR even when the current reuse mode is identity.
- No change to parameter definitions, paths, persistence, CLI flags, the
  accumulation-reset triggers, or any rendering behaviour. This is a
  control-panel grouping change only.

## Capabilities

### New Capabilities

_None._ No new spec capability is introduced; this refines how an existing
capability presents its controls.

### Modified Capabilities

- `restir-di`: the existing **Selectable regimes and front-end selection**
  requirement is extended — the Reuse selector together with the ReSTIR config
  controls SHALL be presented in a dedicated **ReSTIR** UI group, separate from
  the general Render group, on every front-end (and the debug viewport). The
  pre-existing guarantees (selectable + persisted on every front-end, changing
  the mode or any config value resets accumulation) are unchanged.

## Impact

- **Code:** `src/skinny/ui/build_app_ui.py` only — `_classify` gains a rule
  mapping `reuse_index` and `restir_*` paths to `"ReSTIR"`; `_group_params`
  seeds the `"ReSTIR"` group; `build_main_ui` inserts `"ReSTIR"` into
  `section_order`. Because `build_main_ui` is the single source of truth for the
  Qt, Panel/web, and debug-viewport front-ends, all of them inherit the new
  group with no per-backend edits.
- **`scene-sampling` capability:** the Reuse selector it owns moves UI group but
  stays *surfaced and persisted on every front-end* — that requirement is still
  satisfied, so no `scene-sampling` delta is needed.
- **Tests:** add a Vulkan-free unit test for `_classify` covering the ReSTIR
  group; reconcile the stale `tests/test_web.py::TestParamGrouping`, which
  imports the removed `web_app._group_params` (grouping moved to
  `ui/build_app_ui` with a renderer-based signature) and currently fails.
- **No** shader/SPIR-V recompile, **no** `SkinParameters.pack()` / std140
  change, **no** settings-format change, **no** new dependencies.
