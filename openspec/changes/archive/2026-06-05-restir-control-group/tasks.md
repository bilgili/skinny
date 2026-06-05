## 1. Control grouping (`src/skinny/ui/build_app_ui.py`)

- [x] 1.1 In `_classify`, add a rule returning `"ReSTIR"` for
  `path == "reuse_index" or path.startswith("restir_")`, placed before the
  default `return "Render"`.
- [x] 1.2 In `_group_params`, seed the bucket dict with `"ReSTIR": []`.
- [x] 1.3 In `build_main_ui`, insert `"ReSTIR"` into `section_order` immediately
  after `"Render"`; render it expanded (the default).
- [x] 1.4 Update the dangling "Mirrors the rules in `web_app._group_params`"
  docstring on `_classify` to reference the current single source of truth.

## 2. Tests

- [x] 2.1 Added Vulkan-free unit coverage (`tests/test_web.py::TestParamGrouping`,
  rewritten against `build_app_ui._classify`): `reuse_index` and every
  `restir_*` path classify to `"ReSTIR"`; Render-bucket selectors
  (`integrator_index`, `proposal_preset_index`, `tonemap_index`, `exposure`)
  stay in `"Render"`.
- [x] 2.2 Reconciled `tests/test_web.py::TestParamGrouping`: dropped the
  `from skinny.web_app import _group_params` import (symbol removed; grouping
  moved to `ui/build_app_ui`) and the stale pre-`scatter_index→Skin`
  expectations; the class now drives `_classify` directly and passes without a
  Vulkan SDK.
- [x] 2.3 Ran ruff (`build_app_ui.py` clean — also removed a pre-existing unused
  `typing.Sequence` import) and the tests. `TestParamGrouping` + `test_ui_spec.py`
  pass on both the 3.12 `.venv` (Vulkan-free) and the repo-root 3.13 venv with
  `VULKAN_SDK` exported; differential vs main confirms no new failures (the
  remaining `test_web.py` failures are pre-existing `VkErrorIncompatibleDriver`
  GPU-device errors, identical on main).

## 3. Front-end verification

- [ ] 3.1 Live launch of `skinny`, `skinny-gui`, `skinny-web` to eyeball the
  **ReSTIR** group — needs a display; not run headless. Substantively covered by
  `test_ui_spec.py::test_top_level_section_order` (asserts the real
  `build_main_ui` tree yields `… Render, ReSTIR, Skin …`) and the membership
  test above. Final visual smoke-test left to the user.
- [ ] 3.2 Visible-at-identity + accumulation-reset-on-switch: the group always
  renders (tree built with the default renderer includes it regardless of reuse
  mode); accumulation reset on `reuse_index` change is pre-existing
  (`_current_state_hash`) and untouched. Live confirmation left to the user.

## 4. Docs

- [x] 4.1 Reviewed the Markdown docs. `README.md` sidebar enumeration updated
  (`Render / ReSTIR / Skin / Detail / Materials`); `CHANGELOG.md` entry added
  under **UI and interaction**. `docs/Architecture.md` does not enumerate the
  control groups (only generic `build_main_ui` references), so no edit there.
