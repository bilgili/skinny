# Tasks

## 1. Failing test (red)

- [x] 1.1 GPU behavior verified on Metal (repro3: env stays None, contribution 0
  before/after texture-set/toggle on an added dome).
- [x] 1.2 Hostless test `tests/test_dome_texture_authority.py` asserts
  `_usd_scene.environment is not None` + contribution > 0 after set. Confirmed
  RED on `main` (`assert None is not None`).

## 2. Fix (green)

- [x] 2.1 Rework `apply_dome_light_texture` branch: key on `uses_default_lights`,
  construct `LightEnvHDR` for the authored-scene-no-environment case.
- [x] 2.2 Keep `_last_env_index` invalidation, USD prim mirror, `_material_version` bump.
- [x] 2.3 Test passes (4/4).

## 3. Regression

- [x] 3.1 Authored-with-texture swap still replaces in place (test 2).
- [x] 3.2 `ruff check` clean; 47 hostless light/scene-edit tests pass.

## 4. Docs + review

- [x] 4.1 CHANGELOG `### Fixed` entry.
- [x] 4.2 `openspec validate --strict` passes.
- [ ] 4.3 codex review before merge.
