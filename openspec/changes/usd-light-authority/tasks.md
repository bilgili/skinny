## 1. Authored-light authority

- [x] 1.1 Add hostless failing tests for authored-light presence across
  DistantLight, SphereLight, DomeLight, RectLight, DiskLight, emissive material,
  zero-intensity/disabled lights, and a stage with no lighting.
- [x] 1.2 Add authored-light metadata to `Scene` and populate it in
  `usd_loader.py` before area-light conversion; preserve it through animation
  refresh and scene resync.
- [x] 1.3 Replace the cached powered-light policy with one central,
  re-evaluated `uses_default_lights` decision and cover no-USD, initial-load,
  first-light-added, and last-light-removed transitions.

## 2. All-or-nothing render fallback

- [x] 2.1 Add a hostless decision-table test proving that any authored lighting
  suppresses both default DistantLight and default IBL, while a light-less
  scene enables both.
- [x] 2.2 Route distant-light upload through the authority decision: authored
  mode uploads only authored DistantLights and ignores fallback
  `direct_light_index`; fallback mode uploads the built-in DistantLight.
- [x] 2.3 Route environment selection/intensity through the same decision:
  authored mode uses only an authored DomeLight (black environment when
  absent); fallback mode uses the built-in IBL.
- [x] 2.4 Verify Sphere-only, Dome-only, area-light-only, emissive-only,
  zero-intensity-light, and light-less scenes across path, BDPT, SPPM, and MLT
  eligibility without changing shader code.

## 3. Conditional controls and scene graph

- [x] 3.1 Add shared-UI tests requiring the `IBL` and `Direct Light` sections
  and widgets when fallback is active and requiring both headings and bodies
  to be absent when authored lighting is active.
- [x] 3.2 Implement conditional top-level section visibility in the shared UI
  and both Qt/Panel backends, including runtime false→true and true→false
  transitions after asynchronous load/resync.
- [x] 3.3 Gate synthesized `/Skinny/DefaultLight` and
  `/Skinny/DefaultDome` scene-graph injection as a pair; assert both exist only
  in fallback mode and that real USD light nodes remain editable.
- [x] 3.4 Update headless/API tests so `--env-intensity`, `--no-direct`, and
  equivalent `RenderOptions` fields affect fallback scenes only and cannot
  override authored USD lights.

## 4. Validation and documentation

- [x] 4.1 Run focused default-light, USD-loader, scene-editing, UI-spec, Qt,
  web, and headless tests; then run the complete hostless pytest suite and Ruff.
- [x] 4.2 Run one light-less and one authored-light headless smoke render on
  each available backend; verify fallback counts/environment state and
  authored-only output. No GPU backend is available in this sandbox: Metal
  device creation fails and MoltenVK reports `VK_ERROR_INCOMPATIBLE_DRIVER`.
- [x] 4.3 Update `docs/Architecture.md`, `docs/PythonAPI.md`, README, and
  `CHANGELOG.md` with the all-or-nothing fallback and conditional-control
  behavior.
- [x] 4.4 Run `openspec validate usd-light-authority --strict` and review the
  proposal/spec/task alignment before implementation close-out.
