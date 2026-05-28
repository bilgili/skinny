## 1. Control extraction (load time)

- [x] 1.1 Add `ControlSpec` dataclass + `extract_ui_controls(stage) -> list[ControlSpec]` in `usd_loader.py`: scan prims with authored `skinny:ui:type`, parse type/target/label/min/max/step/choices/default/order; skip malformed with a warning; order by `order` then prim path
- [x] 1.2 Unit tests on a synthetic in-memory stage: control prims → correct specs (incl. label fallback, ordering); malformed skipped; empty stage → none

## 2. Binding resolver

- [x] 2.1 Add a resolver mapping `target` prefix → `(getter, setter)`: `renderer:`/`mtlx:` via `_get_nested`/`_set_nested`; `material:M:input` via material lookup + `apply_material_override`; `usd:/p.attr` via attribute `Get`/`Set`
- [x] 2.2 Unresolvable target (unknown prefix, missing material/attr) → inert closures + warning, no crash
- [x] 2.3 Unit tests for each prefix on a stub renderer + synthetic stage (renderer/mtlx get+set, material override called, usd attr written, inert on bad target)

## 3. Renderer wiring

- [x] 3.1 Store `self._usd_controls = extract_ui_controls(stage)` at load (background thread, alongside the anim index)
- [x] 3.2 Apply authored `skinny:ui:default` values to their targets at load
- [x] 3.3 `usd:` setter sets a `_usd_live_dirty` flag; `update()` refreshes lights + instance transforms + camera from the stage when dirty (reuse the animation re-eval helpers)

## 4. Scene Controls UI

- [x] 4.1 Add a "Scene Controls" `DynamicSection` to `build_main_ui` (token = `(id(_usd_scene), len(_usd_controls))`), gated on a non-empty control list
- [x] 4.2 Build slider/checkbox/combo/color widgets per `ControlSpec` via the resolver closures
- [x] 4.3 Confirm it renders in Qt + web panel + debug viewport (shared tree); unit test mirroring the Animation-section test (stub renderer with `_usd_controls`)

## 5. Verification

- [x] 5.1 `ruff check src/` introduces no new errors; new files clean; `pytest -m "not gpu"` green
- [x] 5.2 Headless: load a USD with a `renderer:env_intensity` control + a `usd:` light-intensity control; assert each edit changes the render
- [ ] 5.3 Manual (Vulkan window): load a scene with declared controls, confirm the Scene Controls panel drives the scene
