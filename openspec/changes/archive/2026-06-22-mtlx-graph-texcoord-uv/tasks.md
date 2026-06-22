## 1. Reproduce + failing test (TDD red)

- [x] 1.1 Add a tracked test asset `assets/Usd-Mtlx-Example/materials/
  standard_surface_default_uv_image.mtlx` — a `standard_surface` whose
  `base_color` is driven by a bare `<image>` node (no explicit texcoord input)
  through a nodegraph, reproducing the default-`<texcoord>` UV form
  (`vd.texcoord_0`). Force-add it (`.gitignore` is `*`).
- [x] 1.2 Add a test in `tests/test_materialx_graph.py` that runs
  `generate_for_compute` on that asset and asserts the emitted `slang_source`
  contains `UV_in` and no bare `vd.` identifier. Confirm it FAILS before the fix.

## 2. Fix (TDD green)

- [x] 2.1 In `_emit_graph_fragment` add `body = body.replace("vd.texcoord_0",
  "UV_in")` to the `vd.*` rewrite block.
- [x] 2.2 After all `vd.*` rewrites, guard: if `"vd."` still appears in `body`,
  log the leftover token(s) and `return None` (fall back to the flat path)
  instead of emitting an uncompilable module.
- [x] 2.3 Re-run the new test (green) and the existing
  `tests/test_materialx_graph.py` + `tests/test_material_emitters.py` (no
  regression — marble / brass / wood still produce fragments).

## 3. Render verification (manual)

- [x] 3.1 Run `skinny --integrator sppm assets/bathroom.usda --execution-mode
  wavefront --backend metal` (worktree src via `PYTHONPATH`, repo-root
  python3.13 venv) and confirm the material modules compile and a frame renders.
  Capture the rendered image (inline + clickable path).

## 4. Docs + validate

- [x] 4.1 Update `docs/Architecture.md` (MaterialX graph / per-material codegen
  note) and `CHANGELOG.md` for the default-UV rewrite + fallback guard.
- [x] 4.2 `openspec validate mtlx-graph-texcoord-uv --strict`.
