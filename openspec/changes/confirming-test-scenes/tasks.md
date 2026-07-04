# Tasks — confirming-test-scenes

Every stage ends with a regression checkpoint: hostless tests run automatically; the GPU
sweep is only run after asking the user (exact commands + estimated time in the ask).
Sweep results go to `openspec/changes/confirming-test-scenes/results/stage-<n>.json` and
are reported as before/after deltas with rendered images shown.

## 1. Harness wiring (hostless)

- [x] 1.1 Create `tests/assets/suite/` scaffold + README describing scene layout, budgets, and gate classes
- [x] 1.2 Extend manifest schema handling in `src/skinny/pbrt/parity.py` for suite entries: `tests/assets/suite/` paths, equivalence tolerance / skip-reason, furnace disposition fields
- [x] 1.3 Implement the authoring-equivalence gate class (anchor-combo render of both variants + `metrics.compute_metrics` battery, recorded-skip path)
- [x] 1.4 Extend the coverage meta-test: every suite scene needs a disposition per applicable gate class (pbrt-truth / equivalence / furnace) or the build fails
- [x] 1.5 New hostless suite test module: USD variants open with expected prims/bindings, `.mtlx` parses, `.pbrt` imports, manifest entries schema-valid
- [x] 1.6 Stage checkpoint: run hostless tier (`.venv/bin/pytest tests/pbrt -m "not gpu"` + new module); no GPU needed this stage; report status

## 2. Core material scenes

- [x] 2.1 Author `mat_diffuse`, `mat_conductor`, `mat_dielectric`, `mat_plastic` — each: `<scene>.usda` + `<scene>_mtlx.usda`/`.mtlx` + `<scene>.pbrt`
- [x] 2.2 Author `mat_emissive`, `mat_textured`, `mat_subsurface` the same way (texture scene reuses the existing `texture_uv.png` pattern approach)
- [x] 2.3 Generate pbrt reference EXRs via `tests/pbrt/regen_refs.py` (pinned pbrt, 128×128); verify each `.pbrt` runs unmodified with the user-facing pbrt invocation
- [ ] 2.4 Add manifest entries (spp, tolerances measured-then-pinned, equivalence tolerances); hostless tests green
- [ ] 2.5 Stage checkpoint: hostless tier auto; **ask user** before GPU sweep (matrix over the 7 scenes, both variants, mega+wave); persist `results/stage-2.json`, report deltas + images

## 3. PBR material scenes from assets

- [x] 3.1 Extract the selected OpenPBR Material prims (Gold, Copper, Glass, Plastic PC, Skin I, Gray Card, Musou Black) from `assets/materialxusd/tests/physically_based/*_OPBR_MAT_PBM.usda` into scene dirs as single self-contained prims (no 71-material library, no shaderball/HDR payloads, no `assets/` dependency; record source file per scene)
- [x] 3.2 Author the seven `mat_pbr_*` mini-shaderball scenes (MaterialX-authored variants; plain-USD variant = recorded skip with reason, pbrt gate = recorded skip)
- [x] 3.3 Add the opt-in `pbr_shaderball_smoke` gpu test: render 1–2 original shaderball scenes as-is from `assets/materialxusd/` (skip-if-missing so worktrees pass)
- [x] 3.4 Manifest entries + hostless tests; verify suite scenes load in a worktree without `assets/`
- [ ] 3.5 Stage checkpoint: hostless auto; **ask user** before GPU sweep (self-consistency over the 7 scenes + smoke render); persist `results/stage-3.json`, report deltas + images

## 4. Integrator scenes

- [x] 4.1 Author `int_indirect_box`, `int_caustic`, `int_bleed` (usda + mtlx variant + pbrt counterpart each)
- [x] 4.2 Generate pbrt refs; add manifest entries; record any legitimate integrator exclusions through `combo_is_valid` (e.g. SPPM × subsurface) with reasons
- [ ] 4.3 Hostless tests green (scene validity + coverage meta-test)
- [ ] 4.4 Stage checkpoint: hostless auto; **ask user** before GPU sweep (path/bdpt/sppm × mega/wave over the 3 scenes vs pbrt refs + anchor); persist `results/stage-4.json`, report deltas + images

## 5. Sampling-mode scenes

- [x] 5.1 Author `samp_many_lights` (16 emissive quads, glossy floor) + `samp_env_glossy` (rough conductor, high-contrast env, no analytic lights), dual variants + pbrt counterparts
- [ ] 5.2 Wire ReSTIR DI and proposal/neural axes for these scenes into the matrix (statistical unbiasedness vs analytic anchor, not image equality, for the neural axis)
- [x] 5.3 Generate pbrt refs; manifest entries; hostless tests green
- [ ] 5.4 Stage checkpoint: hostless auto; **ask user** before GPU sweep (sampling axes over the 2 scenes); persist `results/stage-5.json`, report deltas + images

## 6. Furnace closure

- [x] 6.1 Implement the furnace gate harness: headless render with `furnace_index=1`, read linear accumulation image, closure assertion helper (reuses `metrics` battery, no hand-rolled formulas)
- [x] 6.2 Author furnace material variants (Lambert 1.0, smooth conductor 1.0, clear dielectric, rough conductor) as suite scenes with furnace dispositions in the manifest
- [x] 6.3 Sweep furnace × integrator × execution mode through `combo_is_valid`; record exclusions (SPPM decision) with reasons
- [x] 6.4 Per-material furnace test: two-sphere scene, flag bit 10 on one material; flagged closes to 1.0, unflagged matches non-furnace reference
- [x] 6.5 Measure closure values; pin tolerances; record any legitimate energy-loss baselines (rough conductor) — baselines tighten-only
- [x] 6.6 Stage checkpoint: hostless auto; **ask user** before GPU furnace sweep; persist `results/stage-6.json`, report closure table + images

## 7. Final regression + docs + validation

- [ ] 7.1 Full regression comparison: **ask user**, then run the complete matrix (existing corpus + all new scenes) and diff every metric against pre-change values; flag any regression in existing scenes
- [ ] 7.2 Update `docs/Architecture.md` (Parity Matrix Harness: suite location, equivalence + furnace gate classes), `CLAUDE.md` (parity harness section pointer), `README.md` if user-facing
- [ ] 7.3 `.venv/bin/ruff check src/` clean; full hostless `pytest` green; kill-harness check not needed (no dispatch-length changes) — confirm and note
- [ ] 7.4 `openspec validate confirming-test-scenes` passes; results dir committed; ready for archive
