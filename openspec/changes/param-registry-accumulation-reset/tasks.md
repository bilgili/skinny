# Tasks: param-registry-accumulation-reset

## 1. Registry declarations (params.py — inert, nothing reads them yet)

- [ ] 1.1 Add `resets_accumulation: bool = True` to `ParamSpec`; thread it through the `_cont`/`_disc` helpers; mark `tonemap_index` and `exposure` with `resets_accumulation=False` (matching their existing exclusion, renderer.py:1876). Verify all four front-ends still import/construct params unchanged (`.venv/bin/python -m py_compile src/skinny/params.py src/skinny/app.py src/skinny/render_session.py`).
- [ ] 1.2 Declare the legacy-cast coercion override: `restir_m_light`, `restir_m_bsdf`, `restir_spatial_k`, `restir_m_cap` (params.py:127–131) are continuous but `int()`-cast in the hash (renderer.py:10928–10930) — and the difference is observable since `_apply_saved_params` clips without quantizing (params.py:358). Mark them with an explicit `int` hash-coercion beside their reset flag so fractional saved values keep today's no-reset behavior.
- [ ] 1.3 Add `ACCUM_STATE_PROVIDERS` to params.py: named entries with `extractor(renderer)` callables (duck-typed, no new imports) for `camera` (`state_signature()`), `mtlx_overrides` (verbatim sorted-tuple expression from renderer.py:10958, with `covers_prefix="mtlx."`), `material_version`, `volume_grid_key`, `film_max_component`, `camera_mirror`, `usd_time_code` (`clock.current_time_code`), and the three SPPM overrides via `getattr(..., None)`. Move/alias `_hashable_value` so params.py stays hostless-importable.
- [ ] 1.4 `.venv/bin/ruff check src/` clean.

## 2. Hostless invariant test (green-first transcription gate)

- [ ] 2.1 Write `tests/test_accum_reset_registry.py` (pattern: tests/test_cli_common.py — imports only, no GPU): (a) frozen expected contributor set transcribed field-by-field from renderer.py:10906–10968; (b) assert registry-derived identity set (`resets_accumulation` params minus `mtlx.*` covered, plus provider names) equals it; (c) assert the default is `True`; (d) assert the `False` set is exactly `{tonemap_index, exposure}`; (e) assert every `mtlx.*` static param falls under a provider `covers_prefix`; (f) assert the `int` coercion override is exactly `{restir_m_light, restir_m_bsdf, restir_spatial_k, restir_m_cap}` (fractional change within the same integer must not perturb the derived contribution).
- [ ] 2.2 Run `.venv/bin/pytest tests/test_accum_reset_registry.py` — the transcription test is green-first by design: it must pass from the registry alone, before renderer.py is touched, proving the transcription matches the legacy tuple.

## 3. Derive `_current_state_hash` from the registry (renderer.py)

- [ ] 3.1 Rewrite `_current_state_hash` (renderer.py:10904) to build its tuple from: static `resets_accumulation` params (skipping provider-covered prefixes), values coerced by the declared per-field cast — `kind` default (`continuous`→float, `discrete`→int) unless a coercion override applies (the four ReSTIR count params → `int()`, task 1.2) — resolved via a getattr-chain resolver list built once; then the provider extractors in registry order. Keep the method name, docstring intent, and `hash(tuple)` return; keep the consumer at renderer.py:11051 untouched.
- [ ] 3.2 Verify behavior equivalence hostlessly: unit-check the derived contributor list against the frozen set (already gated by 2.1) and the coercion assertions (2.1f; `bool` restir_biased vs `int` is hash-equal, so no override needed there).
- [ ] 3.3 Sweep the scattered comments: replace "hashed into `_current_state_hash` so ..." bodies (renderer.py:1762, 1796, 1919, 4619, 8934) and the exclusion notes (1876, 2132) with one-line pointers to the params.py registry; update the semantics reference in src/skinny/sampling/reuse.py:34; keep the MLT-seed independence note (renderer.py:2452) and presets.py:6 docstring accurate. Coordination: if `renderer-module-carveout` (Stage A) has already moved `_next_mlt_seed` into its `mlt_chain` module, the seed note lives there — update it at its new home (whichever change lands second owns the relocation).
- [ ] 3.4 Hostless suite green: `.venv/bin/pytest tests/test_accum_reset_registry.py tests/test_cli_common.py tests/test_backend_select.py -q` and `.venv/bin/ruff check src/`.

## 4. Re-point the source-inspection tests

- [ ] 4.1 Update `tests/test_sppm_selection.py:40` (integrator_index), `tests/test_volume_grid.py:225` (volume_grid_key), `tests/test_mlt_selection.py:181` to assert registry membership (param flag / provider name) instead of substring-in-method-body; keep `tests/test_mlt_host.py:217`'s "MLT seed NOT derived from the hash" guard working against the new body. Coordination: `tests/test_mlt_host.py` is touched by three concurrent renderer-cluster changes (this one, `renderer-module-carveout`, `reflection-owned-byte-layouts`) — assert against whatever module owns `_next_mlt_seed` at merge time (renderer.py, or `mlt_chain` if the carveout landed first); whichever change lands second owns re-pointing the guard.
- [ ] 4.2 Run those four test modules hostlessly — all green.

## 5. Behavior smoke + docs

- [ ] 5.1 GPU smoke (Metal, one process, per metal-dispatch-hygiene): headless render, flip `integrator_index` mid-run and confirm `accum_frame` resets; flip `exposure` and confirm it does not. (Mirrors the existing headless A/B pattern in tests/test_headless.py.)
- [ ] 5.2 Docs: update `docs/Architecture.md` (renderer state/accumulation section) to describe the registry-derived hash; note the `ParamSpec.resets_accumulation` field wherever ParamSpec is documented. No README/CLI changes (no user-facing surface moved).
- [ ] 5.3 `openspec validate param-registry-accumulation-reset` clean; archive after merge.
