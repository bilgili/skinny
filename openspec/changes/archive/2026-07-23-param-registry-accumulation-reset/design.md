# Design: param-registry-accumulation-reset

## Context

Progressive accumulation resets when `Renderer._current_state_hash()` (renderer.py:10904) changes; the single consumer is `update()` at renderer.py:11051. The hash is a hand-written ~40-element tuple mixing two kinds of contributors:

1. **Param-backed fields** — attributes that also appear as `ParamSpec` entries in `params.py:STATIC_PARAMS` (`light_*`, `env_index`, `integrator_index`, the ReSTIR tuning block, `film.iso`, …). Two params are deliberately *excluded*: `tonemap_index` and `exposure` (post-process, renderer.py:1876).
2. **Non-param state** — `camera.state_signature()`, `mtlx_overrides` (sorted tuple), `_material_version`, `_volume_grid_key`, `film_max_component`, `_camera_mirror`, `clock.current_time_code`, and the three `_sppm_*_override` attributes. (Window resize resets accumulation via separate explicit `accum_frame = 0` sites, e.g. renderer.py:1612/8189 — it was never a hash input and stays out of scope.)

The "every accumulation-affecting field is hashed" invariant lives only in scattered comments (renderer.py:1762, 1796, 1919, 4619, 8934; excluded-by-design notes at 1876, 2132; a semantics reference in src/skinny/sampling/reuse.py:34) plus four *source-inspection* tests that substring-match the method body (test_sppm_selection.py:40, test_volume_grid.py:225, test_mlt_selection.py:181, test_mlt_host.py:217). A new param that forgets the tuple fails silently: stale accumulation, wrong convergence, no crash.

`params.py` is hostless-importable (numpy only) and already the shared registry consumed by app.py, render_session.py, and the web/Qt front-ends — the natural owner for reset semantics.

## Goals / Non-Goals

**Goals**

- Adding a `ParamSpec` IS registering its reset semantics — no second edit site, no comment discipline.
- Non-param contributors are declared in exactly one place, as data, hostless-importable.
- Identical reset behavior: the set of contributing fields is provably unchanged (hash *value* may differ; it is process-local and never persisted).
- A hostless test (no GPU, no Renderer instance) can assert the invariant.

**Non-Goals**

- No new reset behavior, no shader changes, no CLI/GUI changes, no persistence-format changes.
- No redesign of the reset *mechanism* (`_last_state_hash` comparison in `update()` stays).
- Not folding the explicit `accum_frame = 0` sites (resize, scene load) into the hash.
- Not making `execution_mode` hashable — it is fixed per session by design (renderer.py:2132).

## Decisions

### D1: `ParamSpec.resets_accumulation: bool = True`, opt-out only

Each param declares its reset semantics; the default is `True` because the failure modes are asymmetric: a param wrongly defaulted to resetting produces a *visible, benign* extra reset, while a param wrongly defaulted to not-resetting reproduces the exact silent-staleness bug this change removes. Only `tonemap_index` and `exposure` are marked `False`, matching today's exclusions.

*Alternatives considered:* (a) required (no-default) flag — forces every author to think, but 40+ mechanical `resets_accumulation=True` edits add noise for zero information, and a defaulted field keeps all four front-ends and `ParamSpec(...)` call sites source-compatible; (b) separate exclusion set (`NON_RESETTING_PARAMS = {...}`) — smaller diff but splits a param's semantics away from its declaration, recreating the locality problem one level down.

### D2: Non-param contributors = explicit `ACCUM_STATE_PROVIDERS` registry in params.py

A module-level list of named providers, e.g. `(name, extractor)` where `extractor(renderer)` returns a hashable value:

- `camera` → `renderer.camera.state_signature()`
- `mtlx_overrides` → `tuple(sorted((k, _hashable_value(v)) for ...))` (verbatim from today, renderer.py:10958)
- `material_version`, `volume_grid_key`, `film_max_component`, `camera_mirror`, `usd_time_code`
- `sppm_radius_override`, `sppm_photons_override`, `sppm_glossy_roughness_override` (via `getattr(..., None)`, preserving today's absent-attribute tolerance)

This is the "small explicit list of state providers" the deepening calls for: it cannot be eliminated (these are not user-facing params), but it moves from an anonymous tuple inside a 12k-line file to named, individually-documented, hostless-enumerable data next to the param registry. New non-param state gets one obvious place to register, and the invariant test can enumerate provider *names* without a Renderer instance.

*Alternatives considered:* (a) promote everything to synthetic `ParamSpec`s — wrong shape: these are not user-adjustable, have no name/step/lo/hi, and would pollute `build_all_params()` consumers (GUI sliders, snapshots); (b) leave them inline in `_current_state_hash` — keeps the scattered-tuple problem for exactly the fields that historically caused it (volume grid, SPPM overrides were late additions); (c) a decorator/metaclass auto-registration on attribute writes — magic, unauditable, and Ponytail-hostile.

### D3: `mtlx.*` params are covered by the `mtlx_overrides` provider, not hashed per-param

The ~13 `mtlx.*` STATIC_PARAMS (plus all dynamic material params from `build_dynamic_params`) write into `renderer.mtlx_overrides`, which the hash already covers wholesale. Deriving per-param contributions through `_get_nested` would *change* behavior: `_get_nested("mtlx.x")` falls back to the active material's authored default when unset, so material loads would perturb the hash where today they ride `_material_version`. The derivation therefore skips paths under a provider's declared coverage (provider carries `covers_prefix="mtlx."`), and the invariant test counts those params as covered-by-provider. Dynamic params need no registration at all — the dict hash covers any future material uniform automatically.

*Alternatives considered:* per-param `_get_nested` hashing — behavior drift (above) plus per-frame cost of ~50 dict/material-scan resolutions; keeping the dict hash is both cheaper and exactly today's semantics.

### D4: Keep `_current_state_hash` as the entry point; derive its tuple from the registry

The method survives with its name, signature, and `hash(tuple)` return — the consumer (renderer.py:11051), the MLT-seed design note that references it (renderer.py:2452), and the `mirrored-camera-rendering` spec scenario naming it all stay valid. The body becomes: for each `build_all_params(renderer)`-static param with `resets_accumulation=True` and no covering provider, append its value coerced **per the legacy per-field cast, not blindly by `kind`**; then append each provider's extracted value. Coercion defaults from `kind` (`continuous` → `float`, `discrete` → `int`; `hash(False) == hash(0)` makes the one `bool()` cast hash-equal), but four continuous params — `restir_m_light`, `restir_m_bsdf`, `restir_spatial_k`, `restir_m_cap` (params.py:127–131) — are `int()`-cast in today's hash (renderer.py:10928–10930), and that difference is observable: `_apply_saved_params` clips without quantizing (params.py:358), so a fractional value from a hand-edited settings/preset JSON exists in state space, and today `4.2 → 4.7` does NOT reset accumulation while a naive `float()` coercion would. These four therefore carry an explicit `int` coercion override declared beside their `resets_accumulation` flag in the registry, preserving the legacy cast exactly. (The D5 identity-set gate compares field *identity*, not coercion, so the invariant test spot-checks these four coercions explicitly.) Attribute resolution uses the plain getattr chain (what the literal tuple compiles to anyway); the resolver list can be built once at first call since STATIC_PARAMS is fixed.

*Alternatives considered:* new method + shim — churn with no benefit; hashing a dict of `{name: value}` — ordering stability for free with a tuple, and tuples are what `hash()` wants.

### D5: Invariant enforcement = one hostless test module comparing contributor sets

`tests/test_accum_reset_registry.py` (pattern proven by tests/test_cli_common.py, test_backend_select.py — imports only, no GPU, no Renderer instance):

1. **Behavior-preservation gate:** the derived contributor identity set — `{param paths with resets_accumulation=True}` minus provider-covered, plus `{provider names}` — equals a frozen expected set transcribed from the legacy tuple at renderer.py:10906–10968. This is the "resets exactly when it did before" proof at the field-set level.
2. **Fail-safe default:** `ParamSpec("x", "y", "continuous").resets_accumulation is True`.
3. **Exclusion list is closed:** exactly `{tonemap_index, exposure}` carry `False`.
4. **Coverage rule:** every `mtlx.*` static param is covered by a provider with a matching `covers_prefix`.
5. **Coercion spot-check (D4):** `restir_m_light`, `restir_m_bsdf`, `restir_spatial_k`, `restir_m_cap` carry the `int` coercion override (fractional change within the same integer must not perturb the hash), and no other continuous param does.

The four existing source-inspection tests are re-pointed at the registry (e.g. "`integrator_index` is a `resets_accumulation` param" instead of "the string `integrator_index` appears in the method body") — same guarantee, sturdier assertion. After this change the gate for a *new* param is automatic (default True); the frozen set in test 1 is the only thing a legitimate contributor-set change must consciously edit — one file, reviewed, never silent.

## Risks / Trade-offs

- **[Risk] Per-frame cost of registry-driven derivation** (~30 getattr chains + provider calls at up to 60 Hz) → Mitigation: this is what the literal tuple already does; the only formerly-cheap part that could regress is attribute resolution through generic paths, so the resolver list (path → attrgetter) is built once. The expensive contributor (`mtlx_overrides` sort) is byte-identical to today.
- **[Risk] Hash value changes across the refactor** → Mitigation: value is process-local (`_last_state_hash` in-memory only, reset comparison is equality-within-process); the MLT seed is documented as deliberately *not* derived from this hash (renderer.py:2452) and test_mlt_host.py:217 already guards that. Nothing persists the value.
- **[Risk] A future post-process param inherits the True default and needlessly resets** → Mitigation: visible and benign by construction (D1); the closed exclusion-list test (D5.3) forces the opt-out to be explicit and reviewed.
- **[Risk] Frozen expected set in the test rots into a second hand-curated list** → Mitigation: it is a *test fixture*, not a code path — drift fails loudly (set inequality) instead of silently, which inverts today's failure mode; it changes only when the contributor set legitimately changes, in the same commit, in one place.
- **[Risk] Provider extractors touch renderer internals from params.py, inverting the import direction** → Mitigation: extractors are plain callables taking the renderer duck-typed (params.py already does this in `build_dynamic_params` / `_get_nested`); params.py gains no new imports.

## Migration Plan

Single change, no data migration, no flag. Land order (matches tasks.md): registry additions in params.py (inert — nothing reads them yet) → hostless transcription test against the frozen legacy set (green-first: it must pass from the registry alone before renderer.py is touched, proving the transcription) → `_current_state_hash` re-derivation → re-point the four source-inspection tests → comment sweep + docs. Rollback = revert; no persisted state involved.

**Cross-change coordination (renderer cluster):** sibling change `renderer-module-carveout` (Stage A) moves `Renderer._next_mlt_seed` — home of the MLT-seed independence note (renderer.py:2452) — into an `mlt_chain` module, and `tests/test_mlt_host.py` is touched by three concurrent changes (this one, `renderer-module-carveout`, `reflection-owned-byte-layouts`). Rule: whichever change lands second owns relocating/re-pointing the seed note and the test_mlt_host.py:217 guard to the then-current authority; each cluster change's tasks must update that file's assertions against whatever module owns the code at its merge time, not the layout at proposal time.

## Open Questions

- Should `film_max_component` eventually become a real `ParamSpec` (GUI-adjustable clamp) instead of a provider? Out of scope here; the provider entry makes the promotion a two-line move later.
- Should the explicit `accum_frame = 0` sites (resize, scene rebuild) migrate into providers (e.g. a `(width, height)` provider) in a follow-up, so *all* reset causes are registry-visible? Deliberately not folded now — those sites also reallocate buffers, so the hash alone cannot replace them.
