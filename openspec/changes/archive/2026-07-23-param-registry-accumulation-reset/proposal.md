# Change: param-registry-accumulation-reset

## Why

`Renderer._current_state_hash` (renderer.py:10904–10969) is a hand-curated ~40-field tuple that decides when progressive accumulation resets; the invariant "every accumulation-affecting parameter is hashed" is enforced only by ~10 scattered comments (renderer.py:1762, 1796, 1876, 1919, 4619, 8934, …), so a new parameter that forgets to add itself produces a silent stale-accumulation bug — wrong convergence, no crash, no test failure.

## What Changes

- `ParamSpec` (params.py) gains a `resets_accumulation: bool` flag, **default `True`** (fail-safe: forgetting to declare produces a visible spurious reset, never silent staleness). Only the post-process knobs `tonemap_index` and `exposure` are marked `False` (they are already excluded from the hash — renderer.py:1876).
- A small explicit `ACCUM_STATE_PROVIDERS` registry in params.py declares the non-param hash contributors (camera signature, `mtlx_overrides`, `_material_version`, `_volume_grid_key`, `film_max_component`, `_camera_mirror`, USD clock time code, the three SPPM overrides) as named data — one home, hostless-importable.
- `_current_state_hash` is rewritten to DERIVE its tuple from the registry (params with `resets_accumulation=True` + providers) instead of a hand-maintained literal, preserving each field's legacy cast (the four continuous ReSTIR count params keep their `int()` coercion via an explicit override — see design D4). Method name, call site (renderer.py:11051), and reset behavior are unchanged; the hash *value* may differ.
- A hostless invariant test asserts the derived contributor set equals the frozen legacy field set (behavior = "resets exactly when it did before"), and the four existing source-inspection tests (test_sppm_selection.py, test_mlt_selection.py, test_mlt_host.py, test_volume_grid.py) are re-pointed at registry data instead of substring-matching the method body.
- Pure refactor: no shader changes, no new CLI flags, no front-end changes (`ParamSpec` grows a defaulted field only).

## Capabilities

### New Capabilities

- `accumulation-reset-registry` — registry-owned accumulation-reset semantics: every parameter and non-param state contributor declares whether it resets accumulation, the state hash is derived from those declarations, and the invariant is hostless-testable.

### Modified Capabilities

None. Existing specs (scene-sampling, restir-di, mirrored-camera-rendering, usd-animation-playback, photon-mapping, metropolis-light-transport, …) each require *their* state to reset accumulation; those requirements and their scenarios remain true verbatim — this change only relocates the mechanism that satisfies them. `mirrored-camera-rendering`'s scenario referencing `_current_state_hash()` stays valid because the method name is kept.

## Impact

- **Code:** `src/skinny/params.py` (flag + provider registry), `src/skinny/renderer.py` (`_current_state_hash` derivation; the scattered "hashed into `_current_state_hash`" comments become pointers to the registry).
- **Tests:** new hostless `tests/test_accum_reset_registry.py` (pattern: tests/test_cli_common.py — no GPU); updates to the four source-inspection tests listed above.
- **Front-ends:** app.py / render_session.py / web / Qt untouched — they consume `ParamSpec` positionally-compatible fields only.
- **No breaking changes.** Hash value is process-local and never persisted (`_last_state_hash` is in-memory; the MLT seed is deliberately independent of this hash — renderer.py:2452).
