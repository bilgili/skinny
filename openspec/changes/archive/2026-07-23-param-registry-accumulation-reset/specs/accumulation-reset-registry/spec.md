# accumulation-reset-registry

## ADDED Requirements

### Requirement: Parameters declare their accumulation-reset semantics

Every `ParamSpec` in the shared parameter registry (`params.py`) SHALL carry a `resets_accumulation` flag declaring whether a change to that parameter resets progressive accumulation. The flag SHALL default to `True`, so an undeclared new parameter fails safe (spurious visible reset) rather than silently accumulating stale state. Parameters that do not reset accumulation (the post-process controls `tonemap_index` and `exposure`) SHALL opt out explicitly.

#### Scenario: New parameter defaults to resetting accumulation

- **WHEN** a `ParamSpec` is constructed without an explicit `resets_accumulation` argument
- **THEN** `resets_accumulation` is `True` and the parameter contributes to the accumulation state hash

#### Scenario: Post-process parameters opt out explicitly

- **WHEN** the static parameter registry is enumerated for entries with `resets_accumulation=False`
- **THEN** exactly the paths `tonemap_index` and `exposure` are returned, matching their existing exclusion from the state hash

### Requirement: Non-param state contributors are registered as named providers

Accumulation-affecting state that is not a user-facing parameter (camera signature, MaterialX override dict, material version, volume-grid identity, film clamp, camera mirror, USD time code, SPPM tuning overrides) SHALL be declared in a single explicit provider registry (`ACCUM_STATE_PROVIDERS`) beside the parameter registry, each entry naming the contributor and supplying its value extractor. The provider registry SHALL be importable and enumerable without constructing a `Renderer` and without a GPU. A provider MAY declare that it covers a parameter-path prefix (the `mtlx_overrides` provider covers all `mtlx.*` parameters wholesale, preserving today's hash semantics for unset overrides and covering dynamic material parameters automatically).

#### Scenario: Provider registry is hostless-enumerable

- **WHEN** a test imports the provider registry with no GPU and no `Renderer` instance
- **THEN** it can enumerate every provider name and coverage prefix declared for the accumulation hash

#### Scenario: MaterialX parameters are covered by the overrides provider

- **WHEN** the hash contributor set is derived
- **THEN** `mtlx.*` parameters contribute through the `mtlx_overrides` provider's wholesale dict value rather than as individual per-parameter values, so an unset override keeps riding the material version exactly as before

### Requirement: The accumulation state hash is derived from the registry

`Renderer._current_state_hash()` SHALL derive its value from the registry — the `resets_accumulation` parameters (excluding provider-covered ones) plus the registered providers — rather than from a hand-maintained field tuple. The method name and its use as the accumulation-reset trigger SHALL be unchanged. The set of contributing fields SHALL be identical to the pre-change hand-curated tuple, and each field's value coercion SHALL match the legacy per-field cast (declared as a registry override where it differs from the `kind` default — the four continuous ReSTIR count params `restir_m_light`, `restir_m_bsdf`, `restir_spatial_k`, `restir_m_cap` keep their legacy `int()` cast), so accumulation resets exactly when it did before; the numeric hash value itself is process-local and MAY differ.

#### Scenario: Registered parameter change resets accumulation

- **WHEN** any parameter with `resets_accumulation=True` (e.g. `integrator_index`) changes between frames
- **THEN** `_current_state_hash()` changes and the renderer resets `accum_frame` to 0, exactly as before the refactor

#### Scenario: Opted-out parameter change preserves accumulation

- **WHEN** only `tonemap_index` or `exposure` changes between frames
- **THEN** `_current_state_hash()` is unchanged and progressive accumulation continues

#### Scenario: Legacy integer coercion is preserved for ReSTIR count params

- **WHEN** a ReSTIR count param (e.g. `restir_spatial_k`) changes fractionally within the same integer (e.g. 4.2 → 4.7, reachable via an unquantized saved-settings value)
- **THEN** the derived hash contribution is unchanged — the declared `int` coercion override reproduces the legacy `int()` cast, so accumulation does not reset, exactly as before

#### Scenario: Adding a parameter is registering it

- **WHEN** a developer adds a new `ParamSpec` to the static registry with no other edits
- **THEN** the parameter is a hash contributor with no change to `_current_state_hash`'s body

### Requirement: The reset invariant is hostless-testable

The invariant "every accumulation-affecting field contributes to the state hash, and only those" SHALL be assertable by a hostless unit test (no GPU, no `Renderer` construction), by comparing the registry-derived contributor identity set against a frozen expected set transcribed from the pre-change hash tuple. Existing source-inspection tests that substring-match the `_current_state_hash` body (SPPM, MLT, volume-grid selection tests) SHALL be re-pointed at registry data with equivalent or stronger assertions.

#### Scenario: Contributor set matches the legacy hash fields

- **WHEN** the hostless invariant test derives the contributor set from the registry
- **THEN** it equals the frozen field set of the pre-change hand-curated tuple (params minus provider-covered, plus provider names), proving reset behavior is preserved

#### Scenario: Contributor-set drift fails loudly

- **WHEN** a contributor is removed from the registry or a provider is deleted without updating the frozen expected set
- **THEN** the hostless invariant test fails with a set difference naming the drifted field
