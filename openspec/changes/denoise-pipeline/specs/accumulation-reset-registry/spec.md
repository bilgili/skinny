## MODIFIED Requirements

### Requirement: Parameters declare their accumulation-reset semantics

Every `ParamSpec` in the shared parameter registry (`params.py`) SHALL carry a `resets_accumulation` flag declaring whether a change to that parameter resets progressive accumulation. The flag SHALL default to `True`, so an undeclared new parameter fails safe (spurious visible reset) rather than silently accumulating stale state. Parameters that do not reset accumulation SHALL opt out explicitly.

The opt-out set SHALL be exactly the post-process controls — the controls applied over the finished accumulation buffer, which cannot change a sample: `tonemap_index`, `exposure`, `denoise_enabled`, and `denoise_strength`. A test SHALL enumerate the set, so a new opt-out is a deliberate edit rather than a drift.

#### Scenario: New parameter defaults to resetting accumulation

- **WHEN** a `ParamSpec` is constructed without an explicit `resets_accumulation` argument
- **THEN** `resets_accumulation` is `True` and the parameter contributes to the accumulation state hash

#### Scenario: Post-process parameters opt out explicitly

- **WHEN** the static parameter registry is enumerated for entries with `resets_accumulation=False`
- **THEN** exactly the paths `tonemap_index`, `exposure`, `denoise_enabled`, and `denoise_strength` are returned, matching their exclusion from the state hash

#### Scenario: Toggling the denoiser does not discard samples

- **WHEN** `denoise_enabled` or `denoise_strength` changes while a scene is accumulating
- **THEN** the accumulation state hash is unchanged and the accumulation frame counter keeps advancing
