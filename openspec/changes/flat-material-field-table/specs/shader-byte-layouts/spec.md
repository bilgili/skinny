# shader-byte-layouts (delta)

## ADDED Requirements

### Requirement: Flat-material and std-surface records are owned field by field

The byte-layout authority SHALL own the flat-material and std-surface records
down to **named fields**, not only their strides and row offsets. Each field
SHALL be declared once as name → (row, lane, kind, default), with row offsets
derived from the Slang declaration and lane assignment — which cannot be
derived from an opaque `float4` row — declared in the table and pinned by a
permanent golden captured from the pre-change packer. Packing SHALL be keyed by
name rather than by argument position, and an override key absent from the
table SHALL be rejected rather than silently dropped. The MSL packing variants
SHALL consume the same table with MSL offsets; no second table may exist.

#### Scenario: Transposition of two same-typed fields fails

- **WHEN** two fields of the same type exchange positions in the packed record
- **THEN** the name→byte-offset golden fails, whereas the size-equality assert
  that guards these records today would pass

#### Scenario: An unknown override key is rejected

- **WHEN** a material override carries a key that is not in the field table
- **THEN** packing reports the unknown key, instead of ignoring it and
  producing a materially different render with no diagnostic

#### Scenario: Packed bytes are unchanged by the refactor

- **WHEN** the same material inputs are packed before and after the change
- **THEN** the emitted bytes are identical, for both the scalar and the MSL
  variants

### Requirement: One vocabulary for material override keys

The override key vocabulary SHALL have one owner — the field table — and the
dialect alias tables SHALL be projections of it rather than independent
restatements. The three tables that mirror it today
(`usd_loader._STD_SURFACE_TO_FLAT` with 5 entries,
`mtlx_synthesis._STD_SURFACE_TO_FLAT_PACK` with 12, and
`mtlx_synthesis._PREVIEW_SURFACE_FLAT_KEYS`) SHALL read the table, and their
"keep in sync" comments SHALL become assertions. Before collapsing them, each
entry on which the tables currently disagree SHALL be individually verified,
since the disagreement is benign today only because the differing names happen
to be spelled identically across dialects.

#### Scenario: Alias tables cannot drift

- **WHEN** a field is added to or removed from the field table
- **THEN** the dialect alias tables reflect it without a separate edit, and a
  test fails if any table restates a mapping the field table already owns

#### Scenario: Corpus keys are surveyed before rejection is enforced

- **WHEN** the pbrt corpus and the confirming-scene suite are packed in
  report-only mode against the field table
- **THEN** every override key not present in the table is listed and
  individually resolved — author fixed, or key added — before unknown-key
  rejection is enabled

### Requirement: Override merge and derivation are ordered once

Override merge and the derivations that depend on it SHALL be ordered once at
the intake seam, so that no derivation runs twice — that is, the merge of
`customData["skinnyOverrides"]` into a material's overrides, followed by
opacity-from-transmission, opacity-from-subsurface and coat canonicalisation. The current re-run of the
subsurface-to-opacity derivation, needed because the first derivation ran
before the customData merge, SHALL be removed rather than preserved.

#### Scenario: No derivation runs twice

- **WHEN** a material carrying `skinnyOverrides` is read
- **THEN** each derivation runs exactly once, after the merge, and the
  resulting overrides are identical to those the current double-derivation
  produces
