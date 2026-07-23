# pbrt-material-resolution

## ADDED Requirements

### Requirement: Single shared resolver for pbrt material parameters

pbrt-param interpretation SHALL live in exactly one resolver: which params are
read per material type, with what defaults, texture promotion, named-spectrum
substitution, roughness calibration, and subsurface coefficient precedence.
The resolver maps a pbrt material to a target-agnostic resolved intermediate
(lobes as `ParamValue`s, unreduced per-axis roughness, overrides, status,
notes). Both authoring mappers (`map_material` → UsdPreviewSurface and
`map_material_mtlx` → MaterialX standard_surface) MUST consume this resolved
form; neither mapper may read pbrt `ParamSet` values directly except through
the resolver.

#### Scenario: New pbrt param is wired once

- **WHEN** a new pbrt material parameter is added to a material type's
  interpretation in the resolver
- **THEN** both the UsdPreviewSurface export and the MaterialX sidecar export
  reflect it without any per-target duplication of the param read, default, or
  note text

#### Scenario: Adapters contain only target-specific emission

- **WHEN** the material-mapping module is scanned after the refactor for pbrt
  param accesses (`params.get`/`.string`/`.rgb`/`.floats`/`.bool`,
  `get_float_texture`, `get_spectrum_texture`) outside the resolver
- **THEN** none exist — a hostless test asserts the adapters perform zero
  `ParamSet` reads, containing only target-vocabulary translation
  (input renaming, anisotropy reduction policy, transmission/emission
  encoding, value_type derivation, dropping unexpressible lobes)

#### Scenario: One-sided and divergent reads are flavor-gated

- **WHEN** the resolver runs under the `usd` flavor on a material whose
  mtlx-only params (`transmittance`, subsurface `reflectance`/`radius`,
  `interface.eta` on the coated types) are texture-bound or malformed
- **THEN** those params are not read at all — no value, no note, and no
  EXACT→APPROX escalation is produced that today's UsdPreviewSurface path
  does not produce

### Requirement: Refactor preserves importer output byte-identically

The extraction MUST NOT change importer output: for every scene in the
confirming suite and the parity corpus, the `.usda` (plain and `-mtlx`
flavors) and `.mtlx` documents produced after the refactor SHALL be
byte-identical to those produced before it, including report notes and
statuses. Recorded copy drifts between the two pipelines (e.g. the
`coatedconductor` base-metal roughness param spelling) SHALL be preserved
as-is and parameterized explicitly, not silently fixed.

#### Scenario: Byte-identical import diff gate

- **WHEN** the importer is run over the suite and corpus scenes before and
  after the refactor and the outputs are diffed
- **THEN** the diff is empty and no committed `.usda` fixture under `tests/`
  requires regeneration

#### Scenario: Authoring-equivalence gate stays green

- **WHEN** the confirming-suite authoring-equivalence gate
  (`tests/pbrt/test_suite.py`, plain-USD ≡ MaterialX) runs on the refactored
  importer
- **THEN** every equivalence pair passes within its recorded tolerances with
  no baseline or tolerance change

### Requirement: Resolver is hostless-testable

The resolver SHALL be exercisable without a GPU, USD stage, or MaterialX
runtime: unit tests MUST assert the resolved intermediate directly from pbrt
`ParamSet` inputs for every supported material type, including named-spectrum
substitution (7 metals / 7 glasses d-line IOR), texture-bound params,
anisotropic roughness, and subsurface coefficient precedence.

#### Scenario: Hostless resolver unit tests

- **WHEN** the resolver unit tests run under plain `pytest` with no GPU
  markers
- **THEN** they pass, covering each material-type branch and each shared
  accessor path at the resolved-form level
