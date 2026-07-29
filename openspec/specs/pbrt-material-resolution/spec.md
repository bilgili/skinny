# pbrt-material-resolution Specification

## Purpose
TBD - created by archiving change pbrt-material-shared-resolver. Update Purpose after archive.
## Requirements
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

The stage-emission module (`pbrt/media.py`) MUST NOT read pbrt `ParamSet`
values for a material either, and MUST NOT be given one. It SHALL consume the
resolver's resolved intermediate and add only what the USD stage owns: the
mm-per-scene-unit convention, and the `ior` key the renderer reads the boundary
IOR from. The keys it emits SHALL be enumerated at the emission site, so a key
added to the resolver cannot reach `Material.parameter_overrides` unrouted.

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

#### Scenario: Stage emission holds no second copy of the coefficient chain

- **WHEN** `pbrt/media.py`'s subsurface path is scanned for `ParamSet` reads
  (`params.get`/`.string`/`.rgb`/`.floats`/`.float`/`.bool`/`.int`/`.ints`) and
  for a `ParamSet`-shaped parameter
- **THEN** neither exists — hostless tests assert the emission function takes the
  resolved intermediate and nothing else, so the precedence order, the defaults
  and the named-spectrum rule cannot drift between the two modules, and no
  renamed local can smuggle a `ParamSet` past the syntactic check

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

### Requirement: One eta resolution per subsurface material

A pbrt `subsurface` material SHALL resolve its `eta` parameter exactly once.
The resolved value SHALL feed both the boundary IOR lane (`ior`) and the medium
coefficient set (`subsurface_eta`).

The resolution SHALL accept every pbrt parameter type `eta` can carry: a numeric
constant, a named spectrum, and a texture binding. A recognised named glass
SHALL resolve to its d-line (589.3 nm) refractive index. An unrecognised named
spectrum, an unreadable spectrum file, and a texture binding SHALL each degrade
to the pbrt default with a note. No `eta` value SHALL raise.

#### Scenario: Named-spectrum eta imports and reaches the renderer's IOR lane

- **WHEN** a pbrt scene declares
  `Material "subsurface" "spectrum eta" "glass-LASF9"` and the importer runs on
  both the plain and the `-mtlx` flavor
- **THEN** the import completes, and the authored `skinnyOverrides` carry
  `ior == subsurface_eta == 1.85004`, the d-line index of LASF9 — so
  `material_pack.pack_flat_material`, which reads the boundary IOR from `ior`
  and never reads `subsurface_eta`, packs the authored glass

#### Scenario: The two lanes cannot diverge

- **WHEN** a pbrt `subsurface` material is resolved with any `eta` — numeric,
  named glass, unrecognised spectrum, spectrum file, or texture-bound
- **THEN** the resolved `ior` lobe and the `subsurface_eta` override hold the
  same float, because one resolution produced both

#### Scenario: The import performs exactly one eta resolution

- **WHEN** a pbrt scene containing one `subsurface` material is imported, and the
  calls to the resolving accessor for `eta` are counted
- **THEN** the count is one, and the authored UsdPreviewSurface `ior` shader
  input equals the `ior` on `skinnyOverrides` — the emission module consumes the
  resolver's result instead of re-reading the parameter, so the value the loader
  applies last is the value the resolver produced

#### Scenario: A degrading eta is reported once

- **WHEN** a `subsurface` material binds `eta` to a texture, or names a spectrum
  the tables do not carry
- **THEN** the import report carries exactly one note for that read, not one per
  reader

#### Scenario: Numeric eta is byte-identical

- **WHEN** every scene in the parity corpus and the confirming suite is
  imported after the change
- **THEN** the `.usda` (plain and `-mtlx`) and `.mtlx` documents are
  byte-identical to those produced before it, including report notes and status

