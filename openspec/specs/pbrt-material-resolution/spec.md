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

The resolver SHALL NOT read a parameter pbrt itself does not define for that
material type. A read pbrt would ignore is skinny inventing behaviour, and
hardening one cements it.

A flavour gate SHALL NOT be used to freeze a divergence in **which parameter** a
lobe reads. The gate exists for reads that are one-sided because a target has no
input for the value; two flavours resolving one lobe from two different pbrt
parameters is a defect, and one of the two spellings is wrong.

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

#### Scenario: The resolver reads no parameter pbrt does not define

- **WHEN** the `subsurface` branch is compared against pbrt's
  `SubsurfaceMaterial::Create` parameter list, and the `coatedconductor` branch
  against `CoatedConductorMaterial::Create`
- **THEN** neither reads a parameter absent from it — `radius`, which pbrt
  defines only on shapes, is not read and the `subsurface_radius` lobe is derived
  from the resolved `mfp`; a top-level `roughness`, which
  `CoatedConductorMaterial::Create` does not define, is not read either

#### Scenario: No lobe reads a different parameter per flavour

- **WHEN** every flavour-gated read in the resolver is examined
- **THEN** each gate exists because one target has no input for the value, never
  because the two targets disagree about which pbrt parameter the lobe comes from

#### Scenario: One-sided and divergent reads are flavor-gated

- **WHEN** the resolver runs under the `usd` flavor on a material whose
  mtlx-only params (`transmittance`, `interface.eta` on the coated types) are
  texture-bound or malformed
- **THEN** those params are not read at all — no value, no note, and no
  EXACT→APPROX escalation is produced that today's UsdPreviewSurface path
  does not produce
- **AND** `reflectance` is no longer among them: the `usd` path already read it
  for the medium coefficient chain, so the gate suppressed its note rather than
  its read, and one resolution now serves both consumers on both flavors

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

### Requirement: No pbrt parameter binding raises or degrades silently

The resolver SHALL accept every pbrt parameter type a material parameter can
carry: a numeric constant, a `blackbody` temperature, an inline sampled spectrum,
a named spectrum, a spectrum file reference, and a texture binding. No binding
SHALL raise.

A binding whose authored value the resolver cannot carry SHALL degrade to that
parameter's default and record exactly one note that names the parameter and what
was unusable. A degradation SHALL NOT be silent — a substituted value with no
note and an `EXACT` status is a defect, not a degradation.

A parameter value SHALL NOT be read with an accessor that calls `float()` on the
raw token (`ParamSet.rgb`, `.floats`, `.float`, `.int`, `.ints`). Presence tests
and non-numeric reads (`get`, `string`, `bool`) are unaffected.

#### Scenario: The resolver holds no raising accessor

- **WHEN** `resolve_material` and the helpers it calls are scanned for
  `ParamSet` accessors that call `float()` on a token
- **THEN** none exist — a hostless AST check asserts this without naming any
  parameter, so a parameter added to any branch through a raw accessor fails the
  build regardless of which branch or which name it uses

#### Scenario: A subsurface material accepts an unusable binding

- **WHEN** a pbrt scene binds any of `sigma_a`, `sigma_s`, `reflectance`, `mfp`,
  `g` or `scale` on a `subsurface` material to a texture or a named spectrum
- **THEN** the import completes on both flavours, the parameter degrades to its
  pbrt default, and the report carries exactly one note naming it

#### Scenario: A named spectrum is substituted only where it means something

- **WHEN** a recognised metal name is bound to a reflectance-like parameter
- **THEN** its reflectance RGB is substituted with no note — an exact
  substitution, not a fallback
- **AND WHEN** the same name is bound to an absorption or scattering coefficient,
  where a reflectance is meaningless
- **THEN** the parameter degrades to its default with a note, mirroring the
  restriction `_IOR_PARAM_NAMES` places on the float side

#### Scenario: A legal non-numeric binding stops producing garbage

- **WHEN** a coefficient is bound as `"blackbody sigma_a" [6500]` or as an inline
  sampled spectrum `"spectrum sigma_a" [400 .1 700 .9]`
- **THEN** it reduces to the correct RGB — not to the raw tokens, which is what a
  `float()`-on-token accessor yields today for both forms without raising and
  without a note

#### Scenario: A degrading parameter is reported once

- **WHEN** a parameter is read by more than one consumer — `reflectance` feeds
  both the `subsurface_color` lobe and the medium coefficient chain
- **THEN** it is resolved once and the report carries exactly one note for it,
  not one per consumer and not zero

### Requirement: Parameter presence is independent of parameter readability

The precedence branch `subsurface_coefficients` selects SHALL depend only on
whether a parameter is present in the pbrt `ParamSet`, never on whether its value
could be read. This matches pbrt, which branches on
`GetSpectrumTextureOrNull` — non-null for a texture binding too.

A parameter that is present but unusable SHALL keep the branch it selects and
lose only its value. The resolver SHALL NOT treat an unreadable parameter as
absent.

Where pbrt refuses a partially authored parameter group outright, skinny SHALL
degrade that group as a unit rather than mix an authored member with a
substituted one.

#### Scenario: An unusable sigma pair keeps the explicit-sigma branch

- **WHEN** a `subsurface` material authors both `sigma_a` and `sigma_s` and binds
  either of them to a texture
- **THEN** the coefficients come from the explicit-sigma branch — not from the
  `reflectance` inversion and not from the Wholemilk defaults, either of which
  would silently change the physical model

#### Scenario: A half-unusable sigma pair degrades together

- **WHEN** a `subsurface` material binds `sigma_a` to a texture and gives
  `sigma_s` a usable value
- **THEN** both members degrade to the default pair, with one note naming the
  unusable member and stating the pair was replaced as a unit — pairing a
  substituted σ_a with an authored σ_s would combine two different materials, and
  pbrt refuses the half-authored case outright

#### Scenario: An absent parameter still falls through

- **WHEN** a `subsurface` material authors neither `sigma_a` nor `sigma_s`
- **THEN** the precedence falls through to `reflectance` if present, else to the
  Wholemilk defaults — the promoting accessors' defaults do not make an absent
  parameter look present

### Requirement: An unrepresentable binding is reported as skipped

A binding whose authored meaning the renderer cannot express at all SHALL be
reported as `SKIPPED`, not `APPROX`, so `report.has_unsupported` makes the CLI
exit non-zero.

A texture-bound medium coefficient is such a binding: pbrt evaluates it per
intersection, and skinny's imported interior medium is homogeneous. A spectral
reduction to RGB is not — it is a bounded fidelity loss and stays `APPROX`.

#### Scenario: A texture-bound coefficient fails the clean-import gate

- **WHEN** a `subsurface` material binds `sigma_a`, `sigma_s`, `reflectance` or
  `mfp` to a texture, and the scene is imported through the CLI
- **THEN** the material is still authored with its default coefficients, and the
  report records the binding as skipped so the CLI exit code is non-zero

#### Scenario: A spectral reduction stays approximate

- **WHEN** the same parameter is bound to a named or inline spectrum
- **THEN** the report records `APPROX`, not skipped — the value is reduced, not
  discarded

### Requirement: Numeric bindings are byte-identical

Routing the `subsurface` branch through the promoting accessors SHALL NOT change
importer output for any material that binds its parameters as `rgb` or `float`.

Bindings that produce a different value SHALL be covered by explicit fixtures
with recorded values, because the corpus contains none of them and its hash gate
cannot see them.

#### Scenario: The corpus does not move

- **WHEN** every scene in the parity corpus and the confirming suite is imported
  after the change
- **THEN** the `.usda` (plain and `-mtlx`) and `.mtlx` documents are
  byte-identical to those produced before it, including report notes and status

### Requirement: A coated material reads pbrt's spelling for each layer

Each lobe of a coated material SHALL read the parameter pbrt-v4 defines for that
lobe on that material type, identically on both export flavours.

`coatedconductor` SHALL read `conductor.roughness` (with
`conductor.uroughness` / `conductor.vroughness`) for the base metal and
`interface.roughness` for the coat. It SHALL NOT read a top-level `roughness`,
which `CoatedConductorMaterial::Create` does not define — pbrt refuses such a
scene outright (`"roughness": unused parameter`), so no valid pbrt scene carries
the value and no fallback to it can be correct.

`coateddiffuse` SHALL keep reading the top-level `roughness` for its coat, which
is what `CoatedDiffuseMaterial::Create` reads. The two coated types are
asymmetric in pbrt, and the resolver SHALL mirror that asymmetry rather than
unify it.

#### Scenario: Both flavours resolve one metal roughness

- **WHEN** a scene declares a `coatedconductor` with `conductor.roughness` and a
  different top-level `roughness`, and is imported under both flavours
- **THEN** both resolve the metal roughness from `conductor.roughness` alone, and
  the top-level value affects neither export

#### Scenario: The coated types stay asymmetric

- **WHEN** a `coateddiffuse` authors a top-level `roughness`
- **THEN** it still drives that material's coat — the fix to `coatedconductor`
  does not propagate to a type where pbrt reads the top-level spelling

#### Scenario: Metal anisotropy survives the import

- **WHEN** a `coatedconductor` authors `conductor.uroughness` and
  `conductor.vroughness` with different values
- **THEN** the resolved roughness is the unreduced pair, so each adapter applies
  its own reduction policy — the UsdPreviewSurface geometric-mean collapse with
  its note, the standard_surface mean plus `specular_anisotropy` — rather than
  the pair being dropped silently as it is today

### Requirement: One roughness calibration chain, parameterised by spelling

The roughness calibration chain SHALL exist once and be reused for every
roughness spelling, including a prefixed one. The chain is texture promotion,
then `remaproughness`, then alpha, then an unreduced `ResolvedRoughness`. A
material branch SHALL NOT re-implement any part of it.

`remaproughness` SHALL be read unprefixed: pbrt reads one per material, and it
governs every roughness on that material.

#### Scenario: A prefixed roughness gets the whole chain

- **WHEN** the resolver reads a prefixed roughness such as `conductor.roughness`
- **THEN** it goes through the same chain as the top-level spelling — the same
  texture promotion, the same `remaproughness` handling, the same note wording,
  and the same anisotropic pair — because a second copy is what dropped the
  anisotropic spellings in the first place

#### Scenario: The chain has no second implementation

- **WHEN** the material module is scanned for the roughness calibration
  arithmetic (`pbrt_roughness_to_alpha` composed with `alpha_to_usd_roughness`)
- **THEN** it appears only inside the shared resolver, not in any material branch

### Requirement: A material the gates never render is not covered

A material type SHALL NOT be treated as covered by the gates when no corpus
scene and no confirming-suite scene renders it. A defect in such a type SHALL get
a confirming-suite scene before it gets a fix, and that scene SHALL discriminate
the defect: the image changes if the wrong parameter is read.

#### Scenario: The coated-metal scene distinguishes the two spellings

- **WHEN** the confirming-suite scene for `coatedconductor` is authored
- **THEN** its coat roughness and its metal roughness differ by enough that
  resolving one in place of the other changes the rendered image, and a hostless
  assertion pins that the two resolved values are not equal — so the scene cannot
  silently decay into passing whichever parameter is read
- **AND** the scene SHALL NOT author the top-level `roughness` to stage the
  defect: pbrt refuses such a scene, so it would have no reference image. The
  discrimination SHALL come from the parameter's ABSENCE — the wrong read falls
  to its default of 0, so a metal authored ROUGH renders as a mirror

#### Scenario: The material reaches both gate classes

- **WHEN** the scene is registered in the corpus manifest
- **THEN** it carries a pbrt-truth reference from the pinned pbrt v4 build and a
  plain/`-mtlx` authoring-equivalence pair, so the equivalence gate covers this
  material instead of passing vacuously as it does while no scene reaches it

