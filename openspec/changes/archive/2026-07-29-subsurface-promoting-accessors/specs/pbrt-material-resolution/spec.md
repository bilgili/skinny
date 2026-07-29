# pbrt-material-resolution Specification

## ADDED Requirements

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

## MODIFIED Requirements

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
  `SubsurfaceMaterial::Create` parameter list
- **THEN** it reads no parameter absent from it — `radius`, which pbrt defines
  only on shapes, is not read, and the `subsurface_radius` lobe is derived from
  the resolved `mfp` it duplicates

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
