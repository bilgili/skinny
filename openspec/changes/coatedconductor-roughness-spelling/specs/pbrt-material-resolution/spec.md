# pbrt-material-resolution Specification

## ADDED Requirements

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
