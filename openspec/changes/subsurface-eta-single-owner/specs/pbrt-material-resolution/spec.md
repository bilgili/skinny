# pbrt-material-resolution Specification

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
values for a material either. It SHALL obtain subsurface medium coefficients
from the resolver and add only what the USD stage owns: the mm-per-scene-unit
convention, and the `ior` key the renderer reads the boundary IOR from.

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

- **WHEN** `pbrt/media.py` is scanned for `ParamSet` reads of a material's
  parameters (`params.get`/`.string`/`.rgb`/`.floats`/`.float`/`.bool`) inside
  the subsurface path
- **THEN** none exist — a hostless test asserts the subsurface medium payload is
  built by calling the resolver, so the precedence order, the defaults and the
  named-spectrum rule cannot drift between the two modules

## ADDED Requirements

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
