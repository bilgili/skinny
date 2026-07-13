# render-parity-matrix — delta

## ADDED Requirements

### Requirement: Suite scenes under tests/assets are corpus scenes
The corpus manifest SHALL accept scene entries whose sources live under
`tests/assets/suite/` (USD assets and pbrt counterparts), registered in the single existing
manifest (`tests/pbrt/corpus/manifest.json`) with references in the single existing refs
directory. Suite scenes SHALL be swept by the same gates and combo enumeration as existing
corpus scenes — no parallel harness.

#### Scenario: a suite scene is swept like a corpus scene
- **WHEN** the parity matrix runs with a manifest entry pointing at
  `tests/assets/suite/<scene>/<scene>.usda`
- **THEN** the scene is rendered for every valid combo and evaluated by the pbrt-truth
  (when a ref exists) and self-consistency gates identically to a `tests/pbrt/corpus/` scene

### Requirement: Authoring-equivalence gate class
The harness SHALL support a third gate class beside pbrt-truth and self-consistency:
authoring equivalence, which renders a scene's plain-USD and MaterialX variants with the
anchor combo and asserts the standard metric battery between them is within the scene's
recorded equivalence tolerance. Scenes with only one authoring variant SHALL carry a
recorded skip reason for this gate.

#### Scenario: equivalent authorings agree
- **WHEN** a dual-authored scene's variants are both rendered with the anchor combo
- **THEN** the metric battery between the two images passes the recorded equivalence
  tolerance

#### Scenario: single-authoring scenes are recorded skips
- **WHEN** the equivalence gate enumerates a scene with a recorded single-authoring reason
- **THEN** the gate reports a skip with that reason instead of failing or passing silently

### Requirement: Coverage meta-test spans suite gate classes
The coverage meta-test SHALL fail the build when a suite scene lacks a disposition for any
applicable gate class: pbrt-truth (ref EXR or recorded skip reason), authoring equivalence
(tolerance or recorded skip reason), and — for scenes participating in furnace closure — a
furnace disposition (tolerance or recorded baseline).

#### Scenario: a scene missing a gate disposition fails the build
- **WHEN** a new suite scene is added with no pbrt ref and no recorded pbrt skip reason
- **THEN** the coverage meta-test fails on the hostless tier
