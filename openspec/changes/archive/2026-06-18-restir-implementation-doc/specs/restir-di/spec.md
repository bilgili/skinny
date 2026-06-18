## ADDED Requirements

### Requirement: Implementation reference documentation

The ReSTIR DI implementation SHALL be backed by a maintained technical reference
document at `docs/ReSTIR.md`. The document SHALL describe the shipped behaviour
(it is distilled from the shaders, host code, and parameters — not from the
pre-implementation brainstorm) and SHALL cover, at minimum: the rendering stages
and where ReSTIR sits in the wavefront pipeline; the governing equations; the
mapping from each equation to the shader symbol that realizes it; the design
choices and their rationale; the GUI controls and their effects; and references
to the source papers. Diagrams in the document SHALL be SVG. The document SHALL
be kept in sync when the ReSTIR shaders, host pass set, or parameters change.

#### Scenario: Reference document exists and is discoverable

- **WHEN** a contributor looks for how ReSTIR DI works
- **THEN** `docs/ReSTIR.md` exists and is cross-linked from `docs/Architecture.md`,
  `docs/Wavefront.md`, and `README.md`

#### Scenario: Stages, equations, and their implementations are documented

- **WHEN** a reader needs the ReSTIR pipeline and math
- **THEN** the document describes the `fill → spatial → resolve` primary-hit
  passes and the canonical-integration gate, presents the RIS weight and
  contribution weight, the unshadowed mixture-target `p̂`, the multi-reservoir
  merge, the GRIS `m_s` weight, and the biased `ΣM` combination, and maps each to
  the shader symbol/file that implements it
  (`restir/reservoir.slang`, `restir/light_ris.slang`,
  `restir/restir_primary.slang`, `sampling/reuse.slang`)

#### Scenario: Design choices and GUI controls are explained

- **WHEN** a reader needs the rationale or the runtime controls
- **THEN** the document explains the locked design decisions (regimes,
  primary-hit scope, unified light domain, unbiased default + biased toggle,
  canonical integration, wavefront-only) with the shipped deviations (GRIS,
  spatial-only default), and documents each control in the dedicated **ReSTIR**
  group (`Reuse`, regime, combine, `M light`, `M bsdf`, neighbours, radius,
  `M cap`) with the push-constant field/flag it maps to and its effect

#### Scenario: Source papers are cited

- **WHEN** a reader wants the theory behind an equation
- **THEN** the document cites the relevant papers (Talbot 2005 for RIS, Bitterli
  2020 for ReSTIR DI and the reservoir merge, Lin 2022 for the GRIS balance
  heuristic and the reconnection Jacobian, Veach 1997 for MIS) at the equations
  they back, with a collected references section
