## ADDED Requirements

### Requirement: Equations are backed by embedded shader code and a symbol map

`docs/ReSTIR.md` SHALL present, beneath each governing equation, a generated
Slang excerpt of the function that implements it (its signature plus the line(s)
performing the equation's arithmetic, embedded via the
`docs-equation-code-embedding` generator) and a per-equation table mapping each
math symbol to the Slang identifier that carries it. This is in addition to the
existing equation → implementation mapping. The embedded excerpts SHALL be generated from the shipped shaders
(`restir/reservoir.slang`, `restir/light_ris.slang`, `restir/restir_primary.slang`,
`sampling/reuse.slang`), not hand-transcribed.

#### Scenario: Each equation shows the code that computes it

- **WHEN** a reader views an equation in `docs/ReSTIR.md` (e.g. the contribution
  weight `W = Σwᵢ/(M·p̂)` or the GRIS `m_s`)
- **THEN** a generated fenced `slang` excerpt of the implementing function
  (`reservoirFinalize`, `restirSpatial`, …) appears beneath it, tagged with its
  source file

#### Scenario: Math symbols are mapped to Slang identifiers

- **WHEN** a reader needs the variable correspondence for an equation
- **THEN** a per-equation table maps each symbol to its Slang name
  (e.g. `W → r.W`, `p̂(y) → r.pHat`, `M → r.M`, `Σwᵢ → r.wSum`)

#### Scenario: Embedded code cannot silently drift

- **WHEN** a documented ReSTIR shader function is renamed or moved
- **THEN** regenerating the doc (or running the embed check) surfaces the change
  rather than leaving a stale snippet, because the excerpts are generated from
  marked source regions
