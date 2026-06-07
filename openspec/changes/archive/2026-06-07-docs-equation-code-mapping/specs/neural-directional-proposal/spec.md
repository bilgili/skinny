## ADDED Requirements

### Requirement: Equations are backed by embedded shader code and a symbol map

`docs/NeuralGuiding.md` SHALL present, beneath each governing equation, a
generated Slang excerpt of the function that implements it (its signature plus
the line(s) performing the equation's arithmetic, embedded via the
`docs-equation-code-embedding` generator) and a per-equation table mapping each
math symbol to the Slang identifier that carries it. This is in addition to the
existing equation → implementation mapping. The embedded excerpts SHALL be generated from the shipped shaders
(`sampling/neural_flow.slang`, `sampling/neural_proposal.slang`,
`sampling/proposal.slang`, `integrators/path_record.slang`), not hand-transcribed.

#### Scenario: Each equation shows the code that computes it

- **WHEN** a reader views an equation in `docs/NeuralGuiding.md` (e.g. the
  rational-quadratic spline `y = y_k + h_k(sθ²+…)/(…)` or the mixture pdf `p_mix`)
- **THEN** a generated fenced `slang` excerpt of the implementing function
  (`nf_rqs_fwd`, `mixtureProposalPdf`, …) appears beneath it, tagged with its
  source file

#### Scenario: Math symbols are mapped to Slang identifiers

- **WHEN** a reader needs the variable correspondence for an equation whose Slang
  names diverge from the math symbols
- **THEN** a per-equation table maps each symbol to its Slang name
  (e.g. `θ → theta`, `s → delta`, `h_k → hgt`, `w_k → w`, `d_k → d0`,
  `d_{k+1} → d1`, `y_k → y0`)

#### Scenario: Embedded code cannot silently drift

- **WHEN** a documented neural-flow shader function is renamed or moved
- **THEN** regenerating the doc (or running the embed check) surfaces the change
  rather than leaving a stale snippet, because the excerpts are generated from
  marked source regions
