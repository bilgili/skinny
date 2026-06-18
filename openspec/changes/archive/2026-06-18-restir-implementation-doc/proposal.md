## Why

ReSTIR DI shipped (archived changes `restir-di` and `restir-control-group`) and
is the renderer's first non-identity reuse mode, but it has no dedicated
technical reference. Its workings are scattered across an internal brainstorm
design doc (`docs/superpowers/specs/2026-06-02-restir-di-design.md`), dense
shader-comment derivations, and short mentions in `Architecture.md` /
`Wavefront.md`. ReSTIR is also the subtlest estimator in the codebase — RIS
weights, the GRIS generalized balance heuristic, deferred visibility, the
canonical-integration gate — so the absence of a single authoritative,
paper-cited walkthrough makes it the easiest part of the renderer to break or
misuse. This change adds that reference, at the same depth as
`SkinRendering.md`.

## What Changes

- **New `docs/ReSTIR.md`** — a detailed, paper-cited implementation reference for
  ReSTIR DI, structured as:
  - **Stages of rendering** — where ReSTIR sits in the wavefront pipeline and the
    three-pass `fill → spatial → resolve` flow at the primary hit, with the
    canonical-integration gate and deferred visibility.
  - **Equations** — RIS weights and the contribution weight `W`; the unshadowed
    unweighted target `p̂ = lum(f·Le)`; the balance-heuristic mixture source pdf
    over light + BSDF candidates; the multi-reservoir RIS merge; the GRIS
    generalized balance heuristic `m_s`; the biased `ΣM` combination; the DI
    reconnection (identity Jacobian).
  - **Implementations** — each equation mapped to the exact shader symbol/file
    (`restir/reservoir.slang`, `restir/light_ris.slang`,
    `restir/restir_primary.slang`, `sampling/reuse.slang`) and the host
    orchestration (`vk_wavefront.py::RestirDiPass`, `renderer.py`,
    `sampling/reuse.py`, `sampling/registry.py`).
  - **Design choices** — the six locked decisions (regimes, primary-hit scope,
    unified light domain, unbiased default + biased toggle, canonical
    integration, wavefront-only) plus the implementation deviations (GRIS over
    bare `1/Z`, spatial-only default on the progressive accumulator, octahedral
    env refs, separate RNG stream).
  - **GUI** — the dedicated **ReSTIR** control group and each of its eight
    controls (`Reuse`, regime, combine, `M light`, `M bsdf`, neighbours, radius,
    `M cap`), what each maps to in the push constant / flags, and its effect.
  - **References** — the papers the implementation draws on (Talbot 2005, Bitterli
    2020, Lin 2022, Veach 1997, and the Wyman ReSTIR course), cited inline at the
    equations they back.
- **SVG pipeline diagram** — verify and, if stale, update the existing
  `docs/diagrams/restir_pipeline.svg` so it matches the shipped
  `fill → spatial → resolve` passes; embed it in `docs/ReSTIR.md` (SVG only, per
  the repo diagram rule).
- **Cross-links** — point `Architecture.md`, `Wavefront.md`, and `README.md` at
  the new reference, and add it to the doc list in `CLAUDE.md`.
- No code, shader, parameter, or behavioural change.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `restir-di`: adds a requirement that the shipped ReSTIR DI implementation be
  backed by a maintained technical reference document (`docs/ReSTIR.md`) covering
  its stages, equations, implementation mapping, design rationale, GUI controls,
  and source references. No behavioural requirement changes.

## Impact

- **Docs (new):** `docs/ReSTIR.md`.
- **Docs (edit):** `docs/diagrams/restir_pipeline.svg` (verify/update),
  `docs/Architecture.md`, `docs/Wavefront.md`, `README.md`, `CLAUDE.md`
  (cross-links + doc list).
- **Code:** none. No shader recompile, no test or behaviour change.
- **Source of truth:** distilled from the shipped shaders
  (`shaders/restir/*`, `shaders/sampling/reuse.slang`), host
  (`vk_wavefront.py`, `renderer.py`, `sampling/`), params (`params.py`), the
  archived design doc, and the `restir-di` spec. The doc must be kept in sync
  when those change.
