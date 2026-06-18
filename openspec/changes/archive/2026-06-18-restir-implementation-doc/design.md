## Context

ReSTIR DI is implemented and merged. The behaviour is fixed by the `restir-di`
spec; the mechanics live in four shaders (`restir/reservoir.slang`,
`restir/light_ris.slang`, `restir/restir_primary.slang`,
`sampling/reuse.slang`), the host pass set (`vk_wavefront.py::RestirDiPass`),
the renderer glue (`renderer.py`), the reuse plugin/registry
(`sampling/reuse.py`, `sampling/registry.py`), and eight GUI params
(`params.py`, grouped by `ui/build_app_ui.py`). The only existing prose is the
pre-implementation brainstorm (`docs/superpowers/specs/2026-06-02-restir-di-design.md`),
which predates the shipped code and lists deviations after the fact.

This change is documentation only. The design question is not *what to build* but
*how to structure a reference that is detailed, correct, paper-cited, and
resistant to drift* — matching the depth of `SkinRendering.md`.

## Goals / Non-Goals

**Goals:**
- One authoritative reference, `docs/ReSTIR.md`, covering: rendering stages,
  equations, equation→implementation mapping, design choices, GUI controls, and
  paper references.
- Every equation cited to the paper it comes from and mapped to the exact shader
  symbol/file that realizes it, so a reader can verify the doc against the code
  and the code against the doc.
- Honest about the shipped reality: the GRIS deviation from the brainstorm, the
  spatial-only default, the temporal-on-progressive caveat, and the
  wavefront-only capability gate.

**Non-Goals:**
- No code, shader, parameter, behaviour, or test change; no shader recompile.
- Not re-deriving ReSTIR/GRIS from first principles — cite the papers and show
  the renderer's specialization (DI, identity Jacobian).
- Not documenting the reprojected temporal regime as a feature (it is reserved /
  unimplemented; mention only as the P3 follow-on).
- Not a ReSTIR GI / path-reuse doc (out of scope of the shipped feature).

## Decisions

**D1 — Standalone `docs/ReSTIR.md`, not folded into `Wavefront.md`/`SkinRendering.md`.**
ReSTIR spans math, four shaders, the host pass set, and GUI; at the requested
depth it is a peer of `SkinRendering.md`, not a subsection. `Wavefront.md` and
`Architecture.md` get cross-links instead.
_Alternative rejected:_ a long section in `Wavefront.md` — buries the math and
couples two subjects that change independently.

**D2 — Shipped code is the source of truth; the brainstorm doc is historical.**
`2026-06-02-restir-di-design.md` is dated before implementation and its own
"Implementation outcome" section records deviations (GRIS over bare `1/Z`,
canonical integration as shipped, spatial-only default). `docs/ReSTIR.md`
documents what the code does today and references the brainstorm only for the
decision history.

**D3 — Pair every equation with the exact symbol that realizes it.**
Each equation block names its shader function/file (e.g. `W = wSum/(M·p̂(y))` →
`reservoirFinalize` in `restir/reservoir.slang`; `m_s` → `restirSpatial` in
`restir/restir_primary.slang`). This makes the doc verifiable and is the primary
anti-drift mechanism — a reader (or reviewer) can diff prose against the named
function.

**D4 — Cite papers inline at the equation, collect them in a References section.**
The lineage and where each is used:
- **Talbot et al. 2005**, *Importance Resampling for Global Illumination* — RIS,
  the per-candidate weight `w_i = p̂/p_src` and `W = Σw_i/(M·p̂)`.
- **Bitterli et al. 2020**, *Spatiotemporal Reservoir Resampling for Real-Time
  Ray Tracing with Dynamic Direct Lighting* (ReSTIR DI) — streaming weighted
  reservoir sampling and the multi-reservoir RIS merge (Alg. 4).
- **Lin et al. 2022**, *Generalized Resampled Importance Sampling: Foundations of
  ReSTIR* (GRIS) — the generalized balance-heuristic MIS weight `m_s` and the
  reconnection/shift Jacobian (here identity for DI).
- **Veach 1997**, *Robust Monte Carlo Methods…* (thesis) — MIS / the balance
  heuristic, backing the light+BSDF mixture source pdf.
- **Wyman et al.**, *A Gentle Introduction to ReSTIR* (SIGGRAPH course) — the
  approachable secondary reference, named once as further reading.

**D5 — Reuse the existing SVG; verify it against the shipped passes.**
`docs/diagrams/restir_pipeline.svg` already exists. The doc embeds it; a task
verifies it depicts the shipped `fill → spatial → resolve` three-pass flow and
the canonical-integration gate, and updates the SVG if stale. All diagrams stay
SVG per the repo rule — no ASCII art.

**D6 — GUI section is a control→effect table grounded in the host mapping.**
Document the dedicated **ReSTIR** group and its eight controls by reading off
`params.py::STATIC_PARAMS` (paths, ranges) and the renderer mapping
(`renderer.py::_restir_build_config`, the `_RESTIR_REGIME_FLAGS = [0x1,0x3,0x2]`
table, the `biased |= 0x4` flag, the live push-constant vs pass-rebuild split).
Each row: control → param path → push-constant field/flag → effect.

**D7 — State the limits plainly.** Wavefront-only (megakernel/Metal fall back to
identity); flat-material primary hits only (`restirLoadLane` gates on
`MATERIAL_TYPE_FLAT`); directional lights are plain NEE outside the RIS; temporal
reuse fights the progressive accumulator (bias ∝ `M_cap`, glossy-specific) so
spatial-only is the unbiased default; reprojected temporal is reserved.

## Risks / Trade-offs

- **Doc drift** (shaders/params change, doc rots) → D3's equation↔symbol pairing
  makes drift visible; the `restir-di` spec gains a requirement that the doc
  exist and stay current; the proposal lists the exact source files to re-check.
- **Math transcription error** → each equation is cross-checked against its named
  shader function while writing; the mapping table is the review surface.
- **Stale SVG** → D5 verification task before embedding.
- **Scope creep into GI/reprojection** → Non-Goals fence the doc to shipped DI;
  reprojected temporal appears only as a named follow-on.
- **Group-name breadth** (`ReSTIR` group also hosts the general `Reuse` selector)
  → already documented in the control-group design; the doc notes the rename
  trigger (first non-ReSTIR reuse mode).

## Migration Plan

Additive documentation. No data, settings, or format migration. Rollback is
deleting `docs/ReSTIR.md` and reverting the cross-link edits; nothing depends on
it at runtime.

## Open Questions

None blocking. Doc location (`docs/ReSTIR.md`), source of truth (shipped code),
and the paper set are resolved above.
