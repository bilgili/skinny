## Context

Two documents carry most of the reference text in this repository.
`docs/Architecture.md` has 2639 lines in 29 top-level sections.
`README.md` has 1132 lines in 13 top-level sections. The largest single
sections are the Architecture descriptor binding map (288 lines), the
Architecture scene system (330 lines), the Architecture high-level pipeline
(232 lines), the README rendering modes (286 lines), and the README running
guide (269 lines).

The repository already treats a document as a unit of ownership. `CLAUDE.md`
states which document a change must update: a wavefront change updates
`docs/Wavefront.md`, a ReSTIR change updates `docs/ReSTIR.md`. That rule works
for the focused documents. It fails for `docs/Architecture.md`, because almost
every change touches it somewhere.

Three tools read documents by path today:

1. `tests/test_render_envelope.py` reads `README.md` and `CLAUDE.md`. It asserts
   that the compatibility matrix names `render_envelope.py`, calls each
   wavefront-only integrator "wavefront-only" on one line, and states the
   spectral exclusions.
2. `docs/diagrams/embed_code.cjs` reads `docs/ReSTIR.md` and
   `docs/NeuralGuiding.md`. It fills `<!-- CODE:<key> -->` placeholders from
   marked Slang regions.
3. 15 deep anchor links point into `docs/Architecture.md`,
   `docs/Wavefront.md`, `docs/NeuralGuiding.md`, and
   `docs/MetropolisLightTransport.md` from `README.md` and `CLAUDE.md`.

## Goals / Non-Goals

**Goals:**

- Give each document one subject.
- Keep every reference document at or below 700 lines.
- Make `docs/` discoverable from one index.
- Keep every existing link and anchor resolvable after the move.
- Make link breakage a test failure, not a reader's discovery.

**Non-Goals:**

- No prose rewrite. The split moves text verbatim.
- No new technical content, no fact correction, no terminology change. A
  content error found during the split becomes a separate change.
- No split of `CHANGELOG.md`, `docs/Wavefront.md`, `docs/NeuralGuiding.md`, or
  `docs/PythonAPI.md`. The user scoped this change to `docs/Architecture.md`
  and `README.md`.
- No change to any file under `src/`, to any shader, or to any asset.
- No rewrite of archived OpenSpec changes. An archive records what was true
  when the change landed.

## Decisions

### D1: Split by subject, not by size

Each new document holds the sections that a single reader needs together. The
alternative — cut `docs/Architecture.md` into equal parts — gives balanced files
that nobody can name. A named subject is what makes the `CLAUDE.md` upkeep rule
work ("a binding change updates `docs/GpuResources.md`").

The mapping is fixed. Line counts are the current section sizes.

| New document | Sections moved from `docs/Architecture.md` | ≈ lines |
|---|---|---|
| `docs/ShaderPipeline.md` | Pluggable Interface Architecture; Material & Integrator Pipeline; MaterialX Nodegraph Compute Pipeline; Environment Importance Sampling; SlangPile | 360 |
| `docs/SceneSystem.md` | Scene System; Camera, Lens, and Debug Viewport | 395 |
| `docs/GpuResources.md` | Descriptor Binding Map; FrameConstants Layout; Byte-layout ownership; Shader variant key; GPU resource inventory | 630 |
| `docs/HostModules.md` | Python Modules; Front-end bring-up; Renderer carve-out pattern; The device-free pure core | 260 |
| `docs/Backends.md` | Backend selection (with its `MetalContext` subsection); Backend Abstraction (`gfx/`) | 205 |
| `docs/OnlineTraining.md` | Online neural training | 95 |
| `docs/FrontEnds.md` | Web Application Architecture; Headless Render API; Display: Exposure, Tonemap, and Tool Readback | 130 |
| `docs/ParityHarness.md` | Parity Matrix Harness | 120 |

`docs/Architecture.md` keeps High-Level Pipeline, GPU Execution Flow, Shader
Module Dependency Graph, Key Invariants, and a new map of its child documents.
That leaves ≈350 lines.

Two more sections leave `docs/Architecture.md`:

- *Online neural training* (83 lines) becomes `docs/OnlineTraining.md`.
- *File Listing* (127 lines) merges with the README *Implementation Map*
  (103 lines) into `docs/ImplementationMap.md`. The two lists describe the same
  module set today.

**Correction made during implementation.** The first draft of this design folded
*Online neural training* into `docs/NeuralGuiding.md`, because that document
already has a "Running online training" section. The move was made and then
reverted: `docs/NeuralGuiding.md` is already 961 lines, so the fold pushed it to
1044 and broke the 700-line ceiling that this change's own capability states.
`docs/NeuralGuiding.md` sits outside the declared scope, so this change does not
split it. A thin standalone document is the option that respects both the
ceiling and the non-goal. `docs/NeuralGuiding.md`, `docs/Wavefront.md` (1033),
and `docs/PythonAPI.md` (873) stay over the ceiling and are recorded as
follow-ups.

`README.md` loses *Rendering Modes* (286 lines) to `docs/RenderingModes.md` and
*Implementation Map* to `docs/ImplementationMap.md`. It keeps the gallery,
features, requirements, setup, running guide, controls, assets, papers,
testing, development, and license. That leaves ≈750 lines.

### D2: `docs/README.md` is the index; `README.md` links the index

`README.md` gets one paragraph that links `docs/README.md` plus the four or
five entry documents a new reader wants. `docs/README.md` lists **every**
document in `docs/` with a one-line hook.

Alternative rejected: list every document in `README.md`. The link paragraph in
`README.md` already runs nine lines. Each new document would make it longer,
and the list would sit in the file a casual reader scans first.

Alternative rejected: leave `docs/Architecture.md` as a pure table of contents.
That preserves every inbound link path with no edit, but it puts the pipeline
overview nowhere, and it makes `docs/Architecture.md` a second index that must
agree with `docs/README.md`.

### D3: The compatibility matrix moves with its section, and the test follows

`tests/test_render_envelope.py` sets `DOC_TABLES = ("README.md", "CLAUDE.md")`.
The compatibility matrix and the integrator table both sit inside README
*Rendering Modes*. Two options exist:

1. Keep the matrix in `README.md` and move only the prose. That leaves two
   homes for one table, and two homes drift.
2. Move the whole section and set `DOC_TABLES = ("docs/RenderingModes.md",
   "CLAUDE.md")`.

Take option 2. The test guards *the documented table*, wherever the table
lives. `CLAUDE.md` § *Documentation upkeep* must then say that a CLI flag
updates `README.md` **and** that an envelope change updates
`docs/RenderingModes.md`.

The `render-envelope` spec text names "CLAUDE.md and README". This change
updates that requirement to name the documented compatibility table by its new
path.

### D4: Anchors are a checked contract

15 deep anchor links exist. After the move, each target file changes but each
anchor slug stays the same, because the heading text does not change. The
mapping:

| Anchor | New file |
|---|---|
| `#backend-selection`, `#metalcontext-metal_contextpy-metal_computepy` | `docs/Backends.md` |
| `#byte-layout-ownership-…`, `#material-field-table-…`, `#shader-variant-key-…`, `#gpu-resource-inventory-…` | `docs/GpuResources.md` |
| `#renderer-carve-out-pattern-…`, `#front-end-bring-up-…` | `docs/HostModules.md` |
| `#online-neural-training` | `docs/OnlineTraining.md` |

The anchors into `docs/Wavefront.md`, `docs/NeuralGuiding.md`, and
`docs/MetropolisLightTransport.md` do not move. The link test still checks
them, so a later edit to a heading cannot break them silently.

### D5: One hostless link test is the gate

Add `tests/test_docs_links.py`. It walks the live Markdown files —
`README.md`, `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`, `examples/README.md`,
and `docs/**/*.md` — and for each relative Markdown link:

- resolves the path and fails if the file does not exist;
- when the link carries a `#anchor`, slugifies every ATX heading in the target
  file with the GitHub rule (lowercase, drop punctuation, spaces to hyphens)
  and fails if the anchor is absent.

It excludes `openspec/changes/archive/**` and `docs/superpowers/**`, which are
historical records. It excludes absolute `http(s)` links, because the box has
no network.

This test is what makes a verbatim move safe to review: the diff is large, but
a broken pointer is a red test rather than a reader's problem.

Alternative rejected: a Markdown link checker from npm. The box has no network
during a run, and a 60-line test with no dependency does the same job.

### D6: Size ceiling, stated once

The `docs-navigation` capability states a 700-line ceiling for a reference
document in `docs/`. The ceiling is documentation policy in `CLAUDE.md`, not a
test. A hard test would fail on a legitimately long single subject and would
push an author to split a subject for the wrong reason.

`docs/GpuResources.md` lands at ≈630 lines, the closest to the ceiling. The
descriptor binding map alone is 288 lines and grows with every new binding. If
it passes the ceiling later, the map becomes `docs/DescriptorBindings.md`. That
is a follow-up, not this change.

## Risks / Trade-offs

- **A verbatim move is hard to review by diff** → Move each section with an
  exact-text check. After the split, the concatenation of the moved sections
  must equal the removed text, ignoring heading-level shifts. Task 6 records
  this check as a scripted comparison against the pre-change files.
- **A heading-level shift changes an anchor** → Move every section at its
  current level. A `##` section stays `##` in the new file. Do not promote a
  section to a document title. The new document gets a fresh `#` title above
  the moved `##` sections.
- **An inbound link outside the checked set breaks** → The link test covers the
  live tree. Archived OpenSpec changes and `docs/superpowers/` keep stale
  links by design; the test excludes them, so it does not force a rewrite of
  history.
- **`docs/diagrams/embed_code.cjs` drifts** → The generator's document list
  does not change, because `docs/ReSTIR.md` and `docs/NeuralGuiding.md` both
  keep their names. The moved online-training section carries no `CODE:`
  placeholder today. Run `node docs/diagrams/embed_code.cjs --check` after the
  move to prove it.
- **The `render-envelope` doc-sync test fails mid-split** → Move the README
  section and retarget `DOC_TABLES` in the same commit.
- **More files raise the cost of a global rename** → Accepted. The index and
  the link test make the set navigable; a 2639-line file is the larger cost.

## Migration Plan

1. Add the link test first, against the current tree. It must pass before any
   move. That proves the test is honest.
2. Create the new documents one at a time. Each step moves whole sections,
   updates the inbound links for those sections, and re-runs the link test.
3. Retarget `DOC_TABLES` in the same step that moves README *Rendering Modes*.
4. Write `docs/README.md` last, when the document set is final.
5. Update `CLAUDE.md` and `AGENTS.md` upkeep lists last.

Rollback is a `git revert` of the branch. No source file changes, so no
render behaviour can regress.

## Open Questions

None. The document mapping, the index style, and the `DOC_TABLES` target are
decided above.
