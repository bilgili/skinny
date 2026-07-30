## Context

`docs-split-large-docs` (merged `eee259d`, pushed) split `docs/Architecture.md`
into a hub plus eight subject documents and created `docs/README.md` as the
index. That change asked which index style to use and recommended a
`docs/README.md` index; the recommendation was wrong for the stated goal, which
was to refer to the split documents **from `README.md`**. This change corrects
the shape.

The mechanics are already in place and are reused unchanged:

- `tests/test_docs_links.py` resolves every relative link and `#anchor` in the
  live tree, so a moved section with a stale inbound link fails a test.
- The section extractor and the verbatim checker from the previous change prove
  each moved section arrives byte-identical.
- `openspec/specs/docs-navigation/spec.md` states the one-subject-per-document
  rule, the 700-line ceiling, and index completeness.

`README.md` is 750 lines in 13 sections: Gallery 24, Features 95, Requirements
34, Setup 132, Running 269, Controls 53, Assets 60, Rendering Modes 6 (pointer),
Implementation Map 5 (pointer), Papers 17, Testing 17, Development 14,
License 2.

## Goals / Non-Goals

**Goals:**

- `README.md` is the single index. A reader learns where everything lives
  without following a redirect.
- `README.md` is a front door: what this is, what it looks like, what it does,
  how to run it, where to read more.
- Each moved subject answers one question a reader arrives with.
- Every link and anchor still resolves; index completeness is still a test.

**Non-Goals:**

- No prose rewrite of the moved sections. They move verbatim.
- No second index. `docs/README.md` is deleted, not left as a redirect.
- No change to the eight documents `docs-split-large-docs` created, beyond
  repointing links.
- No change to `src/`, shaders, or assets. No change to `DOC_TABLES` — the
  compatibility matrix already lives in `docs/RenderingModes.md`.

## Decisions

### D1: `README.md` holds the index; `docs/README.md` is deleted

The index table moves up verbatim, with its relative targets rewritten from
`Foo.md` to `docs/Foo.md`.

Alternative rejected: keep `docs/README.md` as a stub that links back. That
leaves two files claiming to be the index, and the stub is exactly the redirect
this change exists to remove.

Alternative rejected: keep both, with `docs/README.md` as the detailed index and
`README.md` as a short list. Two lists of the same documents drift — the same
argument that put the compatibility matrix in one place in the previous change.

### D2: Split by the question the reader arrives with

| New document | Sections moved from `README.md` | ≈ lines |
|---|---|---|
| `docs/Install.md` | Requirements; Setup | 165 |
| `docs/Usage.md` | Running (front-ends, CLI flags, headless, pbrt import) | 270 |
| `docs/Controls.md` | Controls (with its Camera Debug viewport subsection) | 55 |
| `docs/Assets.md` | Assets | 60 |
| `docs/Contributing.md` | Testing; Development | 35 |
| `docs/References.md` | Papers and References | 20 |

`docs/Usage.md` lands at ≈270 lines and `docs/Install.md` at ≈165 — both well
under the 700-line ceiling.

Alternative rejected: fold these into two larger documents
(`GettingStarted.md`, `Contributing.md`). `GettingStarted.md` would reach ≈450
lines covering install, invocation, controls, and assets — a mixed document
again, which is the defect this change is fixing one level up.

### D3: Features stays; the quick start is authored

`README.md` keeps Features (95 lines). A front door that does not say what the
renderer does is not a front door, and the feature list is what a visitor scans
after the gallery.

The quick start is the one piece of **new** text in this change: clone, create
the venv, install, run. It is deliberately the shortest path that produces a
frame, and it links `docs/Install.md` for the MaterialX-from-source procedure
and the platform matrix that the real Setup section carries.

This is a stated exception to the verbatim rule. The verbatim checker records it
the same way the previous change recorded its two content edits: named, not
waived.

### D4: The index test moves with the index

`test_index_lists_every_reference_document` reads `docs/README.md` and resolves
targets relative to `docs/`. It now reads `README.md` and resolves relative to
the repo root, counting a target only when it lands directly in `docs/`. The
path-resolution logic added in the previous change (so a nested
`docs/diagrams/<name>.md` cannot stand in for a missing `docs/<name>.md`) is
kept — the base directory changes, the check does not weaken.

### D5: `openspec/config.yaml` is part of the change, not an afterthought

The previous change learned this the hard way: that file carries the
doc-ownership context OpenSpec seeds into every future change, and it named
`docs/README.md` as the index. It is updated in the same commit as the index
move, not left to a later sweep.

## Risks / Trade-offs

- **`README.md` grows back over time** → It is the file every contributor
  reaches for. The `docs-navigation` requirement now states that `README.md` is
  a front door and an index, and names what belongs elsewhere, so the next
  addition has a stated home.
- **A reader looks for install instructions in `README.md`** → The quick start
  is in `README.md` and links `docs/Install.md`. The common case does not leave
  the front page.
- **An external link to `docs/README.md` breaks** → Accepted. That file existed
  for one change and was never released or referenced outside this repository.
- **The moved sections carry inbound links** → The link test covers the live
  tree; every stale pointer fails it. Prose cross-references are not
  mechanically checkable, so they get the same treatment as last time: sweep
  `below` / `above` / `see X` in the split file and convert each to a link.

## Migration Plan

1. Copy `README.md` to the scratchpad as the verbatim reference.
2. Extract the six new documents, each at its current heading level.
3. Move the index table into `README.md`, rewriting its targets to `docs/…`.
4. Author the quick start; delete `docs/README.md`.
5. Repoint the six inbound references and update the spec plus the index test.
6. Run the link test, the verbatim check, the hostless suite, and ruff.

Rollback is a `git revert` of the branch. No source file changes.

## Open Questions

None. Scope, grouping, and the index location are decided above.
