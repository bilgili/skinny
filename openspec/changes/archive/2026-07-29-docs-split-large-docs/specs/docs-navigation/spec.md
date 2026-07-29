## ADDED Requirements

### Requirement: One subject per reference document
Each reference document in `docs/` SHALL cover one subject. A document SHALL
stay at or below 700 lines. When a document passes the ceiling, the author
SHALL split it along a subject boundary, not at an arbitrary line.

`docs/Architecture.md` SHALL be a hub. It SHALL hold the high-level pipeline,
the GPU execution flow, the shader module dependency graph, the key invariants,
and a map of its child documents. It SHALL NOT hold the descriptor binding map,
the scene system, the backend selection logic, the byte layouts, the parity
harness, the web front end, or the module file listing — each of those belongs
to a named child document.

#### Scenario: a new descriptor binding is added
- **WHEN** a change adds a descriptor binding
- **THEN** the author updates the binding map in `docs/GpuResources.md`, and
  `docs/Architecture.md` needs no edit

#### Scenario: a document passes the size ceiling
- **WHEN** a reference document in `docs/` grows past 700 lines
- **THEN** the author splits it at a subject boundary and registers each new
  document in the `docs/README.md` index

### Requirement: The docs index lists every reference document
`docs/README.md` SHALL index every reference document with a one-line hook that
says what the document owns. `README.md` SHALL link `docs/README.md`.

The indexed set is the top level of `docs/`. Nested directories hold generated
artifacts rather than reference documents — `docs/diagrams/` carries the SVG and
equation generators with their result reports, `docs/superpowers/` records
history — and the index SHALL NOT be required to enumerate them. Their links
are still checked by the link-integrity test.

A new reference document SHALL be added to the index in the same change that
creates it. A hostless test SHALL assert that `docs/README.md` links every
`docs/*.md` file, so an unindexed document fails the build rather than becoming
unreachable.

#### Scenario: a new document is added without an index entry
- **WHEN** a change adds a Markdown document at the top level of `docs/` and
  does not list it in `docs/README.md`
- **THEN** the index test fails, naming the unindexed document

#### Scenario: a generated report is added under docs/diagrams/
- **WHEN** a generator writes a result report under `docs/diagrams/`
- **THEN** the index test passes without an entry for it, and the
  link-integrity test still resolves every link the report contains

### Requirement: Every relative Markdown link resolves
A hostless test SHALL resolve every relative Markdown link in the live
documentation set — `README.md`, `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`,
`examples/README.md`, and `docs/**/*.md`. Both link forms count: an inline
`[text](target)` link and the target of a `[label]: target` reference
definition. The test SHALL fail when a link names a file that does not exist. When a link carries a `#anchor`, the test SHALL
slugify every ATX heading in the target file with the GitHub rule and SHALL
fail when the anchor is absent.

The test SHALL exclude `openspec/changes/archive/**` and `docs/superpowers/**`,
which record history and keep the links they had when they landed. The test
SHALL exclude absolute `http` and `https` links, because the test runs with no
network.

#### Scenario: a moved section breaks an anchor
- **WHEN** a section moves to another document and an inbound link still names
  the old file
- **THEN** the link test fails and names the source file, the link, and the
  missing target

#### Scenario: a heading is reworded
- **WHEN** an author rewords a heading that an inbound anchor targets
- **THEN** the link test fails on the stale anchor

#### Scenario: a reference-style link names a missing file
- **WHEN** a document writes `[text][label]` with a `[label]: gone.md`
  definition and `gone.md` does not exist
- **THEN** the link test fails, naming the missing target

#### Scenario: an archived change keeps a stale link
- **WHEN** an archived OpenSpec change links a document path that no longer
  exists
- **THEN** the link test passes, because the archive is excluded by design

### Requirement: A documentation split moves text verbatim
A change whose purpose is to split a document SHALL move text verbatim. It
SHALL NOT rewrite prose, add facts, correct content, or change terminology in
the same change. Section heading levels SHALL be preserved, so that every
anchor slug survives the move.

#### Scenario: a content error is found during a split
- **WHEN** the author finds a wrong statement in a section being moved
- **THEN** the author moves the section unchanged and raises a separate change
  for the correction
