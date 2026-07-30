# docs-navigation Specification

## Purpose
TBD - created by archiving change docs-split-large-docs. Update Purpose after archive.
## Requirements
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
  document in the `README.md` index

### Requirement: The docs index lists every reference document
`README.md` SHALL be the documentation index. It SHALL list every reference
document with a one-line hook that says what the document owns. There SHALL NOT
be a second index: `docs/` SHALL NOT contain a `README.md`, because a reader who
arrives at `README.md` must learn where everything lives without following a
redirect.

The indexed set is the top level of `docs/`. Nested directories hold generated
artifacts rather than reference documents — `docs/diagrams/` carries the SVG and
equation generators with their result reports, `docs/superpowers/` records
history — and the index SHALL NOT be required to enumerate them. Their links
are still checked by the link-integrity test.

A new reference document SHALL be added to the index in the same change that
creates it. A hostless test SHALL assert that the index links every `docs/*.md`
file, so an unindexed document fails the build rather than becoming unreachable.

The test SHALL collect links from the index section **only**. `README.md` links
some documents from its intro and quick start as well, and a prose link SHALL
NOT satisfy the index requirement — otherwise deleting an index row still
passes and the check is decorative. The test SHALL also resolve each target and
count it only when it lands directly in `docs/`, so a link to a nested report
cannot stand in for a missing top-level document.

#### Scenario: a new document is added without an index entry
- **WHEN** a change adds a Markdown document at the top level of `docs/` and
  does not list it in `README.md`
- **THEN** the index test fails, naming the unindexed document

#### Scenario: a document is linked from prose but not from the index
- **WHEN** `README.md` links a document from its intro or quick start and the
  index section has no row for it
- **THEN** the index test fails, naming the unindexed document

#### Scenario: a second index is introduced
- **WHEN** a change adds `docs/README.md`
- **THEN** the index test fails, because the index has one home

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

### Requirement: README.md is a front door, not a mixed document
`README.md` SHALL carry only what a first-time visitor needs: what the project
is, what it looks like, what it does, the shortest path to a rendered frame, the
documentation index, and the licence. It SHALL NOT carry the full installation
procedure, the run-and-CLI reference, the control reference, the asset layout,
the contributor notes, or the paper references — each of those belongs to the
document in `docs/` that owns it.

A quick start in `README.md` SHALL be the shortest sequence that produces a
frame, and SHALL link the installation document for the full procedure. It SHALL
describe the supported-platform path as the normal case, and a from-source build
only as the fallback for a platform outside the wheel matrix. Every navigation
claim it makes SHALL be true of the document it names: it SHALL NOT promise
content the target does not carry.

A platform the quick start names as supported SHALL actually reach a rendered
frame by the sequence shown, including its interpreter-path conventions and every
external prerequisite the backend on that platform needs. The quick start SHALL
NOT be described as complete for a platform that needs a step it omits.

#### Scenario: a new CLI flag is documented
- **WHEN** a change adds a CLI flag
- **THEN** the author documents it in `docs/Usage.md`, or in
  `docs/RenderingModes.md` when it changes what the renderer can be told to do,
  and `README.md` needs no edit

#### Scenario: the quick start omits a platform prerequisite
- **WHEN** the quick start names a platform as supported but the sequence shown
  cannot produce a frame there — a different venv executable path, a missing
  external compiler, an OS version below the wheel target
- **THEN** the quick start is wrong: it either carries the extra step or stops
  calling that platform's install complete

#### Scenario: the install procedure changes
- **WHEN** a dependency or build step changes
- **THEN** the author updates `docs/Install.md`, and touches `README.md` only if
  the quick-start sequence itself no longer produces a frame

