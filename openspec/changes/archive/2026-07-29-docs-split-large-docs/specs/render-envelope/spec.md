## MODIFIED Requirements

### Requirement: Documented compatibility matrix documents the predicate
The human-readable compatibility-matrix tables SHALL be retained as
documentation **of** the predicate, with the predicate module as the stated
source of truth — superseding the prior convention that code mirrors the
documented matrix. The documented tables SHALL live in `CLAUDE.md` and in
`docs/RenderingModes.md`; `README.md` SHALL link `docs/RenderingModes.md`
instead of holding a second copy of the matrix. A hostless doc-sync check SHALL
assert that key envelope facts derived from the predicate (at minimum: the
wavefront-only integrator set, and the axes refused under spectral) are stated
in the documented tables, so envelope edits that skip the docs fail a test. The
checked file set SHALL name the documents that hold the tables, so moving a
table to another document means updating the checked set in the same change.

#### Scenario: doc drift fails the check
- **WHEN** an envelope rule covered by the doc-sync check changes in the
  predicate but the documented compatibility tables are not updated
- **THEN** the hostless doc-sync check fails, naming the stale fact

#### Scenario: the documented table moves to another document
- **WHEN** a change moves a documented compatibility table to a different
  Markdown file
- **THEN** the same change updates the doc-sync check's file set to the new
  path, and the check passes against the new location
