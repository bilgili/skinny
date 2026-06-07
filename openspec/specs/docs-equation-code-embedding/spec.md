# docs-equation-code-embedding Specification

## Purpose
TBD - created by archiving change docs-equation-code-mapping. Update Purpose after archive.
## Requirements
### Requirement: Marker-driven Slang-to-doc code embedding

The repository SHALL provide a doc-build generator (`docs/diagrams/embed_code.cjs`,
Node, alongside the existing equation-render scripts) that embeds excerpts of the
Slang shader sources into the reference Markdown docs, so that the embedded code
is a generated copy of the live shaders and not a hand-maintained transcription.

The generator SHALL slice source regions delimited by `// DOC:<key>#<slice>
start` / `// DOC:<key>#<slice> end` comment pairs in the `.slang` files, and
SHALL replace the body of each `<!-- CODE:<key> <slice,…> -->` … `<!-- /CODE:<key>
-->` region in the Markdown with a fenced `slang` block containing the named
slices in listed order. The markers SHALL be Slang comments only, introducing no
change to shader semantics and requiring no `.spv` recompile.

#### Scenario: A single-slice snippet is embedded verbatim

- **WHEN** a doc contains `<!-- CODE:ris-W core --> … <!-- /CODE:ris-W -->` and
  `restir/reservoir.slang` carries a matching `// DOC:ris-W#core start/end` region
- **THEN** the generator replaces the region body with a fenced `slang` block holding
  exactly those lines, prefixed by a `// from reservoir.slang` provenance comment

#### Scenario: A multi-slice snippet assembles non-adjacent regions with an elision

- **WHEN** a placeholder names several slices (e.g. `<!-- CODE:rqs sig,core -->`)
  whose source regions are not adjacent in the `.slang` file
- **THEN** the generated block contains the `sig` slice, an elision line
  (`    // …`), then the `core` slice, in the listed order

#### Scenario: Regeneration is idempotent

- **WHEN** the generator runs twice with no change to the shaders or placeholders
- **THEN** the second run produces no diff to the Markdown

#### Scenario: Orphaned or unbalanced markers fail the check

- **WHEN** `embed_code.cjs --check` runs and a `CODE:` placeholder names a
  `key#slice` with no matching Slang markers, a slice list is empty, or a
  `// DOC:` marker has a `start` without an `end`
- **THEN** the check exits non-zero and names the offending key, so a removed or
  renamed shader function is caught rather than silently going stale

