## Why

`docs/Architecture.md` holds 2639 lines and 29 top-level sections. `README.md`
holds 1132 lines. Both documents mix many subjects. A reader who wants the
descriptor binding map must scroll past the scene system, the web front end,
and the parity harness. An agent that reads one of these files spends a large
part of its context on unrelated sections. The repository also has no index of
`docs/`, so a new document is only discoverable if somebody adds a link by hand.

## What Changes

- Split `docs/Architecture.md` into a short hub document plus **8** subject
  documents. The hub keeps the high-level pipeline, the GPU execution flow, the
  shader dependency graph, the key invariants, and a map of the child documents.
- Move `README.md` § *Rendering Modes* into a new `docs/RenderingModes.md`.
- Merge `README.md` § *Implementation Map* and `docs/Architecture.md` §
  *File Listing* into one `docs/ImplementationMap.md`. The two lists overlap
  today.
- Move `docs/Architecture.md` § *Online neural training* into a new
  `docs/OnlineTraining.md`. It does **not** fold into `docs/NeuralGuiding.md`:
  that document is already 961 lines, over the ceiling this change states.
- Add `docs/README.md`. It indexes every document in `docs/` with a one-line
  hook. `README.md` links this index.
- Move text verbatim. This change does **not** rewrite prose, add facts, or
  correct content. Only heading levels and cross-document links change.
- Update every inbound link in the live tree: `README.md`, `CLAUDE.md`,
  `AGENTS.md`, `examples/README.md`, the other files in `docs/`, and
  `docs/diagrams/embed_code.cjs`. Archived OpenSpec changes keep their old
  links, because an archive records history.
- **BREAKING** (docs only): 15 deep anchor links change target file. Each anchor
  must resolve in its new document after the move.
- **BREAKING** (test): `tests/test_render_envelope.py` pins
  `DOC_TABLES = ("README.md", "CLAUDE.md")`. The compatibility matrix moves to
  `docs/RenderingModes.md`, so the tuple must name that file instead of
  `README.md`.
- Add a hostless link-integrity test. It resolves every relative Markdown link
  in the live tree and fails on a missing file or a missing anchor.

## Capabilities

### New Capabilities
- `docs-navigation`: one subject per document, a size ceiling for reference
  documents, the `docs/README.md` index, and the link-integrity gate that keeps
  every relative Markdown link and anchor resolvable.

### Modified Capabilities
- `render-envelope`: the doc-sync check names the documented compatibility
  table. That table moves from `README.md` to `docs/RenderingModes.md`, so the
  requirement and the checked file set change.

`docs-equation-code-embedding` is **not** modified. Its generator reads
`docs/ReSTIR.md` and `docs/NeuralGuiding.md`. Neither document is touched by
this change.

## Impact

- **Documents added**: `docs/README.md`, `docs/ShaderPipeline.md`,
  `docs/SceneSystem.md`, `docs/GpuResources.md`, `docs/HostModules.md`,
  `docs/Backends.md`, `docs/FrontEnds.md`, `docs/OnlineTraining.md`,
  `docs/ParityHarness.md`,
  `docs/RenderingModes.md`, `docs/ImplementationMap.md`.
- **Documents changed**: `docs/Architecture.md` (2639 → ≈350 lines),
  `README.md` (1132 → ≈750 lines), `CLAUDE.md` and `AGENTS.md` (the documentation-upkeep list names the new
  documents).
- **Code changed**: `tests/test_render_envelope.py` (`DOC_TABLES`),
  `docs/diagrams/embed_code.cjs` (comment only; its document list is unchanged),
  one new test file for link integrity.
- **Not changed**: no source file under `src/`, no shader, no asset. The
  renderer behaviour is identical.
