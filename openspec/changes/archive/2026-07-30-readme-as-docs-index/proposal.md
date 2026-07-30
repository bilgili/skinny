## Why

The previous change (`docs-split-large-docs`) put the documentation index in
`docs/README.md` and left `README.md` at 750 lines. That is the wrong shape. A
visitor arrives at `README.md`, so `README.md` is the index — a second index one
directory down is a redirect the reader must follow to learn where anything is.

`README.md` also stayed a mixed document. It still carries the full install
procedure (132 lines), the whole run-and-CLI guide (269 lines), the control
reference, the asset layout, and the contributor notes. Each of those answers a
different question, and none of them is what a first-time visitor reads first.

## What Changes

- **`README.md` becomes the index.** The document table from `docs/README.md`
  moves into `README.md`. It lists every reference document in `docs/` with a
  one-line hook.
- **Delete `docs/README.md`.** One index, not two. It is not kept as a redirect.
- Split six subjects out of `README.md`, grouped by the question a reader
  arrives with:
  - `docs/Install.md` — *Requirements* + *Setup* (≈165 lines)
  - `docs/Usage.md` — *Running*: front-ends, CLI, headless, pbrt import
    (≈270 lines)
  - `docs/Controls.md` — keyboard and mouse, camera debug viewport (≈55 lines)
  - `docs/Assets.md` — `hdrs/`, `heads/`, `tattoos/` layout (≈60 lines)
  - `docs/Contributing.md` — *Testing* + *Development* (≈35 lines)
  - `docs/References.md` — *Papers and References* (≈20 lines)
- **`README.md` keeps the front door**: title, project note, intro, Gallery,
  Features, a short quick start, the index, License. Target ≈260 lines.
- The quick start is **authored**, not moved. It is the shortest path from clone
  to a rendered frame, and it links `docs/Install.md` for the full procedure.
  Every other section moves verbatim.
- Repoint the six live references to `docs/README.md`: `README.md`,
  `CLAUDE.md`, `AGENTS.md`, `examples/README.md`, and `openspec/config.yaml`.
- **BREAKING** (docs only): any link to `docs/README.md` now resolves to
  `README.md`. The link test catches every one in the live tree.

## Capabilities

### Modified Capabilities
- `docs-navigation`: the index requirement names `docs/README.md` as the index
  and says `README.md` links it. That inverts — `README.md` IS the index, and
  the index test reads `README.md`. The requirement that `README.md` stay a
  front door rather than a mixed document is new.

## Impact

- **Documents added**: `docs/Install.md`, `docs/Usage.md`, `docs/Controls.md`,
  `docs/Assets.md`, `docs/Contributing.md`, `docs/References.md`.
- **Documents removed**: `docs/README.md`.
- **Documents changed**: `README.md` (750 → ≈260), `CLAUDE.md`, `AGENTS.md`,
  `examples/README.md`.
- **Code changed**: `tests/test_docs_links.py`
  (`test_index_lists_every_reference_document` reads `README.md`; relative
  targets resolve from the repo root, not from `docs/`), and
  `openspec/config.yaml` (the context that seeds future changes).
- **Not changed**: `tests/test_render_envelope.py` — `DOC_TABLES` already names
  `docs/RenderingModes.md`, and the compatibility matrix is not moving again.
  No file under `src/`, no shader, no asset.
