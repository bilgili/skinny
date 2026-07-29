## 1. Baseline and gate

- [x] 1.1 Create a worktree off `main` for this change and record the
      pre-change line counts of `docs/Architecture.md`, `README.md`, and every
      file in `docs/`.
- [x] 1.2 Copy `docs/Architecture.md` and `README.md` to the scratchpad as the
      verbatim-move reference. Every later step compares against these copies.
- [x] 1.3 Write `tests/test_docs_links.py`: walk `README.md`, `CLAUDE.md`,
      `AGENTS.md`, `CHANGELOG.md`, `examples/README.md`, and `docs/**/*.md`;
      resolve each relative Markdown link; slugify ATX headings with the GitHub
      rule to check each `#anchor`. Exclude `openspec/changes/archive/**`,
      `docs/superpowers/**`, and absolute `http(s)` links.
- [x] 1.4 Run the link test against the unchanged tree. Fix any link it finds
      broken today, or record the finding, before any section moves.

## 2. Split `docs/Architecture.md`

Move each section at its current heading level. Do not reword any line.

- [x] 2.1 Create `docs/ShaderPipeline.md`: move *Pluggable Interface
      Architecture*, *Material & Integrator Pipeline*, *MaterialX Nodegraph
      Compute Pipeline*, *Environment Importance Sampling*, and *SlangPile*.
- [x] 2.2 Create `docs/SceneSystem.md`: move *Scene System* and *Camera, Lens,
      and Debug Viewport*.
- [x] 2.3 Create `docs/GpuResources.md`: move *Descriptor Binding Map*,
      *FrameConstants Layout*, *Byte-layout ownership* (with its *Material
      field table* subsection), *Shader variant key*, and *GPU resource
      inventory*.
- [x] 2.4 Create `docs/HostModules.md`: move *Python Modules*, *Front-end
      bring-up*, *Renderer carve-out pattern*, and *The device-free pure core*.
- [x] 2.5 Create `docs/Backends.md`: move *Backend selection* (with its
      *MetalContext* subsection) and *Backend Abstraction (`gfx/`)*.
- [x] 2.6 Create `docs/FrontEnds.md`: move *Web Application Architecture*,
      *Headless Render API*, and *Display: Exposure, Tonemap, and Tool
      Readback*.
- [x] 2.7 Create `docs/ParityHarness.md`: move *Parity Matrix Harness*.
- [x] 2.8 Create `docs/OnlineTraining.md`: move *Online neural training*. Keep
      the heading text, so the `#online-neural-training` anchor still resolves.
      It does **not** fold into `docs/NeuralGuiding.md` — that document is
      already 961 lines, over the ceiling (see design § D1, correction).
- [x] 2.9 Rewrite `docs/Architecture.md` down to the hub: *High-Level
      Pipeline*, *GPU Execution Flow*, *Shader Module Dependency Graph*, *Key
      Invariants*, plus a new section that maps each child document. Confirm
      the file is at or below 400 lines.

## 3. Split `README.md`

- [x] 3.1 Create `docs/RenderingModes.md`: move the whole *Rendering Modes*
      section, including *GPU backend*, *Render resolution*, *Compatibility
      matrix*, *Sampling*, and *Furnace Mode*.
- [x] 3.2 Set `DOC_TABLES = ("docs/RenderingModes.md", "CLAUDE.md")` in
      `tests/test_render_envelope.py` and run that test. The three doc-sync
      cases must pass against the new path.
- [x] 3.3 Create `docs/ImplementationMap.md` from the README *Implementation
      Map* and the Architecture *File Listing*. Merge the two lists; where both
      describe the same module, keep the fuller entry and drop the duplicate.
      Record every dropped line in the commit message.
- [x] 3.4 Replace the two removed README sections with a short pointer each,
      naming the new document. Confirm `README.md` is at or below 800 lines.

## 4. Index and inbound links

- [x] 4.1 Write `docs/README.md`: one line per document in `docs/`, grouped as
      renderer internals, integrators and transport, materials and scene, and
      tooling. Each line says what the document owns.
- [x] 4.2 Update the `README.md` intro paragraph to link `docs/README.md` plus
      the entry documents.
- [x] 4.3 Update every inbound link in the live tree: `README.md`, `CLAUDE.md`,
      `AGENTS.md`, `examples/README.md`, and the other files in `docs/`. Repoint
      all 15 deep anchors to their new documents.
- [x] 4.4 Update the `docs/diagrams/embed_code.cjs` header comment if it names
      a moved section. Do not change its document list.
- [x] 4.5 Update `CLAUDE.md` § *Documentation upkeep* and `AGENTS.md`: name the
      new documents, state the 700-line ceiling, and state that an envelope
      change updates `docs/RenderingModes.md`.

## 5. Verification

- [x] 5.1 Run `tests/test_docs_links.py`. Every relative link and anchor
      resolves.
- [x] 5.2 Run the verbatim-move check: concatenate the moved sections from the
      new documents and compare against the sections removed from the
      scratchpad copies. Only heading-level context and the new document titles
      may differ. Report any other difference before continuing.
- [x] 5.3 Run `node docs/diagrams/embed_code.cjs --check`. No placeholder drift.
- [x] 5.4 Run the hostless suite:
      `.venv/bin/python -m pytest -m "not gpu" -q`. Compare the failure set
      against the recorded `main` baseline; no new failure.
- [x] 5.5 Run `.venv/bin/ruff check src/ tests/test_docs_links.py`.
- [x] 5.6 Confirm `git diff --stat` touches no file under `src/`, no shader,
      and no asset.

## 6. Review and land

- [x] 6.1 Run the codex pre-merge review over the branch. Fix or consciously
      dismiss each finding. Seven passes: 15 findings, all fixed; the seventh
      returned clean. None were in the moved text — every one was in the link
      gate or in a pointer into it.
- [x] 6.2 `openspec validate docs-split-large-docs --strict`.
- [ ] 6.3 Merge to `main`, archive the change, and remove the worktree.
