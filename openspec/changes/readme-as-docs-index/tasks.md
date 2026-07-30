## 1. Baseline

- [x] 1.1 Copy `README.md` and `docs/README.md` to the scratchpad as the
      verbatim-move reference, and record the pre-change line counts.
- [x] 1.2 Run `tests/test_docs_links.py` against the unchanged tree. It must
      pass before anything moves.

## 2. Split `README.md`

Move each section at its current heading level. Do not reword any line.

- [x] 2.1 Create `docs/Install.md`: move *Requirements* and *Setup*.
- [x] 2.2 Create `docs/Usage.md`: move *Running*, including its front-end, CLI,
      headless, and pbrt-import subsections.
- [x] 2.3 Create `docs/Controls.md`: move *Controls* with its *Camera Debug
      viewport* subsection.
- [x] 2.4 Create `docs/Assets.md`: move *Assets*.
- [x] 2.5 Create `docs/Contributing.md`: move *Testing* and *Development*.
- [x] 2.6 Create `docs/References.md`: move *Papers and References*.
- [x] 2.7 Rewrite `README.md` to the front door: title, project note, intro,
      Gallery, Features, an authored quick start, the index, License. Confirm it
      is at or below 300 lines.

## 3. Move the index into `README.md`

- [x] 3.1 Move the document table from `docs/README.md` into `README.md`,
      rewriting each target from `Foo.md` to `docs/Foo.md`. Add a row for each
      of the six documents created in group 2.
- [x] 3.2 Delete `docs/README.md`.
- [x] 3.3 Point `test_index_lists_every_reference_document` at `README.md`:
      resolve targets from the repo root, keep the direct-in-`docs/` check, and
      add a case that fails if `docs/README.md` reappears.

## 4. Inbound references

- [x] 4.1 Repoint `CLAUDE.md` § *Documentation upkeep* to `README.md` as the
      index, and add the new documents to the routing list (a CLI flag →
      `docs/Usage.md`; an install change → `docs/Install.md`).
- [x] 4.2 Repoint `AGENTS.md` and `examples/README.md`.
- [x] 4.3 Repoint `openspec/config.yaml` — the context that seeds every future
      change. Do this in the same commit as the index move, not a later sweep.
- [x] 4.4 Sweep the moved sections for prose cross-references
      (`below` / `above` / `see X`) whose target left `README.md`, and convert
      each to a link.

## 5. Verification

- [x] 5.1 Run `tests/test_docs_links.py`. Every relative link and anchor
      resolves, and the index lists every `docs/*.md`.
- [x] 5.2 Run the verbatim-move check over the pre-change `README.md`: every
      section is byte-identical in its destination, except the authored quick
      start and the index, both recorded by name.
- [x] 5.3 Run `node docs/diagrams/embed_code.cjs --check`.
- [x] 5.4 Run the hostless suite and compare the failure set against the
      recorded `main` baseline; no new failure.
- [x] 5.5 Run `.venv/bin/ruff check src/ tests/test_docs_links.py`.
- [x] 5.6 Confirm `git diff --stat` touches no file under `src/`, no shader, and
      no asset, and that `tests/test_render_envelope.py` is unchanged.

## 6. Review and land

- [x] 6.1 Run the codex pre-merge review. Fix or consciously dismiss each
      finding, and re-run until a pass returns clean. Seven passes: 18 findings
      (2 P1), all fixed; the seventh returned clean. Every one was in text I
      authored or in the gate — none in moved text.
- [x] 6.2 `openspec validate readme-as-docs-index --strict`.
- [ ] 6.3 Merge to `main`, archive, remove the worktree, and push.
