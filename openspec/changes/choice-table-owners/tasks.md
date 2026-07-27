# Tasks: choice-table-owners

## 1. Baseline

- [ ] 1.1 Tabulate every axis's copies and their current contents: integrator
      index (4), integrator labels (2, one missing MLT), tonemap (4, two
      disagree), and the 17 proxy placeholder lists with their 6 divergences.
- [ ] 1.2 For each divergence, decide: drift to fix, or deliberate stub to
      keep. Record the verdict per entry — do not assume drift.
- [ ] 1.3 Tabulate the 34 wavefront kernel entry names across the three
      modules, and the 14 duplicated pass constants, marking which must be
      equal and which are per-backend by design (record-stack sizing formula,
      rebuild-key elements).

## 2. Axis table

- [ ] 2.1 Add the dependency-free axis table: token, index, label per entry.
- [ ] 2.2 Repoint the CLI `choices`, the headless tables and argparse choices,
      and the renderer display lists.
- [ ] 2.3 Repoint `render_session._default_choice_names`; delete the retyped
      lists.
- [ ] 2.4 `render_session._default_values` reads the params registry instead of
      hardcoding 8 defaults.
- [ ] 2.5 Apply the fixes from 1.2; list each in `CHANGELOG.md` (user-visible
      label text).
- [ ] 2.6 Source gate: no axis list literal outside the table.

## 3. Wavefront names and constants

- [ ] 3.1 Kernel entry-name table in `wavefront_driver.py`; both backends and
      the driver import it.
- [ ] 3.2 Shared constants get one home; per-backend ones get a test pinning
      the pair with the stated reason.
- [ ] 3.3 Negative control: rename a kernel in the table and confirm the build
      fails rather than a render.

## 4. Gates

- [ ] 4.1 `ruff check src/`; full hostless `pytest`.
- [ ] 4.2 CLI surface unchanged: every previously accepted flag value still
      accepted, argparse help text checked.
- [ ] 4.3 GPU smoke: one wavefront render per backend (kernel names moved).
- [ ] 4.4 Docs: `README.md` flag choices, `docs/Wavefront.md` kernel table.
- [ ] 4.5 `openspec validate choice-table-owners --strict`.
