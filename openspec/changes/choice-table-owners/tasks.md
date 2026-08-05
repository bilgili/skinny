# Tasks: choice-table-owners

## 1. Baseline

- [x] 1.1 Tabulate every axis's copies and their current contents: integrator
      index (5: cli_common, headless, render_envelope, frame_plan inverse, +
      argparse choices), integrator labels (2, proxy missing MLT), tonemap (4,
      proxy `["Filmic"]`), and the 17 proxy placeholder lists with their 6
      owned-axis divergences.
- [x] 1.2 For each divergence, decide: drift to fix, or deliberate stub to
      keep. Verdicts — all 6 owned-axis proxy lists (integrator, tonemap, reuse,
      detail_maps, restir_combination, proposal_preset) are DRIFT, fixed via the
      table. `restir_regime` `["Initial"]` and `scatter` `["BSSRDF","Volume"]`
      are also drifted but are NOT owned axes (not in the spec's 7); recorded as
      out-of-scope stubs, left as-is (overwritten by the first snapshot).
- [x] 1.3 Tabulate the 34 wavefront kernel entry names across the three
      modules, and the 14 duplicated pass constants, marking which must be
      equal and which are per-backend by design (record-stack sizing formula,
      rebuild-key elements).

## 2. Axis table

- [x] 2.1 Add the dependency-free axis table: token, index, label per entry.
      (`src/skinny/choice_tables.py`.)
- [x] 2.2 Repoint the CLI `choices`, the headless tables and argparse choices,
      and the renderer display lists.
- [x] 2.3 Repoint `render_session._default_choice_names`; delete the retyped
      lists (owned axes only; dynamic/non-owned stubs kept per 1.2).
- [x] 2.4 `render_session._default_values` reads the params registry instead of
      hardcoding defaults (`ParamSpec.default` added; the four override params
      carry it).
- [x] 2.5 Apply the fixes from 1.2; list each in `CHANGELOG.md` (user-visible
      label text).
- [x] 2.6 Source gate: no axis list literal outside the table
      (`tests/test_choice_tables.py`, with a negative control on `9ffd5b0`).

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
