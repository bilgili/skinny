# Proposal: mcp-scene-structure

## Why

The MCP server exposes read/inspect/property-write (`scene_list` / `scene_get` /
`scene_set`) but deliberately excluded structural mutation, so an AI agent can
tune what is already in the scene but cannot compose one: no adding a model, no
primitives, no lights, no removal, no persistence. The renderer already ships
every needed verb (`add_model`, `add_light`, `remove_node`, `save_edits`,
non-destructive USD edit sublayer) and both GUI docks call them — the MCP
surface is the only client without them. The original exclusion rationale
("adds would be discarded at exit because there is no save tool") is resolved
by exposing save alongside the authoring verbs.

## What Changes

- Six new MCP tools: `scene_add_model` (reference a `.usd*` file on disk),
  `scene_add_primitive` (analytic gprim + own `UsdPreviewSurface` material),
  `scene_add_light` (5 UsdLux types, optional intensity/color),
  `scene_remove`, `scene_save`, and `scene_job_status`.
- New renderer verb `add_primitive`: defines a UsdGeom gprim in the edit layer
  together with a bound `UsdPreviewSurface` material so the new object is
  editable from birth (an unbound prim lands on the protected fallback material
  slot and could never be re-colored).
- Path allowlist for all filesystem-touching tool arguments: configurable roots
  (`--mcp-roots` flag > `SKINNY_MCP_ROOTS` env > default of temp dirs + cwd),
  enforced against the argument, the composed USD layer stack, and asset-valued
  attributes of the added subtree, with prim rollback on violation. Asset-typed
  `scene_set` writes get the same check.
- Async job pattern for structural tools: inline grace wait (~2 s) returning the
  result directly when the resync finishes in time, else a `job_id` polled via
  `scene_job_status` — a large model add blows the flat request timeout and a
  cancelled-but-completed add would otherwise invite duplicate retries.
- Transform arguments accept TRS (translate / rotate-euler-degrees / scale) or a
  raw 4×4 row-major matrix, mutually exclusive.
- The `mcp-scene-control` exclusion requirement is narrowed: node authoring and
  save move from "excluded" to "provided"; rendered-image return stays excluded.
- **BREAKING** (spec only): the "no node authoring / no save tool advertised"
  scenarios in `mcp-scene-control` are inverted. No existing client depends on
  the absence of tools.

## Capabilities

### New Capabilities

_None — all changes extend two existing capabilities._

### Modified Capabilities

- `mcp-scene-control`: replace the "Persistence, node authoring, and rendered
  output are excluded" requirement (rendered output stays excluded; authoring
  and save become provided); add requirements for the structural tool set, the
  path allowlist, and the async job pattern.
- `usd-scene-editing`: add requirements for primitive authoring
  (`add_primitive`: gprim + bound preview material in the edit layer), a
  parent-complete `add_model` rollback, a validator seam on `add_model`
  (post-reference, pre-resync `validate(stage, added_prim)` hook the MCP layer
  uses for allowlist enforcement), and `add_light` intensity/color authoring.

## Impact

- `src/skinny/mcp_server.py` — six new tools, job store, path-policy module use.
- New `src/skinny/mcp_paths.py` (or similar) — root resolution + layer/asset
  walk, testable hostless.
- `src/skinny/renderer.py` — `add_primitive` verb; `add_model` rollback fixed to
  remove auto-created parents (matching `add_light`); optional
  `validate(stage, added_prim)` hook on `add_model`; `add_light` gains optional
  `intensity` / `color` authored at define time.
- Front-end CLI plumbing — `--mcp-roots` / `SKINNY_MCP_ROOTS` (all four
  front-ends, shared parser in `cli_common.py`).
- Docs: `docs/PythonAPI.md` (MCP section), `docs/Architecture.md`.
- Tests: hostless fake-renderer suites next to the existing MCP tests
  (`tests/test_mcp_*`), plus scene-editing tests for `add_primitive`.
- No GPU/shader changes; resync and upload paths are reused as-is.
