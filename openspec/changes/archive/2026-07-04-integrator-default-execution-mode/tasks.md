## 1. Shared CLI resolver (cli_common.py)

- [x] 1.1 Add `--execution-mode` choice `auto` and make it the default:
  `choices=("auto","megakernel","wavefront")`,
  `default=os.environ.get("SKINNY_EXECUTION_MODE","auto")`; update the help text
  to describe the integrator-derived default and the override precedence.
- [x] 1.2 Add module-level `DEFAULT_EXECUTION_FOR_INTEGRATOR = {"path":"megakernel",
  "bdpt":"megakernel","sppm":"wavefront"}` and a `resolve_execution_mode(
  execution_mode, integrator) -> str` mapper (auto → derive from integrator;
  explicit mode wins). Export it alongside `resolve_walk`. Also added
  `startup_integrator_name()` (CLI > persisted index > `path`) shared by the
  interactive front-ends.
- [x] 1.3 Update `validate_render_flags`: run after the mode is resolved; the
  `sppm + megakernel` guard now trips only on an explicit megakernel override
  (`== "megakernel"`) — updated its docstring and error message.

## 2. Wire the resolver into every front-end (resolve before validate + before Renderer)

- [x] 2.1 `headless.py` (`skinny-render`): resolve `ns.execution_mode` from
  `ns.integrator or "path"` before `validate_render_flags` and construction.
- [x] 2.2 `app.py` (`skinny`): resolve from `startup_integrator_name(args.integrator,
  saved["params"]["integrator_index"])` before the `Renderer(...)` construction.
- [x] 2.3 Qt GUI front-end (`skinny-gui`, `ui/qt/app.py`): same, before `MainWindow`.
- [x] 2.4 Web front-end (`skinny-web`, `web_app.py`): resolve from `args.integrator
  or "path"` (stateless) before validate + global assignment.

## 3. Tests

- [x] 3.1 Unit tests for `resolve_execution_mode`: `path`/`bdpt` → `megakernel`,
  `sppm` → `wavefront`; explicit overrides; `auto` + `integrator=None` → `megakernel`.
- [x] 3.2 `render-cli` validation tests: `--integrator sppm` alone passes (resolve
  → wavefront); `sppm --execution-mode megakernel` still exits;
  `SKINNY_EXECUTION_MODE=megakernel` + sppm resolves megakernel (env = explicit);
  `path`/`bdpt` defaults unchanged.
- [x] 3.3 Persisted-integrator test via `startup_integrator_name`: index 2 (sppm)
  with no CLI flag → `wavefront`.
- [x] 3.4 `--help` test: `--execution-mode` lists `auto,megakernel,wavefront`,
  default `auto`.
- [x] 3.5 `ruff check src/` clean; `tests/test_cli_common.py` (70),
  `tests/test_execution_mode.py` (6), `tests/test_online_training_observability.py`
  (17) green.

## 4. Documentation

- [x] 4.1 `README.md`: SPPM cell + new "Execution mode follows the integrator"
  paragraph (`{auto,megakernel,wavefront}`, default `auto`, integrator-derived).
- [x] 4.2 `CLAUDE.md`: added the `--execution-mode auto` derivation paragraph
  beside the `--backend` description.
- [x] 4.3 `docs/Wavefront.md`: documented the `auto` → integrator default and the
  session-fixed resolution.
- [x] 4.4 `CHANGELOG.md`: `### Changed` entry under [Unreleased].

## 5. Verify

- [x] 5.1 `openspec validate integrator-default-execution-mode --strict` passes.
- [x] 5.2 CLI-layer smoke through the real `skinny-render` parser: `--integrator
  sppm` → `wavefront`; `path`/`bdpt`/default → `megakernel`; explicit `wavefront`
  overrides; `sppm --execution-mode megakernel` refused. (Full GPU render not run
  in the worktree — no assets/venv; the resolved-mode strings are the same ones
  the render path already consumes, so downstream behavior is unchanged.)
