# Tasks: frontend-bringup-builder

## 1. Baseline capture (before any refactor)

- [ ] 1.1 Capture the current refusal matrix per front-end as test fixtures:
      run each front-end's guard path (sppm+explicit-megakernel, mlt
      out-of-envelope, spectral out-of-envelope, mcp unsupported, unknown/
      unavailable backend, persisted-sppm interactive case) and record exact
      `SystemExit` messages incl. `skinny:` / `skinny-gui:` /
      `skinny-render:` / `skinny-web:` prefixes.
- [ ] 1.2 Record the two current guard orders (interactive: validate → resolve
      → rejects; non-interactive: resolve → validate) in the fixtures so the
      canonical order can be proven refusal-equivalent — including the actual
      equivalence mechanism: the interactive pair feeds the unresolved string
      `"auto"` into guards that compare `(execution_mode or "megakernel") ==
      "megakernel"` (cli_common.py:232, :275), so the pre-resolution check is
      a no-op (NOT a missing-attribute default); fixtures must cover
      `execution_mode="auto"` explicitly.

## 2. Bring-up module + hostless tests

- [ ] 2.1 Add `src/skinny/bringup.py`: `plan_bringup(args, prog, persisted=None)`
      running the canonical sequence (startup_integrator_name →
      resolve_execution_mode → validate_render_flags → reject_sppm_without_wavefront
      → reject_mlt_unsupported → reject_spectral_unsupported →
      reject_mcp_unsupported → select_backend with `{prog}:`-prefixed
      RuntimeError→SystemExit wrap) returning a frozen `BringupPlan`.
- [ ] 2.2 Add `BringupPlan.create(window=None, width, height, gpu_preference=None,
      context_factory=make_context, **renderer_kwargs)` — context +
      `Renderer(...)` with destroy-on-failure (relocated
      `HeadlessRenderer.__init__` pattern). Plan-carried fields
      (execution_mode, spectral, bdpt_walk, neural_config, backend) go to
      `Renderer` from the plan; `**renderer_kwargs` (usd_scene_path,
      use_usd_mtlx_plugin, shader/hdr/tattoo dirs, neural handoff/trainer/
      precision, …) forwarded verbatim; post-construction renderer state stays
      at the call sites.
- [ ] 2.3 Add hostless `tests/test_bringup.py`: guard matrix (integrator ×
      execution mode × spectral × persisted-vs-CLI) against the 1.1 fixtures,
      exact-message assertions, persisted-precedence (flag > env > persisted >
      auto; persisted only when offered), stub-context-factory create +
      destroy-on-failure, and `**renderer_kwargs` pass-through (front-end
      constructor inputs reach `Renderer` unmodified). No GPU.
- [ ] 2.4 `ruff check src/` clean; hostless suite green with no front-end
      migrated yet.

## 3. Migrate skinny-render (no persistence, no deferral)

- [ ] 3.1 Replace `headless.py` `main()` bring-up region with
      `plan_bringup(ns, prog="skinny-render", persisted=None)`; pass the plan
      into `HeadlessRenderer`, whose init calls `plan.create(window=None,
      gpu_preference=gpu)`.
- [ ] 3.2 Refusal-parity check vs the 1.1 fixtures; `--help` unchanged;
      existing headless tests green.

## 4. Migrate skinny-web (deferred, per-session, background thread)

- [ ] 4.1 Replace `web_app.py` `main()` bring-up region with
      `plan_bringup(args, prog="skinny-web", persisted=None)`; collapse the
      resolved module globals into the stored plan.
- [ ] 4.2 `SkinnySession.initialize()` calls `plan.create(window=None,
      gpu_preference=_GPU_PREFERENCE)` on its background thread, keeping its
      own `_log_init`/error capture around the call.
- [ ] 4.3 Refusal-parity check vs fixtures; a persisted `settings.json` on the
      host still does not influence web resolution.

## 5. Migrate skinny (windowed, persisted settings)

- [ ] 5.1 Replace `app.py` `main()` bring-up region with
      `plan_bringup(args, prog="skinny", persisted=saved)`; `plan.create(
      window=window, ...)` after GLFW window creation.
- [ ] 5.2 Keep the six post-construction persisted `Renderer` overrides
      (encoding already resolved pre-construction per design open question;
      neural_handoff/trainer/precision/sppm_glossy/online_training stay local).
- [ ] 5.3 Refusal-parity check incl. the persisted-sppm + forced-megakernel
      case; interactive startup behavior unchanged.

## 6. Migrate skinny-gui (deferred to Qt render thread)

- [ ] 6.1 Replace `ui/qt/app.py` `main()` bring-up region with
      `plan_bringup(args, prog="skinny-gui", persisted=saved_settings)` before
      `QApplication`; hand the plan to `MainWindow` alongside (not through)
      `QtRendererConfig` — `render_session.py` signatures untouched.
- [ ] 6.2 Move `ui/qt/viewport.py` `_build_renderer` (~70–104) onto
      `plan.create(...)` — the actual Qt context/renderer construction site;
      its post-construction state (`_requested_backend`,
      `_online_training_requested`, post-hoc integrator/reuse) stays in
      `viewport.py` after `create` returns.
- [ ] 6.3 Refusal-parity check vs fixtures; offscreen Qt smoke
      (`QT_QPA_PLATFORM=offscreen`) green.

## 7. Docs + close-out

- [ ] 7.1 Update `docs/Architecture.md` with the bring-up module (sequence,
      staging, per-front-end knobs); sweep `README.md`/`CLAUDE.md` for any
      per-front-end resolution wording that moved.
- [ ] 7.2 Full hostless suite + `ruff check src/` green;
      `openspec validate frontend-bringup-builder` clean.
