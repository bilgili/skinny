## 1. Implementation

- [x] 1.1 Import `select_backend` from `skinny.backend_select` in
      `src/skinny/headless.py`.
- [x] 1.2 Change the `HeadlessRenderer.__init__` signature default from
      `backend: str = "vulkan"` to `backend: Optional[str] = None` — unset, so
      `SKINNY_BACKEND` participates (a literal `"auto"` would outrank it).
- [x] 1.3 Inside the `plan is None` branch, resolve the argument with
      `select_backend(backend)` (its own `persisted` default is `None`) and put
      the resolved name in the `BringupPlan`. Leave the `plan is not None` path
      untouched — it already carries a resolved backend from `plan_bringup`.
- [x] 1.4 Update the `__init__` docstring comment to state the precedence
      (argument > `SKINNY_BACKEND` > `auto`) and that the headless API reads no
      persisted settings.

## 2. Hostless test

- [x] 2.1 Add `tests/test_headless_backend_resolution.py`, monkeypatching
      `skinny.headless.select_backend` to record `(prefer, persisted)` and
      return a fixed name, and `BringupPlan.create` to capture the plan and
      return stub `(ctx, renderer)` objects.
- [x] 2.2 Assert the default argument reaches the selector as `prefer=None` or
      `"auto"` (whichever the implementation passes) with `persisted=None`, and
      that the plan carries the selector's return value.
- [x] 2.3 Assert an explicit `backend="vulkan"` reaches the selector as the
      `prefer` value.
- [x] 2.4 Assert `SKINNY_BACKEND` precedence by exercising the real
      `select_backend` with `prefer=None` and a monkeypatched
      `metal_available` (no device constructed).
- [x] 2.5 Assert that passing a `plan=` performs no resolution — the patched
      selector is never called and the plan's backend is used verbatim.
- [x] 2.6 Assert `backend="auto"` no longer raises `unknown backend 'auto'`
      from the context factory.

## 3. Documentation

- [x] 3.1 `docs/PythonAPI.md`: update the `HeadlessRenderer` signature block
      (`backend="vulkan"` → `backend="auto"`), and add a line on the resolution
      precedence next to the `plan` paragraph.
- [x] 3.2 `docs/PythonAPI.md` + `CLAUDE.md`: record that the Vulkan-SDK
      `DYLD_LIBRARY_PATH` export is required to *import* the renderer on
      **either** backend — measured, not assumed: `import skinny.renderer`
      without it raises `Cannot find Vulkan SDK version` because `renderer.py`
      imports `vulkan` at module load. Choosing Metal picks the device, not the
      import graph.
- [x] 3.3 `docs/FrontEnds.md`: change the headless entry's "owns
      `VulkanContext` + `Renderer`" wording to the resolved-context wording.
- [x] 3.4 Regenerate or re-caption `docs/diagrams/headless_api.svg` if its text
      names `VulkanContext`; verify with `node docs/diagrams/embed_code.cjs
      --check` where applicable and with `pytest tests/test_docs_links.py`.

## 4. Verification

- [x] 4.1 Hostless gate: `.venv/bin/python -m pytest -m "not gpu"
      tests/test_headless_backend_resolution.py tests/test_headless_api.py
      tests/test_bringup.py tests/test_docs_links.py`.
- [x] 4.2 Lint: `.venv/bin/ruff check src/skinny/headless.py`.
- [x] 4.3 GPU confirmation on this host: `tests/test_headless_api.py -m gpu`
      → 6 passed. Exit 0 does not name the device, so the device was asserted
      directly: the default `HeadlessRenderer(512, 512)` reports
      `backend auto → metal ON`, `ctx class = MetalContext`,
      `ctx.is_metal = True`, and renders `cornell_box_sphere.usda`.
- [x] 4.4 Vulkan confirmation with `SKINNY_BACKEND=vulkan`:
      `ctx class = VulkanContext`, `ctx.is_metal = False`, renders. This is also
      the env-precedence proof on real hardware. Both images agree in mean
      (144.77412 vs 144.77411).
- [x] 4.5 `openspec validate headless-backend-auto --strict`.
- [x] 4.6 Pre-merge review gate. The codex runtime hit its session limit, so
      the review ran as a review subagent instead (the recorded fallback), over
      the diff at absolute worktree paths. One finding, folded in: a
      self-declared BREAKING change with no `CHANGELOG.md` entry, against a repo
      convention where the preceding change added 37 lines in the same commit.
      A `### Changed` entry now records the default flip, the `backend="vulkan"`
      migration, the `None`-vs-`"auto"` reason, and the fact that Metal does not
      remove the Vulkan-SDK import requirement. The reviewer independently
      re-measured the 8-of-12 negative control by swapping the pre-change module
      in and out. Questions 1-5 and the front-end / parity-harness / docs scope
      came back clean.
- [x] 4.7 `CHANGELOG.md`: the entry above (review finding 1).
