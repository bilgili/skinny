## 1. Derive the declared globals from the compiler (design D1)

The original plan (tasks 1.1–1.6, kept here for the record) was a hand-written
Slang parser that derived a pass's declared globals by reading its `.slang`
sources. That approach was tried and **rejected**: a line/regex parser cannot
separate a file-scope resource global from a resource-typed function parameter
without full scope tracking, and codex pre-merge review kept finding valid Slang
spellings it under-reported — the exact fail-open the gate exists to prevent. So
the declared side comes from the **compiler's own reflection** instead. This
branch was authored against that decision from the start; no parser is committed
here.

- [x] 1.1 Understand the shape of the problem: which globals a compiled kernel
      declares, under which build defines.
      → answered by the compiler, not a survey. `slangc -reflection-json` lists
      the compiled module's top-level `parameters` under the pass's build defines.
- [x] 1.2 Own the build defines from `shader_variants`, not a second list.
      → the generator compiles each pass under `ShaderVariantKey.session_defines()`,
      the same defines the renderer builds with, so `#if`-gated globals resolve
      exactly as the shipped kernel sees them. No gate vocabulary is hand-listed.
- [x] 1.3 Produce a `{pass: [declared globals]}` set with no heuristic.
      → `tests/fixtures/gen_recording_pass_globals.py`: emit the generated
      MaterialX Slang (`emit_megakernel_sources(shader_dir, [])`), compile the
      megakernel to the **Metal** target with `-reflection-json`, take the
      top-level `parameters`, and write the checked-in golden
      `tests/fixtures/recording_pass_globals.json` (37 globals).
- [x] 1.4 Fail closed on anything the reflection cannot express as a top-level
      global.
      → the generator **refuses** on any bindable entry-point parameter (a
      `uniform` that lowers to a push constant, in Slang's entry-point scope, not
      top-level `parameters`); the megakernel takes only system-value thread IDs
      (binding `None`), so a future pass that adds one fails loudly.
- [x] 1.5 Cross-check the golden against an independently maintained table.
      → `test_golden_agrees_with_the_gpu_resource_inventory`: the megakernel
      golden's inventory-named subset must EQUAL `gpu_resources.DECLARATIONS`'
      default-layout Metal names minus the three neural buffers it strips at build
      (equality + disjointness), and `test_the_globals_without_an_inventory_entry
      _are_the_recorded_three` pins `fc` / `flatMaterialTextures` /
      `commonSampler` so a fourth unmatched global cannot appear unnoticed.
- [x] 1.6 Keep the golden honest against drift.
      → a `@pytest.mark.gpu` freshness test re-runs the compiler and diffs the
      golden; the hostless gate trusts the checked-in golden the way the repo
      trusts a checked-in compiled shader artifact.

## 2. Observe the host's real bind map

- [x] 2.1 Survey how each pass's bind map is built and whether any pass
      constructs a binding inside its dispatch call rather than receiving it.
      Answers the design's second open question and decides the provider shape.
      → survey note above the registry. Every Metal dispatch binds through ONE
      builder, `Renderer._build_metal_binds`, which was `metal_binds()` plus two
      renderer-owned globals appended afterwards. No pass constructs a binding
      inside its dispatch; the bindless pool rides the separate `bindless=`
      argument the recorder already counts. So the provider is a plain callable
      returning `(binds, bindless)`.
- [x] 2.2 Define the provider: given a pass, return the name→resource dict the
      host would bind, with no device. Keep it a plain callable so a pass whose
      map is assembled differently can supply its own.
      → `recording_compute.scene_binds(ctx, …)` allocates a `SceneResourceSet`
      against THIS adapter and calls `metal_binds()`. The two renderer-owned
      globals moved INTO `metal_binds` as keyword arguments, so the gate drives
      the same single builder a frame does rather than a copy of it. The
      registry field is `Callable[[RecordingContext], tuple[dict, tuple]]`.
- [x] 2.3 Confirm a key present with a `None` resource counts as UNBOUND, and
      keep the existing test that pins it — that behaviour is what stops a
      skipped bind from reading as a satisfied one.
      → `test_recorder_treats_a_none_resource_as_unbound` kept (renamed only),
      and `test_a_present_but_empty_binding_still_counts_as_omitted` asserts it
      again for a REAL pass through the gate's own call.

## 3. The registry and the gate

- [x] 3.1 Add `RECORDABLE_PASSES`: entry module, entry point, bind-map provider,
      per pass. Register the megakernel first — the pass with the widest binding
      surface and an independently known answer from 1.5.
      → registered first, `main_pass` / `mainImage` under the Metal megakernel
      variant key; 37 declared globals, all bound.
- [x] 3.2 Write `tests/test_recording_pass_coverage.py`: for every registered
      pass, assert `missing_bindings()` is empty, with globals from the golden
      (task 1.3) and binds from the host (task 2.2).
- [x] 3.3 Add the meta-test: every compute entry point in the shader tree is
      registered or excluded with a stated reason. Add `RECORDABLE_EXCLUSIONS`
      with a reason per entry.
      → 59 entries on this megakernel-scope branch: 1 registered (the megakernel),
      58 excluded. Keyed by `(module, entry_point)`, NOT by name: multiple
      modules declare `computeMain`, and a name-keyed map would collapse them and
      report an unregistered pass as covered — the same tautology this change
      removes.
- [x] 3.4 Assert no exclusion names a pass that no longer exists — a stale
      exclusion silently re-admits the gap it was meant to bound.
      → plus a check that no entry is both registered and excluded.
- [x] 3.5 Relabel the existing hand-driven scenarios in `tests/test_gpu_backend.py`
      to say they test the RECORDER, not any pass. Keep them; they are correct
      for what they actually cover.
      → six renamed `test_recording_adapter_*` → `test_recorder_*`, with a
      section note saying why a literal set is right there and where real
      coverage lives.

## 4. Prove the gate can fail

- [x] 4.1 Add the negative control: a fixture pass whose bind map omits exactly
      one declared global. Assert that global is reported.
- [x] 4.2 Route the negative control through the SAME call the real gate uses.
      A parallel hand-built path proves only that the parallel path works.
      → both go through `run_pass()`; the control is a `RecordablePass` whose
      provider wraps the megakernel's and drops one key. The same test then
      re-runs the unmodified pass and asserts it still reports nothing, so the
      control isolates the omission rather than a broken harness.
- [x] 4.3 Confirm that weakening the comparison makes the negative control fail —
      temporarily, by hand, recorded in the commit message, not left in the tree.
      → `Recorder.missing_bindings` had its gap computation replaced with
      `gaps += []`; both negative controls FAILED
      (`assert [] == [('mainImage', 'sphereLights')]`), and every other test in
      the file still passed — so the controls, and only the controls, detect it.
      Reverted; the suite is green.

## 5. Apply to the denoise passes — OUT OF SCOPE on this branch

This change is scoped to the **megakernel** (user decision: the denoise commits
are not part of it, and the `gbuffer` / `display_resolve` shaders do not exist on
this branch). The registry, generator and gate are built so a new pass registers
in one line, so these fold in when denoise lands rather than being reworked.

- [ ] 5.1 Register the auxiliary pass (`gbuffer` / `computeMain`) — deferred to
      `denoise-pipeline` (its own binding-coverage task): add one `RecordablePass`
      with a `scene_binds(ctx, denoise=True, aovs=…)` provider and regenerate the
      golden. Requires a `scene_binds` denoise/aov path (dropped here with the
      denoise `ResourceSizes` fields).
- [ ] 5.2 Register the display pass (`display_resolve` / `computeMain`) — deferred
      to `denoise-pipeline`, same shape as 5.1.

## 6. Verification

- [x] 6.1 Run the hostless suite: `.venv/bin/pytest -m "not gpu"`. Confirm the new
      tests RAN — a skip is not a pass.
      → megakernel-scope branch off main: recording + gpu_backend + gpu_resources
      all green (recording-pass-coverage tests all RAN, 0 skipped in the file);
      the gpu-marked freshness test regenerates the golden via slangc and diffs.
- [x] 6.2 Run the coverage gate in a subprocess with `vulkan` and `slangpy`
      blocked at the meta path, proving it is genuinely hostless.
      → `test_the_gate_runs_with_no_gpu_package_importable`, which drives every
      registered pass end to end and then asserts neither package reached
      `sys.modules`.
- [x] 6.3 Run `ruff check src/` with an explicit target, so the root `.gitignore`
      cannot make it a vacuous pass.
      → `ruff check src/skinny` → "All checks passed!". One F841 exists in
      `tests/test_gpu_backend.py:732`; it reproduces at HEAD, in a function this
      change does not touch.
- [x] 6.4 Confirm `tests/fixtures/gpu_adapter_surface.json` still matches, and
      regenerate it deliberately (`python -m tests.test_gpu_backend`) if the
      adapter surface changed. Never hand-edit it.
      → `test_surface_matches_the_pinned_fixture` passes unregenerated: the
      fixture pins the two DEVICE adapters, and every new member is on the
      recording adapter. The new recording-only members are declared in
      `gpu_backend.ONE_SIDED_MEMBERS`, which the conformance tests demand.

## 7. Documentation and review

- [x] 7.1 Update `docs/Backends.md` — the recording adapter section gains the
      compiler-reflection path (the offline golden), the registry, and the
      negative control.
      → new section "Live bindings on the recording adapter"; the file is 332
      lines, under the 700-line ceiling.
- [x] 7.2 Update `docs/Contributing.md` — adding a compute pass now means
      registering it or excluding it with a reason.
- [x] 7.3 Run `openspec validate recording-adapter-live-bindings`.
      → "Change 'recording-adapter-live-bindings' is valid".
- [ ] 7.4 Run the codex pre-merge review on THIS megakernel-scope branch. Confirm
      from the job log that it read this worktree, not the primary checkout.
      → The full-scope version (megakernel + denoise passes, on the
      `denoise-pipeline` worktree) already passed codex **APPROVE** after seven
      rounds: the first shipped a hand Slang DECLARATION parser that codex kept
      breaking (a line parser cannot separate a file-scope resource global from a
      resource-typed function parameter without scope tracking); the parser was
      DELETED for the compiler's own reflection (`slangc -target metal
      -reflection-json` → checked-in golden, gpu freshness diff, D5
      equality+disjointness vs `gpu_resources.DECLARATIONS`), then rounds 4-6
      hardened the generator and the `compute_entry_points` scanner (single-pass
      comment lexer, block-comment separator, platform-neutral key). This branch
      is that same reviewed code with the denoise passes removed, so a fresh codex
      run is a formality confirming the de-scope introduced no regression.
