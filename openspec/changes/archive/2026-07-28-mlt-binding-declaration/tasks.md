# Tasks: mlt-binding-declaration

## 1. Baseline capture (no code moves)

- [x] 1.1 Record the current pairing from all three host tables and from the
      shader: for each of the six buffers, `(size_key, vulkan_binding,
      metal_global_name, byte_size_at_a_fixed_chain_count)`. Assert the four
      sources agree **today** before changing anything — if they already
      disagree, that is a live bug and this change stops until it is understood.
      → `capture.md`; all sources agree. **Found a sixth declaration site the
      proposal missed** (`vk_compute._create_descriptor_set_layout:688`, the
      set-0 layout entries) — folded into scope, see `capture.md`.
- [x] 1.2 Record the descriptor-write order the Vulkan pass emits for 52–57 and
      the bind-by-name order the Metal pass emits, so the move can be shown to
      preserve both. → `capture.md` §1.2.

## 2. The declaration

- [x] 2.1 Add the declaration table beside `mlt_buffer_sizes` in
      `wavefront_layout.py`: one entry per buffer carrying size key, Vulkan
      binding, Metal global name. Add accessors for the three shapes the
      consumers need (binding numbers; `(binding, key)` pairs; `(name, key)`
      pairs). → `MLT_CHAIN_BUFFERS` (a `NamedTuple` per buffer) +
      `mlt_binding_numbers()`. **Deviation:** only the binding-number shape got
      an accessor — it has two consumers (`vk_compute`'s set-0 layout,
      `gpu_resources`' dummy writes). `(binding, key)` and `(metal_name, key)`
      have exactly one consumer each, so they are projected off the table at
      the call site rather than given a helper that re-derives a shape only to
      hand it back; a projection helper per consumer is the accept-then-drop
      shape this change exists to remove.
- [x] 2.2 Hostless test: the table matches the 1.1 capture entry for entry.
      → `test_table_matches_the_baseline_capture` (keys, bindings, names AND
      scalar/MSL byte sizes at the captured budget).

## 3. The shader-agreement gate

- [x] 3.1 Add a reusable parser for `[[vk::binding(N)]] … <name>;` declarations
      in a shader source (design D2 — reusable because the follow-on may extend
      it to bindings 0–51). → `parse_binding_declarations(path)`, a public
      module-level helper in `tests/test_mlt_binding_declaration.py`. Single-arg
      (set 0) only; two-arg (set 1, pass-owned) deliberately unmatched, proven
      by `test_parser_ignores_set1_and_bodies`.
- [x] 3.2 Gate: the parsed MLT declarations match the table entry for entry.
      Assert the parsed **count** equals the table's length first, so the check
      cannot pass vacuously and a shader-side addition fails loudly.
      → `test_shader_declares_exactly_as_many_as_the_table` +
      `test_table_agrees_with_the_shader`, plus
      `test_parse_finds_the_expected_shape` pinning the parser's shape so a
      stale regex cannot green the suite.
- [x] 3.3 Negative self-test: feed the comparison a deliberately transposed
      table and assert it fails, naming the offending buffer. Without this the
      gate is unproven. → `test_transposed_table_is_rejected` (swaps the Metal
      names of 54 and 56; every number, name and size stays individually valid).

## 4. Adopt

- [x] 4.1 `gpu_resources.MLT_BINDINGS` becomes derived from the table (name and
      public behaviour unchanged, design D3). Existing `gpu_resources` tests
      must pass untouched. → `MLT_BINDINGS = mlt_binding_numbers()`; that
      module's tests pass with no edit.
- [x] 4.2 `vk_wavefront.WavefrontMltPass` consumes the table; delete its local
      `_BINDINGS`. Descriptor-write order unchanged (verify against 1.2).
      → `_BINDINGS` gone; allocation and `descriptor_bindings` both iterate
      `MLT_CHAIN_BUFFERS`, whose declaration order IS the 1.2 write order
      52→57.
- [x] 4.3 `metal_wavefront.MetalWavefrontMltPass` consumes the table; delete its
      local `_BINDINGS`. Bind-by-name set unchanged (verify against 1.2).
      → `_BINDINGS` gone. **Also found a seventh statement of the pairing** the
      capture missed: the MSL stride-validation loop restated
      `(metal_name, size_key)` for three of the six buffers. The source gate
      caught it; the Slang global name now comes off the declaration and only
      the per-element count (a sizing fact, not identity) stays local.
- [x] 4.4 Delete `tests/test_gpu_resources.py::test_mlt_binding_numbers_agree_
      with_the_wavefront_pass` — it compares a value with itself once both
      derive from one table (design D4). → deleted, with a comment in its place
      recording why and pointing at the replacement.
- [x] 4.5 (added) `vk_compute._create_descriptor_set_layout` consumes
      `mlt_binding_numbers()` — the sixth site found in 1.1. A layout that omits
      a binding the shader references is UB on Vulkan and a hard MoltenVK
      `nullptr` conversion error, so this is the highest-consequence copy of the
      six.
- [x] 4.6 (added) Source gate `test_no_consumer_carries_its_own_binding_table`
      makes the requirement structural: no consumer may restate the binding
      numbers or a Metal global. **Proven to fire** — re-introducing the literal
      tuple into `vk_wavefront.py` fails it (then reverted).

## 5. Gates

- [x] 5.1 `ruff` clean over an explicit tracked-file list (bare `ruff check
      src/` inspects 0 files here — the root `.gitignore` is `*`). → clean over
      the five touched `src/` modules + the two touched test modules.
- [x] 5.2 Full hostless `pytest -m "not gpu"` green; the pre-existing failure
      set must be unchanged, not merely small. → **7 failed, 2494 passed**,
      and the 7 are the same 7, name for name, as an unmodified `main` run
      (6 × `test_corpus_scene_imports_cleanly_mtlx[…]` + `test_mcp_tool_
      schemas::test_all_ten_tools_are_advertised`). Zero added.
      A first run showed 10 extra failures that were **worktree asset absence,
      not code** (untracked `.usda`/`.hdr` live only in the primary checkout);
      symlinked in so the gate compares like with like rather than being
      declared green against a smaller corpus.
- [x] 5.3 MLT GPU smoke: one scene, Vulkan and native Metal, RGB and spectral,
      at equal budget — images match to the tolerance recorded before the
      change. This is the gate that would catch a transposition that somehow
      survived 3.2. → `int_caustic` 128×128 @ 512 spp, `mlt|wavefront`, four
      renders per side, each backend in its own process, serially. Rendered the
      **same four on unmodified `main`** rather than asserting the refactor is
      value-neutral: pre-change vs post-change is **maxdiff 0.0, meandiff 0.0**
      on metal-RGB, metal-spectral, vulkan-RGB and vulkan-spectral — bit-exact,
      so no transposition survived. See `gates.md`.
- [x] 5.4 Parity matrix MLT combos unchanged — identical, not close. No
      baseline or self-consistency tolerance edited. → the two MLT-carrying
      suite matrix gates, `test_suite_matrix_gate[int_caustic]` and
      `[spec_prism]`, **2 passed** on Metal (242 s). No `measured`/`baseline`/
      tolerance edited anywhere — `manifest.json` is untouched by this change.
      "Identical, not close" is met in the strong form by 5.3's maxdiff 0.0.
- [x] 5.7 (added) Recorded an **out-of-scope, pre-existing** observation: the
      Metal↔Vulkan MLT cross-backend difference on this scene is NOT zero
      (RGB maxdiff 1.348e-4, spectral 4.181e-3), while the compatibility matrix
      describes the two as "bit-identical at equal budget, RGB and spectral".
      Unchanged by this change — the pre- and post-change cross-backend diffs
      agree to every digit — so it is reported, not fixed here. See `gates.md`.
- [x] 5.5 Docs: `docs/Architecture.md` binding-map rows 52–57 gain a pointer to
      the host-side declaration; the MLT section notes the single owner.
      → `docs/Architecture.md` "One host declaration owns rows 52–57" +
      `docs/MetropolisLightTransport.md` § GPU state and bindings (its table is
      now stated to render the declaration, not to be a fourth copy).
- [x] 5.6 `openspec validate mlt-binding-declaration --strict`. → valid.
