# Tasks: mlt-binding-declaration

## 1. Baseline capture (no code moves)

- [ ] 1.1 Record the current pairing from all three host tables and from the
      shader: for each of the six buffers, `(size_key, vulkan_binding,
      metal_global_name, byte_size_at_a_fixed_chain_count)`. Assert the four
      sources agree **today** before changing anything — if they already
      disagree, that is a live bug and this change stops until it is understood.
- [ ] 1.2 Record the descriptor-write order the Vulkan pass emits for 52–57 and
      the bind-by-name order the Metal pass emits, so the move can be shown to
      preserve both.

## 2. The declaration

- [ ] 2.1 Add the declaration table beside `mlt_buffer_sizes` in
      `wavefront_layout.py`: one entry per buffer carrying size key, Vulkan
      binding, Metal global name. Add accessors for the three shapes the
      consumers need (binding numbers; `(binding, key)` pairs; `(name, key)`
      pairs).
- [ ] 2.2 Hostless test: the table matches the 1.1 capture entry for entry.

## 3. The shader-agreement gate

- [ ] 3.1 Add a reusable parser for `[[vk::binding(N)]] … <name>;` declarations
      in a shader source (design D2 — reusable because the follow-on may extend
      it to bindings 0–51).
- [ ] 3.2 Gate: the parsed MLT declarations match the table entry for entry.
      Assert the parsed **count** equals the table's length first, so the check
      cannot pass vacuously and a shader-side addition fails loudly.
- [ ] 3.3 Negative self-test: feed the comparison a deliberately transposed
      table and assert it fails, naming the offending buffer. Without this the
      gate is unproven.

## 4. Adopt

- [ ] 4.1 `gpu_resources.MLT_BINDINGS` becomes derived from the table (name and
      public behaviour unchanged, design D3). Existing `gpu_resources` tests
      must pass untouched.
- [ ] 4.2 `vk_wavefront.WavefrontMltPass` consumes the table; delete its local
      `_BINDINGS`. Descriptor-write order unchanged (verify against 1.2).
- [ ] 4.3 `metal_wavefront.MetalWavefrontMltPass` consumes the table; delete its
      local `_BINDINGS`. Bind-by-name set unchanged (verify against 1.2).
- [ ] 4.4 Delete `tests/test_gpu_resources.py::test_mlt_binding_numbers_agree_
      with_the_wavefront_pass` — it compares a value with itself once both
      derive from one table (design D4).

## 5. Gates

- [ ] 5.1 `ruff` clean over an explicit tracked-file list (bare `ruff check
      src/` inspects 0 files here — the root `.gitignore` is `*`).
- [ ] 5.2 Full hostless `pytest -m "not gpu"` green; the pre-existing failure
      set must be unchanged, not merely small.
- [ ] 5.3 MLT GPU smoke: one scene, Vulkan and native Metal, RGB and spectral,
      at equal budget — images match to the tolerance recorded before the
      change. This is the gate that would catch a transposition that somehow
      survived 3.2.
- [ ] 5.4 Parity matrix MLT combos unchanged — identical, not close. No
      baseline or self-consistency tolerance edited.
- [ ] 5.5 Docs: `docs/Architecture.md` binding-map rows 52–57 gain a pointer to
      the host-side declaration; the MLT section notes the single owner.
- [ ] 5.6 `openspec validate mlt-binding-declaration --strict`.
