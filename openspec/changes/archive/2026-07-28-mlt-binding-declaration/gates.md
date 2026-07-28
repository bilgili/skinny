# Gate results — mlt-binding-declaration

## 5.1 ruff

Clean over the seven touched tracked files (bare `ruff check src/` inspects
**0** files here — the root `.gitignore` is `*`).

## 5.2 Hostless suite

`pytest -m "not gpu"`: **7 failed, 2494 passed, 35 skipped**.

The 7 are the same 7, name for name, as an unmodified `main` run: six
`test_corpus_scene_imports_cleanly_mtlx[…]` and
`test_mcp_tool_schemas::test_all_ten_tools_are_advertised`. **Zero added.**

A first run showed 10 additional failures. They were **worktree asset absence,
not code** — untracked `.usda`/`.hdr` corpus assets live only in the primary
checkout (`bathroom.usda`, `dragon_sss.usda`, the `light_infinite_*_env.hdr`
set, …). Symlinked in so the comparison is like-for-like; declaring the gate
green against a silently smaller corpus is the failure mode this project has
hit before.

## 5.3 MLT GPU smoke — the decisive gate

`int_caustic`, 128×128 @ 512 spp, `mlt | wavefront`. One backend per process,
strictly serial, both backends × RGB and spectral.

Rather than assert that a table-for-table refactor cannot change values, the
**same four renders were run on unmodified `main` (8fd1b61)** and diffed
against the post-change images:

| Render | pre-change vs post-change |
|--------|---------------------------|
| metal · RGB      | maxdiff **0.0**, meandiff **0.0** |
| metal · spectral | maxdiff **0.0**, meandiff **0.0** |
| vulkan · RGB     | maxdiff **0.0**, meandiff **0.0** |
| vulkan · spectral| maxdiff **0.0**, meandiff **0.0** |

Bit-exact on every combo. Means, for the record:

| | metal | vulkan |
|--|-------|--------|
| RGB      | 0.2519518466 | 0.2519517879 |
| spectral | 0.2531574248 | 0.2531572506 |

(identical pre and post on both sides).

## 5.4 Parity matrix, MLT combos

`test_suite_matrix_gate[int_caustic]` and `[spec_prism]` — the two
MLT-carrying suite scenes, both RGB and spectral — **2 passed** on Metal in
242 s. No `measured`, no `baseline`, no self-consistency tolerance edited;
`tests/pbrt/corpus/manifest.json` is untouched by this change.

## 5.7 Out-of-scope observation: cross-backend MLT is not bit-identical here

Measured Metal ↔ Vulkan on this scene at this budget:

| Mode | maxdiff | meandiff |
|------|---------|----------|
| RGB      | 1.3481080532e-4 | 2.7933e-6 |
| spectral | 4.1812062263e-3 | 1.9059e-6 |

The compatibility matrix (`CLAUDE.md`, `docs/MetropolisLightTransport.md`)
describes Vulkan and native Metal MLT as "bit-identical at equal budget, RGB
**and** spectral". On `int_caustic` at 512 spp that does not hold.

**This is pre-existing and untouched by this change**: the pre-change and
post-change cross-backend differences agree to every printed digit
(1.3481080532073975e-4 both sides), which is why 5.3 gates on the pre/post
comparison instead. Not diagnosed here — the recorded bit-identity measurement
was taken at a different budget (the scene's spp was raised 256 ⇒ 512 by change
`spectral-mlt`), so the claim may simply be stale rather than wrong. Flagged
for a separate change; deliberately not folded in, and no tolerance was
loosened to accommodate it.
