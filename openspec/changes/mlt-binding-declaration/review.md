# Pre-merge review — mlt-binding-declaration

Codex (`codex:codex-rescue`) over `903bfca` vs base `main` `8fd1b61`.
Verdict: **REQUEST CHANGES** — ownership and cross-backend data flow judged
sound; one load-bearing gate gap. All findings resolved below.

## Confirmed sound

`MLT_CHAIN_BUFFERS` is the single canonical owner, correctly kept **out** of
`gpu_resources.DECLARATIONS` (design D1) because these buffers are pass-owned,
not renderer-lifetime. All four consumers derive; declaration order preserves
52→57. The raw `VkWriteDescriptorSet` at `vk_wavefront.py:2122` is pre-existing
debt, not a regression from this change.

## BLOCKER (fixed) — the source gate did not gate what the change claims

`test_no_consumer_carries_its_own_binding_table` regex-matched only the
contiguous sequence `52, 53, 54, 55, 56, 57`. **The two tables this change
actually removed never contained that string**: the Vulkan pass used
`((52, "mlt_primary_samples"), (53, …))` and the Metal pass
`(("mltPrimarySamples", "mlt_primary_samples"), …)`. Either could have been
restored with every new gate green — the exact "a test that cannot fail reads
as coverage" failure this change removed the old cross-check for, reintroduced
in the replacement.

Fix: `consumer_violations()` is now **AST-based**. Any integer literal in 52…57
reaching executable code is a violation (verified: the four consumer modules
contain **zero** legitimate ones), as is any of the six Metal globals outside a
docstring. Comments and docstrings are excluded by construction, so the
surrounding prose may still name a binding.

Proven, not assumed:
- `test_gate_catches_the_exact_pre_change_table_shapes` drives the checker with
  all three real pre-change shapes and asserts each is rejected.
- `test_gate_ignores_prose` pins the docstring/comment exemption.
- Live check: appending the old `(52, "mlt_primary_samples")` tuple to
  `vk_wavefront.py` fails the gate naming file and line (then reverted).

## Nits (all fixed)

| Finding | Fix |
|---------|-----|
| Both pass docstrings say "five" chain buffers; there are six | corrected in `vk_wavefront.py` and `metal_wavefront.py` |
| Parser probe wrote into `tests/` instead of `tmp_path` | now takes the `tmp_path` fixture |
| `gpu_resources` module doc says binding numbers are never derived, unqualified | records the MLT exception and why it runs the other way |

## Re-gate

Hostless suite after the review round: **7 failed, 2496 passed** — the same 7
as unmodified `main`, name for name. ruff clean.

GPU gates were **not** re-run and did not need to be: the review round changed
test code and docstrings only — `git diff` over `src/` since `903bfca` is three
docstring edits with no executable line touched, so 5.3's maxdiff-0.0 pre/post
result and the two suite matrix gates stand.
