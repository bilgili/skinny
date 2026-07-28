# Gate results — mlt-cross-backend-equivalence

## Repro

Scratch script and arrays under the session scratchpad. One backend per process,
serial, no Metal process ever killed:

```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
SKINNY_BACKEND={metal|vulkan} PYTHONPATH=src ./bin/python3.13 mlt_render.py \
  int_caustic {rgb|spectral} out.npy [W H SPP]
```

The script calls `parity.render_combo` with
`RenderCombo(integrator="mlt", execution_mode="wavefront", spectral=…)` on the
`int_caustic` spec, and records the host normalization `b` by wrapping
`mlt_chain.run_bootstrap`. `MLT_CHAINS_OVERRIDE` sets `mlt_num_chains` for the
historical-budget run (there is no CLI or env knob for it).

## Results

| Run | Metal mean | Vulkan mean | max abs diff | relMSE | PSNR | FLIP |
|---|---|---|---|---|---|---|
| RGB, 128×128, 512 spp, 16384 chains | 0.2519518466 | 0.2519517879 | 1.3481e-4 | 5.302e-10 | 135.67 | 2.949e-6 |
| spectral, same budget | 0.2531574248 | 0.2531572506 | 4.1812e-3 | 5.150e-08 | 114.94 | 1.779e-6 |
| RGB, 64×64, 8 spp, 512 chains | 0.2631299827 | 0.2631291333 | 7.7280e-4 | 1.770e-08 | 119.85 | — |

Same-backend control, second process, manifest budget: Metal `maxdiff 0.0`,
Vulkan `maxdiff 0.0` — pixel-identical.

Host `b`: Metal 0.2525743171504189, Vulkan 0.25257431707728983 (RGB, manifest
budget), a relative difference of 2.9e-10.

Attribution: max abs diff / splat quantum = 69.959 and 6.000 — integers to
within float32 accumulation rounding. Per-pixel histogram at the manifest budget
peaks at +1q (2520 pixels) and −1q (2349 pixels).

## Gates

- `openspec validate --all --strict` — 85 passed, 0 failed.
- Hostless suite, stashed control versus applied change — `17 failed, 2476
  passed, 34 skipped` on both, **identical FAILED sets**. The 17 are pre-existing
  in this worktree (missing corpus assets, one MCP schema test).
- `git diff --stat` — 7 files, docs + spec + one shader comment.
  `tests/pbrt/corpus/manifest.json` untouched. No tolerance or baseline edited.
- No GPU gate re-run is owed: no shader code, host code, or dispatch shape
  changed (the only shader edit is a comment).

## Still owed before merge

- codex pre-merge review (the standing gate for anything landing on `main`).

## Review round 1 (codex) — findings folded

The codex wrapper stalled mid-run, but its log had already surfaced two
leftovers, both confirmed independently:

1. `CHANGELOG.md` — the `spectral-mlt` entry repeats the cross-backend
   over-claim. Qualified with the budget it was measured at and pointed at the
   superseding entry. The file's two other `bit-identical` hits
   (`reflection-owned-byte-layouts`, `renderer-module-carveout`) are
   **same-backend** pre/post claims: true, left alone.
2. `docs/Wavefront.md` had a **second** untouched site — the spectral-MLT
   "spectral Metal ≡ spectral Vulkan" sentence. Qualified the same way.

The same review flagged that `main` has advanced (`1652374`,
`mlt-binding-declaration`) and overlaps this branch in
`docs/MetropolisLightTransport.md` and the living MLT spec. Checked:
`git merge-tree --write-tree main HEAD` reports a **clean** merge — main's
additions land in different regions of both files.

## The same bug class at a sibling site (probed, not assumed)

`docs/Wavefront.md` also claimed path / all three BDPT walk modes / ReSTIR DI are
"bit-identical to the Vulkan wavefront render". Re-measured `path` + wavefront on
`int_caustic` at the manifest budget: **not bit-identical** — maxdiff 6.556e-7,
2.3 % of pixels differing, relMSE 5.023e-15, PSNR 184.18.

This is the same cause (two compilers for one Slang source) at float-ULP scale
rather than quantized: the path integrator accumulates in float, so it has no
fixed-point splat quantum to snap the difference onto. That is *why* MLT's
difference is 1e-4-sized and quantized while path's is 1e-7-sized and smooth —
the mechanism is shared, the amplifier is MLT's alone.

The doc now says so, and explicitly records that the **BDPT walk modes and
ReSTIR DI were not re-measured** — their recorded identity is flagged, not
restated. Re-measuring those (and deciding whether the `metal-wavefront-parity`
spec needs the same equivalence-class treatment MLT just got) is a follow-up,
deliberately out of this change's scope.
