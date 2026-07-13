# SPPM photon-dispatch tiling (bound the watchdog by breadth, not by starving photons)

## Why

`--integrator sppm` wedges the Metal GPU on caustic scenes (e.g.
`assets/glass_caustics_test.usda`). The phase-3 photon pass
(`wfSppmPhotonTrace`) is committed as **one** command buffer whose total work is
`photons × (visible points gathered per photon)`. The per-photon deposit loop
(`wavefront_sppm.slang` `for k in 0..cnt` over every VP in the grid cell) is
**unbounded by `cnt`**, and a caustic scene clusters visible points into the
focus cell — so a single photon touches tens of thousands of VPs and the command
buffer blows past the macOS GPU watchdog.

The prior fix (`9142fe8`, change `spectral-wavefront`) worked around the wedge by
**capping photons per pass** (`SKINNY_SPPM_METAL_PHOTON_CAP`, default 262144,
needing lower on this scene). That trades a wedge for **wrong output**: SPPM is a
consistent estimator only in the photon budget, and starving it leaves visible
points with no photon in radius contributing zero — a **dark bias** worst where
photon density is lowest.

Measured on `glass_caustics_test` at 256², RGB, Metal, ACES, matched exposure vs
the `path` anchor:

| photons/pass (cap) | SPPM/path mean ratio |
|---|---|
| 2048 (64 spp)      | 0.917 |
| 8192               | 0.931 |
| 32768              | 0.944 |
| 32768 (400 spp)    | 0.953 |

Ratio climbs monotonically with photons — i.e. the reported "SPPM and path don't
match" is **the cap starving SPPM**, not a normalization bug. Flat lit surfaces
match to ~7 % and improve with photons; the sparse-photon far-ground corner sits
at ratio **0.73**; the residual diff is localized to the sphere / caustic /
shadow (where SPPM legitimately captures the caustic path finder misses). The
crash and the "mismatch" are the **same** root cause: the crippling cap.

## What Changes

- **Bound the phase-3 command buffer by breadth (tiling), not by count** — the
  same shape as `metal-megakernel-watchdog-tiling`'s row bands. Split the single
  photon dispatch into a sequence of **flushed photon sub-batches**; each command
  buffer traces at most `batch` photons, independent of how many VPs cluster in
  the focus cell. Photons-per-pass returns to the full `width × height`.
- **Unbiased.** Deposits are additive `InterlockedAdd`s into `sppmAccum`; the
  resolve divides by the total `sppmPhotonsEmitted` (unchanged). Splitting the
  dispatch and flushing between batches changes only *which command buffer* a
  deposit lands in, not the sum. `clear_accum` runs **once** before the batch
  loop.
- **Reuse the existing `SppmTilePC.streamBase` push-constant** as the photon base
  offset: `wfSppmPhotonTrace` reads `pid = sppmTile.streamBase + tid.x`. Base `0`
  reproduces today's dispatch exactly, so the RGB behavior is unchanged (Vulkan
  runs one full-width batch, base 0). No new binding, no new struct field.
- **Env knobs (Metal only):** add `SKINNY_SPPM_METAL_PHOTON_BATCH` (per-dispatch
  breadth; default hardware-calibrated) and drop the crippling default of
  `SKINNY_SPPM_METAL_PHOTON_CAP` to `0` (unlimited per pass; the batch now bounds
  the buffer). The cap env stays honored as an optional per-pass ceiling.

## Impact

- Affected specs: `metal-dispatch-hygiene` (new tiling requirement, mirroring the
  megakernel-tiling requirement).
- Affected code: `shaders/integrators/wavefront_sppm.slang` (1-line photon base),
  `wavefront_driver.py` (`record_sppm_loop` phase-3 tiled loop),
  `renderer.py` (photon-count cap → batch plumbing), the wavefront `.spv`
  recompile for `wfSppmPhotonTrace`.
- Parity: SPPM re-enters the parity sweep on caustic scenes without wedging;
  the `(sppm, wavefront)` self-consistency gate is expected to **tighten** toward
  the `(path, wavefront)` anchor now that photons are no longer starved. No gate
  tolerance is loosened.
- Backward compatible: Vulkan path byte-identical in behavior; RGB `.spv` differs
  only by the `+ streamBase` (base 0 at runtime).
