# Design — SPPM photon-dispatch tiling

## Constraint

macOS cannot cancel another process's GPU work: a command buffer that outruns the
watchdog wedges the device until reboot. `metal-dispatch-hygiene` requires every
committed command buffer to complete within a watchdog-safe budget. The SPPM
phase-3 photon dispatch violates this: its cost is `photons × VPs-in-cell`, and
the VP-in-cell factor is unbounded and *maximized* on exactly the scenes SPPM is
for (caustics cluster visible points in the focus cell).

## Options considered

1. **Cap photons per pass** (the shipped workaround, `9142fe8`). Bounds the buffer
   but **biases the estimator** — starves visible points → dark image. Rejected as
   the primary mechanism (it is the bug this change fixes).
2. **Cap the per-cell VP-gather loop** (`for k in 0..min(cnt, MAX)`). Bounds the
   buffer, needs no host change, but **drops deposits** in dense cells → permanent
   under-count exactly in the caustic focus → biased, wrong. Rejected.
3. **Tile the photon dispatch by breadth** (chosen). Split the full photon count
   into `ceil(N/batch)` sub-dispatches, flush between. Each command buffer is
   `≤ batch × VPs-in-cell`; `batch` is chosen so even the worst cell keeps the
   buffer watchdog-safe. **Unbiased** — every photon is still traced, every
   deposit still lands, only spread across command buffers. This is the direct
   analogue of `metal-megakernel-watchdog-tiling`'s row bands (bound by breadth,
   bit-identical to one dispatch modulo float non-associativity of the atomics).

## Mechanism

`SppmTilePC { streamBase, shadeSlot, streamSize }` is already a per-dispatch
push-constant patched by the recorder's `push_tile`. The eye/update kernels read
`streamBase`/`streamSize` to tile over the *pixel* stream. `wfSppmPhotonTrace`
currently ignores it (`pid = tid.x`, guard `pid >= sppmPhotonsEmitted`).

- **Shader:** `let pid = sppmTile.streamBase + tid.x;` The guard against
  `fc.sppmPhotonsEmitted` is unchanged, so the last (rounded-up) batch's overshoot
  is masked. `streamBase = 0` ⟹ `pid = tid.x` ⟹ today's behavior. RNG is seeded
  from the *global* `pid`, so disjoint batches trace disjoint photons — no
  correlation, no double-count.
- **Driver (`record_sppm_loop` phase 3):**
  ```
  rec.clear_accum(); rec.barrier()
  base = 0
  while base < photons:
      n = min(batch, photons - base)
      rec.push_tile(base)                       # streamBase = base
      rec.dispatch_count("wfSppmPhotonTrace", n, 64)
      rec.barrier(); rec.flush()                # one command buffer per batch (Metal)
      base += n
  ```
  `flush()` is a no-op off Metal, so Vulkan runs a single `batch = photons`
  dispatch (base 0) exactly as before. `clear_accum` stays **outside** the loop.
- **Renderer:** `sppm_photons` returns to `width × height` (the
  `SKINNY_SPPM_METAL_PHOTON_CAP` default drops to `0` = unlimited; still honored
  if set). New `_sppm_metal_photon_batch` reads
  `SKINNY_SPPM_METAL_PHOTON_BATCH` (Metal only; off Metal the batch is the full
  count). The driver receives `batch` alongside `photons`.

## Why the atomics stay correct

`sppmAccum[j].phi{R,G,B,W}` accumulate `uint(phi * SPPM_FLUX_FIXED_SCALE)` via
`InterlockedAdd`; `.m` counts photons. These are commutative/associative over the
integers, so partitioning the photons into batches yields the identical fixed-point
sum regardless of batch boundaries. `clear_accum` once per pass (not per batch) is
required so batches accumulate rather than overwrite. The resolve
(`denom = sppmPhotonsEmitted · π · r²`) uses the unchanged per-pass total, so the
estimator is identical to the single-dispatch pass.

## Batch size — a calibration knob, not a constant

The safe `batch` depends on GPU throughput and the worst-case VPs-per-cell, which
varies by scene and radius. Ship a default sized to the megakernel band budget
(~2·10⁵ work units) and expose `SKINNY_SPPM_METAL_PHOTON_BATCH` so a wedge on a
pathological scene is a one-env-var fix, not a rebuild. Start conservative
(e.g. 65536 photons/dispatch) and raise after the GPU gate confirms headroom.

## Verification

- **Hostless:** driver emits `ceil(photons/batch)` dispatches + flushes with the
  right `streamBase` sequence; single-dispatch when `batch ≥ photons`; Vulkan
  path unchanged (one dispatch, base 0). Renderer batch/cap plumbing.
- **GPU (Metal):** `glass_caustics_test` at 256² renders without a wedge at the
  full `width × height` photon budget; SPPM/path mean ratio ≥ the capped baseline
  and rising with spp (no dark-starvation plateau); GPU-usable probe after the run.
- **Parity:** `(sppm, wavefront)` self-consistency vs `(path, wavefront)` anchor
  does not regress; re-measure any manifest entry that legitimately shifts.
