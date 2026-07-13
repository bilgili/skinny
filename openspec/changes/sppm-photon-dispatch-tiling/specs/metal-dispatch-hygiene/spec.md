# metal-dispatch-hygiene — SPPM photon-dispatch tiling delta

## ADDED Requirements

### Requirement: SPPM photon dispatch is watchdog-bounded by tiling

On the Metal backend, the SPPM phase-3 photon pass (`wfSppmPhotonTrace`) SHALL be
committed as a sequence of photon sub-batches — one command buffer per batch,
flushed at each boundary — so that each command buffer's work is bounded to
`batch × (visible points gathered per photon)`, independent of how many visible
points cluster into a grid cell (caustic focus). The per-pass photon count SHALL
NOT be reduced to bound the command buffer; tiling SHALL bound it by breadth while
the full `width × height` photon budget is emitted per pass. The batch size SHALL
be a `SKINNY_METAL`-gated, env-overridable calibration knob
(`SKINNY_SPPM_METAL_PHOTON_BATCH`).

The tiling SHALL be unbiased: photon flux is deposited by additive atomics into a
per-pass accumulator that is cleared once before the batch loop, and the radiance
resolve SHALL divide by the unchanged per-pass total `sppmPhotonsEmitted`, so the
tiled pass produces the same estimator as a single full-photon dispatch (modulo
integer-atomic ordering, which is exact). Off Metal (Vulkan, no watchdog) the pass
SHALL run as a single full-photon dispatch with base offset zero, behaviorally
unchanged.

#### Scenario: caustic SPPM scene never trips the watchdog

- **WHEN** `assets/glass_caustics_test.usda` renders on Metal under
  `--integrator sppm` with the full `width × height` per-pass photon budget for a
  gate-length accumulation
- **THEN** every command buffer completes without a GPU fault / watchdog kill, and
  a fresh probe subprocess afterward constructs a Metal device and completes a
  trivial dispatch within its time budget

#### Scenario: tiled photon pass is unbiased vs a single dispatch

- **WHEN** the same SPPM pass is rendered with the photon batch set larger than the
  photon count (single dispatch) and again with a smaller batch (multiple flushed
  dispatches), all else equal
- **THEN** the resolved radiance images match to within integer-atomic ordering,
  and neither exhibits the dark-starvation bias produced by reducing the per-pass
  photon count

#### Scenario: photon count is no longer starved to avoid the wedge

- **WHEN** SPPM renders on Metal after this change with
  `SKINNY_SPPM_METAL_PHOTON_CAP` unset
- **THEN** the per-pass photon count is the full `width × height` (not the prior
  262144 cap), and the mean radiance of a lit diffuse region converges toward the
  `path` anchor as samples accumulate rather than plateauing dark
