## Why

The scene-sampling change reserved a **reuse hook** — a `ReusePlugin` socket
with a `reuseMode` uniform, `IdentityReuse` baseline, and pass-structural
switching — but shipped only identity (stock NEE). This change lands the first
real reuse plugin: **ReSTIR DI**, reservoir-based spatiotemporal resampling of
direct lighting at the primary visible point. It reuses the best light sample
per pixel across spatial neighbors and accumulation frames, cutting
direct-lighting variance (largest with many emissive triangles / large area
lights / env). Full design:
`docs/superpowers/specs/2026-06-02-restir-di-design.md`.

skinny is a **progressive accumulator with no temporal infrastructure** (no
motion vectors, reprojection, G-buffer, or history; any `_current_state_hash`
change resets accumulation). So ReSTIR here targets convergence-acceleration on
a static camera; the game-style reprojection regime is a separate later change.

## What Changes

- **`reuseMode = RESTIR_DI` (1)** realized by `RestirDiReuse(ReusePlugin)`,
  owning a stack of **wavefront compute passes** + per-pixel **reservoir**
  (double-buffered) + **G-buffer** buffers + bindings. Wavefront-only;
  megakernel/Metal fall back to identity (capability gate).
- **Primary-hit screen-space reservoirs.** ReSTIR runs once per pixel at the
  primary visible point; secondary vertices keep stock NEE.
- **Unified light-domain RIS.** One reservoir resamples directional + sphere +
  emissive-triangle + env; a sample is `(lightType, lightId, point-on-light)`.
  Candidate generation mixes **light-sampled + BSDF-sampled** candidates;
  visibility is **deferred** (unshadowed target `p̂`; one shadow ray for the
  survivor).
- **Unbiased default + biased toggle.** Default unbiased combination (GRIS-lite:
  per-neighbor MIS weights + `1/Z` domain-aware normalization + the reconnection
  Jacobian); `biased` toggle sums + normalizes by `ΣM` (faster, discontinuity
  darkening).
- **Canonical integration.** ReSTIR owns primary direct: the bounce **skips its
  depth-0 sphere-light / env-miss direct term** (ReSTIR counted it) but still
  spawns the indirect ray via the proposal mixture; `depth ≥ 1` unchanged.
- **Selectable regimes.** `spatial` (on/off) × `temporal` (off / progressive /
  reprojected). This change implements **spatial + progressive-temporal +
  biased**; **reprojected is reserved in the enum** but deferred to a follow-on
  change. Config in a small ReSTIR UBO the plugin owns; front-end-selectable
  (`reuse_modes` gains `"ReSTIR DI"`) + persisted; any change resets accumulation.

## Capabilities

### Added Capabilities

- `restir-di`: reservoir-based spatiotemporal resampling of primary-hit direct
  lighting — the first non-identity reuse mode for the scene-sampling reuse
  hook. Unified light domain, unbiased (with a biased toggle), wavefront-only,
  spatial + progressive-temporal regimes (reprojected reserved). Converges to
  the stock-NEE reference with lower variance.

## Impact

- **Shaders (new):** `shaders/sampling/reuse.slang` (the reuse-seam interface +
  `identityReuseDirect`, reserved by scene-sampling, now realized);
  `shaders/restir/{reservoir,initial,temporal,spatial,resolve}.slang`.
- **Shaders (edit):** the wavefront primary path + `integrators/path.slang` /
  `wf_shade_common.slang` — the depth-0 "primary direct owned by reuse" gate;
  `common.slang` — `RESTIR_DI` constant + the G-buffer record.
- **Host:** `sampling/reuse.py` (`RestirDiReuse` + config); `sampling/registry.py`
  (`reuse_modes` += `"ReSTIR DI"`); `vk_wavefront.py` / new `vk_restir.py` (pass
  set + buffers); `renderer.py` (per-pixel buffer alloc/resize, state-hash,
  wavefront capability gate); `params.py` (ReSTIR sub-config params).
- **Tests:** `tests/test_restir_di.py` — converge-to-reference (unbiased gate),
  variance reduction, temporal-beats-spatial, biased bounded, capability gate,
  furnace + determinism. Reuses the `test_sampling_parity::_accumulate` harness.
- **Behaviour:** `reuse=none` (default) is unchanged; ReSTIR DI is opt-in and
  wavefront-only. ReSTIR is **not** bit-identical to stock NEE (different
  estimator) but converges to the same reference.
- **Risk:** the unbiased spatial-merge math (per-neighbor `m_i` + Jacobian +
  domain check) is the correctness hot spot; the depth-0 integration gate must
  avoid double-counting env between ReSTIR and the proposal mixture.

## Notes

Depends on the merged `scene-sampling` reuse hook (`ReusePlugin`, `reuseMode`,
`IdentityReuse`, the pass-structural rebuild). Reprojected temporal (motion
vectors + prev-frame G-buffer + disocclusion) is a follow-on change.
