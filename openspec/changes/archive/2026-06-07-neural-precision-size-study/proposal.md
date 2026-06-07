## Why

The neural directional proposal (Stage 1) is unbiased but **lost the 6.3 equal-time
gate on Mac entirely to inference cost** — the MLP pre-pass is ~28× a BSDF bounce on
MoltenVK/MPS (95.8 vs 3.4 ms/frame); the bulk image converges identically, so the
loss is pure cost, not quality. Yet the network is **hardcoded**: `neural_flow.slang`
is all `float` (fp32) and `NF_LAYERS=6 / NF_BINS=24 / NF_HIDDEN=96` are `static const`.
There is no way to ask the two questions that decide whether neural can ever be
real-time on this hardware: *how small can the net get before it stops guiding*, and
*can Apple-Silicon fp16 (native ~2× ALU throughput, half the bandwidth) cut the MLP
cost*. Apple GPUs have first-class `half`; MoltenVK exposes `shaderFloat16` +
`storageBuffer16BitAccess`; slangc emits `half` — none of it is wired.

This change makes network **size and floating-point precision tunable** and adds a
**study harness** that maps quality-vs-cost across the size×precision grid on
MoltenVK. It is a measurement, not a product: the output is a Pareto table that
informs which config to ship and what the eventual CUDA real-time target should be.
Tunability lands as the byproduct.

## What Changes

- **Configurable size.** `NF_LAYERS / NF_BINS / NF_HIDDEN` become `#ifndef`-guarded
  defines, overridable via `slangc -D`; the dims thread from a renderer config into
  every module that imports `neural_flow` (the wavefront pre-pass, the inline inverse,
  the record entry) and fold into the pipeline cache key. The host NFW1 loader already
  asserts `(layers, bins, hidden, cond)` — host is size-ready.
- **Configurable precision (mixed fp16).** Two compile-time type aliases in
  `neural_flow.slang`: `NF_WT` (weight *storage*, `StructuredBuffer<NF_WT>`) and
  `NF_CT` (MLP GEMM *accumulate*). The RQ-spline math (softmax / cumsum / exp / log +
  the analytic inverse-quadratic solve) stays `float` always. Three modes: **fp32**
  (`float`/`float`), **fp16-storage** (`half`/`float`), **fp16-compute**
  (`half`/`half`).
- **Device + format plumbing.** `vk_context` probes and enables `shaderFloat16`
  (`VK_KHR_shader_float16_int8`) + `storageBuffer16BitAccess` (`VK_KHR_16bit_storage`);
  fp16 configs **skip gracefully** (fall back to fp32) where unsupported. **NFW1 stays
  fp32 on disk** — the host casts fp32→half at upload for the fp16 modes. No new file
  format.
- **Two-track study harness.**
  - *Precision track:* one trained net run at all three modes → pdf-parity drift vs the
    fp32 / PyTorch reference (extends `test_neural_parity`) + a headless
    unbiased/firefly check + **ms/frame (MoltenVK) + buffer bytes**.
  - *Size track:* retrain per size in `spline_flow` (`render_records.py` /
    `export_weights` already parametrize) → held-out NLL + the same render check + cost.
  - Emits a **quality-vs-cost table** over a bounded grid (cuts, not the full
    cross-product) → a results doc with the Pareto + a recommended ship config.

- **Not in scope:** int8 / quantised weights; CUDA performance; online/dynamic
  training; a new weights-file format; *shipping* a final size/precision config (the
  study only recommends one).

## Capabilities

### New Capabilities
- `neural-precision-size-study`: the neural directional proposal's network size and
  floating-point precision become build-time configurable (including mixed fp16 on
  Apple-Silicon Metal via MoltenVK, with graceful fp32 fallback), and a study harness
  measures quality (pdf-parity drift / held-out NLL / in-renderer unbiased+firefly)
  against cost (MoltenVK ms/frame + buffer bytes) across the size×precision grid,
  preserving the mixture-MIS unbiasedness in every mode.

## Impact

- **Code:** `shaders/sampling/neural_flow.slang` (`NF_WT`/`NF_CT` typedefs +
  `#ifndef` size defines); `vk_context.py` (fp16 device-feature probe/enable);
  `sampling/neural_weights.py` (fp32→half upload cast); `vk_compute.py` /
  `vk_wavefront.py` (thread `-D` dims+precision into the neural-pass + record-entry
  compiles + the cache key); `renderer.py` (size/precision config + the study driver).
- **Descriptor bindings:** none new — weight buffers 33/34/35 change element type
  (fp32↔fp16) but not their binding slots.
- **Backends:** wavefront-primary (inference); the record-dump megakernel entry is
  precision-agnostic (records stay fp32).
- **Dependencies:** offline training stays in `spline_flow` (retrain per size); no new
  runtime Python dependency.
- **Tests:** fp16 pdf-parity bars; the two-track study harness; unbiasedness in every
  precision mode.
- **Docs:** `Architecture.md` (binding element-type note), `Wavefront.md` (size +
  precision config + the study finding), `README.md` (any config flag), `CHANGELOG.md`.
