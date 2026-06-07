## Context

The frozen neural proposal shipped (change `neural-directional-proposal`) with a
fixed-architecture flow: `neural_flow.slang` is entirely `float`, and
`NF_LAYERS=6 / NF_BINS=24 / NF_HIDDEN=96 / NF_COND=9` are `static const`. The 6.3
equal-time gate showed the proposal is unbiased but ~28× too slow on MoltenVK/MPS —
the loss is **inference cost**, not quality (clamped bulk MSE matched ReSTIR). Apple
GPUs run `half` at ~2× fp32 ALU throughput with half the bandwidth; MoltenVK exposes
`shaderFloat16` + `storageBuffer16BitAccess`; slangc emits `half`. The host side is
already size-parametric: `export_weights.export_flow` bakes `(layers,bins,hidden,cond)`
and `neural_weights.load_neural_weights` asserts them. Only the **shader** is fixed.

## Goals / Non-Goals

**Goals:**
- Build-time-configurable `NF_LAYERS/BINS/HIDDEN` (one shader, `-D` per config).
- Mixed-precision fp16 (fp32 / fp16-storage / fp16-compute) with the RQ-spline math
  kept fp32; graceful fp32 fallback where MoltenVK lacks fp16.
- A study harness mapping quality (parity-drift / NLL / render) vs cost (MoltenVK
  ms/frame + bytes) across a bounded size×precision grid → a Pareto + ship recommendation.
- Unbiasedness preserved in every precision mode (mixture-MIS).

**Non-Goals:**
- int8 / quantised weights; CUDA performance tuning; online/dynamic training; a new
  NFW1 format; *shipping* a chosen config (the study recommends, a later change ships).

## Decisions

- **Precision via `-D` typedefs in one file, not a fork.** `NF_WT` (weight storage)
  and `NF_CT` (MLP GEMM accumulate) are compile-time aliases; the spline stays `float`.
  A forked `neural_flow_fp16.slang` would duplicate the numerically-tricky inverse and
  drift out of sync. *Alt rejected:* fork.
- **Mixed-precision boundary = the linear-layer GEMMs only.** That is the cost (3
  Linears × `NF_LAYERS` couplings) and the fp16-tolerant part. The RQ-spline
  softmax/cumsum/exp/log + analytic inverse-quadratic solve are catastrophic-cancellation
  prone in fp16 → kept fp32. This is the safe knee of the accuracy/speed curve.
- **NFW1 stays fp32 on disk; host casts to half at upload.** Keeps one weights format
  + the existing parity goldens valid; the precision mode is a *runtime/build* property
  of the renderer, not of the file. fp16-storage uploads `half` bytes to bindings
  33/34/35 (element type changes, slot doesn't).
- **Size via `-D` recompile, not specialization constants.** Per-lane arrays are sized
  `float[NF_HIDDEN]`; SPIR-V spec-constant-sized arrays lower unreliably through
  MoltenVK. `-D` matches the existing recompile-per-scene pattern and is deterministic;
  a config = one `.spv` keyed in the pipeline cache. *Alt rejected:* spec constants;
  fixed-MAX + runtime bound (can't show the real cost shrink).
- **Feature probe + graceful fallback.** `vk_context` queries
  `VkPhysicalDeviceShaderFloat16Int8Features.shaderFloat16` +
  `VkPhysicalDevice16BitStorageFeatures.storageBuffer16BitAccess`; absent → fp16
  configs are skipped (logged), never a hard failure. Keeps the renderer portable.
- **Two-track quality metric (size vs precision are different questions).** Precision =
  the *same* trained net at three modes → parity drift vs its fp32 self / the PyTorch
  reference (numerical fidelity, scene-independent). Size = a *retrained* net per size →
  held-out NLL (its fit), since a smaller net is a different model, not a degraded one.
  Both add a headless unbiased+firefly render check and report MoltenVK ms/frame +
  buffer bytes. This sidesteps Cornell's flat render-variance signal (6.3).
- **Bounded grid, logged cuts.** Not the full cross-product: e.g.
  `hidden∈{48,96,144}`, `layers∈{4,6,8}`, `bins∈{16,24,32}` swept one-axis-at-a-time
  from the baseline, × the 3 precision modes. The harness `log()`s exactly which cells
  it ran so the table never reads as exhaustive when it isn't.

## Risks / Trade-offs

- **fp16 numerical drift** (spline-adjacent values, very small pdfs) → mitigated by the
  fp32 spline boundary; the fp16 parity bars are relaxed vs fp32 and the *measured*
  drift is reported, not hidden. If fp16-compute breaks parity badly, fp16-storage is
  the fallback win.
- **fp16 fireflies** — lower-precision pdf valleys can deepen the heavy tail; the study
  measures the firefly tail per config (mixture-MIS still bounds bias).
- **MoltenVK fp16 absence / partial support** → probed; fp16 cells skipped with a log,
  fp32 always runs.
- **Retrain-per-size cost** — bounded grid; each size trains in minutes on MPS.
- **slangc fp16 capability emission** — `half` should auto-pull SPIR-V `Float16` /
  `StorageBuffer16BitAccess`; if not, add an explicit `-capability`. Verified in 1.x.

## Migration Plan

Additive and gated: the default config is `fp32 @ 6/24/96` → **byte-identical to the
shipped proposal** (defines + typedefs default to the current values/`float`). fp16 and
off-default sizes are opt-in via the renderer config / harness. No descriptor-binding
changes; rollback = build the default config. Shader change requires recompiling the
neural `.spv`s with slangc.

## Open Questions

- Does the per-lane fp16 GEMM actually beat fp32 on M-series here, or is the pre-pass
  dominated by dispatch/bandwidth (in which case fp16-storage is the real lever)? — the
  study answers it.
- Pareto knee: smallest `(size, precision)` that keeps pdf-parity within bar AND stays
  firefly-bounded — feeds the follow-up *ship* change + the CUDA target.
- Should the chosen config be expressed in the NFW1 header (precision tag) so a baked
  net pins its intended mode? — deferred to the ship change.
