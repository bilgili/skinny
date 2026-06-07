## 1. Precision plumbing (mixed fp16)

- [x] 1.1 `shaders/sampling/neural_flow.slang`: introduce `NF_WT` (weight storage) and `NF_CT` (MLP GEMM accumulate) type aliases, `#ifndef`-guarded, defaulting to `float`/`float`. Change weight/bias buffers to `StructuredBuffer<NF_WT>` and the linear-layer accumulate to `NF_CT`; keep the RQ-spline math (softmax/cumsum/exp/log + inverse-quadratic solve) and the returned solid-angle pdf in `float`. Verify default build is byte-identical to the shipped fp32 net.
- [x] 1.2 `vk_context.py`: probe `VkPhysicalDeviceShaderFloat16Int8Features.shaderFloat16` + `VkPhysicalDevice16BitStorageFeatures.storageBuffer16BitAccess`; enable them on the device when present; expose a capability flag. Confirm slangc emits the SPIR-V `Float16` / `StorageBuffer16BitAccess` capabilities from `half` usage (add an explicit `-capability` only if needed).
- [x] 1.3 `sampling/neural_weights.py`: add an fp32→half upload path (cast `weight_bytes`/`bias_bytes` to `float16`) selected by the precision mode; NFW1 on disk stays fp32. The buffers at bindings 33/34/35 hold half bytes in fp16 modes (element type changes, slots don't).
- [x] 1.4 Thread the precision mode into the neural-pass + inline-inverse + record-entry compiles (`-DNF_WT=… -DNF_CT=…`) and fold it into the pipeline cache key; fp16 modes auto-fall-back to fp32 when 1.2's capability is absent (logged).

## 2. Size plumbing

- [x] 2.1 `neural_flow.slang`: make `NF_LAYERS / NF_BINS / NF_HIDDEN` `#ifndef`-guarded defines (default 6/24/96). Confirm the inline inverse (`proposal.slang` via `neural_proposal`), the wavefront pre-pass, and the record entry all inherit the dims through the import.
- [x] 2.2 Thread the size dims from a renderer config into every neural `.spv` compile (`-DNF_*=…`) + the pipeline cache key; reuse the existing NFW1 loader arch assert as the mismatch guard.
- [x] 2.3 `spline_flow`: confirm `render_records.py` + `export_weights.export_flow` produce a net at an arbitrary `(layers,bins,hidden)` (already parametrized) and bake a few off-default sizes for the size track.

## 3. Study harness (two-track)

- [x] 3.1 Precision track: extend `tests/test_neural_parity.py` numpy mirror with an fp16-round path + (where bindable) the GPU `sampleNeural`/`pdfNeural`; report pdf-parity drift of fp16-storage / fp16-compute vs the fp32 / PyTorch reference with mode-specific tolerance bars.
- [x] 3.2 Size+precision render+cost driver (extends `tests/test_neural_headless.py`): for each grid cell, measure MoltenVK ms/frame + weight-buffer bytes, and a headless unbiased + firefly-tail check on the flat Cornell box; `log()` exactly which cells ran.
- [x] 3.3 Size track: retrain each grid size in `spline_flow`, record held-out NLL + pdf normalization, bake NFW1, feed 3.2.
- [x] 3.4 Emit the quality-vs-cost table (CSV + a results doc) over the bounded grid (`hidden∈{48,96,144}`, `layers∈{4,6,8}`, `bins∈{16,24,32}` one-axis sweeps × 3 precision modes); identify the Pareto knee + a recommended ship config.

## 4. Verification + docs

- [x] 4.1 Gate: default `fp32 @ 6/24/96` build is byte-identical to the shipped proposal (extends the existing `{bsdf}` ≡ wavefront / `{bsdf,neural}`-unbiased gates).
- [x] 4.2 Gate: each fp16 mode stays unbiased (converges to the fp32 reference within noise) where the device supports it; fp16-unsupported devices skip cleanly.
- [x] 4.3 Docs: `Architecture.md` (bindings 33/34/35 element-type note), `Wavefront.md` (size + precision config + the study finding/Pareto), `README.md` (any new config flag), `CHANGELOG.md`, `PythonAPI.md` (any new config symbols).
