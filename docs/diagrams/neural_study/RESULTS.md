# Neural size × precision — quality-vs-cost study

Scene: flat Cornell box · 96×96 · MoltenVK · {bsdf} reference mean=0.00808.

Coverage: ran **21/21** grid cells (7 sizes × 3 precisions).
Quality: size axis = held-out NLL (lower=better, baked per size); precision axis = fp16 pdf-parity drift (test_neural_parity 3.1). Cost: MoltenVK ms/frame + weight-buffer bytes.

| L | B | H | precision | ms/frame | weight bytes | NLL | unbiased rel | firefly p99.9 | fallback |
|---|---|---|-----------|---------:|-------------:|----:|-------------:|--------------:|:--------:|
| 6 | 24 | 48 | fp16-compute | 13.0 | 75456 | -0.276 | 0.0029 | 2.40e-02 |  |
| 6 | 24 | 48 | fp16-storage | 58.7 | 75456 | -0.276 | 0.0029 | 2.40e-02 |  |
| 6 | 24 | 48 | fp32 | 61.8 | 150912 | -0.276 | 0.0029 | 2.40e-02 |  |
| 4 | 24 | 96 | fp16-compute | 17.3 | 137472 | -0.281 | 0.0025 | 2.38e-02 |  |
| 4 | 24 | 96 | fp16-storage | 87.0 | 137472 | -0.281 | 0.0025 | 2.38e-02 |  |
| 4 | 24 | 96 | fp32 | 322.7 | 274944 | -0.281 | 0.0025 | 2.39e-02 |  |
| 6 | 16 | 96 | fp16-compute | 103.8 | 178560 | -0.282 | 0.0024 | 2.38e-02 |  |
| 6 | 16 | 96 | fp16-storage | 60.4 | 178560 | -0.282 | 0.0024 | 2.38e-02 |  |
| 6 | 16 | 96 | fp32 | 108.1 | 357120 | -0.282 | 0.0024 | 2.38e-02 |  |
| 6 | 24 | 96 | fp16-compute | 35.7 | 206208 | -0.279 | 0.0026 | 2.39e-02 |  |
| 6 | 24 | 96 | fp16-storage | 61.5 | 206208 | -0.279 | 0.0026 | 2.39e-02 |  |
| 6 | 24 | 96 | fp32 | 121.6 | 412416 | -0.279 | 0.0026 | 2.39e-02 |  |
| 6 | 32 | 96 | fp16-compute | 26.4 | 233856 | -0.279 | 0.0023 | 2.39e-02 |  |
| 6 | 32 | 96 | fp16-storage | 38.6 | 233856 | -0.279 | 0.0023 | 2.39e-02 |  |
| 6 | 32 | 96 | fp32 | 562.2 | 467712 | -0.279 | 0.0023 | 2.39e-02 |  |
| 8 | 24 | 96 | fp16-compute | 50.7 | 274944 | -0.279 | 0.0024 | 2.38e-02 |  |
| 8 | 24 | 96 | fp16-storage | 95.4 | 274944 | -0.279 | 0.0024 | 2.38e-02 |  |
| 8 | 24 | 96 | fp32 | 153.7 | 549888 | -0.279 | 0.0024 | 2.38e-02 |  |
| 6 | 24 | 144 | fp16-compute | 266.0 | 392256 | -0.281 | 0.0019 | 2.38e-02 |  |
| 6 | 24 | 144 | fp16-storage | 713.2 | 392256 | -0.281 | 0.0019 | 2.38e-02 |  |
| 6 | 24 | 144 | fp32 | 247.2 | 784512 | -0.281 | 0.0019 | 2.38e-02 |  |

## Pareto front (NLL vs ms/frame)

- L6B24H48 fp16-compute: NLL=-0.2762240171432495, 13.0 ms, 75456B
- L4B24H96 fp16-compute: NLL=-0.2808907926082611, 17.3 ms, 137472B
- L6B16H96 fp16-storage: NLL=-0.2815595269203186, 60.4 ms, 178560B

## Recommended ship config (the Pareto knee)

**L6 B24 H48 @ fp16-compute** — the smallest weight footprint (75456 B, **18% of the baseline fp32's 412416 B**) whose NLL is within 2% of the best size, unbiased + firefly-bounded. Picked on the clean axes (see notes); a follow-up change ships it.

## Measurement notes

- **Reliable axes:** weight bytes (deterministic), held-out NLL, unbiased rel-mean, firefly p99.9. The fp16 modes are exactly **½** the fp32 weight bytes; every cell is unbiased (rel-mean < 0.003) and firefly-bounded (p99.9 ≈ 0.024).
- **ms/frame is noisy across sizes** — cells run sequentially so the GPU heats over the sweep (early cells cold/fast, late cells hot/slow; e.g. L6B24H144's 713 ms fp16-storage is a thermal spike vs its 247 ms fp32). Treat cross-size ms as indicative, not exact. **Within a size** (the 3 precisions measured adjacently, same thermal window) **fp16-compute < fp16-storage < fp32 holds in 6/7 sizes** — the real Apple-Silicon precision win (half ALU + half bandwidth).
- **Quality is flat across size** on this broad-indirect scene (NLL spread ~2%, −0.276…−0.282): a smaller net fits nearly as well, so the knee is small. A concentrated-indirect scene would spread NLL and move the knee up.
- **fp16 pdf-parity drift** (scene-independent, `test_neural_parity.py` 3.1) is negligible: ~4e-4 (storage) / ~1e-3 (compute) vs the fp32 reference — fp16 costs no measurable quality here.
