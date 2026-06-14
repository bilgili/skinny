# Guiding variance — `cornell` (vulkan)

Reference: {bsdf} @ 512 spp, two-witness converged (rel-err 0.0001167 ≤ 0.02), hash `58e545881002f86b`, mean=0.00079, 96×96.

Seeds: **4** independent; budgets [16, 64, 256] spp; variance = MSE vs reference over the linear-HDR accumulation image, mean ± spread over seeds.

**Neural cells use the dummy (untrained, zero) net** — this slice measures *cost + unbiasedness*, not a guiding win: the dummy net is a valid-but-poor proposal, so mixture-MIS keeps it unbiased (same variance as `bsdf`) while adding the MLP pre-pass cost. A guiding win needs a trained `.nrec` (set `net:` per cell; produced by `spline_flow`), which moves the variance column.

### Equal-time / equal-variance / efficiency — cornell

% source: scripts/guiding_variance_sweep.py --backend vulkan --scene cornell (seeds=4, budgets=[16, 64, 256], ref_spp=512, res=96)

Equal-variance target = slice median eq-time var = 8.556e-07 MSE. `in-grid=n` ⇒ extrapolated (logged).

| config | eq-time var (MSE) | ± spread | firefly p99.9 | time (s) | 1/(var·t) | eq-var spp | eq-var s | in-grid |
|---|---|---|---|---|---|---|---|---|
| bsdf|V1/E0/fp32 | 8.610e-07 | 1.590e-09 | 0.002301 | 0.8053 | 1.442e+06 | 221.6 | 0.6987 | y |
| bsdf,env|V1/E0/fp32 | 8.620e-07 | 1.261e-09 | 0.002347 | 0.8161 | 1.421e+06 | 221.6 | 0.7112 | y |
| bsdf,neural|V1/E0/fp32 | 8.556e-07 | 9.643e-10 | 0.002302 | 25.98 | 4.499e+04 | 221.6 | 22.59 | y |
| bsdf,neural|V1/E1/fp32 | 8.556e-07 | 9.645e-10 | 0.002302 | 77.35 | 1.511e+04 | 221.6 | 66.76 | y |
| bsdf,neural|V1/E3/fp32 | 8.556e-07 | 9.639e-10 | 0.002302 | 53.5 | 2.185e+04 | 221.6 | 46.32 | y |
| neural|V1/E0/fp32 | 8.392e-07 | 1.736e-09 | 0.002394 | 20.78 | 5.735e+04 | 221.7 | 17.98 | y |
| neural|V1/E1/fp32 | 8.392e-07 | 1.736e-09 | 0.002394 | 60.4 | 1.973e+04 | 221.7 | 52.3 | y |
| neural|V1/E3/fp32 | 8.392e-07 | 1.736e-09 | 0.002394 | 40.33 | 2.955e+04 | 221.7 | 34.92 | y |
| bsdf,env|V1/E0/fp32+restir-di | 8.657e-07 | 1.619e-09 | 0.002351 | 0.9208 | 1.255e+06 | 221.6 | 0.8002 | y |
| bsdf,neural|V1/E1/fp16 | 8.556e-07 | 9.639e-10 | 0.002302 | 13.66 | 8.559e+04 | 221.6 | 11.82 | y |


### Skipped cells (coverage gaps, not hidden)

% source: scripts/guiding_variance_sweep.py --backend vulkan --scene cornell (seeds=4, budgets=[16, 64, 256], ref_spp=512, res=96)

- `bsdf,neural|V2/E0/fp32` — chart V2 not built (renderer-chart-selection not landed)
- `bsdf,neural|V1/E0/fp32/T` — temporal=on not built (neural-temporal-conditioning not landed)


![equal-time](./cornell_equal_time.svg)
![efficiency](./cornell_efficiency.svg)
![equal-variance](./cornell_equal_variance.svg)
![variance-vs-spp](./cornell_variance_vs_spp.svg)
