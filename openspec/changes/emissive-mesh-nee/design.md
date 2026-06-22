## Context

Emissive-triangle area lights flow through one host builder and one shader
selector:

- **Host** `Renderer._upload_emissive_triangles` (`renderer.py:5070-5123`): walks
  every enabled instance whose material has non-zero `emissiveColor`,
  world-transforms each source triangle, packs a 64-byte record `(v0, v1, v2,
  emission, area)` into the emissive-triangle buffer (binding 18,
  `EMISSIVE_TRI_STRIDE = 64`), and sets `_num_emissive_tris`. It caps at
  `EMISSIVE_TRI_CAPACITY = 256` via `n = min(len(records), 256)` — silent.
- **Shader** `allLightsNEE` (`nee.slang:90-95`): when `numEmissiveTriangles > 0`,
  picks `idx = min(uint(rng.next()·N), N-1)` and calls `neeLightEstimator` on
  `emissiveTriangles[idx]`. `EmissiveTriangleLightImpl.samplePoint`
  (`emissive_triangle_light.slang`) sets `selectionPdf = 1/N` and
  `pdfArea = selectionPdf / triArea`. The estimator already does area→solid-angle
  conversion, visibility, and MIS (`pdfSolidAngle`, power heuristic).

There is already a worked pattern for a discrete importance distribution: the
**environment** map. `environment.build_env_distribution()` builds a
piecewise-constant 2D distribution; `_upload_env_distribution` concatenates a
`[marginal | conditional]` CDF into one buffer (binding 31, with `numEnvDist*`
metadata at binding 32); `environment.slang` samples it with an
`upperBound`-style binary search (`cdf` has `n+1` entries, find largest `i` with
`cdf[i] <= u`). This change reuses that exact shape for a 1D power CDF over
emissive triangles.

ReSTIR DI (`restir/light_ris.slang`) draws emissive-triangle candidates through
the same `light.samplePoint` / `selectionPdf` seam and is primary-hit only, so it
inherits a better selection distribution with no ReSTIR-specific edit.

## Goals / Non-Goals

**Goals:**
- Emissive-mesh NEE is **unbiased at any triangle count** — remove the 256 cap so
  no light energy is silently dropped (fixes the dark interior).
- Emissive-mesh NEE is **low-variance** — select triangles ∝ power
  (`area × luminance`) instead of uniformly by index (fixes the noise).
- Keep small-light behaviour and all MIS / no-double-count invariants unchanged
  (the `diffuse_arealight` corpus must not regress).
- Reuse the env importance-sampling pattern (CDF + binary search) rather than
  inventing a new structure.

**Non-Goals:**
- Alias-table O(1) sampling (CDF O(log N) is enough for the target counts; no
  existing alias analogue in skinny to reuse).
- A light BVH / spatial light clustering (per-shading-point importance).
- Re-balancing MIS across env + emissive + sphere lights as a joint distribution.
- Any ReSTIR-internal change beyond inheriting the new per-triangle distribution.
- Changing the BSDF-hit emission accounting (owned by `integrator-convergence`).

## Decisions

### D1 — Weight = area × Rec.709 luminance(emission)
Per emissive triangle `i`, `w_i = area_i · (0.2126·Er + 0.7152·Eg + 0.0722·Eb)`.
Area-only weighting (a fallback) ignores that one mesh may be far brighter than
another; power weighting puts samples where the radiance actually is. Computed on
the host (areas + emission are already known in `_upload_emissive_triangles`).

### D2 — Cumulative-power CDF, binary-search selection (mirror env)
Host builds `cdf[0..n]` with `cdf[0]=0`, `cdf[k]=Σ_{i<k} w_i`, normalized by
`cdf[n]=Σw` (store the unnormalized total or normalize to 1.0 — match the env
convention). Upload as a `StructuredBuffer<float>` at a **new binding**, sized
`n+1`. The shader draws `u ∈ [0,1)`, finds the largest `i` with `cdf[i] <= u`
(reuse the env `upperBound` helper), and selects triangle `i` with discrete
probability `p_i = w_i / Σw = (cdf[i+1]-cdf[i]) / cdf[n]`.

### D3 — `selectionPdf = p_i` everywhere it matters
`EmissiveTriangleLightImpl` carries `p_i` (read from the CDF, or passed in) and
sets `selectionPdf = p_i`, `pdfArea = p_i / triArea_i`. The estimator's
area→solid-angle conversion and the NEE-side MIS pdf (`pdfSolidAngle`) consume
`pdfArea` unchanged, so the only behavioural change is the selection probability.
A degenerate all-zero-power scene (no emissive triangles) keeps the
`numEmissiveTriangles == 0` early-out.

### D4 — Remove the cap: dynamic buffer sizing
Drop `n = min(len(records), 256)`. Size the emissive-triangle buffer and the CDF
buffer to the actual `len(records)` (round up to a growth quantum), reallocating +
rebinding when the count exceeds the current capacity — the same grow-and-rebind
pattern `material_capacity` uses. `log` the final emissive-triangle count so a
large emissive mesh is visible, never silently clipped.

### D5 — Verification reuses the pbrt parity harness on native Metal
All GPU tests go through `skinny.pbrt.parity.render_linear` (backend resolved by
`select_backend` → native Metal on Apple Silicon), comparing exposure-aligned
relMSE / FLIP. Three checks: (a) **correctness** — a synthetic >256-triangle
emissive mesh: pre-change biased dark (energy lost), post-change converges to the
expected (uniform-emitter) energy; (b) **no regression** — `diffuse_arealight`
stays within the corpus parity tolerance; (c) **variance** — equal-spp relMSE on
a multi/uneven-emitter scene drops materially (power-weighted < uniform).

## Risks / Trade-offs

- **Buffer growth churn** — a scene with a huge emissive mesh allocates a large
  per-triangle buffer + CDF. Mitigate with a growth quantum and only on count
  increase; log the size. (The data is small: 64 B/tri + 4 B/CDF entry.)
- **CDF rebuild cost on scene edits** — the CDF is rebuilt whenever emissive
  geometry/material changes, like the triangle buffer already is; negligible vs a
  bake. Static during accumulation, so no per-frame cost.
- **Zero-luminance emissive** (emission color present but luminance 0, e.g. pure
  values that Rec.709 weights to ~0) — guard `Σw > 0`; fall back to area-only or
  the `numEmissiveTriangles == 0` path so the CDF is never degenerate.
- **MIS consistency** — if `selectionPdf` is updated in `samplePoint` but the
  BSDF-side MIS companion still assumes uniform, the estimate biases. The
  power-heuristic companion uses the light's `pdfSolidAngle`, which derives from
  the same `selectionPdf`, so they stay matched — covered by the no-regression +
  energy tests.
- **Cap removal changes a constant other code may assume** — audit any reader of
  `EMISSIVE_TRI_CAPACITY` / `_num_emissive_tris` (descriptor range, struct-layout
  tests) so dynamic sizing doesn't desync a fixed range.
