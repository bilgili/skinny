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

### D2 — Cumulative-power CDF packed **inline** into the emissive buffer (same slot as env)
The CDF rides in the **existing** emissive-triangle buffer (binding 18), the way
the env distribution concatenates its `[marginal | conditional]` CDFs into one
buffer slot — **no new descriptor binding**. Each 64-byte `EmissiveTriangle`
record has three unused `.w` padding lanes (`_v0.w/_v1.w/_v2.w`; the host writes
0.0 there today). The host packs, per triangle `i`:
- `_v0.w = cw[i]` — the **inclusive** normalized cumulative weight
  `cw[i] = Σ_{j≤i} w_j / Σw` (so `cw[n-1] = 1.0`), for binary-search selection.
- `_v1.w = p_i` — the per-triangle selection probability `p_i = w_i / Σw`, read
  directly as `selectionPdf` (no neighbour fetch).

A shared `sampleEmissiveTriangle(u, n)` helper in `scene_lights.slang` draws
`u ∈ [0,1)` and binary-searches for the smallest `i` with `cw[i] > u` (same shape
as the env `envSearchCdf` upper-bound), selecting triangle `i`.

**Why inline, not a separate `StructuredBuffer<float>` at a new binding (revised
from the original proposal):** the native-Metal wavefront programs (neural
`SKINNY_METAL_NEURAL`, records `SKINNY_METAL_RECORDS`) already sit at **exactly**
Metal's 31-buffer argument-table cap (see `docs/Architecture.md` § MetalContext
and the `metal-graphparam-fold` history), and the wavefront path/BDPT
integrators' NEE reads `emissiveTriangles` (binding 18) — so it is already in that
count. A separate CDF buffer would add a live 32nd buffer and crash those builds.
Packing into the existing record's dead lanes adds **zero** buffers and keeps the
typed `StructuredBuffer<EmissiveTriangle>` (no `ByteAddressBuffer` rewrite, no
descriptor-map change).

### D3 — `selectionPdf = p_i` everywhere it matters
`loadEmissiveTriangleLightImpl` sets `EmissiveTriangleLightImpl.selectionPdf =
tri.pi` (the packed `_v1.w`, was `1 / numEmissiveTriangles`) and
`pdfArea = selectionPdf / triArea_i`. The estimator's area→solid-angle conversion
and the NEE-side MIS pdf (`pdfSolidAngle`) consume `pdfArea` unchanged, so the
only behavioural change is the selection probability. A degenerate all-zero-power
scene (no emissive triangles) keeps the `numEmissiveTriangles == 0` early-out.

### D3a — ReSTIR draws through the **same** power-CDF helper (revised)
The original proposal claimed ReSTIR needs *no* change. That is not quite right:
`light_ris.slang` draws its emissive candidate index **uniformly** and reports the
source pdf via the shared `loadEmissiveTriangleLightImpl().selectionPdf`. Once
that seam reports power `p_i`, a uniform draw would mismatch the reported pdf and
bias the RIS estimate. So ReSTIR's candidate *index draw* switches to the same
`sampleEmissiveTriangle` helper (one line) — the reservoir / RIS / GRIS reuse code
is untouched, and the change is strictly an improvement (importance-sampled
primary-hit candidates).

### D4 — Remove the cap: dynamic buffer sizing (single buffer)
Drop `n = min(len(records), 256)`. Size the **one** emissive-triangle buffer
(records + inline CDF lanes) to the actual `len(records)`, doubling capacity
(`max(n, cap*2)`) and reallocating + rebinding when the count exceeds the current
capacity — the same grow-and-rebind pattern `material_capacity` uses (Vulkan
re-writes binding 18; native Metal re-reads the buffer reference fresh at every
dispatch, so the realloc is picked up automatically). `print` the final
emissive-triangle count so a large emissive mesh is visible, never silently
clipped.

### D5 — Verification reuses the pbrt parity harness on native Metal
All GPU tests go through `skinny.pbrt.parity.render_linear` (backend resolved by
`select_backend` → native Metal on Apple Silicon), comparing exposure-aligned
relMSE / FLIP. Three checks: (a) **correctness** — a synthetic >256-triangle
emissive mesh: pre-change biased dark (energy lost), post-change converges to the
expected (low-poly-emitter) energy; (b) **no regression** — `diffuse_arealight`
stays within the corpus parity tolerance; (c) **variance** — equal-spp relMSE on
a multi/uneven-emitter scene drops materially (power-weighted < uniform).

The power-vs-uniform A/B (checks c + unbiasedness) needs the **uniform** baseline
from the *same* binary. Rather than gate it with a shader flag, the host exposes a
test-only `Renderer._emissive_uniform_selection` toggle: when set, the inline CDF
is built uniform (`cw[i] = (i+1)/n`, `p_i = 1/n`) so the *unchanged* shader path
reproduces exact uniform-by-index selection. `render_linear` grows an
`emissive_uniform=` parameter that flips it before the scene build. Power and
uniform thus differ only in the packed CDF, nothing else.

## Risks / Trade-offs

- **Buffer growth churn** — a scene with a huge emissive mesh allocates a large
  per-triangle buffer. Mitigate by doubling capacity and reallocating only on
  count increase; log the size. (The data is small: 64 B/tri, CDF packed inline
  in the existing record lanes — no extra buffer.)
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
  tests) so dynamic sizing doesn't desync a fixed range. Audited: `_num_emissive_tris`
  feeds `fc.numEmissiveTriangles` (loop bound, scales with the buffer) and the
  Vulkan descriptor `range` is read live from `emissive_tri_buffer.size`; the
  `test_struct_layout` emissive case only checks the 64-byte size (the `.w` lanes
  we pack into are within it — no assertion on their being zero), so packing is
  safe. `EMISSIVE_TRI_CAPACITY` becomes only the *initial* capacity seed.
