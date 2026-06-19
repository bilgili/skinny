# PM-1 implementation blueprint (post adversarial review)

Authoritative authoring plan for the GPU SPPM kernels + dual-backend host
wiring, produced by the `sppm-blueprint` workflow (recon → integrate → 3-lens
adversarial review) and reconciled with the review verdicts. The
`gpu-portability` and `integration` lenses passed; the `sppm-math` lens returned
**flawed** with 5 blocking issues — all resolved below. Author against THIS
file; it supersedes the open questions in `design.md`.

## Buffers (3 folded GPU-internal buffers, sized by num_pixels = W·H)

- **`sppmStateCombined`** (`RWByteAddressBuffer`): `VisiblePoint[num_pixels]`
  region followed by `SppmAccum[num_pixels]` region. `Load<T>/Store<T>`, scalar
  layout identical Metal/SPIR-V (the `graphParamsCombined` fold trick). 1 Metal
  slot for 2 logical buffers. **Sized + indexed BY PIXEL** (`lane`), never by
  `stream_size` (fix #3).
- **`sppmGridCombined`** (`RWByteAddressBuffer`, all-uint → layout-agnostic):
  sub-ranges `gridCount | gridOffset | gridCursor | sortedIdx` over
  `numCells = next_pow2(2·num_pixels)`. 1 slot for 4 arrays.
- **`sppmScanScratch`** (`RWStructuredBuffer<uint>`): block sums for the parallel
  prefix scan, `ceil(numCells/256)` uints.

Host sizing helpers in `wavefront_layout.py`: `sppm_buffer_sizes(num_pixels,
msl=…)` (VP region uses the MSL stride on Metal) and
`sppm_grid_buffer_sizes(num_pixels)` (all uint).

`VisiblePoint` carries a per-pass `ld` (direct radiance this pass, written by the
eye stage — single site for direct, fix for the eye/update contradiction) plus
the persistent estimator `tau/radius/n`. `SppmAccum` is the per-pass
fixed-point atomic flux accumulator, cleared each pass.

## Kernels — `shaders/integrators/wavefront_sppm.slang`

All entries `wfSppm*` (never `main` — Metal reserves it). All dispatches have
host-known counts → plain `dispatch_full`/`dispatch_one`, no indirect dispatch
(no Metal CPU-readback stall).

1. **`wfSppmEye`** (over num_pixels): clear this pixel's `VP_ACTIVE` first; trace
   the camera path through the **flat delta lobe** (`bs.valid && bs.pdf<=0`) —
   glass/mirror ARE flat materials in skinny (UsdPreviewSurface opacity=0+ior),
   so there is **no** non-flat-specular branch. At the first `bs.valid &&
   bs.pdf>0` flat hit, store `VisiblePoint{pos, ns=post-normal-map N, beta=eye
   throughput EXCLUDING this vertex's BSDF, wo=V, materialId, flags|=VP_ACTIVE}`;
   on pass 0 / `n==0` init `radius=fc.sppmInitialRadius, n=0, tau=0`. Accumulate
   the full per-pass **direct** term (NEE at the VP + any emitter/env reached
   through the specular chain) into `vp.ld`. Branch order: test `bs.valid` FIRST
   (terminate if invalid), THEN `pdf<=0` (delta → bounce) vs `pdf>0` (store VP).
   Skip cutout-transparent hits (cap 32, like `wfPathIntersect`). Non-flat first
   hit (skin/python/debug) → terminate inactive (flat-only PM-1).
2. **`wfSppmGridCount`** (over num_pixels): active VPs only; `cell =
   hash(floor((pos-boundsMin)/cellSize)) & (numCells-1)`;
   `InterlockedAdd(gridCount[cell],1)`.
3. **`wfSppmGridScan*`**: blocked exclusive prefix sum count→offset —
   `wfSppmGridScanBlock` (per-workgroup scan + block total), `wfSppmGridScan`
   (`dispatch_one`, scan block sums), `wfSppmGridScanAdd` (add block base). NOT
   the single-thread `buildArgs` scan (numCells is W·H-scale).
4. **`wfSppmGridScatter`** (over num_pixels): active VPs; `dst =
   gridOffset[cell] + InterlockedAdd(gridCursor[cell],1); sortedIdx[dst]=pixel`.
5. **`wfSppmPhotonTrace`** (over `fc.sppmPhotonsEmitted`): emit power-weighted
   (`sampleEmissiveTriangle` pSel-weighted, **not** the uniform
   `bdpt.sampleLightOrigin`; divide by the matching pSel pdf), cosine emission
   about light normal; `beta = Le·cos/(selectPdf·pdfArea·pdfDir)` (distant:
   pdfDir=1, account 1/A_disk). Trace with RR (reuse `path.slang` RR), skip
   cutout-transparent. **Track `depth` (prior scatter count, specular or
   diffuse).** At a `bs.valid && bs.pdf>0` flat hit: **deposit only if `depth≥1`**
   (fix #1 — `depth==0` is light→VP direct that NEE owns; `depth≥1` is caustic
   `L S+ D` or indirect `L D+ D`, disjoint from NEE), then scatter
   (`beta*=bs.weight`, RR, `depth++`). Delta hit (`pdf<=0`): never deposit,
   scatter, `depth++`. **Deposit:** 3×3×3 neighbor-cell scan via the grid; for
   each active VP `j` within `||Δ||≤vp.radius` and same hemisphere
   (`dot(vp.ns,photonNs)>0`): `f = mat_vp.evaluate(woVP, wiVP).response /
   max(wiVP.z, 1e-4)` (**bare f_r, no cosine** — fix #2),
   `phi = beta·f`; guard NaN/inf/≥0 (copy `atomicSplatRadiance`);
   `InterlockedAdd(accum[j].phi{R,G,B}, uint(phi·SPPM_FLUX_FIXED_SCALE))`,
   `InterlockedAdd(accum[j].m,1)`.
6. **`wfSppmUpdate`** (over num_pixels): if `VP_ACTIVE`,
   `sppmUpdate(vp, sppmDecodeFlux(accum), accum.m, 2/3)`; store vp;
   `L_indirect = vp.tau/(max(fc.sppmPhotonsEmitted,1)·π·vp.radius²)` (else 0).
   `sample = vp.ld + L_indirect`; sanitize; running-mean composite into
   `accumBuffer` (`n=fc.accumFrame; (n==0)?sample:(prev·n+sample)/(n+1)`);
   `wfWriteDisplay`. Clear `accum[i]=0` for next pass.

## Per-pass dispatch order (one SPPM pass == one accumulation frame)

**Mandated split ordering (fix #4)** — NO per-tile interleave; tiles==1 is the
degenerate case:

1. **all eye tiles** (write every pixel's VP) → barrier
2. clear `gridCount`+`gridCursor` → `wfSppmGridCount` → scan(block/one/add) →
   `wfSppmGridScatter` (single global grid over all pixels) → barrier
3. clear `SppmAccum` region → `wfSppmPhotonTrace` (single global photon pass) →
   barrier
4. **all update tiles** → barrier

`record_sppm_loop` in `wavefront_driver.py` encodes exactly this; the
`WavefrontRecorder` protocol gains `clear_grid()` / `clear_accum()` primitives.

## Cell size (fix #5)

`fc.sppmCellSize = fc.sppmInitialRadius`, fixed for the session. Radius is
monotonically non-increasing per VP, so the initial radius is a valid upper
bound on every active VP's current radius → the 3×3×3 scan never misses a
photon. Do NOT recompute cell size from a per-pass max radius (the deleted
"max active VP radius" variant would under-cover large-radius pixels). A tighter
GPU max-reduce is a possible later optimization, out of PM-1 scope.

## Host wiring

- **`wavefront_layout.py`**: `sppm_buffer_sizes(num_pixels,…)` (param renamed
  from stream_size — fix #3) + new `sppm_grid_buffer_sizes(num_pixels)`.
- **`wavefront_driver.py`**: `record_sppm_loop(...)` (split ordering) +
  `clear_grid`/`clear_accum` recorder primitives.
- **`vk_wavefront.py`**: `WavefrontSppmPass` (compile entries, set-1 layout with
  the 3 buffers as RWByteAddressBuffer/RWStructuredBuffer, real `vkCmdDispatch`,
  `vkCmdFillBuffer` clears, allocate state buffer by num_pixels).
- **`metal_wavefront.py`**: `MetalWavefrontSppmPass` (3 buffers + `_bind_map`,
  size VP region `msl=True`). **No `SKINNY_METAL_SPPM` gate needed** — the new
  `wavefront_sppm.slang` never imports neural/record/splat globals, so they
  dead-strip and the kernel sits ~15/31 buffer slots. A Metal-compile smoke test
  with ≥2 graph materials confirms the budget before broad wiring.
- **`renderer.py`**: `_ensure_wavefront_sppm_pass` **split by backend** (Vulkan
  guards on `descriptor_sets`; Metal does not — `descriptor_sets` is None on
  Metal). Branch `integrator_index==2` in the 4 seams (Metal ~8156, Vulkan
  windowed ~8742, headless ~8979) and tighten `_resolve_record_source` (~9437)
  to `==0`. Pack the SPPM FrameConstants tail in `_pack_uniforms` AND
  `_pack_uniforms_msl` via `_FC_SCALAR_FIELDS`: `sppmInitialRadius(float),
  sppmCellSize(float), sppmGridRes(uint3), sppmPhotonsEmitted(uint)` after
  `cameraMirror` (24 B, fits the 768 B UBO). Add the sppm radius/photon tuning to
  `_current_state_hash`. Compute `sppmInitialRadius` from the pbrt `radius` param
  else a scene-bbox heuristic (~0.1 % of bbox diagonal); `sppmGridRes` from
  `sceneBoundsExtent/cellSize`.
- **`cli_common.py`**: add `--sppm-radius`, `--sppm-photons-per-pass`
  (selection + wavefront gate already done in group 1).
- **`pbrt/emit.py`+`metadata.py`+`report.py`**: map `Integrator "sppm"` → USD
  metadata + select INTEGRATOR_SPPM; report mapped not skipped (D6).

## FrameConstants tail additions

`sppmInitialRadius` (float), `sppmCellSize` (float), `sppmGridRes` (uint3),
`sppmPhotonsEmitted` (uint). Append to `_FC_SCALAR_FIELDS` so both the scalar
and offset-driven MSL packers relocate them; the import-time
`_VK_UNIFORM_BUFFER_BYTES` assert self-guards the 768 B budget.

## Non-blocking refinements carried from review

- Distant-light photon emission: account the `1/A_disk` factor in `beta`.
- Fixed-point flux: watch BOTH overflow (hot HDR cells) and **underflow** (dim
  caustics `phi·2^20 < 1` truncates to 0); per-pass clear bounds magnitude.
- `selectPdf` must be the full power-weighted light-selection pdf (group power ×
  within-group pSel), matching `emissive_triangle_light.slang`.
- Lock the parallel prefix-sum kernel against a CPU exclusive-scan in a unit test
  before wiring.
