# Design — NanoVDB Heterogeneous Volume Rendering

## Context

The volumetric machinery shipped with `pbrt-subsurface-volumetric` was deliberately built as an
additive-extension seam (living spec `subsurface-scattering`, requirement *"Volume transport is
forward-compatible with heterogeneous free-standing media"*):

- Transport is majorant / null-collision (Woodcock) tracking; a constant σ_t is the degenerate
  case of a varying density field.
- The walk reads the medium only through `densityAt(medium, p)` and
  `mediumMajorant(medium, segment)` dispatched on `MediumParams.kind`
  (`materials/subsurface/medium.slang`), with `MEDIUM_NANOVDB = 1u` reserved in `bindings.slang`
  and `MediumParams.gridHandle` reserved in `common.slang`.
- The pbrt importer (`src/skinny/pbrt/media.py`, `api.py:258`) already parses `MakeNamedMedium` /
  `MediumInterface`; heterogeneous types (`nanovdb`, `uniformgrid`, …) are detected and skipped
  with a report entry.

Target scenes:

| Scene | Grid | Coefficients | Lights | Notes |
|---|---|---|---|---|
| `disney-cloud` | `wdas_cloud_quarter.nvdb` (~0.1 G voxels) | σ_a=0, σ_s=1, g=0.877, scale=4 | infinite (rgb) + distant | `Material "interface"`, sphere-bounded medium, `Scale -1 1 1` mirrored camera, ground disk |
| `bunny-cloud` | `bunny_cloud.nvdb` | σ_s=10, σ_a=0.5, g=0 | infinite (`sky.exr` ×4) | `Material "interface"`, sphere r=45, rotated medium CTM, Nikon d850 film sensor |

Both use `Integrator "volpath"` — importer already maps integrators onto skinny's path tracer.
Walk NEE currently supports distant + env lights only, which is exactly what both scenes use.

Separately: on macOS, abandoned Metal compute work (process killed mid-dispatch, watchdog-length
kernels) is not reliably cleaned up — the GPU stays wedged until reboot (observed repeatedly during
prior changes: "Metal slang-rhi WEDGED, needs reboot"). Long per-pixel delta-tracking marches make
this failure mode much more likely, so dispatch hygiene ships in the same change.

## Goals / Non-Goals

**Goals:**

- Render `disney-cloud` and `bunny-cloud` after `.pbrt → .usda` conversion, visually and
  radiometrically comparable to pbrt v4 (dual parity gates: pbrt-truth + mega≡wave
  self-consistency).
- Fill the reserved `MEDIUM_NANOVDB` kind through the existing `densityAt`/`mediumMajorant` seam —
  **no change to the transport equations, NEE, RR, or integrator wiring** (that's the contract the
  subsurface spec promised).
- Free-standing `MediumInterface` attachment: pbrt `Material "interface"` shapes route the path
  into the medium walk with index-matched (pass-through) boundaries.
- Metal dispatch hygiene: every GPU dispatch bounded below the macOS watchdog, teardown guaranteed
  on all exit paths, regression harness proving no kernel outlives its process.

**Non-Goals:**

- Spectral σ (RGB only — same 0.875-ish RGB-vs-spectral floor as `diffuse_arealight` applies).
- Emissive volumes (temperature/blackbody grids), `LeGrid`/RGB grids — density (`FloatGrid`) only.
- Nested/overlapping media, medium priority stacks (explicitly out of scope in the subsurface spec).
- Macrocell/DDA local majorants — global majorant first; a residual-ratio/macrocell follow-up is a
  separate change if convergence demands it.
- Camera-in-medium (both target scenes have the camera outside the bound sphere).
- Vulkan-specific optimizations beyond parity; BDPT/SPPM media support (Path first, mirroring the
  emissive-MIS precedent; other integrators are recorded exclusions in the parity matrix).
- Nikon d850 film sensor response (bunny-cloud) — approximated by existing film exposure params;
  recorded as importer `approx`, not spec'd.

## Decisions

### D1 — NanoVDB parsing: pure-Python reader, dense-decode at import

A small pure-Python + numpy reader (`src/skinny/pbrt/nanovdb.py`) parses the NanoVDB file layout
(magic, grid metadata, tree: root → upper 32³ → lower 16³ → leaf 8³ nodes) for **FloatGrid /
FogVolume** class grids only, and decodes to a dense `numpy.float32` array plus the
index→world transform and value min/max.

- *Why not OpenVDB/NanoVDB native bindings:* heavy new dependency (C++ build, like the MaterialX
  saga) for two files; the repo precedent is pure-Python importers (`loopsubdiv.py`, `read_ply`
  gzip sniffing). NanoVDB is a flat, pointer-free, mmap-friendly format — designed to be trivially
  walkable; a read-only float-grid decoder is a few hundred lines.
- *Why dense-decode:* the GPU side wants a 3D texture anyway (D3); sparse GPU traversal of NanoVDB
  nodes in Slang is a large follow-up, not needed for correctness.
- Handle both uncompressed and the codec field (NONE/ZIP/BLOSC): the two target files ship
  uncompressed or ZIP (zlib — stdlib); BLOSC → clear "unsupported codec" error.

### D2 — USD representation: `UsdVolVolume` + `OpenVDBAsset` field pointing at the `.nvdb`

The importer emits a `UsdVol.Volume` prim with a `UsdVol.OpenVDBAsset` field (`filePath` →
the original `.nvdb`, `fieldName "density"`), CTM from the pbrt medium/shape transform stack, and
the medium coefficients (`volume_sigma_a/_s`, `volume_g`, `volume_scale`, `pbrt_medium`) as
`skinnyOverrides` customData — the same override channel every other pbrt medium key rides.

- *Why reference, not bake into USD:* keeps `.usda` small and the grid authoritative;
  `usd_loader.py` resolves the asset path and calls the D1 reader at load. USD does not validate
  that an OpenVDBAsset's file is `.vdb`, and skinny is the only consumer.
- The bounding shape (pbrt `Shape "sphere"` with `Material "interface"`) is still emitted as
  geometry with a null/interface material binding carrying the medium reference — the walk needs a
  boundary surface, and this matches how pbrt structures it.

### D3 — GPU density field: one `Texture3D<float>` (R16F) binding + inline grid params

Density uploads as a single 3D sampled texture, R16F, normalized so texel values ∈ [0,1] with the
scale folded into σ (majorant = σ_t · scale · maxDensity). New descriptor binding (next free slot
per the Architecture.md binding map; update the map). `MediumParams` gains real fields where
`gridHandle` was reserved: grid slot + world→index transform (3×4) + majorant density, packed
inline in the material record like the existing subsurface medium (no new SSBO — Metal 31-slot
argument-table pressure, same reasoning as emissive-mesh-nee's inline CDF).

- *Why one texture, not a bindless pool:* both target scenes have exactly one grid; a pool is
  additive later (mirror the flat-texture pool pattern) without touching the walk.
- *Why R16F:* wdas quarter cloud ≈ 0.1 G voxels → 200 MB at R16F vs 400 MB fp32; density is
  normalized so fp16 quantization error is ≤ 2⁻¹¹ of max — invisible under Monte Carlo noise.
  `metal_compute.py`/`vk_compute.py` `SampledImage` grows a 3D variant (both APIs support it; on
  Metal this is a plain `TextureType.texture_3d` through slang-rhi).

### D4 — Transport: `MEDIUM_NANOVDB` = two new `case` bodies, walk untouched

`densityAt` → trilinear `Texture3D.SampleLevel` of the density grid in index space;
`mediumMajorant` → global majorant (σ_t · scale · maxDensity). Boundary mode: index-matched
pass-through (η=1, no Fresnel) — the mode parameter the subsurface spec already reserved. The
existing walk, HG phase, per-channel throughput, NEE (distant+env), and RR are reused verbatim.
This is precisely the additive contract: a new `kind` + two `case` bodies.

Free-standing routing: `Material "interface"` imports as a material whose `parameter_overrides`
carry `volume_*` keys; `renderer.py` routes it to `MATERIAL_TYPE_SUBSURFACE`'s walk path with
`mediumKind = MEDIUM_NANOVDB` and pass-through boundary (a new `_material_is_volume` predicate
beside `_material_is_subsurface`, which currently only matches `subsurface_*` keys). Megakernel and
wavefront both go through the shared `evaluateBounce`/medium resolve — one case, both modes, same
as the subsurface phase-4 precedent.

### D5 — Metal dispatch hygiene: bounded dispatches + guaranteed teardown + kill harness

Three layers:

1. **Bounded work per command buffer.** No single Metal command buffer may exceed a watchdog-safe
   budget. The megakernel already tiles for SSS (`MAX_VOLUME_STEPS=16` under `SKINNY_METAL`); the
   volume walk gets the same treatment (capped steps/bounces per dispatch under `SKINNY_METAL`,
   loop continued across accumulation frames — biased caps only where the subsurface precedent
   already accepts them, re-tuned so parity gates pass). Prefer wavefront for heavy volume renders
   (documented; wavefront's staged kernels are naturally short).
2. **Guaranteed teardown on every exit path.** `MetalContext` gains context-manager support and an
   `atexit` + SIGINT/SIGTERM hook that calls the existing `destroy()` (`wait_for_idle` +
   `device.close()`), unregistered on clean shutdown. Renderer/headless entry points
   (`render_headless`, tests, `app.py`) acquire the context through this scope, so an exception or
   Ctrl-C between dispatches never leaks an open device with queued work.
3. **Kill harness.** New `tests/test_metal_cleanup.py` (gpu-marked): (a) subprocess renders then
   exits cleanly → device in a fresh subprocess constructs and dispatches within a time budget;
   (b) subprocess is SIGKILLed mid-render → same probe must still pass, proving queued work dies
   with the process (bounded command buffers are what makes this true — the OS reclaims the queue
   at process exit; what wedges the GPU is a single over-long kernel, which layer 1 forbids);
   (c) leak probe asserts `destroy()` ran via the atexit path on normal interpreter exit.

- *Why not try to cancel in-flight work from outside:* macOS gives no supported way to kill another
  process's GPU work; the only robust fix is never enqueueing unbounded work in the first place,
  which is what layer 1 enforces and layer 3 verifies.
- Honors the standing thermal rule: one guarded Metal-compile process at a time; harness runs
  under `guarded_metal.sh` conventions and is `-m gpu`.

### D6 — Parity gates: add both scenes to the corpus, Path-first

`tests/pbrt/corpus/manifest.json` gains `disney_cloud` and `bunny_cloud` (pbrt `file` assets);
refs regenerated via `regen_refs.py` against the pinned pbrt v4 (`--outfile`, spp tuned for volume
noise). Dual gate as always: pbrt-truth (expect an initial `baseline` around the RGB-vs-spectral
floor, recorded not hidden) + self-consistency vs the `(Path, wavefront)` anchor. `combo_is_valid`
excludes BDPT/SPPM × volume scenes (recorded exclusion, follow-up change). Full-res disney cloud
render time is bounded by rendering the gate at 256² like the rest of the corpus.

## Risks / Trade-offs

- [NanoVDB layout drift across file versions (32.x vs 33.x headers)] → reader checks magic +
  version and fails loudly with the version found; the two target files pin what we must support;
  unit tests fixture tiny synthetic grids we author ourselves.
- [wdas quarter grid memory (~200 MB texture + transient dense numpy)] → decode streams
  leaf-by-leaf into the preallocated dense array; ZERO-SWAP rule respected (gpu tests stay out of
  default `pytest`, guarded runner budgets RAM); if the quarter cloud still stresses CI, gate on a
  downsampled copy and keep full-res as a manual scene.
- [Global majorant on a sparse cloud → many null collisions → slow convergence] → correctness
  first (unbiased regardless); noted follow-up for macrocell majorants; gate spp chosen for the
  variance we actually get.
- [Metal watchdog vs long volume marches in the megakernel] → capped-steps tiling (D5.1), continue
  across frames; wavefront documented as the preferred mode for volumes (same guidance as the
  28.8M-tri dragon).
- [Mirrored camera in disney-cloud (`Scale -1 1 1`)] → already handled by
  `pbrt-mirrored-camera-flip` (`CameraOverride.mirrored`); parity gate will catch a regression.
- [Interface-material routing accidentally capturing genuine cutout/glass materials] → predicate
  keys off `volume_*` overrides + pbrt `interface` material type only; existing corpus (glass,
  subsurface) is the regression net.
- [Bunny-cloud film sensor (`nikon_d850`, iso, whitebalance) not modeled] → imported as `approx`
  with the existing film-exposure fold; absolute-radiance gate uses the recorded baseline mechanism
  if the sensor response shifts brightness.
- [atexit/signal handlers interacting with GLFW/slangpy teardown order] → handlers only drain +
  close the device if not already destroyed (idempotent `destroy()`), never touch window state;
  covered by the clean-exit probe in the harness.

## Open Questions

- Does slang-rhi's Metal path expose 3D texture upload through the existing `SampledImage` seam
  without new API? (Expected yes — `texture_3d` type exists; verify early, it gates D3.)
- **RESOLVED (task 1.1 recon, 2026-07-02):** both target files' headers were parsed byte-exact
  (recon script consumed every byte of both files). Findings:
  - Both files are NanoVDB **32.3.3** (pbrt v4's pinned ABI; file `Version` = major<<21 |
    minor<<10 | patch), file magic `0x304244566f6e614e` ("NanoVDB0"), **1 grid** each, codec
    **ZIP** (zlib stream with a uint64 compressed-size prefix per grid blob — single frame per
    grid, uncompressed size = MetaData.gridSize; confirms the D1 zlib assumption).
  - `wdas_cloud_quarter.nvdb`: grid `density`, **Float / FogVolume**, index bbox
    min (−261, −81, −358) max (236, 256, 254) → dims **498×338×613** (103.2 M voxels →
    **394 MiB fp32 dense / ~197 MiB R16F**, confirming the D3 sizing), voxel size 0.8333…,
    Map = uniform scale 0.8333… with zero translation, background 0, value range **[0, 1]**,
    24.1 M active voxels, nodes (leaf/lower/upper/root) 41312/52/8/1, 16309 active 8³ value
    tiles (tile broadcast is required for a correct dense decode, not just leaf voxels).
  - `bunny_cloud.nvdb`: grid `density`, **Float / FogVolume**, index bbox min (−300, −47, −208)
    max (276, 524, 229) → dims **577×572×438** (144.6 M voxels → 551 MiB fp32 / ~276 MiB R16F),
    voxel size 0.08, Map = uniform scale 0.08 with zero translation, background 0, value range
    **[0, 2.792]** (⇒ D3's normalize-by-max before R16F upload matters here), 19.2 M active
    voxels, nodes 66212/73/8/1, 2 active value tiles.
  - File layout (v32.x `util/IO.h`): per-segment `Header{u64 magic, u32 version, u16 gridCount,
    u16 codec}` then per-grid `MetaData` (176 B: gridSize, fileSize, nameKey, voxelCount,
    gridType, gridClass, worldBBox, **indexBBox**, voxelSize, nameSize, nodeCount[4],
    tileCount[3], codec, version) + gridName, then the grid blobs. Note this differs from the
    later (v33+) layout floating around in docs — the pinned headers are authoritative.
  - Grid blob: `GridData` 672 B (magic/checksum/version/flags/gridIndex/gridCount/gridSize,
    gridName[256], `Map` 264 B with row-major double 3×3 `mMatD` + translation `mVecD` applying
    as world = M·ijk + t, worldBBox, voxelSize, gridClass, gridType, blind-meta offset/count),
    `TreeData` 64 B (node byte-offsets **relative to the tree**, ordered leaf/lower/upper/root),
    root (`RootData` 64 B + 32 B tiles), upper `InternalData` 270 400 B (32³ children of 128³),
    lower 33 856 B (16³ children of 8³), leaves 2 144 B (8³ fp32 + 64 B value mask). All node
    offsets verified against both files (exact byte accounting to EOF).
