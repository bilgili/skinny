# Tasks — nanovdb-volume-rendering

Worktree change (per CLAUDE.md). GPU work: Metal backend, one guarded Metal process at a time,
gpu tests never in default `pytest` (ZERO-SWAP rule). Every render produced for verification is
shown to the user (SendUserFile + clickable path).

## 1. Recon + Metal teardown foundation (unblocks everything GPU)

> 1.3–1.5 were extracted into the standalone `metal-dispatch-hygiene` change and MERGED to main
> (`c4159bb`, archived to `openspec/specs/metal-dispatch-hygiene/`, CLAUDE.md standing rule).
> Kill harness GPU-verified: 3/3 gpu + 13 hostless green. This change consumes the capability;
> the volume-march watchdog scenario stays here as task 6.5.

- [x] 1.1 Read both `.nvdb` headers (magic, version, grid class, codec, dims, value range,
      index→world map) with a throwaway script; confirm D1 (ZIP/NONE, FloatGrid) and D3 (R16F
      size) assumptions; record findings in design.md Open Questions
- [x] 1.2 Verify slang-rhi Metal + Vulkan expose 3D texture create/upload/sample through the
      existing `SampledImage` seam (tiny probe kernel both backends); note any API gap
      — PROBE PASS both backends: `TextureType.texture_3d` + `Format.r16_float`,
      `copy_from_numpy` accepts `(D,H,W)` float16, `Texture3D.SampleLevel` exact at voxel
      centers (max err 0.0). No API gap; 4.2 is a straight `SampledImage3D` sibling.
- [x] 1.3 `MetalContext`: idempotent `destroy()`, context-manager protocol, `atexit` +
      SIGINT/SIGTERM teardown hooks (unregister on clean close); wire headless entry points +
      `app.py` through the scope
- [x] 1.4 Unit tests for teardown paths (no GPU where possible: hook registration/idempotency;
      gpu-marked: exception-between-dispatches drains + closes)
- [x] 1.5 Kill harness `tests/test_metal_cleanup.py` (gpu): clean-exit probe, SIGKILL-mid-render →
      fresh-subprocess probe dispatch within time budget

## 2. NanoVDB reader (pure Python, hostless)

- [x] 2.1 `src/skinny/pbrt/nanovdb.py`: header + grid metadata parse (magic, version, grid class,
      codec), fail-loud errors for unsupported class/codec/version
- [x] 2.2 Tree decode (root → upper 32³ → lower 16³ → leaf 8³) for FloatGrid → dense
      `numpy.float32` + index→world transform + min/max; ZIP (zlib) codec support
- [x] 2.3 Unit tests against tiny authored synthetic grids (known voxels round-trip) + both target
      files decode (dims/value-range sanity, no full-array golden)

## 3. pbrt import → USD volume

- [x] 3.1 `media.py`: `nanovdb` leaves `_HETEROGENEOUS`-skip path; heterogeneous overrides
      (`volume_sigma_a/_s`, `volume_g`, `volume_scale`, `pbrt_medium`, grid asset path)
- [x] 3.2 `materials.py`/`api.py`: `Material "interface"` → null-boundary material carrying the
      `MediumInterface` interior medium overrides (no diffuse/dielectric fallback);
      `_resolve_medium` returns heterogeneous overrides
- [x] 3.3 `emit.py`: `UsdVol.Volume` + `OpenVDBAsset` field (filePath → `.nvdb`, fieldName
      `density`), medium CTM from the transform stack (respect `B=diag(1,1,-1)` axis convention)
- [x] 3.4 Importer unit tests: disney-cloud + bunny-cloud convert with zero unsupported skips
      (approx allowed: bunny film sensor); volume prim, transform, overrides, interface binding
      asserted; `rgbgrid` still skips

- [x] 3.5 pbrt `Shape "disk"` import (both target scenes' ground): tessellate disk (radius,
      height, innerradius, phimax) to a triangle mesh at import, same path as other shapes;
      unit tests + both scenes re-checked for zero skips (surfaced by 3.4 — pre-existing gap)

## 4. Loader + renderer plumbing

- [x] 4.1 `usd_loader.py`: ingest `UsdVol.Volume` (resolve `.nvdb` asset → nanovdb reader →
      density array + transform + majorant density), hand to renderer scene state
- [x] 4.2 `metal_compute.py` + `vk_compute.py`: 3D `SampledImage` variant (R16F upload, trilinear
      sampler)
- [x] 4.3 `renderer.py`: density-grid upload (normalize to [0,1], fold max into majorant), new
      descriptor binding (next free slot; both backends; watch Metal 31-slot table),
      `_material_is_volume` predicate routing interface+`volume_*` materials to the walk with
      `mediumKind=MEDIUM_NANOVDB` + pass-through boundary; grid fields packed into the inline
      `MediumParams` (world→index 3×4 + grid slot + majorant)
- [x] 4.4 Accumulation-reset coverage: volume state in `_current_state_hash()`; pack/stride unit
      tests (MSL stride check under guarded Metal — gpu-marked test WRITTEN in
      `tests/test_metal_flat_material_layout.py`; its guarded-Metal run is still pending)

## 5. Shader transport (the two case bodies)

- [x] 5.1 `bindings.slang`: density-grid binding + `MEDIUM_NANOVDB` un-reserved; `common.slang`
      `MediumParams` grid fields (replacing reserved `gridHandle`)
- [x] 5.2 `materials/subsurface/medium.slang`: `densityAt` trilinear grid sample (index space) +
      `mediumMajorant` global-majorant case; index-matched boundary mode for free-standing media
- [x] 5.3 Megakernel + wavefront routing: interface material hit → medium walk (shared
      `evaluateBounce` case, mirroring subsurface phase-4); `SKINNY_METAL` watchdog caps for the
      volume march (D5.1), continued across accumulation frames
- [x] 5.4 Recompile `main_pass.spv` (slangc, Vulkan byte-path); Metal in-process compile smoke
      under guarded runner
- [x] 5.5 GPU sanity gates: constant-density grid ≡ homogeneous medium (relMSE under
      self-consistency threshold); zero-density interface sphere ≡ no-sphere scene; homogeneous
      subsurface scenes byte-unchanged (same seed) — render and show images

> 5.4/5.5 verified on GPU (Metal): slangc spv recompiled (needs `-I src/skinny/mtlx/genslang`,
> CLAUDE.md command update pending in 7.2); Metal in-process megakernel compile smoke OK; MSL
> stride 240 pinned. Sanity gates: const-grid ≡ homogeneous relMSE 4.1e-7; zero-grid sphere ≡
> no-sphere relMSE 9.1e-4 (ground visible through volume = escape continuation);
> subsurface_infinite relMSE IDENTICAL to HEAD to the last digit (byte-unchanged).
> ROOT-CAUSED + FIXED during 5.5: adding the volumeDensity texture made EVERY Metal pipeline
> fail silently (all-black) — the 128-texture argument table was exactly full (120 pool + 5
> discrete + output/accum/hud). Fix: bindless pool trimmed 120→119 on Metal
> (bindings.slang + metal_compute.BINDLESS_TEXTURE_CAPACITY).

## 6. Target scenes end-to-end + parity gates

- [x] 6.1 Convert both scenes to `.usda`; headless Metal wavefront render each (bounded res/spp);
      show labelled renders vs pbrt references side-by-side at shared tonemap
- [x] 6.2 Corpus: add `disney_cloud` + `bunny_cloud` to `tests/pbrt/corpus/manifest.json`;
      regenerate refs with pinned pbrt v4 (`regen_refs.py --outfile`, spp tuned for volume noise)
- [x] 6.3 `parity.py`: `combo_is_valid` exclusions for BDPT/SPPM × volume scenes (recorded, not
      silent); coverage meta-test green
- [x] 6.4 Run dual gate (pbrt-truth + mega≡wave self-consistency) on Metal for both scenes;
      record measured baselines honestly (RGB-vs-spectral floor expected); never loosen
      self-consistency
- [x] 6.5 Watchdog soak: full gate-length disney-cloud accumulation on Metal megakernel + wavefront
      completes with no GPU fault (Requirement: watchdog-bounded)

> G6 DONE (GPU-verified Metal, both scenes 256²/256spp):
> - **disney_cloud**: CLEAN dual gate — mega≡wave EXACT (relMSE 0.0), pbrt-truth relMSE 0.077 /
>   mean 0.87× (RGB-vs-spectral + scatter-cap floor).
> - **bunny_cloud**: CLEAN dual gate — mega≡wave EXACT (relMSE 0.0), pbrt-truth relMSE 0.117 /
>   mean 0.735× (bounded multi-scatter cap 64 vs pbrt maxdepth 50 + RGB-vs-spectral; recorded).
> - `combo_is_valid`: BDPT/SPPM × volume + ReSTIR×volume excluded (recorded), coverage meta green.
> - 6.5 watchdog soak: all full-gate Metal accumulations (mega+wave, 206–289 MB grids) completed
>   with NO GPU fault under guarded_metal.sh.
> - ROOT-CAUSED bugs fixed en route: (1) 128-texture argument table full → volumeDensity black
>   (pool 120→119); (2) null-collision double-attenuation → analog (1−pReal)/(1−pRealMax);
>   (3) `direct_light_index=1` zeroed AUTHORED sun → keep 0 when scene authors a distant light;
>   (4) unbounded step budget → invisible cloud (VOLUME_MAX_STEPS + uvw-slab clip); (5) RR
>   forced-death on σ_a=0 albedo≈1 → pbrt survival RR; (6) volume pass-through delta continuation
>   took env at full weight → preserveMisState; (7) bunny env `Rotate 10` dropped → light-CTM
>   baked into equiarea reprojection; (8, code-review) uninitialized `preserveMisState` in
>   `evaluateFlatBounce` → flat wavefront read garbage, diverged from megakernel (this was the
>   TRUE cause of the bunny mega≠wave + fireflies I'd first mis-attributed to a pre-existing
>   env difference — fixing it made BOTH scenes mega≡wave EXACT and killed the fireflies).

## 7. Validation, docs, wrap-up

- [x] 7.1 Full hostless suite (`pytest -m "not gpu"`), ruff, `openspec validate
      nanovdb-volume-rendering`
- [x] 7.2 Docs: `docs/Architecture.md` binding map + volume module notes; README + CLAUDE.md
      compatibility matrix (volume × integrator/backend rows); CHANGELOG; `docs/Wavefront.md` /
      `docs/Megakernel.md` where the march caps land
- [x] 7.3 Final review pass (`/code-review` on the branch), then merge flow per
      finishing-a-development-branch; archive change after merge
