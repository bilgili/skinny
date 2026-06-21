# RESUME POINTER (new session)

- Worktree: `.claude/worktrees/pbrt-subsurface-unit-scale` (shared across the pbrt
  fixes this session). Main is at origin `300b69f` (units + env + walk-energy all
  merged + pushed).
- **Prototype lives on branch `worktree-pbrt-subsurface-3d-walk`** (WIP commit):
  `subsurface_walk.slang` has `import scene_trace;` + `subsurfaceRadiance3D(...)`;
  `path.slang evaluateBounce` MATERIAL_TYPE_SUBSURFACE case calls it (passing
  `fc`). Metal `SSS_MAX_BOUNCES` left at **16** (test value — RE-TUNE). The merged
  slab `subsurfaceRadiance` is intact in the same file (revert = swap the dispatch
  call back).
- Env for headless renders: `export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS`
  then separate `export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib`; run `./bin/python3.13`
  with `sys.path.insert(0, worktree/src)`. Furnace harnesses: `/tmp/furnace_eta.py`,
  `/tmp/furnace_tau.py`. Dragon: `/tmp/dragon_3d.py`. WAVEFRONT only (megakernel OOMs
  the 28.8M dragon + watchdog). pbrt ref `/tmp/dragon_ref.npy` (mean 0.211).

## 1. Execution-mode gate

- [x] 1.1 Branch the SUBSURFACE dispatch in `path.slang evaluateBounce`: wavefront →
  `subsurfaceRadiance3D`, megakernel → `subsurfaceRadiance` (slab). Use the
  existing wavefront/megakernel define (check how other shaders distinguish, e.g.
  `SKINNY_METAL_RECORDS` / a wavefront build define) or thread an execution-mode
  flag through `FrameConstants`.
- [x] 1.2 Confirm the megakernel path still compiles + renders the slab (no 3D
  trace in the single-dispatch megakernel).

## 2. Distant-light NEE in 3D

- [x] 2.1 At each real scatter in `subsurfaceRadiance3D`, shadow-trace from the
  scatter vertex toward `lightDir` via `traceScene` to the boundary; attenuate by
  `exp(-σ_t · dist)` over the in-medium distance × HG phase × exit Fresnel × the
  distant-light radiance. Restore the `lightDir`/`lightRadiance` params the
  prototype dropped.
- [x] 2.2 Add a distant-lit subsurface gate scene (sphere + distant light) and
  confirm NEE is non-zero and unbiased (matches a brute-force long-walk reference).

## 3. Exit refraction + cap

- [x] 3.1 Refract the escape direction (Snell, `refractInto` inverse) for the env
  lookup instead of the raw interior direction.
- [x] 3.2 Sweep `SSS_MAX_BOUNCES` (16 / 32 / 48) on the dragon (wavefront) for the
  best energy within the watchdog; pick the final value, document the headroom.

## 4. Gates

- [x] 4.1 Furnace energy across τ (1 / 20 / 200) and η (1.0 / 1.5) → ~unity, η-
  independent (prototype already ~0.94/0.83 at cap 16 — should improve with the
  final cap).
- [x] 4.2 PT≡BDPT relMSE on a subsurface sphere (BDPT excludes subsurface → both
  run PathTracer; must still match).
- [x] 4.3 Metal↔Vulkan wavefront parity on a subsurface sphere.
- [x] 4.4 Non-subsurface corpus unchanged (glass stays flat; pbrt parity corpus).
- [x] 4.5 sssdragon A/B vs pbrt (mean + exposure-matched relMSE + hue): closer than
  the slab; capture pbrt | slab | 3D side-by-side. Watchdog: full-res dragon
  renders without a hang.

## 5. Wire-up + docs + OpenSpec

- [x] 5.1 Update `docs/Subsurface.md` (3D walk for wavefront, slab for megakernel,
  the path-length rationale).
- [x] 5.2 `openspec validate pbrt-subsurface-3d-walk --strict`; ruff clean.
- [ ] 5.3 Commit → archive → merge → push (worktree `.gitignore` is `*` → `git add -f`;
  merge from the primary checkout `git -C /Users/ahmetbilgili/projects/skinny`).
