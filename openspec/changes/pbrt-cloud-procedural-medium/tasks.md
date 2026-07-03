# Tasks — pbrt-cloud-procedural-medium

Worktree change (per CLAUDE.md). Builds on `nanovdb-volume-rendering` (living specs
`heterogeneous-media`, `pbrt-volume-import`). GPU: Metal, one guarded process at a time, gpu tests
out of default `pytest`. Every render shown to the user. slangc needs
`-I src/skinny/shaders -I src/skinny/mtlx/genslang`.

## 1. Recon + noise port (hostless)

- [x] 1.1 Read the exact 240 B `FlatMaterialParams` layout (`common.slang` + `renderer.py`
      `pack_flat_material`); identify free lanes for `cloud_density`/`wispiness`/`frequency` or
      decide to append one float4 (record in design Open Questions)
- [x] 1.2 `src/skinny/shaders/materials/subsurface/cloud_noise.slang`: port pbrt `NoisePerm[512]`,
      `Grad`, quintic `NoiseWeight`, `pbrtNoise(float3)`, `pbrtDNoise(float3)`,
      `cloudDensity(pLocal, density, wispiness, frequency)` — line-for-line vs pbrt
      `util/noise.cpp` + `media.h:493`
- [x] 1.3 Hostless noise unit test: numpy mirror of the SAME constants (perm table + algorithm),
      assert Slang-intended math matches at a grid of points (compile the tiny kernel via slangc or
      mirror purely in numpy against a captured Slang eval); include the altitude falloff + wispiness

## 2. Importer

- [x] 2.1 `media.py`: `cloud` leaves the unsupported-skip; new `cloud_overrides(medium)` →
      `volume_cloud`=True, `cloud_density`/`cloud_wispiness`/`cloud_frequency`, `volume_sigma_a/_s`,
      `volume_g` (defaults wispiness 1, frequency 5); `rgbgrid` etc still skip
- [x] 2.2 `materials.py`/`api.py`: `Material ""` + a `MediumInterface` → lobe-less null boundary +
      `volume_interface` marker (bring code to the living spec); `Material ""` without a medium
      keeps the default
- [x] 2.3 Importer hostless tests: `clouds.pbrt` imports with zero unsupported skips, sphere binds
      the `c` medium overrides (`cloud_density`=2), `Material ""` is null-boundary not grey;
      empty-material-without-medium keeps default; `rgbgrid` still skips

## 3. Renderer

- [x] 3.1 `renderer.py`: `_material_is_volume` already matches `volume_interface`; extend the pack
      to route `volume_cloud` → `mediumKind=MEDIUM_CLOUD` and pack `density`/`wispiness`/`frequency`
      into the chosen lanes (D1); world→medium-local affine reuses the `worldToUvw` rows (verify
      p.y∈[0,1] over the sphere so the altitude falloff reads); no new descriptor binding
- [x] 3.2 Pack/layout hostless unit test (offsets, identity rows for non-cloud, `MEDIUM_CLOUD`
      tag); MSL stride pin unchanged (no new fields) or updated if a float4 was appended

## 4. Shader transport

- [x] 4.1 `bindings.slang`: `static const uint MEDIUM_CLOUD = 2u`
- [x] 4.2 `materials/subsurface/medium.slang`: `densityAt` `MEDIUM_CLOUD` case → transform p by the
      world→local rows, `cloudDensity(...)`; `mediumMajorant` `MEDIUM_CLOUD` → σ_a+σ_s (packed σ_t);
      `resolveMedium` fills the cloud scalars
- [x] 4.3 Recompile `main_pass.spv` (slangc, both `-I`); guarded Metal in-process compile smoke
- [x] 4.4 GPU sanity (Metal, shown): zero-`density` cloud ≡ no-sphere (relMSE under threshold);
      cloud renders structured (fBm + falloff); homogeneous + nanovdb scenes byte-unchanged

## 5. Target scene + parity gates

- [x] 5.1 Convert `clouds.pbrt` → `.usda`; headless Metal wavefront + megakernel render; show
      labelled render vs pbrt reference at shared tonemap
- [x] 5.2 Corpus: add `clouds` to `tests/pbrt/corpus/manifest.json`; regen ref with pinned pbrt v4
      (`regen_refs.py`, spp tuned)
- [x] 5.3 Dual gate on Metal (pbrt-truth + mega≡wave); record measured baselines honestly; never
      loosen self-consistency; `combo_is_valid` volume exclusions already cover it

## 6. Validation, docs, wrap-up

- [x] 6.1 Full hostless suite (`pytest -m "not gpu"`), ruff, `openspec validate
      pbrt-cloud-procedural-medium`
- [x] 6.2 Docs: `docs/Architecture.md` medium-kind list (`MEDIUM_CLOUD`) + `cloud_noise.slang`
      module note; README + CLAUDE.md compatibility matrix (procedural cloud row); CHANGELOG
- [x] 6.3 Code-review pass on the diff (adversarial — Perlin parity + pack lanes are the risk),
      then merge flow per finishing-a-development-branch; archive after merge
