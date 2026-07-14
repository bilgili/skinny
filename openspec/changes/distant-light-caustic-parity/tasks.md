## 1. Default-light synthesis policy (host, no shader change)

- [x] 1.1 Add `_scene_authors_lights(scene)` to `renderer.py`: true when the USD
  scene has any **powered** `lights_dir`/`lights_sphere` entry (reuse
  `_has_power` so zero-intensity-only scenes count as unlit), emissive-material
  triangles, or `scene.environment is not None` (authored DomeLight ONLY — the
  built-in HDRI backdrop must not count).
- [x] 1.2 Gate the per-frame distant-light mirror (~line 9764): authored powered
  `lights_dir` → upload those; else authored-light scene → upload zero records;
  else → slider default light (unchanged). `direct_light_index` semantics
  preserved on top. Authority = load-time `_usd_scene` (live scene-graph edits
  don't re-derive in v1 — documented).
- [x] 1.3 Hostless unit tests: predicate truth table (powered/zero-power dir,
  sphere, emissive, authored dome, built-in HDRI only, none) and the mirror
  outcome for each (records uploaded vs zeroed vs slider).
- [x] 1.4 Confirm all four front-ends route through the same mirror (no separate
  synthesis site); grep for other `scene.lights_dir` fallbacks.

## 2. BDPT distant-light subpath walk (3 shader files, ~8 skip sites, 3 seeds)

- [x] 2.1 `bdpt.slang` origin seed (~:247): disk origin over scene bounds
  (`fc.sceneBoundsMin/Extent`, mirror `sppmEmitPhoton`),
  `beta = L·πR²/lightPickPdf`, `pdfPos = lightPickPdf/(πR²)`, `pdfDir = 1`,
  `isDelta = true`; **set `vertex.position` to the disk point** (today never
  assigned) and **`dirOut = -L.direction`** (today `+dir`, which points toward
  the light); walk-ray origin offsets `+N·0.002` with `N = -dir`.
- [x] 2.2 Extend `convertSAtoArea` (~:98) with the **DIR-as-source** case: the
  first walked vertex's `pdfFwd` = parallel-projection area density
  `pdfPos_disk / |N_A·dir|` — no `cos/d²` falloff (the disk distance is a
  placement artifact). pbrt `Pdf_Le`/`InfiniteLightDensity` analog.
- [x] 2.3 Remove the `BDPT_VK_LIGHT_DIR` walk skip in `bdpt.slang` (~:1089) and
  launch `randomWalk` from the disk ray.
- [x] 2.4 **connectT1 distant MIS** (~:693-707): replace the unconditional
  full-weight distant NEE with the `misWeight` partition, mirroring the emissive
  branch (~:776-817): `lit[0]` distant pseudo-vertex (`BDPT_VK_LIGHT_DIR`,
  `pdfFwd = lightPickPdf`, `isDelta = true`), endpoint reverse pdfs,
  `misWeight(eye, s, lit, 1, …)`. Verify the weight degrades to exactly 1.0 when
  no walked partner exists (subpath escaped) so pure-direct distant scenes stay
  byte-consistent.
- [x] 2.5 Replicate seed + skip-removal + connectT1 MIS in
  `bdpt_spectral.slang` (`sampleLightOriginS` ~:190, skip ~:883) with per-λ
  recolor via the authored SPD slot (`distantLightSpd`, mirror the SPPM
  emitter), and in `wavefront/wavefront_bdpt.slang` (`sampleLightOriginS`
  ~:227; six skips ~:937, 1096, 1151, 1180, 1931, 1949).
- [x] 2.6 Wavefront lane budgeting: distant subpaths now walking flips
  `WF_BDPT_SLOT_NEE` → `SLOT_FULL` lanes (gate keys off `lightLen`); verify
  stream-size/heavy-eye tiling bounds still hold under the Metal watchdog rules.
- [x] 2.7 Recompile and check in the `.spv` set (RGB + `-DSKINNY_SPECTRAL`, all
  wavefront BDPT kernels + megakernel); RGB kernels byte-identical under the
  spectral split guard.

## 3. Regression gates (GPU, Metal headless; one guarded process at a time)

- [x] 3.1 Gate A — phantom policy on `assets/glass_caustics_spectral.usda`
  (authored SphereLight only): `numDistantLights == 0`; `bdpt/path` full-image
  ≈ 1.00; SPPM zero local fireflies at default radius; record the honest SPPM
  baselines (caustic-mask ≈ 0.94, shadow-box ≈ 0.78 vs bdpt — env-indirect
  follow-up gap; follow-up must lower, never hide).
- [x] 3.2 Gate B — new confirming-suite scene: **non-dispersive** glass sphere +
  authored DistantLight + diffuse ground (extend `_gen/build.py`; regen refs).
  Compare `bdpt` vs `sppm` on the caustic region at **equal-time or a recorded
  high-spp budget using firefly-robust same-budget region medians** (BDPT
  splats have no spatial reuse — matched-spp equality is unrealistic); record
  the honest number. `path` = recorded per-component exclusion. Distant
  *direct* regions unchanged before/after (the no-double-count check for 2.4).
- [ ] 3.3 Re-run the full parity matrix (bathroom/dragon author distant lights)
  — no combo regresses; distant direct lighting energy unchanged (the 2.4
  degradation-to-1.0 check at scale); update manifests only with recorded
  baselines, never loosened tolerances.
- [ ] 3.4 Megakernel ≡ wavefront BDPT self-consistency green (both now walk
  distant subpaths); spectral bdpt smoke on Gate B's scene.

## 4. Docs + close-out

- [x] 4.1 `docs/Architecture.md`: default-light synthesis policy (+ the
  geometry-only-USD corollary and the live-edit limitation);
  `docs/PhotonMapping.md`: post-mortem note (SPPM speckle was the phantom
  light's caustic — SPPM was correct).
- [x] 4.2 `CHANGELOG.md`: **BREAKING** phantom-sun policy + BDPT distant-caustic
  support.
- [ ] 4.3 `openspec validate distant-light-caustic-parity --strict`; codex (or
  review-subagent fallback) pre-merge; merge from the worktree; archive.
- [ ] 4.4 File follow-up notes: SPPM env-indirect dimness (env photon emission /
  eye-walk continuation), SPPM deposit `vp.beta`, optional disk-radius
  tightening (variance), spectral dispersion on the light-side specular chain.
