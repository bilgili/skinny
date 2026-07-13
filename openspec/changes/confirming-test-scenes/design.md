# Design — confirming-test-scenes

## Context

Current regression coverage: `tests/pbrt/corpus/` has 6 pbrt scenes (diffuse/mirror/conductor/
glass/subsurface/texture) swept by the parity matrix (`src/skinny/pbrt/parity.py` —
`INTEGRATORS = path|bdpt|sppm`, `EXECUTION_MODES = megakernel|wavefront`, proposal axis
`neural`, reuse axis `restir-di`) under two gates: pbrt-truth (checked-in EXR refs) and
self-consistency (anchor = `(path, wavefront)`). Heavy scenes (bathroom, dragon) cover
integration but are slow and noisy — bad at *localizing* a regression.

Furnace plumbing already exists in the renderer and is untested:
- global `renderer.furnace_index` → `_furnace_environment()` swaps in a constant-white env
  and disables lights (`renderer.py:2606`, `furnaceMode` uniform at `renderer.py:87`);
- per-material furnace flag, bit 10 of the material flag word
  (`toggle_material_furnace`, `renderer.py:6134`).

Available material assets: `assets/materialxusd/tests/physically_based/` — 77 measured
physically-based OpenPBR materials (`*_OPBR_MAT_PBM.usda`: metals Gold/Copper/Silver/…,
dielectrics Glass/Diamond/Water, plastics, six Skin presets, radiometric extremes
Gray_Card/Musou_Black/MIT_Black), each file embedding the full 71-material library plus a
shaderball test scene that references heavy shared data (`../data/shaderball.usd` 5.8 MB,
`san_giuseppe_bridge.hdr` 5.7 MB); `stdsurf_materials/` holds standard_surface versions.
Also `assets/Usd-Mtlx-Example/materials/*.mtlx` (untextured standard_surface) and heavy
ASWF assets in `assets/assets-main/` (not used here; wrong size class).

Host constraints: Metal backend always (`select_backend()`), headless runs via
`./bin/python3.13` with `VULKAN_SDK`/`DYLD_LIBRARY_PATH` exported, metal-dispatch-hygiene
for every GPU dispatch, one guarded Metal process at a time. Repo `.gitignore` is `*`
(new files need `git add -f`). Worktree gotcha: `assets/` and venvs exist only in the
main checkout — the suite must not depend on `assets/` at test time.

## Goals / Non-Goals

**Goals:**
- Minimal, discriminating scene per renderer axis (materials, integrators, execution
  modes, sampling modes), all under `tests/assets/`, each rendering in seconds.
- Every scene dual-authored (plain `UsdPreviewSurface` `.usda` + MaterialX variant) with an
  equivalence gate; every pbrt-expressible scene gets a runnable `.pbrt` counterpart plus a
  checked-in reference EXR.
- Automated furnace-closure gates (global + per-material) over the valid
  material × integrator × execution-mode matrix.
- A per-stage regression checkpoint: hostless tests run automatically; GPU sweeps run only
  after asking the user; results reported as before/after metric deltas.

**Non-Goals:**
- No renderer/shader behavior changes. A divergence exposed by a new gate is recorded as a
  baseline (or xfail with a tracking note) and fixed in its own follow-up change.
- No skin-pipeline scenes (head assets are huge; skin has its own estimator-chain tests).
- No heavy ASWF assets (StandardShaderBall, OpenChessSet) — wrong size class for this suite.
- No new CLI flags or GUI work.

## Decisions

### D1 — Scene inventory (fixed list, one discriminator per axis)

| Scene | Axis | Contents | pbrt? |
|---|---|---|---|
| `mat_diffuse` | materials | albedo-0.5 sphere on plane, area light | ✅ `diffuse` |
| `mat_conductor` | materials | smooth + rough (0.3) metal spheres, IBL const env | ✅ `conductor` |
| `mat_dielectric` | materials | clear glass sphere over checker plane (transmission + caustic) | ✅ `dielectric` |
| `mat_plastic` | materials | coated diffuse sphere, area light | ✅ `coateddiffuse` |
| `mat_emissive` | materials | emissive quad lighting a diffuse box interior | ✅ `arealight` |
| `mat_textured` | materials | image-textured quad (reuse `texture_uv.png` pattern) | ✅ `imagemap` |
| `mat_subsurface` | materials | small SSS blob, area light | ✅ `subsurface` |
| `mat_pbr_gold`, `mat_pbr_copper`, `mat_pbr_glass`, `mat_pbr_plastic_pc`, `mat_pbr_skin1`, `mat_pbr_graycard`, `mat_pbr_musou_black` | materials (PBR from assets) | mini-shaderball (sphere on plane) with the named OpenPBR material extracted from `assets/materialxusd/tests/physically_based/` | ⛔ MaterialX-only |
| `pbr_shaderball_smoke` | materials (PBR from assets) | 1–2 original full shaderball scenes rendered as-is from `assets/materialxusd/` — opt-in smoke test, skip-if-missing | ⛔ |
| `int_indirect_box` | integrators | Cornell-style box, light hidden from camera → indirect-dominant (BDPT advantage) | ✅ |
| `int_caustic` | integrators | glass sphere focusing area light onto floor (SPPM advantage) | ✅ |
| `int_bleed` | integrators | two-wall color bleed, multi-bounce | ✅ |
| `samp_many_lights` | sampling (ReSTIR DI) | 16 small emissive quads over glossy floor | ✅ |
| `samp_env_glossy` | sampling (proposals/neural) | rough conductor under high-contrast env, no analytic lights | ✅ |

Execution modes need no dedicated scenes — every scene above is swept mega≡wave by the
existing self-consistency anchor.

Budget: 128×128, ≤ 1k triangles, per-scene spp recorded in the manifest (start 256 like the
existing corpus, lower where converged). All geometry is USD prims / tiny inline meshes —
no external model files.

### D2 — Directory layout and manifest registration

```
tests/assets/suite/<scene>/
  <scene>.usda        # plain UsdPreviewSurface authoring
  <scene>_mtlx.usda   # MaterialX authoring (references <scene>.mtlx)
  <scene>.mtlx
  <scene>.pbrt        # when pbrt-expressible; user-runnable with pinned pbrt v4
```

Registration stays in the **single existing manifest** `tests/pbrt/corpus/manifest.json`
(it already supports `usd` asset entries); refs stay in `tests/pbrt/corpus/refs/`. One
manifest, one refs dir, one gate implementation — no parallel harness.
*Alternative rejected*: a second manifest under `tests/assets/` — would fork the gate code
and the coverage meta-tests.

### D3 — Self-contained material extraction, heavy payloads stay out

The selected OpenPBR materials (Gold, Copper, Glass, Plastic PC, Skin I, Gray Card,
Musou Black — chosen to span conductor / dielectric / coated / subsurface lobes plus the
radiometric extremes) are **extracted** as single Material prims (~40-line untextured
parameter blocks) from `assets/materialxusd/tests/physically_based/*_OPBR_MAT_PBM.usda`
into self-contained mini-shaderball scenes under `tests/assets/suite/` — the source files
embed a 71-material library plus references to a 5.8 MB shaderball and 5.7 MB HDR, all of
which stay out of `tests/assets/` (and `assets/` does not exist in worktrees). Each
extracted scene records its source file for provenance.
*Alternatives rejected*: referencing `assets/` at test time — breaks every worktree run;
copying whole scene files — 8k lines + heavy payloads per material.

Additionally an **opt-in smoke test** (`pbr_shaderball_smoke`, gpu-marked, skip-if-missing)
renders 1–2 of the original shaderball scenes as-is from `assets/materialxusd/` in the main
checkout, so the untouched upstream authoring is exercised end-to-end too.

### D4 — Equivalence gate (plain-USD vs MaterialX variant)

New third gate class beside pbrt-truth and self-consistency: render `<scene>.usda` and
`<scene>_mtlx.usda` with the anchor combo and compare with the standard
`metrics.compute_metrics` battery under a per-scene tolerance (same shape as
self-consistency; codegen paths differ, so bit-equality is not expected). Where a material
is not expressible in `UsdPreviewSurface` (the OpenPBR PBR-material scenes), the plain
variant is a recorded skip with reason — mirroring how `combo_is_valid` records exclusions.

### D5 — pbrt counterparts and refs

Material mapping: diffuse→`diffuse`, metal→`conductor`, glass→`dielectric`,
plastic→`coateddiffuse`, SSS→`subsurface`, emissive→`arealight`, texture→`imagemap` on
`diffuse`. Refs regenerated via the existing `tests/pbrt/regen_refs.py` against the pinned
pbrt (`~/projects/pbrt-v4/build/pbrt`); `.pbrt` files are written so the user can run pbrt
manually on them unmodified. MaterialX-only scenes (OpenPBR / standard_surface with no
UsdPreviewSurface or pbrt counterpart) carry no pbrt gate (recorded
skip), and are still covered by self-consistency + equivalence.

### D6 — Furnace-closure gate

- **Global**: headless render with `renderer.furnace_index = 1` and a lossless material
  (Lambert albedo 1.0; smooth conductor reflectance 1.0; clear glass). Read the **linear
  accumulation image** (not tonemapped output); assert per-pixel mean ≈ 1.0 within
  tolerance (target ~1e-2 relative; final value measured during implementation and pinned).
- **Per-material**: two-sphere scene, furnace bit enabled on one material only; assert the
  furnace-flagged object closes to 1.0 while the other renders normally.
- **Matrix**: sweep material-variant × integrator × execution-mode through `combo_is_valid`
  so exclusions are recorded, not silent (SPPM validity for a pure-env furnace resolved at
  implementation time via the validity table).
- **Expected true failures**: rough conductor without multiple-scattering energy
  compensation legitimately darkens. That is a *finding*, recorded as a numeric baseline
  (`furnace_closure` value in the manifest entry), never a loosened tolerance — identical
  policy to pbrt-truth baselines.

### D7 — Regression checkpoint protocol (per implementation stage)

1. Hostless tier runs automatically (`pytest -m "not gpu"` for matrix/metrics/scene-import
   plus new hostless suite tests).
2. GPU sweep is **proposed, never auto-run**: the stage ends with an explicit ask
   ("run GPU sweep now?") listing exact commands and estimated wall time.
3. When a sweep runs, per-scene metrics are written to
   `openspec/changes/confirming-test-scenes/results/stage-<n>.json` and diffed against the
   previous stage; the report is a before/after delta table (regressions bolded), with
   rendered images shown.

### D8 — Hostless validity tier

Every scene gets hostless tests (no GPU): USD opens and contains expected prims/materials;
mtlx variant parses; `.pbrt` counterpart imports through the pbrt importer; manifest entry
schema-valid; coverage meta-test extended so a suite scene missing a gate class (pbrt-truth
/ equivalence / furnace where applicable) **fails the build**.

## Risks / Trade-offs

- [Rough-lobe furnace legitimately fails] → baseline-recording policy (D6); gate still
  catches *changes* in energy behavior.
- [Neural proposal is stochastic → flaky gates] → sampling-mode gates assert unbiasedness
  vs the analytic anchor with statistical tolerance (existing proposal-unbiasedness
  requirement), not image equality; fixed seeds where the harness supports them.
- [mtlx-vs-plain never bit-equal] → tolerance-based equivalence gate (D4); tolerance pinned
  from measured values at implementation, tightened not loosened later.
- [GPU sweep time grows with 15+ new scenes] → 128×128 budget, spp trimmed per scene,
  gpu-marked tests stay out of the default pytest sweep, sweeps user-gated (D7).
- [pbrt RGB-vs-spectral residual on new scenes] → same tolerance discipline as the existing
  corpus (measured + headroom, baselines for known mismatches).
- [SPPM lacks volume transport; subsurface random walk under SPPM may be excluded] →
  resolved through `combo_is_valid`, recorded exclusion — consistent with the existing
  compatibility matrix.

## Migration Plan

Additive only: new scenes/tests land behind the existing harness; no existing scene,
tolerance, or baseline is touched except manifest additions. Rollback = remove the new
manifest entries and test modules. Implementation happens in a dedicated worktree per the
repo workflow; suite files are committed with `git add -f` (`.gitignore` is `*`).

## Open Questions

- Final per-scene spp / tolerances — measured during implementation, pinned in the
  manifest (start from the existing corpus's 256 spp and tighten from measurements).
- SPPM × furnace and SPPM × subsurface validity — decided at the validity table with a
  recorded reason either way.
