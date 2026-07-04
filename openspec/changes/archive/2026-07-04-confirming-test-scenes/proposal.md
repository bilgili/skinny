# Confirming Test Scenes

## Why

The standing regression corpus (`tests/pbrt/corpus/`, 6 pbrt scenes) plus a handful of heavy
assets (bathroom, dragon) is how renderer regressions get caught today. There is no systematic
set of *minimal, discriminating* scenes per renderer axis — materials, integrators, execution
modes, sampling modes — so a regression in one lobe/strategy either hides inside a big scene's
noise or is not exercised at all. Furnace (energy-conservation) closure exists in the renderer
(`furnace_index`, per-material furnace bit) but has **zero automated tests**, and there is no
gate that the MaterialX authoring of a material matches its plain-USD authoring.

## What Changes

- New scene suite under **`tests/assets/`**: tiny, fast-to-render (128×128, low spp,
  few-hundred-triangle) scenes, each designed to *discriminate* one renderer axis:
  - **Materials**: diffuse, conductor (smooth+rough), dielectric/glass, plastic, emissive,
    textured, subsurface — one scene per lobe family, plus scenes using existing
    physically-based materials from `assets/materialxusd/tests/physically_based/`
    (77 measured OpenPBR materials): a selected subset spanning the lobe space
    (e.g. Gold, Copper, Glass, Plastic PC, Skin I, Gray Card, Musou Black) extracted onto a
    simple mini-shaderball prop, plus an opt-in smoke render of 1–2 of the original
    full shaderball scenes as-is.
  - **Integrators**: scenes where path / BDPT / SPPM legitimately differ in convergence but
    must agree in expectation — indirect-dominant box, specular-chain caustic, many-bounce
    color bleed.
  - **Execution modes**: every scene doubles as a megakernel ≡ wavefront self-consistency
    probe (existing anchor mechanism).
  - **Sampling modes**: many-small-lights scene (ReSTIR DI), env-dominant glossy scene
    (bsdf/env proposal mixture + neural proposal), verified unbiased vs the analytic-proposal
    anchor.
- **Dual authoring per scene**: a plain `UsdPreviewSurface` `.usda` **and** a MaterialX
  variant (`.usda` + `.mtlx`), with an equivalence gate between the two where the material is
  expressible in both (skip-with-reason where OpenPBR / standard_surface has no
  UsdPreviewSurface counterpart).
- **pbrt counterparts**: a `.pbrt` file per scene wherever the scene is expressible in pbrt v4
  (diffuse/conductor/dielectric/coateddiffuse/subsurface + area/infinite lights), runnable by
  the user with the pinned pbrt build and consumed by the existing pbrt-truth gate via
  checked-in reference EXRs.
- **Furnace closure gates**: automated furnace tests — constant-white environment, lossless
  material ⇒ accumulated image ≡ 1.0 within tolerance — swept over materials × integrators ×
  execution modes (valid combos only), both global furnace mode and per-material furnace.
- **Regression sweep checkpoint**: each implementation stage ends with a hostless run
  (automatic) and an optional GPU sweep (megakernel + wavefront over the new scenes plus the
  existing corpus) that is **only run after asking the user**; results reported as
  before/after metric deltas so regressions are visible per stage.

## Capabilities

### New Capabilities
- `confirming-scene-suite`: the minimal per-axis scene corpus under `tests/assets/` — scene
  inventory, dual USD/MaterialX authoring, pbrt counterparts, size/speed budget, and the
  equivalence gates between authoring variants.
- `furnace-closure`: furnace-mode closure testing — global and per-material furnace gates
  across the material × integrator × execution-mode matrix.

### Modified Capabilities
- `render-parity-matrix`: corpus grows to accept suite scenes living under `tests/assets/`
  (today scenes sit in `tests/pbrt/corpus/`); adds the MaterialX-vs-plain-USD equivalence
  gate as a third gate class beside pbrt-truth and self-consistency; coverage meta-test
  extends to the new scene axes.

## Impact

- **New files**: `tests/assets/**` (usda + mtlx + pbrt scene sources, README),
  `tests/pbrt/corpus/refs/*.exr` (new pbrt refs), new pytest modules (furnace gates, suite
  gates, hostless scene-validity tests).
- **Modified**: `tests/pbrt/corpus/manifest.json` (new scene entries),
  `src/skinny/pbrt/parity.py` (scene axis / gate wiring), `tests/pbrt/test_matrix.py`
  coverage meta-tests, `tests/pbrt/regen_refs.py` (path handling for `tests/assets/`).
- **Docs**: `docs/Architecture.md` (Parity Matrix Harness section), `CLAUDE.md`
  (compatibility-matrix pointer to the suite), `README.md` if any user-facing flag appears.
- **No renderer/shader behavior changes** — this change only adds scenes, gates, and
  harness wiring; any real divergence it exposes becomes its own follow-up change.
- **GPU cost**: all GPU sweeps follow metal-dispatch-hygiene; scenes budgeted to render in
  seconds each; GPU runs gated on explicit user approval at each checkpoint.
