# RESUME POINTER (new session)

- Worktree `~/projects/skinny-radiometric` (branch `pbrt-radiometric-parity` off
  main `a1e4234`). Headless GPU rules: Metal backend, `VULKAN_SDK`+
  `DYLD_LIBRARY_PATH` exported even on Metal; `./bin/python3.13` + `sys.path` to the
  worktree `src`; pbrt v4 `~/projects/pbrt-v4/build/pbrt`; scenes
  `~/projects/pbrt-v4-scenes/sssdragon`.
- Diagnosis already done (see proposal): imaging ratio `exposureTime·ISO/100`
  (`api.py:_film_exposure_scale`) is BAKED into emitters/env at import, NOT a live
  knob; `.hdr`-direct env (`lights.py:98-100`) drops `scale`; a separate
  light-type abs offset (area-lit ~1.6× vs pbrt, env-lit metal ~1.0×).
- Key files: `src/skinny/pbrt/api.py` (`_emit_camera`, `_film_exposure_scale`,
  `_emit_shape`, `add_light` call), `src/skinny/pbrt/lights.py`,
  `src/skinny/usd_loader.py` (`_extract_camera`), `src/skinny/renderer.py`
  (`self.exposure` display EV ~1495/8566; CameraOverride), `src/skinny/scene.py`
  (`CameraOverride`), `src/skinny/pbrt/parity.py` (`render_linear`).

## 1. Film params on the scenegraph camera (live, not baked)

- [x] 1.1 `CameraOverride` (scene.py) + `_extract_camera` (usd_loader.py): carry
  `iso` (default 100) + `exposure_time` (default 1.0) from `skinny:film:iso` /
  `skinny:film:exposureTime` (fallback: standard `exposure` attr, folded
  stops→seconds as 2^stops). Pure-Python tests in `test_radiometric_parity.py`.
- [x] 1.2 `api.py _emit_camera`: authors `skinny:film:iso` + `skinny:film:exposureTime`
  + standard `exposure` (= log2(ratio)) on the camera prim. STOPPED baking
  `exposure_scale` into emitters/env (`_emit_shape`, `add_light` passed `1.0`);
  `report.exact` note → "authored on camera (live output scale)". Unit-tested.
- [x] 1.3 Renderer: `FilmParameters` on `self.film`, set by `_apply_camera_override`.
  Imaging ratio applied as a **live linear output scale** — multiplies the
  linear-HDR read (`save_screenshot` EXR/HDR) and folded into the packed display
  exposure (`+log2(ratio)` stops) for the on-screen path (no shader/UBO change).
  iso/exposure_time added to `_current_state_hash` (retune resets accum).
  ⚠ GPU end-to-end verify pending (§4 sweep).
- [x] 1.4 `render_linear` (parity.py): applies `r.renderer.film.imaging_ratio()` to
  the returned linear-HDR so headless A/B sees pbrt-equivalent absolute radiance.

## 2. UI — retune ISO/exposure on the fly

- [x] 2.1 Added `film.iso` + `film.exposure_time` to `STATIC_PARAMS` (params.py)
  with `film.*` nested paths (resolve on `renderer.film`); change resets accum via
  the hash. Surfaces in both front-ends via `build_all_params`.
- [x] 2.2 USD-driven control UI: `resolve_control_binding` already routes a
  `renderer:film.iso` / `renderer:film.exposure_time` target through
  `_get/_set_nested` → `renderer.film.*` (no code change needed). A scene can author
  a `skinny:ui:*` control prim targeting those paths and it retunes the live film
  params + resets accum, consistent across both front-ends (shared `build_app_ui`).
  STATIC_PARAMS sliders (2.1) cover the always-on path + settings.json persistence.

## 3. Fix the `.hdr`-direct env scale drop

- [x] 3.1 `lights.py` infinite `.hdr`-direct branch now authors DomeLight
  `intensity = scale` (loader collapses color×intensity×2^exposure → env scalar),
  so the pbrt `scale` (and any film ratio) is no longer dropped. Regression tests
  (`scale 2` → intensity 2; no-scale → 1) in `test_radiometric_parity.py`.

## 4. Diagnose the light-type absolute offset — DONE (premise corrected)

The proposal's "~1.6× **bright** area-lit" premise did **not** reproduce (stale; refs
regenerated 2026-06-26). Measured current state (Metal, anchor path|wavefront, GPU):
area-lit **0.866–0.875** (skinny *dimmer*), env-lit const 0.957, env-lit diffuse 0.996,
subsurface 1.245 (known SSS residual, out of scope). Full diagnosis below.

- [x] 4.1 Cross-rendered area lights vs pbrt v4 at matched res/camera, film neutral.
  Pinned the factor: **0.866** (diffuse), 0.874 (mirrored), 0.875 (glass) — tight,
  L- and material-independent. Direct emitter view = **1.000** (emission magnitude
  correct). Achromatic gray = colored = per-channel flat ⇒ NOT colored RGB-vs-spectral.
- [x] 4.2 Traced to source — **two stacked effects**:
  (a) **Fixable**: the emissive-triangle **BSDF-sampling MIS complement is dropped**
  (path.slang:213 gates raw emission off for bounce>0 but adds no MIS-weighted
  replacement; only sphere lights get `w_bsdf`). Size-dependent (large light 0.806).
  This is the **separate change `emissive-triangle-bsdf-mis` (commit `6e66706`)**,
  already implemented + GPU-verified, **not yet merged to main** → absent here.
  Verified via the fix worktree: large light 0.806→0.873, removing the size term.
  (b) **Inherent**: a constant ~0.875 RGB-vs-spectral illuminant residual (= 1/1.14,
  the documented post-MIS-fix bathroom residual) — a proposal **non-goal**.
  → No in-this-change source fix; the fixable part lands by merging `6e66706`.
- [x] 4.3 Env-lit re-checked: conductor_infinite 0.957 (≈1.0), texture_quad 0.996.
  Confirms the offset is area-light-transport-specific, not a global scale.

## 5. Absolute-radiance gate

- [x] 5.1 Absolute (un-exposure-aligned) mean-ratio + relMSE gate added beside the
  exposure-blind gate: `metrics.mean_ratio`, `parity.absolute_radiance_result`,
  `SceneSpec.absolute` (+ unit tests), wired into `test_parity` matrix (anchor-only;
  self-consistency ties the rest). Configured on `diffuse_arealight` (area) and
  `conductor_infinite` (const-env). HDRI corpus scene deferred to the
  `pbrt-env-float-precision` follow-up (it adds the HDRI parity scene + ref).
- [x] 5.2 Recorded `absolute` baselines = the measured current state: area-lit
  0.866, env-lit 0.957 (the proposal's "0.9–1.1×" target is **gated by merging
  `6e66706`** for the fixable part + the inherent RGB-vs-spectral floor). The gate
  guards these and the windows admit the post-`6e66706` shift to ~0.875.
  Exposure-matched FLIP unchanged (the change is output-equivalent for ratio=1).
- [x] 5.3 No corpus combo shifted from this change — the imaging ratio is applied at
  the output (was baked into emitters), algebraically identical for a linear path
  tracer; default-film scenes (all corpus scenes, iso 100) are byte-unchanged. No
  `measured` re-record needed; no self-consistency tolerance loosened.

## 6. Docs + OpenSpec

- [x] 6.1 `docs/PbrtImport.md`: film params live on the camera + output imaging
  ratio (not baked), `.hdr`-direct `scale` on DomeLight intensity, absolute-radiance
  gate + the two-component area-light note; `docs/Subsurface.md` brightness note
  links here + scopes the SSS residual; `docs/PythonAPI.md` documents `Renderer.film`.
- [x] 6.2 `openspec validate pbrt-radiometric-parity --strict` valid; ruff clean.
- [ ] 6.3 Commit → archive → merge → push (worktree `.gitignore` is `*` →
  `git add -f`; `main` usually not checked out → merge via temp
  `git worktree add ../skinny-main-merge main`).
