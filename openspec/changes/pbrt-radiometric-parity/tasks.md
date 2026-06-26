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

- [ ] 1.1 `CameraOverride` (scene.py) + `_extract_camera` (usd_loader.py): carry
  `iso` (default 100) + `exposure_time` (default 1.0) read from `skinny:film:iso`
  / `skinny:film:exposureTime` on the `UsdGeom.Camera` (fallback to the standard
  `exposure` attr). Pure-Python unit tests for the read.
- [ ] 1.2 `api.py _emit_camera`: author `skinny:film:iso` + `skinny:film:exposureTime`
  on the camera prim from `scene.film` / shutter. STOP baking `exposure_scale` into
  emitters/env (`_emit_shape`, `add_light`) — pass `exposure_scale=1.0`; the ratio
  is applied live instead. Update the `report.exact("film:exposure", …)` note.
- [ ] 1.3 Renderer: compute `imagingRatio = exposure_time · iso / 100` from the
  applied camera override and apply it as a **live linear output scale** on the
  accumulation-HDR read (and the display path), re-read each frame. A change to
  iso/exposure resets accumulation via `_current_state_hash`.
- [ ] 1.4 `render_linear` (parity.py): apply the camera imaging ratio to the
  returned linear-HDR so headless A/B sees pbrt-equivalent absolute radiance.

## 2. UI — retune ISO/exposure on the fly

- [ ] 2.1 Add `film.iso` + `film.exposure_time` to `ALL_PARAMS` (app.py) with
  `_get/_set_nested` paths; accumulation resets on change.
- [ ] 2.2 Wire into the USD-driven control UI so the knob round-trips to the
  authored `skinny:film:*` attrs (usd-driven-control-ui), consistent across the
  front-ends (per the GUI-consistency rule).

## 3. Fix the `.hdr`-direct env scale drop

- [ ] 3.1 `lights.py:92-124` infinite branch: when a pbrt infinite light points at
  a `.hdr` directly, its pbrt `scale` must still apply (author DomeLight intensity
  = `scale`, or fold into the loader). Add a regression test (pbrt `scale 2` on a
  `.hdr` infinite → 2× DomeLight intensity).

## 4. Diagnose + fix the light-type absolute offset (~1.6× area-lit)

- [ ] 4.1 Cross-render a diffuse-white patch under a single area light
  (`AreaLightSource "diffuse" "rgb L"`) in pbrt and skinny at matched res/camera;
  measure the flat-patch radiance ratio with film params neutral (iso 100). Pin
  the factor.
- [ ] 4.2 Trace the factor to its source (emissive-triangle radiance pack / NEE
  power normalization / area-light pdf) and fix at source; re-measure → ~1.0.
- [ ] 4.3 Re-check env-lit (conductor_infinite stays ~1.0; dragon abs improves).

## 5. Absolute-radiance gate

- [ ] 5.1 Add an absolute (un-exposure-aligned) mean-ratio + relMSE gate to
  `tests/pbrt` on `diffuse_arealight`, a constant-env scene, and an HDRI scene —
  alongside the existing exposure-blind gate (retain it).
- [ ] 5.2 sssdragon (const + dragon10) absolute mean within tolerance of pbrt
  (target 0.9–1.1×) without exposure matching; exposure-matched FLIP unchanged.
- [ ] 5.3 Re-record `measured` baselines for any corpus combo that legitimately
  shifts; never loosen a self-consistency tolerance to hide a shift.

## 6. Docs + OpenSpec

- [ ] 6.1 `docs/PbrtImport.md`: film params now live on the camera (`skinny:film:*`)
  + applied as output imaging ratio, not baked; resolve the brightness note in
  `docs/Subsurface.md` (link here); `docs/PythonAPI.md` for any new public symbol.
- [ ] 6.2 `openspec validate pbrt-radiometric-parity --strict`; ruff clean.
- [ ] 6.3 Commit → archive → merge → push (worktree `.gitignore` is `*` →
  `git add -f`; `main` usually not checked out → merge via temp
  `git worktree add ../skinny-main-merge main`).
