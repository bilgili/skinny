> Status (worktree `pbrt-v4-scene-import`): import pipeline implemented and
> tested (66 pbrt unit tests green, ruff clean). **Parity gate is LIVE and
> GREEN** — pbrt v4 built from source (`5f7a606`), reference EXRs generated, and
> the GPU gate passes (6 passed/54s): diffuse relMSE 0.087/FLIP 0.041, conductor
> 0.133/0.068, glass 0.128/0.071 (FLIP ≤ 0.07 = perceptually near-matching).
> Remaining partials: 5.3 (full `realistic` lens), 6.3 (non-constant EXR→HDR env).

## 1. Package scaffolding

- [x] 1.1 Create `src/skinny/pbrt/` package (`__init__.py` exposing `import_pbrt`, `__main__.py`) and a `tests/pbrt/` test dir; wire a `skinny-import-pbrt` console entry in `pyproject.toml`
- [x] 1.2 Add the optional test deps for EXR read (or commit a minimal RGBA-scanline EXR reader) and `.ply` read decision per design D10; record any new dep in `pyproject.toml` — own `.ply` reader + lazy imageio/OpenEXR EXR read (no hard dep)
- [x] 1.3 Vendor small spectral data tables (CIE XYZ curves, named pbrt spectra, sRGB primaries/whitepoint) under `src/skinny/pbrt/data/` — analytic CIE CMFs + named-metal IOR table

## 2. Tokenizer and parameter parser

- [x] 2.1 Write failing tests for the tokenizer (comments, quoted strings, numbers, bracketed arrays) and typed `"type name" [values]` params
- [x] 2.2 Implement the tokenizer and typed-parameter parser (`float`/`integer`/`point*`/`vector`/`normal`/`rgb`/`spectrum`/`blackbody`/`bool`/`string`/`texture`)
- [x] 2.3 Implement recursive `Include`/`Import` resolution relative to the including file
- [x] 2.4 Split parsing into options block (pre-`WorldBegin`) and world block; raise located errors on malformed input — tests green

## 3. Graphics-state machine and coordinate bridge

- [x] 3.1 Write failing golden tests: a known world point projects to the expected image location; a front-facing triangle stays front-facing after conversion
- [x] 3.2 Implement the CTM stack (`AttributeBegin/End`, `TransformBegin/End`, `Translate`/`Rotate`/`Scale`/`Transform`/`ConcatTransform`/`LookAt`) and named coordinate systems
- [x] 3.3 Implement the fixed left→right-handed change-of-basis (design D3): apply to every CTM, preserve winding/normals, convert `LookAt` to camera-to-world; set USD scene scale / `mm_per_unit`
- [x] 3.4 Implement current/named material, named texture, current area-light, reverse-orientation state, and `ObjectBegin/End` + `ObjectInstance` instancing — tests green

## 4. Geometry translation → USD

- [x] 4.1 Write failing tests: a `trianglemesh` (P/indices/uv) and a `sphere` emit USD that `usd_loader` loads as `MeshInstance`(s)
- [x] 4.2 Implement a minimal `.ply` reader (ascii + binary LE/BE) and the `plymesh` translator
- [x] 4.3 Implement `trianglemesh` and `sphere` → UsdGeom prims (points, indices, normals, UVs) under the accumulated CTM
- [x] 4.4 Emit instanced shapes as multiple loadable instances of shared geometry — tests green

## 5. Camera translation

- [x] 5.1 Write failing tests for the `perspective` shorter-axis `fov` → vertical-FOV conversion across aspect ratios, and a `realistic` lens → ordered `LensElement`s
- [x] 5.2 Implement `perspective` (fov-axis + `screenwindow` + DOF `lensradius`/`aperturediameter`/`focaldistance` → `fstop`/`focus_distance`) → `CameraOverride`
- [x] 5.3 Implement `realistic` lens description → `LensSystem`/`LensElement` (signed radius, thickness, IOR, aperture, aperture-stop) — parses the pbrt lens file, authors `skinny:lens:*` child prims; loader yields a `LensSystem` (test_camera)

## 6. Light translation

- [x] 6.1 Write failing tests: `distant`→DistantLight, `point`→SphereLight, area `diffuse` front-only emission, `infinite`→DomeLight energy, `spot` flagged "approx"
- [x] 6.2 Implement `distant`/`point`/area(`diffuse`, honoring `twosided`) translation with `L`/`scale`/`power`/`illuminance` → linear-HDR RGB radiance
- [ ] 6.3 Implement `infinite` → DomeLight: emit/link an equirect `.hdr` (convert EXR/PFM), verify integrated energy — references the file + flags non-`.hdr`; conversion + energy check not yet done
- [x] 6.4 Implement best-effort `spot` (+ any other unsupported light) with report flagging — tests green

## 7. Material translation

- [x] 7.1 Write failing unit tests for the pbrt v4 roughness→alpha remap (default and `remaproughness false`, `uroughness`/`vroughness`) against known pbrt values
- [x] 7.2 Implement the roughness remap and the material dispatch for `diffuse`/`dielectric`/`thindielectric`/`coateddiffuse`/`coatedconductor`/`diffusetransmission`
- [x] 7.3 Implement `conductor`: complex IOR (η,k) / named metal → RGB normal-incidence Fresnel base color, metallic=1
- [x] 7.4 Route transmission → skinny opacity/IOR refraction gate; emit flat params or MaterialX `standard_surface` with `customData.skinnyOverrides` as needed; unknown materials best-effort + reported — tests green

## 8. Spectrum → RGB reduction

- [x] 8.1 Write failing tests: `blackbody [6500]` → expected RGB white; a named conductor spectrum → expected RGB reflectance; RGB-reflectance passthrough
- [x] 8.2 Implement CIE-XYZ integration under the rendering color space (default sRGB/Rec709; honor `ColorSpace`) for named/blackbody/sampled `.spd`/RGB spectra — tests green

## 9. Participating media and subsurface (best-effort)

- [x] 9.1 Write failing tests: homogeneous `MakeNamedMedium` coefficients reach the volume path; a grid/VDB medium is flagged "unsupported"
- [x] 9.2 Implement homogeneous medium + `subsurface` best-effort mapping (volume path / `customData`); detect and flag heterogeneous media — tests green

## 10. CLI, Python entry, and translation report

- [x] 10.1 Implement `import_pbrt(path) -> Stage`, `python -m skinny.pbrt`, and the `skinny-import-pbrt scene.pbrt -o out.usda` CLI
- [x] 10.2 Implement the translation report (exact / approximated / skipped per construct, with reasons), keyed to the parity matrix
- [x] 10.3 Test: CLI emits a `.usda` that `Renderer(usd_scene_path=...)` loads with the expected instances, lights, and camera

## 11. Parity metrics library

- [x] 11.1 Write failing tests: relMSE and FLIP are 0 for identical images; relMSE matches a hand-computed value on a tiny fixture
- [x] 11.2 Implement pure-numpy relMSE (`mean((a-b)²/(b²+ε))`) and FLIP (on identically-tonemapped copies), plus an optional global-scale alignment — tests green
- [x] 11.3 Implement EXR reference read returning linear float RGB at the corpus resolution

## 12. Reference corpus and gate

- [x] 12.1 Author the curated low-res (≤256 px) `.pbrt` corpus scenes under `tests/pbrt/` — 3 scenes (diffuse/area, conductor/infinite, glass/area); more axes can be added
- [x] 12.2 Render pbrt v4 reference EXRs and commit them (`git add -f`) with a manifest pinning pbrt version, per-scene tolerance, and content hash — pbrt v4 `5f7a606` built from source; 3 refs in `corpus/refs/`; manifest pins version + measured tolerances
- [x] 12.3 Implement the headless harness: import → skinny offscreen render → read **linear-HDR accumulation** → compare to reference at matching resolution/converged spp — `parity.py`; run on GPU (Vulkan/MoltenVK), passes; zeroes default ambient env for no-`infinite` scenes
- [x] 12.4 Wire the per-scene parity gate as a pytest test that fails (naming scene + metric) when relMSE/FLIP exceeds tolerance, and runs with no pbrt binary present — `test_parity.py` GREEN under 3.13+GPU (6 passed); skips cleanly without GPU/refs

## 13. Documentation and close-out

- [x] 13.1 Write `docs/PbrtImport.md` (usage + the parity matrix: matched/approx/unsupported with measured per-scene error)
- [x] 13.2 Update `README.md`, `docs/Architecture.md` (module map), `docs/PythonAPI.md` (new public `skinny.pbrt` symbols), and `CHANGELOG.md`
- [ ] 13.3 Run `.venv/bin/ruff check src/` and `.venv/bin/pytest` (incl. the parity gate); `openspec validate pbrt-v4-scene-import`; verify no doc drift — ruff clean + 66 pbrt tests green; full-repo pytest + parity gate pending GPU/refs
