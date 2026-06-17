## Why

pbrt v4 is the de-facto reference physically-based renderer and ships a large
corpus of public test scenes. skinny has no way to consume them, so every
cross-check against "the book renderer" is manual and ad hoc. A pbrt v4 importer
that converts scenes into skinny's existing USD pipeline gives us (a) a ready
supply of non-trivial test content and (b) a credible ground-truth for
validating skinny's transport, materials, lights, and camera against an
independent implementation — the single best lever for catching energy and bias
bugs. The explicit goal is **near-matching images** for the same scene in skinny
and pbrt v4, gated by an automated error metric.

## What Changes

- **New pbrt v4 reader/translator.** A self-contained Python parser for the pbrt
  v4 text format (directives, graphics-state stack, named objects/instances,
  `Include`/`Import`, `.pbrt`/`.ply` assets) that translates a scene into a
  skinny-loadable **USD** stage (reusing the existing `usd_loader` pipeline —
  no new in-memory scene type). The user-facing "exporter" is this
  pbrt→skinny converter; direction is **pbrt → skinny only**.
- **Broad directive coverage.** Translate the common pbrt v4 surface: triangle/
  ply/sphere shapes and object instancing; `diffuse`/`conductor`/`dielectric`/
  `coateddiffuse`/`coatedconductor`/`diffusetransmission`/`thindielectric`
  materials; `distant`/`point`/`spot`/area(`diffuse`)/`infinite` lights;
  `perspective` (+ thick-lens `realistic`) cameras; `constant`/`scale`/`mix`/
  `imagemap`/`checkerboard` textures; named/blackbody/RGB **spectra** reduced to
  the renderer's RGB color space; and homogeneous participating **media** +
  `subsurface`. Features skinny cannot render faithfully are imported best-effort
  and flagged, never silently dropped.
- **Spectrum → RGB reduction.** A documented conversion (CIE + pbrt's default
  rendering color space) from pbrt spectra to skinny's linear-RGB inputs, with
  the residual divergence called out (skinny is RGB; full spectral parity is out
  of scope).
- **Coordinate / units / film bridge.** Handle pbrt's left-handed look-at
  convention, world scale, and film/`gbuffer` settings so geometry, camera, and
  exposure line up with skinny's conventions.
- **CLI + Python entry.** `skinny-import-pbrt scene.pbrt -o scene.usda` (and a
  `skinny.pbrt` Python API) that emits a USD stage plus a translation report
  (what mapped exactly, what was approximated, what was skipped).
- **Parity validation harness + corpus.** A curated set of small low-resolution
  scenes with **checked-in pbrt v4 reference EXRs**, plus a headless harness that
  renders the imported scene in skinny and asserts a per-scene perceptual/error
  gate (FLIP and relMSE) against tolerance. No pbrt binary required in tests.
- **Parity matrix doc.** A living table of which pbrt features match, approximate,
  or are unsupported, with measured error per corpus scene.

## Capabilities

### New Capabilities
- `pbrt-scene-import`: Read a pbrt v4 scene file and produce a skinny-loadable USD
  stage — tokenizer/parser, graphics-state + named-object/instance semantics,
  asset/`Include` resolution, and the translation of pbrt shapes, materials,
  lights, cameras, textures, spectra, and homogeneous media into USD (UsdGeom /
  UsdLux / UsdPreviewSurface + MaterialX, with `customData` carrying skinny
  extensions). Includes the CLI/Python entry point and the human-readable
  translation report.
- `pbrt-parity-validation`: Define the image-parity contract — the curated
  low-res scene corpus, the checked-in pbrt v4 reference EXRs, the headless
  compare harness (FLIP + relMSE against linear-HDR accumulation), per-scene
  tolerance gates, and the maintained parity matrix documenting matched /
  approximated / unsupported features.

### Modified Capabilities
<!-- None. The importer emits USD consumed by the existing loader and is invoked
     through the existing headless render path; neither's spec-level requirements
     change. New behavior is additive and lives in the two new capabilities. -->

## Impact

- **New code:** `src/skinny/pbrt/` (tokenizer, parser, graphics-state machine,
  per-directive translators, spectrum→RGB, USD emitter, report); a
  `skinny-import-pbrt` console entry in `pyproject.toml` + `python -m skinny.pbrt`.
- **Reuses unchanged:** `usd_loader.py`, `scene.py`, the headless render path
  (`render_headless` / accumulation readback per CLAUDE.md), MaterialX runtime,
  thick-lens camera (`LensSystem`/`LensElement` — already PBRT-convention),
  `volume_render.slang` for homogeneous media.
- **Dependencies:** OpenEXR/Imath read for reference EXRs (test-time);
  `.ply` mesh read. The pbrt parser is pure-Python, no new runtime parser dep.
  Spectrum tables (CIE + named-spectra) vendored as small data files.
- **Tests/assets:** new `tests/pbrt/` (corpus `.pbrt` + scenes) and reference
  EXRs under `tests/pbrt/refs/`; kept tiny and low-res to bound size. Note the
  repo `.gitignore` is `*`, so corpus + refs must be force-added (`git add -f`).
- **Docs:** new `docs/PbrtImport.md` (usage + parity matrix); cross-links from
  `README.md` (new CLI flag/entry), `docs/Architecture.md` (module map),
  `docs/PythonAPI.md` (new public symbols), and `CHANGELOG.md`.
- **Out of scope:** skinny → pbrt export; full spectral rendering; heterogeneous
  (grid/VDB) media parity; bidirectional/photon pbrt integrators.
