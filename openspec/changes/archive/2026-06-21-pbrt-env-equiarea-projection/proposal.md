## Why

pbrt v4 `ImageInfiniteLight` **always** interprets its image with the equal-area
octahedral parameterization (`EqualAreaSquareToSphere`); there is no
equirectangular option. The importer's `_convert_env_to_hdr`
(`src/skinny/pbrt/lights.py`) copies the source `.exr`/`.pfm` pixels **verbatim**
into the `.hdr` that skinny's dome-light path loads, and skinny's
`environment.slang` samples that map as **equirectangular**
(`directionToEquirectUV`: `u = atan2(dx,dz)/2π+0.5`, `v = acos(dy)/π`). So every
image-based infinite light is mis-projected: octahedral pixels read as lat-long
scramble all incoming directions.

This is the dominant reason `sss_dragon_small.pbrt` looks wrong when imported to
`dragon_sss.usda` — the subsurface dragon's appearance is almost entirely
transmitted environment light, so a scrambled env reshapes its color, highlights,
and shadow falloff. Material export is already correct (Skin1 σ_a/σ_s × `scale`
match the sidecar). Existing parity gates only ever use a **constant** infinite
light (`rgb L`) or a **uniform** PFM, so the bug has never been exercised.

## What Changes

- Reproject the source map from pbrt's **equal-area octahedral square** to a
  skinny-convention **equirectangular** `.hdr` inside `_convert_env_to_hdr`,
  instead of copying pixels verbatim. The reprojection ports pbrt's exact
  `EqualAreaSquareToSphere` / `EqualAreaSphereToSquare` and inverts skinny's
  `directionToEquirectUV` so a direction sampled by the shader lands on the same
  radiance pbrt would return for that direction.
- Apply the reprojection only for **square** image maps (equal-area octahedral is
  square by construction). Non-square maps (already lat-long, e.g. a hand-authored
  panorama) and constant infinite lights are passed through unchanged, with a
  report note when a non-square map is assumed equirectangular.
- New module `src/skinny/pbrt/equiarea.py`: pure-numpy `equal_area_square_to_sphere`,
  `sphere_to_equal_area_square`, and `equiarea_to_equirect(img, height)` resampler
  (bilinear, wrap-aware), with no GPU/USD deps so it is unit-testable in `.venv`.
- Output equirect resolution: height = source square edge, width = 2·height
  (preserves angular detail; bilinear source sampling).
- Refresh `docs/PbrtImport.md` (env-map conversion row) and `CHANGELOG.md`.

## Capabilities

### New Capabilities
- `pbrt-env-equiarea`: equal-area-octahedral → equirectangular reprojection for
  pbrt `infinite` light image maps in the importer, so image-based environment
  lighting matches pbrt v4 orientation and radiance.

## Impact

- **Code:** new `src/skinny/pbrt/equiarea.py`; `src/skinny/pbrt/lights.py`
  (`_convert_env_to_hdr` calls the resampler for square maps). No shader, USD
  loader, descriptor-binding, or render-path changes — the dome-light path and
  `environment.slang` are unchanged.
- **Tests:** new `tests/pbrt/test_equiarea.py` (round-trip + known-direction unit
  tests, torch/USD-free); a directional-consistency check that a delta in the
  source square emerges at the matching equirect direction; optional GPU parity
  corpus scene with a non-uniform equal-area map.
- **Docs:** `docs/PbrtImport.md`, `CHANGELOG.md`.
- **Back-compat:** constant and uniform infinite lights (the whole existing parity
  corpus) are byte-unchanged; only non-uniform square image maps change output.

## Open Risk — orientation

The hard part is matching pbrt's light-space axis convention to skinny's world
(`+y` up, `phi` about `+y` from `+z`). The unit tests pin the square↔sphere math;
the **world-axis permutation** (pbrt env direction → skinny env direction) is
fixed by a render A/B against the pbrt-v4 reference and is the explicit go/no-go
gate. A residual whole-map rotation, if any, is recorded — not silently shipped.
