## 1. Equal-area chart module (TDD, RESEARCH)

- [x] 1.1 Failing test `tests/pbrt/test_equiarea.py`: `sphere_to_square(square_to_sphere(p)) == p` and the dual round-trip, over a grid of `p ∈ [0,1]²`, float tolerance
- [x] 1.2 Failing test: the six axis directions (`±x,±y,±z`) map to expected square corners / edge midpoints
- [x] 1.3 New `src/skinny/pbrt/equiarea.py`: `equal_area_square_to_sphere`, `sphere_to_equal_area_square` (numpy, vectorized, no USD/torch); ports pbrt math in design.md
- [x] 1.4 Tests 1.1–1.2 green; `ruff check src/skinny/pbrt/equiarea.py` clean

## 2. Reprojection resampler (TDD, RESEARCH)

- [x] 2.1 Failing test: `equiarea_to_equirect(img, height)` of a known directional delta puts the bright pixel at the equirect cell whose `directionToEquirectUV⁻¹` direction round-trips into the source texel (directional consistency)
- [x] 2.2 Failing test: output shape is `(H, 2H, 3)`; a uniform source maps to a uniform equirect (no energy distortion beyond bilinear)
- [x] 2.3 Implement `equiarea_to_equirect`: build equirect grid → invert `directionToEquirectUV` → apply axis permutation `P` → `sphere_to_equal_area_square` → bilinear sample source (clamp border)
- [x] 2.4 Tests green; ruff clean

## 3. Importer wiring (TDD, ENGINEERING)

- [x] 3.1 Failing test: a square non-uniform PFM infinite light → emitted `.hdr` differs from a verbatim copy and equals `equiarea_to_equirect(src)`; report entry says "equal-area → equirect"
- [x] 3.2 Failing test (back-compat): non-square PFM and constant `rgb L` infinite light → output byte-identical to current behavior
- [x] 3.3 `_convert_env_to_hdr` (`src/skinny/pbrt/lights.py`): reproject when `img` is square, else pass through; thread the report note
- [x] 3.4 Tests green; full `tests/pbrt/` suite (`-m "not gpu"`) green (1 pre-existing `.venv`-USD mtlx failure, unrelated); ruff clean

## 4. Orientation gate (RESEARCH — go/no-go)

- [x] 4.1 Reproject the **actual** scene asset (`small_rural_road_equiarea.exr`, 2048² equal-area) on CPU and inspect before (octahedral square) vs after (equirect panorama) at shared Reinhard+sRGB tonemap — avoids the 28.8M-tri dragon megakernel OOM
- [x] 4.2 Fix the world-axis permutation `P` from the A/B: P0/identity put sky on the horizon; **P1 = `Rx(+90)` → `(x,-z,y)`** gives a flat horizon band (sky up, road down); P2/`Rx-90` upside down. `P` baked into `_apply_axis`; recorded in design.md
- [x] 4.3 Showed the labelled before/after to the user; structural orientation correct. **Residual** azimuth-rotation / mirror-handedness vs pbrt is NOT yet pinned (needs the full dragon A/B against a pbrt-v4 reference EXR) — deferred to §5
- [~] 4.4 Full `sss_dragon_small` Metal **wavefront** render (342×256, spp 24, RAM-guarded — survived, no OOM, ~18GB free) vs pbrt-v4 `sss_dragon_small.exr`. **Orientation structurally confirmed**: dragon lit coherently (bright top/front, dark underside) consistent with the sky-up/road-down panorama, pose aligned, NOT scrambled. relMSE 0.52 / FLIP 0.26 is dominated by three **out-of-scope** confounds, not the env: (a) subsurface volumetric walk does not engage in this import path — `materialx=True` rendered **bit-identical** to UsdPreviewSurface (mean 0.03880461276461626 both) → dragon opaque vs pbrt's translucent SSS; (b) env ~5.4× too dim (known env-intensity issue, see `pbrt-camera-up-axis`); (c) ground-plane shading. Precise azimuth/handedness not numerically isolable until (a)+(b) resolved; gross orientation correct. Side-by-side: `/tmp/dragon_mtlx_sidebyside.png`

## 5. Optional GPU corpus scene (ENGINEERING)

- [ ] 5.1 Add a small corpus scene with a non-uniform equal-area map + its pbrt-v4 reference EXR under `tests/pbrt/corpus/` (`git add -f`), tolerance + hash in the manifest
- [ ] 5.2 Wire it into the parity gate (skips if no GPU)

## 6. Docs

- [ ] 6.1 `docs/PbrtImport.md`: env-map conversion row — equal-area→equirect reprojection, square-aspect gate, non-square passthrough
- [ ] 6.2 `CHANGELOG.md` entry
- [ ] 6.3 `openspec validate pbrt-env-equiarea-projection --strict`
