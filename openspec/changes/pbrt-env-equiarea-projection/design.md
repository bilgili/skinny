# Design — pbrt equal-area env reprojection

## Problem

pbrt v4 `ImageInfiniteLight::ImageLe(w)` computes `uv =
EqualAreaSphereToSquare(wLight)` and samples the image at `uv`. The image is an
**equal-area octahedral square**. skinny loads the dome texture and samples it
equirectangularly via `directionToEquirectUV`. The importer must therefore bake
an equirectangular `.hdr` whose pixel at equirect-uv `e` holds the radiance pbrt
returns for the direction that skinny's shader maps `e` back to.

## Pipeline

For each output equirect pixel `(row i, col j)` in an `H × 2H` image:

1. **equirect-uv → direction** (invert `directionToEquirectUV`):
   - `u = (j + 0.5) / (2H)`, `v = (i + 0.5) / H`
   - `phi = (u − 0.5) · 2π`, `theta = v · π`
   - `dy = cos(theta)`, `s = sin(theta)`, `dx = s·sin(phi)`, `dz = s·cos(phi)`
2. **direction → pbrt light-space direction**: apply the fixed world-axis
   permutation `P` (skinny `+y`-up world ↔ pbrt env light space). `P` is the one
   unknown; candidate is identity vs a y/z swap — pinned by the render gate.
3. **direction → square-uv**: `sq = EqualAreaSphereToSquare(P · d)` ∈ [0,1]².
4. **sample source**: bilinear sample the equal-area source at `sq` (clamp at the
   square border — equal-area has no wrap; the octahedral seam is handled by the
   border being the back hemisphere fold, sampled by clamp which is exact at the
   edge midpoints and a sub-texel approximation elsewhere).

Output `H = source_edge`, width `2H`.

## pbrt math (ported verbatim)

`EqualAreaSquareToSphere(p∈[0,1]²) → unit vector`:
```
u = 2p.x − 1 ; v = 2p.y − 1
up = |u| ; vp = |v|
sd = 1 − (up + vp) ; d = |sd| ; r = 1 − d
phi = (r==0 ? 1 : (vp − up)/r + 1) · π/4
z = copysign(1 − r², sd)
cosPhi = copysign(cos phi, u) ; sinPhi = copysign(sin phi, v)
x = cosPhi · r · √(2 − r²) ; y = sinPhi · r · √(2 − r²)
return (x, y, z)
```

`EqualAreaSphereToSquare(d) → [0,1]²` (inverse; real `atan` instead of pbrt's
polynomial approx — exactness over bit-parity):
```
x=|d.x| ; y=|d.y| ; z=|d.z|
r = √(max(0, 1 − z))
a = max(x,y) ; b = (a==0 ? 0 : min(x,y)/a)
phi = atan(b) · (2/π)
if x < y: phi = 1 − phi
vv = phi · r ; uu = r − vv
if d.z < 0: (uu, vv) = (1 − vv, 1 − uu)
uu = copysign(uu, d.x) ; vv = copysign(vv, d.y)
return ((uu + 1)/2, (vv + 1)/2)
```

`sphere_to_square(square_to_sphere(p)) == p` is the primary unit invariant.

## Why reproject at import, not in-shader

- The dome-light path, env importance-sampling CDF builder, and
  `environment.slang` all assume equirectangular. Adding an octahedral sampling
  mode would touch the shader, the CDF builder, both backends, and the descriptor
  set — large blast radius for a rare input. Baking equirect at import keeps the
  change to the importer and leaves the GPU path byte-identical.

## Decisions

- **D1 — gate on square aspect, not filename.** pbrt always uses equal-area for
  infinite-light images; `_equiarea` in a filename is incidental. A square map is
  the reliable signal; a non-square map is already lat-long → pass through + note.
- **D2 — resampler is pure numpy, no USD/torch.** Lives in its own module so it is
  importable and unit-testable under `.venv` (the renderer is not).
- **D3 — bilinear, output H = source edge.** Equal-area octahedral packs the full
  sphere into the square; `H × 2H` equirect over-samples slightly near the poles
  and matches near the equator — no detail loss versus a same-edge square.
- **D4 — orientation permutation `P` is the verification gate, not a guess.** Unit
  tests lock the chart math; the world-axis mapping is confirmed by A/B render
  against the pbrt-v4 reference of `sss_dragon_small`.

## Verification

- Unit: square↔sphere round-trip; the six axis directions map to the expected
  square corners/edge-midpoints; a single bright source texel reprojects to the
  equirect pixel whose direction round-trips back into that texel.
- Render A/B: import `sss_dragon_small.pbrt`, render headless on Metal (wavefront —
  the dragon mesh OOMs the megakernel cold-compile per memory), compare to the
  pbrt-v4 reference EXR at shared exposure. Gate: env-lit orientation matches
  (silhouette/feature alignment), relMSE within the corpus tolerance. Show the
  labelled side-by-side (reference · before · after).
