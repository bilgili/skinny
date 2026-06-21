# Design — pbrt-subsurface-volumetric (Stage-2 Ch5)

## Where this sits

pbrt-mtlx Stage-2 roadmap, Ch5 (XL / research). Glass (Ch2 tints, shipped), rough
glass BTDF (Ch4b), and this — true subsurface — are the transmissive tail. pbrt-v4
`subsurface` is a **random walk in a homogeneous interior medium** behind a smooth
dielectric boundary; this change reproduces that, rather than the separable
diffusion BSSRDF the skin path uses (kept as a possible fast follow-up).

## What exists to reuse (from the landscape survey)

- `volume_render.slang` — Woodcock/delta-tracking + Henyey-Greenstein phase
  (`henyeyGreenstein`, `sampleHenyeyGreenstein`) + transmittance. Standalone today
  (only `skin_transmission.slang` imports it); not attached to geometry interiors.
- `flat_shading.slang` — `fresnelDielectric`, the refract/reflect boundary logic
  already used by the flat opacity<1 delta-transmission branch.
- Material dispatch scaffold — `MATERIAL_TYPE_*` (`bindings.slang`), the
  `main_pass.slang` megakernel switch, the `scene_trace.slang` wavefront switch,
  and `materialTypes[]` packing (`renderer.py`).
- Param extraction — `map_material_mtlx` already emits `subsurface_color`,
  `subsurface_radius`; `pack_std_surface_params` packs them (binding 19).
- Per-instance buffer — TLAS instance records already exist (animated-xform path
  re-uploads them), a place to carry an interior-medium handle/coefficients.

## What's missing (the work)

1. **`MATERIAL_TYPE_SUBSURFACE`** constant + a dispatch case in both the megakernel
   and wavefront material switches.
2. **A medium registry + handle** — a `Medium` entry `(σ_a, σ_s, g, eta,
   majorant σ_max, densityFieldHandle)` in a small SSBO, referenced by a handle.
   This change populates it from subsurface materials and resolves the handle by
   material id; the **registry + handle indirection is deliberate** so a future
   free-standing `MediumInterface` can register a named medium and attach it to an
   instance/camera without reworking transport (see Forward compatibility). For a
   homogeneous interior, `densityFieldHandle = none` and `σ_max = max-channel σ_t`.

   **Storage (refined during implementation — Metal 31-buffer cap):** the megakernel
   already sits in the low-20s of distinct buffer args against slang-rhi's hard
   31-buffer Metal limit (the codebase has repeatedly folded to fit). Adding a new
   SSBO registry risks re-breaching it when neural+graph stack. So — following the
   emissive-mesh-NEE precedent (CDF packed inline, no new buffer) — the homogeneous
   medium is **packed into the existing `FlatMaterialParams` buffer (binding 13)**
   that every non-skin/python material already owns; `resolveMedium(handle)` reads
   it from `flatMaterials[matId]`. The seam is unchanged — a future free-standing
   `MediumInterface` adds a dedicated registry buffer (folding to fit) *then*,
   behind the same `resolveMedium`. No new buffer now.
3. **The medium random walk** — a shared `mediumWalk(rayIn, medium, boundaryMode,
   rng)`: cross the boundary (mode *dielectric* ⇒ Fresnel refract, the subsurface
   case; mode *index-matched* ⇒ enter the bounding AABB unrefracted, the
   free-standing case) then loop **null-collision (Woodcock/ratio) tracking against
   `σ_max`** — at each tentative collision look up the local `σ_t` (constant for a
   homogeneous interior; a density-field lookup for a grid), accept a real scatter
   with prob `σ_t/σ_max` (HG `g`, throughput `σ_s/σ_t`) else a null collision; on a
   boundary hit apply the boundary mode (Fresnel split vs pass-through). Bounded
   (throughput ≤ 1/event), Russian-roulette-terminated — single pdf, no clamp.
   **Null-collision is chosen over closed-form homogeneous transmittance precisely
   because it IS the heterogeneous algorithm** — constant `σ_t` is its degenerate
   case, so the Disney-cloud grid reuses this loop verbatim.
4. **Integrator wiring** — call the walk from the megakernel estimator and the
   wavefront `path.slang` / `bdpt.slang` (the skin BSSRDF is wired the same way;
   reuse that structure).
5. **Importer mapping** (`materials.py` + `usd_loader.py`) — derive `(σ_a, σ_s, g)`
   and route to the new type instead of flat-opacity-0.

## Coefficient derivation (importer)

pbrt's own precedence, reproduced so renders match:
- explicit `sigma_a` + `sigma_s` (mm⁻¹) → use directly (× `scale`).
- named preset (`"Skin1"`, `"Marble"`, …) → pbrt's measured `GetMediumScatteringProperties`
  table (port the constants).
- `reflectance` + `mfp` → invert the diffusion albedo: solve single-scatter albedo
  `α` from diffuse reflectance `R_d` (the standard Jensen/Christensen inversion),
  then `σ_t = 1/mfp`, `σ_s = α·σ_t`, `σ_a = σ_t − σ_s`.
- std_surface (`-mtlx`) path: `subsurface_color` = diffuse reflectance, per-channel
  `subsurface_radius` = `mfp`, `subsurface_scale` scales `1/mfp`. Same inversion →
  identical `(σ_a, σ_s)` so native-USD and `-mtlx` agree (a parity requirement).
- `g` from `subsurface_anisotropy` (default 0 = isotropic).

## Invariants & parity (the tax)

The walk is an unbiased estimator with a single sampling pdf and bounded per-event
throughput (no clamp), so the path/BDPT/ReSTIR/neural invariants hold by
construction — re-verified by:
- **Furnace / energy** — a homogeneous SSS sphere with `σ_a → 0` in a constant
  environment returns ~unity (no energy created/lost); albedo inversion round-trips.
- **sssdragon parity** — the reduced subsurface dragon converges to a milky result
  matching a pbrt v4 reference within relMSE/FLIP (replacing the glass look). Both
  execution modes, both backends.
- **Back-compat** — non-subsurface scenes (the pbrt parity corpus, true glass)
  byte-unchanged; flat opacity/refraction path untouched.
- **PT ≡ BDPT** — both integrators converge to the same image on the SSS sphere.

## Risks

- **High-albedo deep walks** — `σ_a → 0`, `α → 1` makes walks very long (hundreds
  of collisions); needs Russian roulette + a step/bounce cap that stays unbiased,
  or the megakernel will stall / firefly. This is the main research risk.
- **Boundary total-internal-reflection bookkeeping** — repeated internal Fresnel
  reflections must conserve energy and not bias; the dielectric η²-on-refraction
  radiance scaling (the BDPT adjoint subtlety from Ch4b) applies on exit.
- **Per-channel σ** — RGB-split mean free paths mean three transmittances; keep
  the throughput a float3 and the pdf scalar (Hero-wavelength not in scope).
- **Wavefront cost** — the walk is an unbounded inner loop; on Metal the indirect-
  dispatch CPU-readback fallback already makes wavefront slow on big meshes
  (sssdragon = 28.8M tris), so expect long parity-gate renders (reduce spp/res).
- **Megakernel size** — adding a third heavy material path may push the Metal
  in-process compile RAM (guarded_metal.sh) — watch the floor.

## Forward compatibility: Disney cloud / heterogeneous free-standing media

The `disney-cloud` pbrt scene is the canonical next consumer: `MakeNamedMedium
"cloud" "string type" "nanovdb"` (a heterogeneous NanoVDB density grid), spectral
`sigma_a`/`sigma_s`, `float scale`, attached to a bounding box via `MediumInterface
"cloud" ""` (interior = cloud, exterior = vacuum, **index-matched** — no dielectric
boundary). This change implements only the homogeneous, dielectric-bounded
subsurface case, but its abstractions are chosen so cloud drops in **without
reworking transport**. The deliberate choices:

| Cloud needs | This change bakes in | Cloud-only follow-up |
|-------------|----------------------|----------------------|
| Spatially-varying σ_t (grid) | null-collision tracking vs `σ_max` (heterogeneous algorithm; homogeneous = constant degenerate case) | `densityAt(p)` grid lookup behind the existing handle |
| Free-standing volume (not a surface's interior) | medium **registry + handle** (not hardwired to material-interior) | resolve `MediumInterface` → register named medium, attach to instance/camera |
| Index-matched boundary (vacuum exterior) | `mediumWalk(boundaryMode)` with a *dielectric* and an *index-matched* mode | enter/exit the bounding AABB in index-matched mode (already a parameter) |
| Strong forward scattering | HG phase already general (`g` from `subsurface_anisotropy`; cloud `g≈0.85`) | — |
| Per-channel coefficients | throughput is `float3`, σ per-channel | spectral→RGB resample at import (concession) |
| Huge optical depth (~0.99 albedo) | unbiased Russian roulette in the walk | majorant grids / decomposition for efficiency (optional) |

So the **only** cloud-specific work left after this change is: a NanoVDB→grid
importer + `densityAt` lookup, `MediumInterface`/named-medium resolution in the
pbrt importer, and a spectral→RGB σ resample. The walk loop, the medium registry,
the handle indirection, the boundary-mode split, the HG phase, and the float3
per-channel transport are all reused unchanged. **Explicit guardrail (in the spec):
the transport SHALL be majorant/null-collision based and the medium SHALL be a
handle-referenced registry entry, so this forward path is not designed out.**
NanoVDB ingestion, `MediumInterface` free-standing media, and spectral σ remain
out of scope here (their own change).

### The density seam (so adding a volume kind is purely additive)

The whole point: the random walk reads the medium **only** through two functions,
and every transport equation is written against them — so a new volume source is a
new `case`, never an equation change. Following the codebase idiom (a runtime
`kind` tag + `switch`, exactly like `flatSampleLobe(lobeKind, samplerId, …)` — not
existential dynamic dispatch, because a scene mixes media kinds per-hit at
runtime):

```
// Medium carries a kind tag + a resource handle (grid index; unused for homogeneous).
struct Medium { float3 sigma_a, sigma_s; float g, eta; uint kind; uint gridHandle; float sigmaMaxScale; }

// THE seam. Returns the local density multiplier in [0,1] (× base sigma_t).
float densityAt(Medium m, float3 pWorld) {
    switch (m.kind) {
        case MEDIUM_HOMOGENEOUS: return 1.0;                       // this change
        // case MEDIUM_NANOVDB:  return sampleGrid(m.gridHandle, worldToGrid(m, pWorld)); // later
        default: return 1.0;
    }
}
// Majorant for the null-collision accept prob. Homogeneous: the constant. A grid
// returns its (optionally per-segment) max so thin clouds stay efficient.
float3 mediumMajorant(Medium m, float3 pA, float3 pB) {
    switch (m.kind) {
        case MEDIUM_HOMOGENEOUS: return (m.sigma_a + m.sigma_s);   // this change
        default: return (m.sigma_a + m.sigma_s) * m.sigmaMaxScale;
    }
}
```

`mediumWalk` calls **only** `densityAt` and `mediumMajorant` (plus the shared HG
phase + Fresnel) — `sigma_t(p) = (sigma_a+sigma_s)·densityAt(p)`, accept a real
collision with prob `sigma_t(p)/majorant`. This change implements the
`MEDIUM_HOMOGENEOUS` case (densityAt ≡ 1, majorant = σ_t) and routes the walk
through the seam from day one, so the homogeneous render is itself proof the seam
works. Adding NanoVDB = one enum value + the two `case` bodies + grid ingestion;
the equations, the walk, NEE, RR, and the integrator wiring are untouched.

Honest caveat on "just add a kind": correctness is purely additive, but
*efficient* heterogeneous media want a **spatial majorant** (super-voxel /
majorant grid) so `mediumMajorant(pA,pB)` is tight along a segment rather than the
global grid max — that accel structure is real work, but it lives entirely behind
the `mediumMajorant` seam (a global max is correct, just slower), so it is an
optimization, never an equation or walk change. Grid resource binding (a bindless
3D-texture/buffer pool) and the per-medium world→grid transform are likewise
isolated additions behind the handle.

### Multiple coexisting media (e.g. a cloud AND a subsurface dragon)

Different media of different kinds in one scene **do not conflict**: each is its
own registry entry, resolved by its own `handle`→`kind`, so the dragon walks as
`MEDIUM_HOMOGENEOUS`+dielectric and the cloud as `MEDIUM_NANOVDB`+index-matched
through the *same* `mediumWalk(medium, boundaryMode)` with different args — no
shared mutable state. The density seam already isolates them.

The one thing the seam does **not** provide, and that free-standing media require,
is **current-medium tracking**: a ray traversing the cloud between surface hits
must know it is *inside* the cloud (and a dragon may sit inside it). That is pbrt's
`MediumInterface` (inside/outside medium per surface) + a ray-carried current-medium
handle — a *control-flow* addition to the integrator path loop, **not** an equation
or walk change. This change does not need it: homogeneous subsurface is entered by
refracting through its own boundary, exterior is always vacuum, and the whole
enter→walk→exit happens in one self-contained `mediumWalk` call at the hit.

To keep the future additive, this change **factors the per-segment traversal** (the
null-collision loop over `densityAt`/`mediumMajorant`/HG/RR) as its own function,
separate from the "subsurface hit triggers it" caller. A later free-standing-media
change then adds: (1) a ray current-medium handle, (2) `MediumInterface` resolution
at surfaces (set current medium from the surface's inside/outside), and (3) a path-
loop step that runs the *same* segment traversal through the current medium between
hits. The subsurface entry is re-expressed as the degenerate `MediumInterface`
(inside = its medium, outside = vacuum). So coexisting cloud+dragon costs:
current-medium bookkeeping + `MediumInterface` resolution + the grid `kind` — the
segment transport, seam, registry, phase, and RR are reused unchanged.

**Genuinely deferred (hard even for the cloud change): overlapping / nested media**
(a dragon *inside* the cloud, or two clouds intersecting) needs a medium **priority
/ stack** to decide which medium owns an overlap region — pbrt solves this with
medium priorities. Out of scope for this change and flagged as the hard part of the
free-standing-media follow-up; the non-overlapping cloud+dragon case (disjoint
volumes) only needs the current-medium tracking above.

## Phasing within this change

1. Plumbing: `MATERIAL_TYPE_SUBSURFACE`, `MediumParams` SSBO, importer mapping +
   routing, param round-trip tests (no GPU).
2. Megakernel walk + dispatch; furnace + energy gates (Vulkan).
3. Wavefront walk in path/bdpt; PT≡BDPT + Metal↔Vulkan parity.
4. sssdragon pbrt parity gate + reference; docs.
