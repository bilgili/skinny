## 1. Coefficient mapping (importer, no GPU — TDD first)

- [x] 1.1 Unit tests: pbrt `subsurface` → `(σ_a, σ_s, g)` for each input form —
  explicit `sigma_a`/`sigma_s` (×`scale`); named preset `Skin1`; `reflectance` +
  `mfp` via albedo inversion. Assert the `-mtlx` (`subsurface_color`/`radius`/
  `scale`/`anisotropy`) path yields identical coefficients. Inversion round-trips.
- [x] 1.2 `materials.py`: port pbrt's named-medium scattering table (`Skin1`, …)
  and the `reflectance`+`mfp` → albedo inversion; emit `(σ_a, σ_s, g, eta)`.
- [ ] 1.3 `usd_loader.py`: route subsurface to the new material type (stop the
  `opacity = 0` flat lowering for it); carry the medium coefficients.

## 2. Material type + medium registry plumbing

- [ ] 2.1 `bindings.slang`: add `MATERIAL_TYPE_SUBSURFACE`; a `Medium` registry
  struct `(σ_a, σ_s, g, eta, uint kind, uint gridHandle, sigmaMaxScale)` + a
  registry SSBO referenced by a **medium handle** (NOT hardwired material-indexed —
  the handle indirection is what lets free-standing `MediumInterface` media reuse
  it later). Define `MEDIUM_HOMOGENEOUS` (+ reserve the enum for `MEDIUM_NANOVDB`).
  Document in the binding map.
- [ ] 2.2 `renderer.py`: pack the `Medium` registry; resolve the subsurface
  material's medium handle; tag the new type in `materialTypes[]`; add the medium
  fields to `_current_state_hash`.
- [ ] 2.3 Struct-layout test for `Medium` (scalar/Metal byte parity), incl. the
  density-handle + majorant fields (present, `none`/`σ_t` for homogeneous).

## 3. Volume random walk (null-collision) + density seam

- [ ] 3.0 Density seam (the additive-extension contract): `densityAt(Medium m,
  float3 pWorld) -> float` and `mediumMajorant(Medium m, float3 pA, float3 pB)
  -> float3`, dispatched by `m.kind` (switch idiom, like `flatSampleLobe`). Implement
  `MEDIUM_HOMOGENEOUS` (densityAt ≡ 1, majorant = σ_a+σ_s). The walk SHALL read the
  medium ONLY through these two — adding a kind = one enum + two `case` bodies.
- [ ] 3.1 `mediumWalk(rayIn, medium, boundaryMode, rng)` (new shader module):
  cross boundary by mode (*dielectric* Fresnel refract for subsurface; *index-
  matched* AABB pass-through stub for the free-standing case) → **null-collision
  (Woodcock) loop**: `σ_max = mediumMajorant(...)`; tentative collision, local
  `σ_t = (σ_a+σ_s)·densityAt(m, p)` **via the 3.0 seam**, accept real scatter w/
  prob σ_t/σ_max (HG `g`, throughput σ_s/σ_t) else null → boundary hit applies the
  mode (Fresnel split vs pass-through). Russian roulette; bounded (no clamp); exit
  applies the η² refraction radiance scaling (dielectric mode). The walk references
  the medium ONLY through `densityAt`/`mediumMajorant`.
- [ ] 3.2 Reuse `volume_render.slang` (HG phase/sampling) + `flat_shading.slang`
  Fresnel; throughput float3, pdf scalar. **No closed-form homogeneous-only
  transmittance path** (would block heterogeneous reuse) — homogeneous is the
  constant-density degenerate of the null-collision loop.
- [ ] 3.3 Factor the null-collision **segment traversal** (the loop over
  `densityAt`/`mediumMajorant`/HG/RR for one ray segment through a medium) as a
  standalone function that `mediumWalk` calls — separate from the "subsurface hit
  triggers it" caller. This is what lets a later free-standing-media change reuse
  the exact segment transport from a path-loop current-medium step (cloud+dragon)
  without touching the equations. No current-medium tracking implemented here.

## 4. Integrator wiring

- [ ] 4.1 Megakernel: dispatch `MATERIAL_TYPE_SUBSURFACE` → `subsurfaceWalk` in the
  `main_pass.slang` material switch.
- [ ] 4.2 Wavefront: wire the walk into `integrators/path.slang` and
  `integrators/bdpt.slang` (mirror the skin-BSSRDF wiring).
- [ ] 4.3 Recompile SPIR-V + confirm the Metal in-process compile (watch the
  guarded_metal.sh RAM floor — third heavy material path).

## 5. Verification

- [ ] 5.1 Furnace / energy: homogeneous SSS sphere, `σ_a → 0`, constant env →
  ~unity; bounded throughput, no firefly, no clamp.
- [ ] 5.2 PT ≡ BDPT on the SSS sphere, both execution modes.
- [ ] 5.3 Metal ↔ Vulkan parity on the SSS sphere.
- [ ] 5.4 Back-compat: pbrt parity corpus + a true `dielectric` glass scene
  byte/relMSE-unchanged (flat path untouched).
- [ ] 5.5 Forward-compat guard (Disney cloud): assert the walk takes a `σ_max`
  majorant + a `boundaryMode` and resolves the medium via a handle — i.e. the
  homogeneous case runs through the null-collision path (a unit/shader check that
  setting a uniform density field equal to the constant σ_t yields the identical
  result), proving a grid lookup would slot in. No NanoVDB/MediumInterface impl —
  just the abstraction is in place and exercised.

## 6. pbrt parity gate + docs

- [ ] 6.1 Add a subsurface parity scene (reduced `sssdragon` Skin1, or a simpler
  SSS sphere) + a pbrt v4 reference EXR; gate relMSE/FLIP and assert it is milky
  (differs from the pre-change glass render in the expected direction). Show the
  pbrt-vs-skinny image at shared exposure (note env-intensity is out of scope).
- [ ] 6.2 Docs: `docs/SkinRendering.md` or a new subsurface doc — the volumetric
  random-walk model + coefficient derivation; `docs/Architecture.md` binding map
  (`MediumParams`, `MATERIAL_TYPE_SUBSURFACE`); `docs/PbrtImport.md` (subsurface
  now volumetric, no longer glass); `README.md` compatibility matrix + `CHANGELOG.md`.
