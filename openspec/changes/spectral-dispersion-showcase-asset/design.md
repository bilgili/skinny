# Design — Spectral Dispersion Showcase Asset

## Context

Hero-wavelength dispersion is live (spectral-rendering spec, Group 6.4): the
importer preserves a named-glass key on `skinnyOverrides["glass_dispersion"]`,
`renderer.py` resolves it through `named_glass_cauchy()` into `glassCauchyB`
(packed into `_normalBiasPad.w`), and the spectral integrators evaluate
`n(λ) = A + B/λ_µm²` with pbrt secondary-wavelength termination. What exists
today for demonstration:

- `tests/assets/suite/spec_prism/` — a 128² gate scene (area-light slab over a
  BK7 prism) built for the parity matrix, not for looking at.
- `assets/glass_caustics_spectral.usda` — BK7 spheres; fringing is subtle.
- `_GLASS_CAUCHY = {"default": (1.5046, 0.00420), "bk7": (1.5046, 0.00420)}` —
  BK7 only. Crown glass B is small; at demo scale the fan barely separates.

Constraints: SPPM has no dispersion in v1 (hero-λ collapse would break photon/VP
coherence); delta glass occludes NEE shadow rays, so projected caustics under the
plain path tracer come only from BSDF-sampled paths (noisy); BDPT's light-side
splat renders projected dispersed caustics well.

## Goals / Non-Goals

**Goals:**
- One checked-in `.usda` under `assets/` that renders an unmistakable rainbow
  under `--spectral` and a gray/white fan under RGB — a single-command demo.
- A physically real high-dispersion glass in the named-Cauchy table so the
  effect is strong without faking physics.
- Keep it asset + table + hostless test; zero shader / binding / pipeline work.

**Non-Goals:**
- No SPPM dispersion, no new light types, no importer changes.
- No new parity-matrix scene or GPU gate (suite `spec_prism` already gates
  dispersion correctness); this asset is a showcase, not a metric scene.
- No exhaustive glass catalog — one flint entry is enough; more can ride later.

## Decisions

1. **Glass = SF11 (Schott dense flint), added to `_GLASS_CAUCHY` as `"sf11"`.**
   Cauchy fit from the Sellmeier curve over 400–700 nm gives approximately
   A ≈ 1.737, B ≈ 0.0166 µm² — ~4× BK7's B, so the visible-band IOR spread is
   ~0.047 (vs ~0.012 for BK7). Real glass, real fit (least-squares over the
   sampled Sellmeier curve at implementation time, mirroring how BK7's fit is
   stated). Alternative considered: exaggerating BK7's B or adding a synthetic
   "demo" glass — rejected; the table is spec-visible and should stay physical.
   Fallback semantics unchanged: unknown names still resolve to `"default"`.

2. **Scene = classic dark-room prism.** Triangular-cross-section prism
   (hand-authored mesh, ~6 verts, oriented near minimum deviation),
   SF11 delta glass (`UsdPreviewSurface` opacity 0, roughness 0, ior 1.78,
   `skinnyOverrides.glass_dispersion = "sf11"` — same authoring seam as
   `glass_caustics_spectral.usda`). The slit is **emissive mesh geometry**
   (a small bright `emissiveColor` quad strip), NOT a UsdLux analytic light:
   camera rays must be able to hit it directly (the through-prism cue) and the
   BDPT light-tracing splat is built around emissive triangles. A matte white
   screen/floor ~2–3 prism-heights past the exit face catches the fan (60° SF11
   prism spreads ~5–6° across the visible band → clearly separated bands at
   that throw). **TIR watch-out:** at n≈1.8 a 60° apex leaves only ~3° margin
   to the exit-face critical angle at the blue end — implementation must verify
   the blue band actually exits, and drop to a ~45° apex (≈10° margin) if the
   equilateral orientation proves fragile. Near-black room (diffuse ~0.02
   walls, no/negligible env) for contrast. Camera framed to see BOTH cues: the
   dispersed slit image seen *through* the prism (an eye-side refraction path
   the plain path tracer resolves cleanly) and the projected rainbow fan on the
   screen (light-side caustic, BDPT's splat territory). Rationale: two
   independent dispersion signatures in one frame; at least one is low-noise
   under every supported integrator.
   **Authored `ior 1.78` is deliberate and decoupled from Cauchy A:** under
   `--spectral` the renderer overwrites the scalar IOR with Cauchy A (≈1.737),
   so the authored value only drives the RGB comparison render. 1.78 ≈ SF11's
   d-line index (A + B/0.589²), so the RGB baseline bends by the same *mean*
   angle as the spectral hero-mean and the A/B isolates dispersion, not mean
   bending. The asset carries a comment saying so — do not "fix" it to 1.737.

3. **Authored directly as USD, no `.pbrt` source.** The suite generator exists
   for pbrt-truth gate scenes; this asset has no pbrt reference requirement.
   Direct authoring keeps it editable and avoids extending `_gen`.

4. **Recommended command is BDPT; path documented as supported.**
   `skinny-render --spectral --integrator bdpt assets/dispersion_prism.usda` is
   the headline demo (clean caustic fan). Path works (through-prism view clean,
   fan noisier). SPPM is an accepted `--spectral` combo but has no dispersion
   in v1 — it renders the scene without a rainbow; docs say exactly that
   (not "refused").

5. **Testing = hostless integrity, GPU visual check manual.** One test module
   asserts: asset opens via the USD loader, the prism material carries
   `glass_dispersion = "sf11"`, `named_glass_cauchy("sf11")` returns a fit with
   `B > named_glass_cauchy("bk7")[1]`, and `cauchy_ior` monotonically decreases
   over 400→700 nm for the entry. Verification renders (spectral vs RGB
   side-by-side, shared tonemap) are done during implementation on Metal and
   shown, but not added as a standing GPU gate — the parity suite already owns
   dispersion correctness.

## Risks / Trade-offs

- [Path-traced fan is noisy at low spp (delta glass blocks NEE)] → Camera
  framing guarantees the through-prism dispersed slit is visible regardless;
  README names BDPT as the demo integrator.
- [Slit too wide → bands overlap into washed-out white] → Keep slit width ≪
  the per-band displacement at the screen; tune during implementation renders.
- [Blue end TIRs at the exit face (60° apex ≈3° margin at n≈1.84)] → Verify
  the full 400–700 nm band exits during implementation renders; fall back to a
  ~45° apex if the equilateral orientation is fragile.
- [`"sf11"` name collides with future full Sellmeier import] → Table stores the
  Cauchy *fit* exactly like BK7 today; a later Sellmeier upgrade replaces fits
  wholesale, not this entry specially.
- [Asset drifts from loader expectations (past lesson: deleted baked HDRs broke
  scenes)] → No external file dependencies at all: geometry, light, and
  materials are self-contained in the one `.usda`; hostless integrity test loads
  it in CI.

## Migration Plan

Additive only: new asset file, new table entry, new test. No persisted-settings,
binding, or shader impact. Rollback = delete the three files.

## Open Questions

- Exact SF11 Cauchy coefficients: computed at implementation from the Sellmeier
  fit; recorded in the table docstring next to BK7's.
- Final camera/slit tuning is empirical — settled by the implementation renders.
