## Context

The path tracer's emissive-triangle direct-lighting estimator combines two
sampling strategies — NEE and BSDF sampling — with MIS (change
`emissive-triangle-bsdf-mis`, living spec `emissive-light-sampling`). That fix
was scoped "Path + Metal only"; BDPT was left with its pre-fix emission handling.

BDPT builds a full MIS partition over many strategies. For a path
`camera → … → surface → light`, the strategies that can generate it are: `t = 0`
(the eye walk hits the light directly), `t = 1` (`connectT1` NEE from the last
eye surface), `t ≥ 2` (connect the eye subpath to a traced light-subpath vertex),
and `s = 1` (the light-tracer splat). BDPT already routes `connectT1`, the
generic connections, and the splat through one shared `misWeight()` so their
weights sum to 1 — the codebase explicitly moved `connectT1` off a standalone
2-strategy power heuristic for this reason (it ignored the `t≥2`/`s=1`
alternatives and ran ~2% bright).

The `t = 0` term never joined that partition. In all three transports
(`bdpt.slang` megakernel, `bdpt_spectral.slang` megakernel-spectral,
`wavefront/wavefront_bdpt.slang` staged) it carries a binary gate:

```
if (noNeePartner) L += throughput · emission;   // else: dropped
```

`noNeePartner` is `s == 2 || numEmissiveTriangles == 0 || eye[s-2].isDelta`. When
false, the emission is discarded — the BSDF-sampling strategy's share of the
direct-lighting estimate is lost, so BDPT reads ~3% dim vs the path tracer on
`mat_emissive` (0.9451 vs 0.9743 mean/pbrt; 0.1292 vs 0.0522 relMSE).

The three transports are hand-duplicated (the staged wavefront kernels cannot
call the megakernel struct method) but produce bit-identical RGB output
(`bdpt|megakernel` ≡ `bdpt|wavefront` at 0.1292), so the fix must land in all
three, identically.

## Goals / Non-Goals

**Goals:**
- Remove BDPT's ~3% direct/one-bounce area-light dim bias; converge to the path
  tracer's mean.
- Keep the estimator unbiased and the full MIS partition summing to 1.
- Preserve RGB `bdpt|megakernel ≡ bdpt|wavefront`; keep spectral tracking its RGB
  sibling.
- Lower the recorded pbrt-truth baselines (tighten-only) for the BDPT combos.

**Non-Goals:**
- SPPM and MLT (MLT is not on this branch) — separate transports, own follow-ups.
- Recompiling the Vulkan `main_pass.spv` (megakernel) — a Vulkan-side follow-up
  boundary, exactly as the path-tracer fix left it; Metal compiles Slang
  in-process and is the validation backend.
- Any change to `connectT1`, the generic connections, the splat, or the light
  distribution — only the `t = 0` weight changes.

## Decisions

### D1: Weight `t = 0` via the full `misWeight` partition, not a 2-strategy heuristic

The correct weight for the `t = 0` strategy is `misWeight(eye, s, ·, t = 0, …)`,
the same partition `connectT1` uses. Alternatives considered:

- **path.slang-style `powerHeuristic(bsdfPdf, pdfLightSA)`** (the 2-strategy
  BSDF-vs-NEE balance). Rejected: it only balances `t = 0` against `t = 1`,
  ignoring `t ≥ 2` and `s = 1`. Since `connectT1` already down-weights `t = 1` to
  account for all alternatives, adding a `t = 0` term from a *different* partition
  makes the sum ≠ 1 → residual bias. This is exactly the inconsistency the repo
  removed from `connectT1`.
- **Full `misWeight`** (chosen). Both estimators (path 2-strategy, BDPT
  many-strategy) are unbiased for the same integral, so their means agree; BDPT's
  `t = 0` weight is smaller than path's `wBSDF` but the `t ≥ 2`/`s = 1` strategies
  contribute the remainder. The partition sums to 1 and BDPT's mean matches path.

`misWeight` already supports `t = 0`: the `s + t == 2` guard returns 1 for the
`s == 2` primary hit, and the light-side ratio loop (`for i = t-1 … 0`) is empty
for `t = 0`, so only the eye-side ratios contribute — precisely the alternatives
that could have sampled the emitter vertex from the surface side.

### D2: Reconstruct the reverse pdfs index-free, mirroring `connectT1`

`misWeight`'s eye-side ratio at the emitter vertex is
`eyeC[i].pdfRev / eyeC[i].pdfFwd`. `pdfFwd` (the real BSDF-sampled area pdf that
reached the emitter) is already stored during `randomWalk`. The two `pdfRev`
values are injected as the `pdf_zsm1_rev` / `pdf_zsm2_rev` arguments:

- `z.pdfRev = Rec709-lum(z.emission) / emissiveTotalPower`. Under the
  power-weighted CDF `p_i = area_i·lum_i / ΣW` and uniform-area sampling
  (`pdfArea = p_i / area_i`), the per-triangle area cancels to `lum_i / ΣW` —
  the same cancellation the path tracer relies on, so no buffer index is needed.
  `samplePoint` reports `pdfArea = selectionPdf / triArea` with
  `selectionPdf = area·lum/ΣW`, confirming the reconstruction matches the NEE
  strategy's actual pdf.
- `eye[s-2].pdfRev = convertSAtoArea(cosOut/π, z.position, eye[s-2])`, the
  diffuse-emitter directional pdf — identical in form to `connectT1`'s
  `pdf_zsm1_rev = (cosLight/π)·NdotL/d²` but anchored at the emitter `z` rather
  than a separately sampled light point.

### D3: One shared helper, three call sites

Add `emitterHitMisWeightT0(BDPTVertex eye[BDPT_MAX_VERTS], int s)` in `bdpt.slang`
beside `misWeight`. It reads `eye[s-1].emission/N/position` and
`eye[s-2].position`, builds the two reverse pdfs, and returns
`misWeight(eye, s, eye /*unused for t=0*/, 0, pdf_zsm1_rev, pdf_zsm2_rev, 0, 0)`.
`wavefront_bdpt.slang` and `bdpt_spectral.slang` already `import integrators.bdpt`,
so the helper is visible to all three. The megakernel-spectral transport uses
`SpectralBDPTVertex`; it already builds an RGB `mirrorRgb(eye, eyeRgb)` for its
generic connections — that mirror is moved above the `t = 0` loop and passed to
the helper (the RGB projection carries the emission luminance the reconstruction
needs). The MIS weight itself is scalar (pdfs are wavelength-independent), so one
helper serves RGB and spectral.

### D4: Harness-first, tighten-only baselines

Re-measure `bdpt|megakernel`, `bdpt|wavefront` (+`|spectral`) on `mat_emissive`
and `mat_emissive_mtlx` on the Metal backend and record the measured relMSE/FLIP.
A baseline may only move **down** (toward the path anchor 0.0522). If any BDPT
baseline would rise, the fix is wrong — do not raise it. `sppm|*` is not touched
by this change and stays. The self-consistency anchor is `(Path, wavefront)`; BDPT
now tracks it more closely, so the self-consistency gate should also improve.

## Risks / Trade-offs

- **A back-facing or grazing emitter hit** (`cosOut → 0`) → `convertSAtoArea` uses
  `abs()` (matching how `pdfFwd` was computed at both endpoints), so the ratio
  stays finite and consistent; the dominant front-facing case is exact. Not a
  correctness risk for the estimator (MIS weights only affect variance split).
- **`emissiveTotalPower == 0` with an emissive material not in the light set** →
  guarded: the `numEmissiveTriangles == 0` full-weight branch already covers "no
  registered emissive NEE", and the helper divides only when `emissiveTotalPower
  > 0`. Same assumption the path tracer already makes.
- **Three duplicated call sites drift** → mitigated by the single shared helper;
  only the accumulation line differs per transport (float3 vs Spectrum). The
  parity harness's `bdpt|megakernel ≡ bdpt|wavefront` self-consistency gate
  catches any divergence.
- **Vulkan megakernel `main_pass.spv` left stale** → intentional follow-up
  boundary (Metal validates in-process); the Vulkan recompile is a mechanical
  follow-up, exactly as the path-tracer fix handled it.
- **Textured emitters** (codex review, P2) → the helper reloads the per-uv
  emission (`bdptSurface`), but the host NEE CDF / `emissiveTotalPower` are built
  from the constant `emissiveColor`. For an emitter with an emission texture
  (`emissiveTextureIdx`, `flat_shading.slang`) the reconstructed `pdf_zsm1_rev`
  is not exactly `connectT1`'s sampling pdf → a small MIS-weight error on
  textured area lights. **Kept as-is deliberately:** this mirrors the path
  tracer's `pdfLightSA` (`path.slang`, also per-uv), so `BDPT ≡ path` is
  preserved; using the constant `emissiveColor` here would *break* that parity
  invariant on textured emitters. It is exact for every constant emitter (all
  gated scenes have constant emitters), and a consistent fix (texture-aware NEE
  + per-triangle power in the host CDF, applied to path + bdpt together) is a
  separate cross-integrator change. Marked with a `ponytail:` note at the helper.

## Migration Plan

1. Add the shared helper and patch the three `t = 0` branches.
2. Recompile the affected wavefront kernels' `.spv` if `slangc` is available;
   otherwise note the recompile boundary. Metal needs no precompiled `.spv`.
3. Re-measure the four (×2 scenes) BDPT baselines on Metal; lower them in the
   manifest. Run the `mat_emissive` matrix gate and the hostless suite.
4. Rollback: revert the three shader hunks + the helper + the manifest baseline
   edits (single commit).

## Open Questions

- None blocking. The Vulkan `main_pass.spv` recompile timing (this change vs a
  batched shader-rebuild follow-up) mirrors the path-tracer fix's precedent and
  is not a correctness question.
