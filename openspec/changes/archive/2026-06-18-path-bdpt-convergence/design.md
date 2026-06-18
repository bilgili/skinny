## Context

`shaders/integrators/path.slang` is the unidirectional path tracer (shared by the
megakernel and, via `evaluateBounce`/`nee.slang`, the wavefront shade);
`bdpt.slang` is the bidirectional tracer. Both estimate the same rendering
equation and must converge to the same image. On `glass_arealight` they don't:
the path tracer misses the caustic and the area-light reflection on a smooth
dielectric sphere (measured mean 82.5 vs BDPT 110), so the path tracer is biased.

The bias is the emissive-triangle accounting in `path.slang`:

```slang
radiance += throughput * br.fullRadiance;
if (bounce == 0u || fc.numEmissiveTriangles == 0u)   // line 171
    radiance += throughput * br.bsdfSample.emission;
radiance += throughput * br.directLight;             // NEE
```

When emissive triangles exist, a BSDF ray that lands on the area light adds its
emission only at the primary bounce; otherwise the area light is supposed to come
in via `directLight` (NEE). NEE samples a point on the light and evaluates the
surface BSDF toward it — which is **zero for a delta lobe** (a smooth dielectric's
mirror reflect/refract). So a specular reflection/refraction that hits the area
light contributes through **neither** path → dropped → bias. The sphere-light
branch lower in the same loop already handles this (`w_bsdf = 1` when
`pdf == 0 || transmitted`); `bdpt.slang` flags `deltaBounce = (pdf <= 0)`. Only the
emissive-triangle emission line is missing the rule.

## Goals / Non-Goals

**Goals:**
- Make the path tracer **unbiased** for specular→area-light transport so it
  converges to BDPT (and pbrt) on `glass_arealight` — caustic and area-light
  reflection appear, mean energy matches.
- Keep non-specular transport bit-unchanged (no regression on diffuse/glossy).
- Apply the fix to every path-integrator surface: `path.slang`,
  `path_record.slang`, and the wavefront path.
- Lock it with an A/B path-vs-BDPT convergence regression test.

**Non-Goals:**
- Caustic **variance** — the path tracer stays noisier than BDPT/SPPM on caustics;
  converging slowly (same expected image) is the contract, not equal noise.
- Adding SPPM or full BSDF↔light MIS for non-delta area-light hits (optional
  variance reduction, separate change).
- Any change to BDPT (it is the correct reference) or to the pbrt importer.

## Decisions

### D1 — Gate emissive-triangle emission on the spawning bounce's delta-ness
Carry a loop variable `spawnedBySpecular` (false for the primary ray). After each
bounce's BSDF sample, set `spawnedBySpecular = (br.bsdfSample.pdf <= 0.0)` — the
same delta test the sphere-light branch and `bdpt.slang` use (a smooth
dielectric's reflect and refract are both `pdf == 0`). Change the emission line to:

```slang
if (bounce == 0u || fc.numEmissiveTriangles == 0u || spawnedBySpecular)
    radiance += throughput * br.bsdfSample.emission;
```

A delta bounce that lands on an emissive triangle now contributes its emission at
full weight — the missing specular reflection/refraction of the area light, and
the specular leg of the caustic path (floor → BSDF ray → glass refract → light).

### D2 — No double-count: delta vs non-delta is a clean partition
NEE at a surface samples the area light **only when that surface's lobe is
non-delta** (a delta lobe yields zero BSDF response toward the sampled point). So:
spawning bounce non-delta → NEE already counted the light, BSDF emission stays
skipped (unchanged); spawning bounce delta → NEE counted nothing, BSDF emission is
added. Mutually exclusive → unbiased, no MIS weight needed (a delta lobe has no
NEE partner, so the BSDF estimate owns the path at weight 1). This mirrors the
sphere-light branch exactly.

### D3 — Apply to all three path surfaces
`path.slang` (megakernel) and `path_record.slang:89` are textually identical at
the bug site — same edit. The **wavefront** path showed the identical mean (82.5)
and bias, so its emission accumulation needs the same delta-gated rule; locate it
in the wavefront shade/accumulate stage and apply D1 with the same delta flag
carried through the wavefront path state. Recompile `main_pass.spv`.

### D4 — Reference and verification via the pbrt reference (NOT BDPT)
**Superseded during apply.** The original plan used BDPT as ground truth. Apply
measurement disproved it: skinny's BDPT renders ~1.7× brighter than the pbrt v4
reference on BOTH `glass_arealight` AND the purely diffuse `diffuse_arealight`
(no delta bounces), so the over-brightness is a BDPT normalization bug, not the
specular caustic gap — BDPT cannot anchor the gate. The path tracer, by contrast,
matches pbrt closely (glass FLIP 0.058 → 0.025 across the fix; diffuse FLIP
0.020). So the regression renders `glass_arealight` with the **path tracer** and
asserts exposure-aligned relMSE / FLIP versus the checked-in **pbrt reference
EXR** are within the corpus manifest tolerance (the same metric the pbrt parity
gate uses). The fix is what moves it from failing (FLIP 0.058) to passing (FLIP
0.025). The harness is the pbrt parity render path (`parity.render_ab` /
`render_linear`, `read_accumulation_hdr`, default key light disabled, env off).

**Follow-up (out of scope here):** BDPT's ~1.7× over-brightness vs pbrt is a real,
separate bug (visible on a diffuse scene, independent of this delta fix). Tracked
as its own change; this one does not touch BDPT.

## Risks / Trade-offs

- **Double-counting → energy too high** if the delta test is wrong → mitigate by
  the strict delta/non-delta partition (D2) and a furnace/white-room energy check;
  the A/B mean-ratio assertion catches over- or under-counting.
- **Wavefront emission site differs** from the megakernel loop (NEE/emission may
  live in a separate kernel) → the wavefront fix must add the delta-gated emission
  in the correct stage; the wavefront A/B test (path vs bdpt, wavefront) catches a
  miss.
- **Rough transmission** (`pdf > 0` refraction) still relies on NEE, which samples
  the upper hemisphere and can miss a light behind a rough dielectric — a separate,
  lesser variance/bias issue; smooth dielectric (the glass case) is `pdf == 0` and
  fully covered. Documented as a follow-up.
- **`main_pass.spv` staleness** — the checked-in SPIR-V must be regenerated; the
  headless/Vulkan path runtime-compiles from source, but commit the recompiled
  `.spv` so non-recompiling consumers get the fix.

## Migration Plan

Pure correctness fix, no flag. Order: (1) `path.slang` + `path_record.slang` edit
+ recompile; (2) A/B convergence test (megakernel path vs BDPT) — should pass;
(3) locate + fix the wavefront path emission site; (4) wavefront A/B test; (5)
diffuse-unchanged regression; (6) docs + `main_pass.spv` commit.

## Open Questions

- Add full BSDF↔light MIS for **non-delta** area-light hits at bounce > 0 (power
  heuristic, as the sphere-light branch does) to cut glossy-near-light variance?
  Out of scope for convergence; worthwhile follow-up.
- Exact wavefront emission-accumulation site — to be located during apply.
