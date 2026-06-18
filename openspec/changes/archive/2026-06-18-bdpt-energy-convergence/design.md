# Design — BDPT absolute-energy convergence

## D1. Which strategies reach the measured signal

`read_accumulation_hdr` reads `accumBuffer`, written in `main_pass.slang` **before**
the light-tracer splat is composited (the splat is added only to a local copy for
the tonemapped display path). So the BDPT strategies that reach the gate are, per
eye path of `L` edges:

- `t = 0` — eye/BSDF subpath ends on an emissive triangle (`z.onLight` loop).
- `t = 1` — `connectT1`, NEE from the last eye vertex (power-heuristic weighted).
- `t ≥ 2` — `connectGeneric`, eye↔light-bounce connections with full `misWeight`.
- env / sphere-light escapes — handled in `randomWalk` with their own power
  heuristic (≈ 0 for the area-light corpus: no env when `env_off`, no sphere lights).

The `s = 1` splat is **not** in the measured buffer, so the user-hypothesised
"splat Q22.10 scaling" is not the gate's bug. For a 1-bounce direct path
(camera → diffuse → light, 3 vertices / 2 edges) only `t = 0` and `t = 1` apply:
`connectGeneric` needs ≥ 4 vertices. So the measured direct lighting is a clean
two-strategy estimate — and the two-strategy power heuristic is exact for it,
*provided both strategies are weighted*.

## D2. The double-count

`connectT1` weights NEE by `w_light = powerHeuristic(pdfLightSA, bsdfPdf)`,
expecting the BSDF-hit strategy to carry the complementary `w_bsdf`. But the
`t = 0` loop added the BSDF hit at weight `1.0`, not `w_bsdf`. So:

```
measured_direct = w_light·D (NEE) + 1.0·D (emissive hit) = (1 + w_light)·D
correct_direct  = w_light·D       + w_bsdf·D             = D        (w_light+w_bsdf=1)
```

Ratio `(1 + w_light)`. For a diffuse surface under an area light, light sampling
is the lower-variance strategy so `w_light ≈ 0.75`, giving the observed ×1.76.
The glass scene's ×1.49 is lower because its specular legs mix in differently —
but the diffuse scene (no delta transport) is the unambiguous proof it is a
direct-lighting double-count, not a specular bug.

## D3. The fix mirrors the path tracer, not full BDPT MIS

The path tracer (`path.slang`, the in-repo pbrt-converged reference) captures
emissive triangles via NEE only, gating the BSDF-sampled emission to full weight
only when there is no NEE partner:

```slang
if (bounce == 0u || fc.numEmissiveTriangles == 0u || spawnedBySpecular)
    radiance += throughput * br.bsdfSample.emission;
```

BDPT's `connectT1` already mirrors `path.slang`'s NEE (same per-triangle pick, same
`powerHeuristic`), so the minimal, provably-parity fix is to give the `t = 0`
emissive loop the **same gate**, translated to BDPT vertex bookkeeping:

- `bounce == 0` → `s == 2` (z is the first hit `eye[1]`; the camera is delta, so
  no NEE partner).
- `spawnedBySpecular` → `eye[s - 2].isDelta` (the delta flag is set on the vertex
  where the delta bounce was sampled, i.e. the origin of the ray that reached z).
- `numEmissiveTriangles == 0` → unchanged.

This makes BDPT's direct area-light estimate identical to the path tracer's in
expectation, so it converges to the same pbrt-matching mean. We deliberately do
**not** rewrite `connectT1` / the `t = 0` strategy onto the full `misWeight`
partition: for the corpus's direct-lighting-dominated scenes the two-strategy
heuristic is exact, the change is one gate per shader (low risk), and it keeps
BDPT byte-for-byte consistent with the path tracer it must match. A residual
indirect (`t ≥ 2`) over-count is bounded and well within the gate tolerance
(post-fix `mean(bdpt)/mean(path)` ≈ 1.00–1.01).

## D4. Why a new gate (and why absolute energy)

`align_exposure` scales the candidate by the least-squares scalar that best
matches the reference, so a *uniform* 1.76× error is divided straight out — the
corpus parity gate stayed green (relMSE ≈ 0.02) all along. The new gate compares
un-aligned mean energy. It carries two assertions: the headline
`mean(bdpt)/mean(ref)` (absolute), and the sharper `mean(bdpt)/mean(path)` which
strips the shared path-tracer-vs-spectral-pbrt ≈ 0.87× offset so the BDPT-specific
bias can be gated tightly (≈ 2.0 pre-fix → ≈ 1.0 post-fix).

## D5. Measurements (128², native Metal)

| scene / mode | spp | path ×ref | bdpt ×ref pre | bdpt ×ref post | bdpt/path post |
|--------------|-----|-----------|---------------|----------------|----------------|
| diffuse / megakernel | 256 | 0.867 | **1.756** | 0.876 | 1.01 |
| diffuse / wavefront  | 256 | 0.867 | (same bug) | 0.875 | 1.01 |
| glass / megakernel   | 512 | 0.875 | **1.49**  | 0.874 | 1.00 |

Vulkan (MoltenVK) megakernel diffuse matches Metal exactly (bdpt ×0.876),
confirming the fix is backend-agnostic. Raw (un-aligned) relMSE vs pbrt fell
0.346 → 0.017 on diffuse.
