# Design — constant-spectrum-achromatic-rgb

## Context

`sampled_spectrum_to_rgb` (`src/skinny/pbrt/spectra.py:95`) interpolates a pbrt
`spectrum [l v l v …]` onto the internal 360–830 nm grid, integrates against
the Wyman analytic CIE CMFs, and maps XYZ → linear sRGB with the D65 matrix.
The docstring documents the equal-energy simplification for reflectance, but
the D65 matrix maps the equal-energy whitepoint (E) to a tinted RGB: a flat
SPD of value `v` becomes roughly `[1.20v, 0.95v, 0.91v]`. pbrt itself keeps
constant spectra achromatic (a `ConstantSpectrum` scales all channels equally
after its own spectral pipeline), so skinny's imported media coefficients and
constant reflectances are visibly red-shifted against pbrt ground truth.

Observed baked values:

- `assets/clouds.usda` (main checkout): `sigma_s = [12.002, 9.497, 9.084]`
  from `"spectrum sigma_s" [200 10 900 10]` — should be `[10, 10, 10]` (×unit
  scale as applicable).
- `assets/disney_cloud.usda`: `sigma_s = [4.801, 3.799, 3.633]` from a
  constant spectrum — same tint ratio.

Corpus scenes render from the **baked `.usda` assets** (manifest `usd:` field,
resolved against the repo root by `parity._scene_source`), so fixing
`spectra.py` alone does not change the parity render — the assets must be
re-imported.

## Goals / Non-Goals

**Goals:**

- Constant sampled spectra reduce to exact achromatic `[v, v, v]`.
- Colored spectra keep the existing XYZ→sRGB projection bit-for-bit.
- Regenerate baked cloud assets; re-measure `disney_cloud`/`bunny_cloud`
  pbrt-truth baselines (must improve); re-verify mega≡wave self-consistency.

**Non-Goals:**

- No change to the projection math for genuinely colored spectra (the residual
  RGB-vs-spectral divergence remains documented in the parity matrix).
- No chromatic-adaptation (Bradford/E→D65) rework — out of scope; the shortcut
  sidesteps the whitepoint question for the constant case only.
- No changes to `blackbody_rgb`, named conductor spectra, or shader code.

## Decisions

**D1 — Detect constancy on the input samples, not the interpolated grid.**
Check `np.all(val == val[0])` on the parsed `[l v]` pairs. If all authored
sample values are equal, the linearly-interpolated SPD is constant everywhere
(interpolation between equal values, `left=`/`right=` clamp to those same
values), so the input-side check is exact and cheaper than comparing 95 grid
values. Exact equality, not `isclose`: pbrt authors constant spectra with
literally identical values (`[200 10 900 10]`); a near-constant colored
spectrum should keep the projection.

**D2 — Return `[v, v, v]` for both reflectance and illuminant modes.**
An achromatic value has no chromatic content to project in either mode. For
`illuminant=True` the current path returns the integrated magnitude
(`∫v·ȳ dλ` scaled through the matrix) — but every call-site that passes
`illuminant=True` (lights `L`/`I`, area-light `L`) treats the result as a
color that multiplies a separate `scale`/intensity, and pbrt's own semantics
for a constant illuminant spectrum is "achromatic with this level". `[v, v, v]`
is correct for both.

**D3 — Regenerate baked assets rather than hand-editing the numbers.**
Re-run the importer (`import_pbrt`) on the pbrt sources
(`~/projects/pbrt-v4-scenes/{disney-cloud,bunny-cloud,clouds}`) to produce the
`.usda` assets, same as the original nanovdb change did. Hand-editing the
sigma arrays would drift from what a fresh import produces and would miss the
disney-cloud ground-plane reflectance (`"spectrum reflectance" [200 0.2 900
0.2]`, also constant, also currently tinted).

**D4 — Baselines re-measured with the standard harness, never hand-tuned.**
Re-render via the parity harness at the manifest's recorded resolution/spp and
update `measured` (and `notes`) from `metrics.compute_metrics` output.
Direction gate: new relMSE/FLIP must be ≤ the recorded values; if any metric
regresses, stop and diagnose — do not raise the baseline (standing rule).

## Risks / Trade-offs

- **[Asset/worktree split]** `.usda` cloud assets and pbrt sources live in the
  main checkout, not the worktree (worktree `.gitignore` is `*`). → Regenerate
  into the worktree `assets/` for the corpus pair (they are manifest-referenced
  repo paths), and update the main-checkout viewer assets (`clouds.usda`) in
  place; `git add -f` anything git-tracked.
- **[Hidden dependents]** Some other scene may rely on the tinted constant
  conversion. → Corpus sweep is the guard: run the full matrix gate, not just
  the volume scenes; hostless spectra/media/materials tests run first.
- **[Exact-equality misses]** Authors could write `[200 10.0 900 10.000001]`.
  → Accepted: that is a colored spectrum by construction; projection path
  handles it as today.
- **[mega≡wave]** Both modes consume the same imported RGB, so self-consistency
  should hold trivially; re-verify anyway because the baked assets change
  (EXACT 0.0 expected, as recorded in the manifest notes).

## Migration Plan

1. Land `spectra.py` shortcut + unit tests (hostless, fast).
2. Regenerate baked `.usda` assets from pbrt sources.
3. GPU sweep: re-render `disney_cloud`/`bunny_cloud` combos, compare to
   checked-in pbrt refs (refs themselves unchanged — pbrt truth is unaffected).
4. Update manifest `measured` + `notes`; run full parity gates.
5. Rollback = revert the commit; no persisted-state or schema migration.

## Open Questions

None blocking. (Whether to also re-import `bathroom`/`dragon` assets: no —
neither uses constant sampled spectra; leave untouched.)
