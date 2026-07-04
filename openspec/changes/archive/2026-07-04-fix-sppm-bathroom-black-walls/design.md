# Design — restore the SPPM photon term (VisiblePoint ⊇ FlatHitMat)

## Context

SPPM composites `vp.ld` (eye-pass direct: NEE + specular-chain emitters) with a
photon-map indirect estimate resolved from per-pass deposited flux
(`wfSppmUpdate`). The deposit path is pbrt's store-the-BSDF approach: the eye
pass stores the evaluated material in the `VisiblePoint`
(`sppmStoreVisiblePoint`), and `sppmDepositPhoton` rebuilds a `FlatMaterial`
from those stored fields (`sppmLoadMaterial`) to evaluate f_r — no texture
refetch, no graph re-run.

Change `flat-lobes-rich-inputs` (af4ffb5, 2026-06-20) added three fields to
`FlatHitMat` — `transmissionColor`, `specularColor`, `diffuseRoughness` — that
the lobe model reads in `flatBsdfResponse` / `FlatMaterial.evaluate`
(`specularColor` tints the spec response; `diffuseRoughness` drives the
Oren-Nayar factor; `transmissionColor` weights transmission). PM-1's SPPM
landed from a branch authored the day before; `sppmLoadMaterial` was never
taught the new fields, so the rebuilt struct left them **undefined** on the
GPU. Every deposit evaluated to zero: GPU readback shows `tau == 0` at 100% of
visible points on bathroom AND cornell_box_sphere, while the photon count `m`
(incremented regardless of flux value) keeps growing and shrinking the radius.
The two MoltenVK-backed SPPM GPU regression tests that would have caught this
fail to compile on this machine (pre-existing, filed separately), so the
regression shipped silently.

Diagnostics that pinned it (scratchpad, GPU visible-point readback):
- VP albedo image: fully textured, walls 0.5 gray — graph/texture evaluation in
  the eye pass is CORRECT (the initially-suspected graph gap does not exist;
  `loadFlatMaterial` already dispatches `evalSceneGraphBaseColor`, and the
  per-scene `generated_materials` emit happens before the SPPM pass compiles).
- VP ld image: interior direct-only lighting — matches the black-wall pattern.
- tau image: identically zero — photon term dead.
- Setting the three fields to their documented defaults in `sppmLoadMaterial`
  flipped cornell tau from 0.0 to mean 8.51 (nonzero at 81% of VPs) and
  bathroom `black_frac` 0.35 → 0.00.

## Goals / Non-Goals

**Goals:**

- Photon deposit evaluates the *exact* BSDF the eye pass stored — including
  authored rich inputs (bathroom authors `specular_color` ×25,
  `transmission_color` ×16, so defaults-only rebuild would diverge from path).
- A structural guard so `FlatHitMat` can never again grow a field that the
  SPPM deposit silently drops.
- Honest parity-manifest records for bathroom SPPM.

**Non-Goals:**

- Environment-light photon emission (`sppmEmitPhoton` has no env group;
  env-lit-interior indirect stays a recorded follow-up — bathroom is
  bulb-dominated and now matches path to mean-ratio 1.007).
- The MoltenVK SPPM GPU-test compile failure, the `skinny-render` wavefront
  readiness gate, and the 10 pre-existing hostless failures (all filed as
  separate sessions).
- Skin/BSSRDF or volumetric photon transport (unchanged PM-1 scope).

## Decisions

### D1 — Store the rich inputs in the VisiblePoint (reject defaults-only)

`sppmLoadMaterial` could rebuild with the documented defaults
(`transmissionColor = albedo`, `specularColor = white`,
`diffuseRoughness = 0`) — 3 lines, no layout change, and it provably restores
the photon term. Rejected as the shipped fix because the deposit would then
evaluate a *different* BSDF than the eye pass / path tracer on any material
that authors the inputs (bathroom does, pervasively), reintroducing a quiet
sppm-vs-path bias. The `VisiblePoint` instead grows the three slots
(scalar 152→180 B, MSL 192→240 B); cost is ~25% on the largest SPPM buffer,
which stays far below the wavefront path-state footprint at equal resolution.

### D2 — Keep the struct-mirror discipline, add a completeness lock

The Slang struct, `wavefront_layout.VISIBLE_POINT_FIELDS`, and the stride
constants update in lockstep (existing parse-lock tests enforce equality).
Two NEW locks in `tests/test_sppm_state.py` enforce the *deposit contract*
itself: every `FlatHitMat` field (exempt list: `emission`, documented as
direct-not-BRDF) must (a) have a VP slot and (b) appear as `vp.<f> =` in
`sppmStoreVisiblePoint` and `m.<f> = vp.<f>` in `sppmLoadMaterial`. This turns
the exact failure mode of this bug into a hostless test failure.

### D3 — Manifest: update `measured`, never touch `baselines`

`sppm_vs_path` measured 64.97→2.59 relMSE (10.42→0.297 MSE). The pbrt-truth
`sppm|wavefront` measured number moves 0.364→1.71 — WORSE on paper because the
old number was an artifact (black walls contribute exactly 0 to relMSE where
the dark-pixel /b² amplification lives; a lit noisy wall scores worse than a
black one), and the path anchor itself currently measures 0.83 against its
0.342 record (pre-existing anchor drift, separate follow-up). Baselines stay
as-is (never raised); bathroom remains `known_divergent` → the matrix gate
xfails, and the manifest notes record the full story.

## Risks / Trade-offs

- [VP stride change breaks a stale consumer] → both backends size from
  `wavefront_layout`; Metal additionally asserts the reflected MSL stride
  equals the mirror (`MetalWavefrontSppmPass`), which was exercised in the GPU
  runs. No other consumer reads VP bytes host-side.
- [Eye kernel grew 3 stores → Metal watchdog] → kill harness re-run green
  (13 hostless + 3 gpu-marked).
- [Vulkan behavior unverified on this machine] → MoltenVK cannot compile the
  SPPM kernels at HEAD (pre-existing, filed); the change is
  backend-neutral Slang + a layout mirror both backends already derive from.
- [relMSE regression optics on `sppm|wavefront` pbrt-truth] → explained and
  recorded in the manifest notes; gate unaffected (known_divergent xfail).

## Migration Plan

Shader + host-mirror + test change. No persisted state, no API change. The
SPPM kernels compile at pipeline build on both backends (no checked-in .spv
touches SPPM). Rollback = revert.

## Open Questions

None — both open questions from the pre-investigation draft were resolved:
the wavefront graph seam already covers SPPM (per-scene `generated_materials`
emit precedes the pass compile), and bathroom's graphs drive `base_color`
only, so no graph-driven input lacks a VP slot after this change.
