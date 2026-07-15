# sppm-vp-beta-resolve

## Why

The wavefront SPPM update stage resolves the per-pass photon flux as
`L_indirect = Φ / (N_emitted · π · r²)` without applying `vp.beta` — the eye
throughput from the camera to the visible point that `wfSppmEye` stores at
`sppmStoreVisiblePoint` (`sppm_state.slang` `VisiblePoint.beta`) and never
reads again. pbrt-v4 folds `vp.beta` into the flux at pass end
(`integrators.cpp:3148`, `Phi` is multiplied by the visible point's `beta`
when folded into τ / the pixel estimate).

The omission is invisible when the visible point is directly viewed
(`beta ≈ lensWeight ≈ 1`) — and ALSO through clear Fresnel-sampled delta glass,
where the sample weight cancels to exactly 1 (verified: bit-identical A/B on
`glass_caustics_test.usda`). It mis-counts the photon-map indirect term
whenever the eye chain is lossy or tinted (`beta ≠ 1`): tinted glass, rough
glossy chains, absorbing media. With `beta < 1` the un-multiplied photon term
is OVER-counted and un-tinted (verified: tinted-glass A/B — through-glass
region mean 0.00102 before vs 0.0007 bdpt reference; 0.00067 after). Every
`ld` (direct) add already multiplies `throughput`; the photon term is the only
radiance component that skips it.

## What Changes

One shader, one site: in `wfSppmUpdate` (`wavefront_sppm.slang`), multiply the
decoded per-pass flux by `vp.beta` in the per-pass radiance estimate
`lIndirect` (per-λ, inside the spectral resolve). The `sppmUpdate` τ fold is
deliberately left untouched: unlike pbrt (where τ IS the radiance
accumulator), this renderer's τ is write-only — radiance comes from the film
running mean, and the radius-reduction rule is flux-independent
(`ratio = nNew/(nOld+m·f)`, `sppm_state.slang`), so a τ multiply would be
inert (design-review finding). Applied at resolve, not at deposit, so:

- one multiply per pixel per pass instead of one per photon-deposit;
- the fixed-point `sppmAccum` deposits stay in raw photon-flux scale
  (multiplying tiny `beta`-scaled values at deposit would waste
  `SPPM_FLUX_FIXED_SCALE` quantization precision);
- per-λ coherence holds in spectral mode: `vp.beta` is stored at the same
  shared pass wavelengths (`sppmPassWavelengths()`, D5) the photons and the
  resolve use, so `beta ⊗ Φ` is a per-λ product at matching λ.

The RGB block in `wfSppmUpdate` loses its "textually verbatim / byte-identical
.spv" status from change `spectral-wavefront` — that guard certified the
*spectral* change was a no-op for RGB; this change intentionally alters RGB
behavior, so the RGB and spectral blocks change together and the stale comment
is updated.

## Impact

- Affected spec: `photon-mapping` (update-stage requirement wording).
- Affected code: `src/skinny/shaders/integrators/wavefront_sppm.slang`
  (`wfSppmUpdate`), recompiled wavefront kernel caches.
- No host/Python change; no binding change; megakernel untouched (SPPM is
  wavefront-only).
- Renders: no change for directly-viewed visible points or clear delta glass
  (weight ≡ 1); correctly attenuated/tinted photon-indirect term behind lossy
  or tinted eye chains (was over-bright and desaturated).
