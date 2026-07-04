# Fix SPPM black walls (zero photon flux on bathroom.usda)

## Why

Rendering `assets/bathroom.usda` with the SPPM integrator produces black walls
while the path tracer renders the same scene correctly. Investigation (GPU
visible-point readback) found the SPPM **photon term is dead scene-wide on
every scene**, not just bathroom: `tau == 0` at 100% of visible points even on
`cornell_box_sphere.usda`, while the per-point photon *count* still grows.
Root cause: `sppmLoadMaterial` (`wavefront_sppm.slang`) rebuilds the
deposit-time `FlatMaterial` field-by-field from the `VisiblePoint`, and the
Stage-2 rich inputs added to `FlatHitMat` by change `flat-lobes-rich-inputs`
(**after** PM-1 shipped) — `transmissionColor`, `specularColor`,
`diffuseRoughness` — were never given VisiblePoint slots. The rebuilt material
feeds **undefined values** into `evaluate()` at every photon deposit, zeroing
the deposited flux (`beta·f = 0`) while `m` still increments. SPPM therefore
renders only its eye-pass direct term (`ld`); surfaces lit mostly indirectly —
the bathroom walls, ceiling, tub — go black. Bathroom is simply the scene
where the loss is unmissable, and the recorded `sppm_vs_path` relMSE 64.97 in
the parity manifest was this bug.

(The originally-suspected mechanism — graph materials not evaluated in the SPPM
eye pass — was disproved: `loadFlatMaterial` already dispatches
`evalSceneGraphBaseColor`, and the VP albedo readback shows correctly textured
albedos on every bathroom surface.)

## What Changes

- `VisiblePoint` grows three slots — `transmissionColor`, `specularColor`,
  `diffuseRoughness` — stored by the eye pass from the evaluated material and
  rebuilt by `sppmLoadMaterial` at photon deposit, restoring the store-the-BSDF
  contract (deposit evaluates the *exact* eye-pass BSDF; defaults-only rebuild
  was rejected because bathroom authors `specular_color` ×25 /
  `transmission_color` ×16). Stride: scalar 152→180 B, MSL 192→240 B
  (host mirror `wavefront_layout.VISIBLE_POINT_FIELDS` updated in lockstep).
- New hostless parse-lock tests: every `FlatHitMat` field (except documented
  `emission`) must have a VisiblePoint slot AND be written by
  `sppmStoreVisiblePoint` AND rebuilt by `sppmLoadMaterial` — a future
  `FlatHitMat` field can never silently kill the photon term again.
- Parity-manifest `measured` records for bathroom SPPM updated to post-fix
  truth: `sppm_vs_path` relMSE 64.97→2.59 (MSE 10.42→0.297, linear-mean ratio
  1.007); baselines untouched (never raised), scene stays `known_divergent`.
- The `photon-mapping` spec gains a requirement that the visible point mirrors
  the full deposit-relevant `FlatHitMat` surface, with the parse-lock as its
  scenario.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `photon-mapping`: the per-pass pipeline requirement is amended — the
  visible point SHALL store every `FlatHitMat` field the lobe model reads
  (evaluated, including graph-driven values) and the photon deposit SHALL
  rebuild the BSDF exclusively from those stored fields; a structural lock
  SHALL fail the build when `FlatHitMat` grows a field with no VP slot.

## Impact

- `src/skinny/shaders/integrators/sppm_state.slang` — `VisiblePoint` struct
  (+3 fields, stride docs).
- `src/skinny/shaders/integrators/wavefront_sppm.slang` —
  `sppmStoreVisiblePoint` + `sppmLoadMaterial`.
- `src/skinny/wavefront_layout.py` — `VISIBLE_POINT_FIELDS` mirror + stride
  comments (buffer sizing on both backends derives from it).
- `tests/test_sppm_state.py` — stride locks 152/192→180/240 + the two new
  FlatHitMat⊆VisiblePoint completeness locks.
- `tests/pbrt/corpus/manifest.json` — bathroom measured records + notes.
- Both backends share the Slang source; Metal reflects the new stride
  (`MetalWavefrontSppmPass` stride assert) and Vulkan sizes from
  `VISIBLE_POINT_STRIDE`. Metal kill harness re-run (eye kernel grew 3 stores).
- Out of scope, filed as separate follow-ups: `skinny-render` wavefront
  readiness-gate bug, MoltenVK SPPM GPU-test compile failure (why the energy
  gate never caught this), 10 pre-existing hostless test failures, no
  environment-light group in `sppmEmitPhoton` (env-lit-interior indirect).
