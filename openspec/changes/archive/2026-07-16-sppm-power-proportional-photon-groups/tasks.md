# Tasks — sppm-power-proportional-photon-groups

## 1. Host: group powers + pmf

- [x] 1.1 `environment.build_env_distribution` returns the sinθ-weighted luminance
      grid total as a third value (∫L dω = total·(π/H)·(2π/W)); update both call
      sites; `_ensure_env_uploaded` caches `self._env_lum_integral`.
- [x] 1.2 `_upload_sphere_lights` caches `self._sphere_power_sum = Σ lum(rad)·r²`
      over the enabled, capacity-clamped set; `_upload_distant_lights` caches
      `self._distant_lum_sum = Σ lum(rad)` likewise. Initialize both (and the env
      integral) to 0.0 at construction.
- [x] 1.3 Pure helper `_sppm_photon_group_pmf(powers, present)` (module-level,
      renderer.py): clamp non-finite/negative to 0, zero absent groups, normalise;
      uniform-over-present fallback when the sum is ≤ 0 or non-finite.
- [x] 1.4 `_pack_uniforms`: in the SPPM branch compute the four powers
      (Φ_E = π·`_emissive_total_power`; Φ_S = 4π²·sphere sum; Φ_D = πR²·distant sum;
      Φ_env = πR²·envIntensity·env integral, R = max(0.5·‖ext‖, 1e-4) from
      `_neural_scene_bounds`), presence-gate with the shader predicates, call the
      helper; pack the 4 pmf floats after `filmMaxComponent`, before the trailing
      `tileOriginY` u32 (zeros when integrator ≠ SPPM). Presence gates read the
      *packed* counts (the same attrs that feed fc.numSphereLights etc.) so host
      and shader predicates agree byte-for-byte.
- [x] 1.5 Probe hook: `_sppm_group_pmf_override` (None default; 4-tuple packs
      verbatim) so the forced-env flux probe stays expressible (pmf [0,0,0,1]).

## 2. Shader

- [x] 2.1 `common.slang`: append `float sppmGroupPmfE/S/D/Env` to `FrameConstants`
      after `filmMaxComponent`, before the `SKINNY_METAL`-gated `tileOriginY`;
      document the packer-order contract.
- [x] 2.2 `wavefront_sppm.slang` `sppmEmitPhoton`: replace the uniform pick with a
      CDF walk over the four fc pmf entries (last positive group absorbs the float
      residual; return false when all ≤ 0); each branch divides by its own `p_g`
      instead of `gsel`. Keep all geometry/flux/guards unchanged.
- [x] 2.3 Recompile all shaders (fc struct change: megakernel `main_pass.spv` via
      slangc + wavefront kernel set per the repo build path). Verify compile clean.

## 3. Tests (hostless)

- [x] 3.1 pmf helper units: proportionality, single-group ⇒ 1, absent ⇒ 0,
      all-zero ⇒ uniform-over-present, non-finite input ⇒ fallback.
- [x] 3.2 env-integral formula unit (synthetic constant map: ∫L dω → 4π·L).
- [x] 3.3 Wiring guards (test_sppm_selection.py pattern): fc fields present in
      common.slang in order before `tileOriginY`; `_pack_uniforms` packs 4 pmf
      floats; `wavefront_sppm.slang` no longer divides by `1.0 / float(G)` uniform
      `gsel` and references the fc pmf fields.
- [x] 3.4 Run hostless suites: `tests/pbrt` non-gpu, `tests/test_metal_cleanup.py`
      hostless, full `.venv/bin/pytest` sweep clean.

## 4. GPU verification (Metal, guarded runner)

- [x] 4.1 A/B `glass_caustics_test.usda` 384² 48 spp: SPPM before (main) vs after
      vs path reference at matched spp; `compute_metrics` noise_sigma + relMSE +
      median-of-ratio (unbiasedness), **whole-image AND caustic-region (masked to
      the caustic footprint — whole-image is background-dominated and blind to a
      caustic regression)**. Show labelled side-by-side grid, shared tonemap.
      Gate: whole-image noise_sigma drops markedly; caustic-region noise_sigma
      does not regress vs the uniform baseline; median-of-ratio ~1.0. Pre-agreed
      remedy on caustic regression: p_min floor (0.05/present group) in the pmf
      helper, re-measure within this change.
- [x] 4.2 SPPM parity/suite gates: sppm scenes of tests/pbrt matrix + suite
      (Metal, guarded). Re-measure/record any legitimately shifted sppm baseline
      (must improve, never loosen self-consistency).
- [x] 4.3 Metal kill-harness gpu tests unaffected dispatch shape — hostless part
      mandatory, gpu-marked run if dispatch/kernel length judged touched (it is
      not; photon kernel unchanged in length).

## 5. Docs + ship

- [x] 5.1 Update docs/PhotonMapping.md (emission section), CHANGELOG.md, CLAUDE.md
      compatibility notes if wording mentions uniform group selection; regen
      embedded code blocks if a DOC-marked region changed.
- [x] 5.2 `openspec validate sppm-power-proportional-photon-groups`.
- [ ] 5.3 Codex pre-merge review (fallback review subagent if codex unavailable);
      fold findings; merge --no-ff to main; archive change; update memory.
