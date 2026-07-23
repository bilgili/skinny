# Tasks — renderer-module-carveout

Ordering is load-bearing: group 2 (Stage B) must merge before the sibling
`reflection-owned-byte-layouts` rewrites the packer; group 1 (Stage A) lands
first as the smallest-blast-radius template. Each group is one PR, gated by
D3's bit-identity checks. GPU runs follow CLAUDE.md Metal dispatch hygiene
(guarded runner, one Metal process, render progress log).

## 1. Stage A — MLT chain-state module

- [ ] 1.1 Add `src/skinny/mlt_chain.py`: pure `next_seed(frame_index)`
      (crc32 formula moved verbatim from `Renderer._next_mlt_seed`),
      `iterations_per_frame(width, height, num_chains)`,
      `uniform_tail_active(integrator_index, is_metal, execution_mode_is_wavefront,
      pass_built)`, and the shared `run_bootstrap(mlt, *, seed,
      upload_uniforms, submit)` round-trip (seed → uniforms upload →
      bootstrap → readback → `mlt_bootstrap.resample_chain_seeds` → seed
      upload → init → publish `b` → uniforms re-upload).
- [ ] 1.2 Hostless tests (`tests/test_mlt_chain.py`, no GPU): pin exact
      `next_seed` integers for several `frame_index` values against the
      pre-carve-out formula; cover the tail-predicate truth table (Vulkan
      always-on at integrator 3, Metal gated on wavefront + pass built) and
      the iterations budget; drive `run_bootstrap` with a stub pass to assert
      call order and `b`/`seeded` publication.
- [ ] 1.3 Rewire `renderer.py`: `_next_mlt_seed`, `_mlt_uniform_tail_active`,
      `_mlt_iterations_per_frame` delegate to (or are replaced by calls into)
      `mlt_chain`; `_run_wavefront_mlt_bootstrap` and
      `_run_wavefront_mlt_bootstrap_metal` collapse onto
      `mlt_chain.run_bootstrap` with backend-supplied callables
      (`_submit_one_shot_compute` + scene-set lambdas on Vulkan; the pass's
      own submits on Metal). Delete the superseded bodies.
- [ ] 1.3a Re-point `tests/test_mlt_host.py` at the new authority: the
      `inspect.getsource(Renderer._next_mlt_seed)` assertion and the
      seed-independence checks (:214–226) move to `mlt_chain.next_seed`;
      relocate the seed-independence docstring note (renderer.py:2452) with
      the function. Cross-change note: this file is also touched by
      `param-registry-accumulation-reset` (which pledges to preserve that
      note) and `reflection-owned-byte-layouts` — whichever change lands
      second owns the note's pointer.
- [ ] 1.4 Gate: hostless suites green (`tests/test_mlt_chain.py` + existing
      MLT hostless tests); one MLT suite scene (`int_caustic`) rendered at
      equal budget bit-identical to pre-stage on Metal; parity-matrix MLT
      combos pass with unchanged measured values; no `src/skinny/shaders/`
      diff.
- [ ] 1.5 Docs: `docs/MetropolisLightTransport.md` host-orchestration section
      + `docs/Architecture.md` module map entry for `mlt_chain.py`.

## 2. Stage B — frame-constant derivation module

- [ ] 2.1 Catalogue `_pack_uniforms` side effects (`_sync_lens_buffer`,
      `_warn_neural_megakernel_once`, `_sppm_metal_photon_batch` stash, any
      others found) in a short comment block; these keep their exact call
      sites.
- [ ] 2.2 Golden-blob capture test first (same commit, red/green): snapshot
      `_pack_uniforms()` and `_pack_uniforms_msl()` bytes across the state
      matrix (lens on/off, detail maps on/off, each integrator, both
      execution modes, neural-on-megakernel strip case) and assert byte
      equality across the refactor. This is a **gpu-marked** test, not
      hostless: it constructs `Renderer`s, and execution mode is fixed per
      session, so it runs as two guarded processes (megakernel + wavefront)
      under the metal-dispatch-hygiene runner, one Metal process at a time.
- [ ] 2.3 Add `src/skinny/frame_derive.py`: pure `detail_flags(master,
      nrm_ok, rgh_ok, dsp_ok, baked)`, `film_half_height_world(va_mm,
      focal_mm, mm_per_unit, lens_active_count, lens_film_distance_world)`,
      `exposure_stops(exposure_ev, imaging_ratio)`,
      `fold_sampling_capabilities(mask, alpha, reuse_mode,
      execution_mode_is_wavefront) -> (mask, alpha, reuse_mode,
      neural_stripped)`. No dataclass bundle (design D2); camera inverses
      stay as the two `np.linalg.inv` lines unless a helper falls out free.
- [ ] 2.4 Hostless tests (`tests/test_frame_derive.py`): lens framing ratio
      on/off, missing-map masking in `detail_flags`, imaging-ratio fold edge
      (ratio ≤ 0), neural strip + renormalise incl. the neural-only →
      `{bsdf}` fallback and the empty-mask guard.
- [ ] 2.5 Rewire `_pack_uniforms` to call the pure functions at the existing
      append sites — no append statement moves (Metal MSL relocation table
      depends on append order); the warn-once fires off the returned
      `neural_stripped` flag at the same site.
- [ ] 2.6 Gate: golden-blob test green on both packing paths; parity matrix
      green with unchanged values; no shader diff. Ordering with
      `reflection-owned-byte-layouts` is soft (merge-conflict avoidance
      only); note the land order in both changes when they overlap in time.
- [ ] 2.7 Docs: `docs/Architecture.md` module map entry for
      `frame_derive.py`.

## 3. Stage C — wavefront pass-object seam

- [ ] 3.1 Move pass construction into per-backend factories:
      `vk_wavefront.build_pass(integrator, ...)` absorbs the bodies of
      `_ensure_wavefront_{path,bdpt,sppm,mlt}_pass` (including the MLT
      descriptor-set 52–57 rebind, relocated next to `WavefrontMltPass`);
      `metal_wavefront.build_pass(integrator, ...)` absorbs the `_metal`
      siblings. Rebuild-key computation moves verbatim (values unchanged).
      Enumerate and preserve every None-fallback gate exactly — behavior,
      not construction: `_ensure_wavefront_mlt_pass` returns None on
      `is_metal`, missing `compute_queue`, `_scene_bindings is None or
      descriptor_sets is None`, and missing `mlt_bindings`
      (renderer.py:2495–2502); SPPM's analogous unbuildable→None. A dropped
      gate turns megakernel-mode MLT selection from path-fallback into a
      crash. Re-point the `tests/test_mlt_host.py` greps for the MLT
      ensure/destroy/dispatch wiring at the factory/new call sites.
- [ ] 3.2 Collapse the renderer to one `_ensure_wavefront_pass(integrator)`
      (cache + key compare + factory call + destroy-on-key-change) and fold
      the 3 Metal wavefront frame-dispatch bodies (`_render_wavefront_metal`
      — shared by path/BDPT — plus the SPPM and MLT variants) into one
      parameterised by per-integrator kwargs (photons/first_frame/photon_batch for SPPM; iterations + the
      bootstrap round-trip for MLT), leaving `_record_wavefront_dispatch`
      (Vulkan) and the Metal encoder entry points separate per the
      metal-backend spec.
- [ ] 3.3 Key-equality unit tests: new key tuples equal the pre-carve-out
      values for representative states (reuse none↔ReSTIR, neural on/off,
      record mode, dims, spectral).
- [ ] 3.4 Gate: full parity-matrix wavefront sweep green on both backends
      (path/BDPT/SPPM/MLT, RGB + spectral) with unchanged measured values;
      runtime-toggle smoke per backend (none↔ReSTIR, neural on/off,
      integrator cycling) recorded in the PR; `tests/test_metal_cleanup.py`
      -m gpu under the guarded runner (context lifecycle touched); no shader
      diff.
- [ ] 3.5 Docs: `docs/Wavefront.md` pass-construction/seam wording;
      `docs/Architecture.md` module map.

## 4. Stage D — extraction pattern + follow-on ordering

- [ ] 4.1 Write the one-page extraction pattern in `docs/Architecture.md`
      (pure core → unchanged call sites → backend factories → bit-identity
      gate → one stage per PR) and the follow-on order: detail maps, gizmo
      overlay, USD live-edit — each a future OpenSpec change with its own
      gate.
- [ ] 4.2 Update CLAUDE.md architecture notes (renderer module carve-out
      state + new module pointers); sweep the remaining Markdown docs per the
      documentation-upkeep rule.
- [ ] 4.3 `openspec validate renderer-module-carveout` clean; record final
      renderer-resident `is_metal`/`_metal` site count (baseline 117) in the
      change notes.
