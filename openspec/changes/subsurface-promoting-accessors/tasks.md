# Tasks: subsurface-promoting-accessors

Groups 1–2 are prerequisites: D1 is unsafe until `get_spectrum_texture` notes.

## 1. Record the defect

- [ ] 1.1 Capture the byte-identity baseline: import every parity-corpus and
      confirming-suite scene on both flavours, hash the `.usda` / `.mtlx` output.
      Run it in ONE worktree — the emitted `.usda` embeds absolute asset paths,
      so a texture-bearing scene hashes differently elsewhere.
- [ ] 1.2 Write the structural gate as a FAILING test: an AST check that
      `resolve_material` and its helpers call no `float()`-on-token `ParamSet`
      accessor. Reuse `_reads_in` from `tests/pbrt/test_material_resolve.py`.
      Expect it to fail on the `subsurface` branch only.
- [ ] 1.3 Write the silent-degradation tests, all FAILING today:
      a named spectrum on a spectrum lane produces no note (`status == EXACT`);
      `"spectrum reflectance" "metal-Au-eta"` silently yields gold's RGB;
      `"blackbody sigma_a" [6500]` yields `[6500, 6500, 6500]`;
      `"spectrum sigma_a" [400 .1 700 .9]` yields `[400.0, 0.1, 700.0]`.
- [ ] 1.4 Write the precedence tests: a texture-bound `sigma_a` with a numeric
      `sigma_s` selects the explicit-sigma branch AND degrades both members.
      Assert the branch and both values, not that the import survives.

## 2. `get_spectrum_texture` gains a note path (prerequisite for group 3)

- [ ] 2.1 Note every degradation: unrecognised name, spectrum file reference, and
      a resolvable texture whose value the target drops. Reuse
      `spectra.looks_like_spectrum_file` and the wording from
      `_named_spectrum_scalar`. A recognised name on a valid lane stays unnoted.
- [ ] 2.2 Make the substitution lane-aware — the spectrum-side mirror of
      `_IOR_PARAM_NAMES`. A named metal's reflectance is legal on a
      reflectance-like lane, and degrades with a note on a coefficient lane.
- [ ] 2.3 Confirm zero corpus drift: the only named-spectrum material parameter
      in the corpus is `all_mtypes.pbrt:83`, a recognised name on a reflectance
      lane. If any other scene drifts, stop and re-measure before continuing.

## 3. The subsurface branch joins the promoting layer

- [ ] 3.1 `sigma_a`, `sigma_s`, `reflectance`, `mfp` read through
      `get_spectrum_texture`; `g` and `scale` through `get_float_texture`.
- [ ] 3.2 Presence comes from `p.get(<name>) is not None`, value from the
      accessor — two separate reads (design D3).
- [ ] 3.3 σ pair coherence: if either member is unusable, both degrade to the
      default pair with one note (design D4).
- [ ] 3.4 Correct the `_IOR_PARAM_NAMES` comment — it claims every other float
      parameter through `get_float_texture` is a roughness; `g` and `scale` make
      that false. Record that accepting a texture there is deliberately more
      permissive than pbrt, which `ErrorExit`s.

## 4. One resolution for `reflectance`

- [ ] 4.1 Resolve `reflectance` once, outside the mtlx flavour gate; feed both
      the `subsurface_color` lobe and the coefficient chain.
- [ ] 4.2 Use `(0.5, 0.5, 0.5)` as the present-but-unusable default (design D8).
      The lobe keeps `[1.0, 1.0, 1.0]` for an ABSENT `reflectance` — a different
      question. No test can catch this choice; it must be made deliberately.
- [ ] 4.3 Pin the subsurface note order on both flavours. Nothing pins it today —
      `test_notes_are_in_read_order` covers `conductor` and `coatedconductor`
      only, so a reordering currently lands green.

## 5. Delete the phantom `radius` read

- [ ] 5.1 Remove the `radius` read: pbrt's `SubsurfaceMaterial::Create` has no
      such parameter. Derive `subsurface_radius` from the resolved `mfp`, which
      is the same physical quantity.
- [ ] 5.2 Confirm the measured cost holds — no corpus or suite scene authors a
      subsurface `radius`; only `all_mtypes.pbrt` does, and it has no test
      consumers. If a parity scene turns out to author one, split this group out.

## 6. Skipped vs approximate

- [ ] 6.1 A texture-bound coefficient records `SKIPPED`; a spectral reduction
      records `APPROX` (design D5).
- [ ] 6.2 Confirm no corpus scene starts failing
      `test_corpus_scene_imports_cleanly`, which asserts `report.count("skipped")
      == 0`.

## 7. Gates

- [ ] 7.1 The structural gate from 1.2 passes.
- [ ] 7.2 Every test from 1.3 and 1.4 passes.
- [ ] 7.3 Add fixtures with recorded values for the `blackbody` and
      inline-sampled-spectrum forms. The corpus contains neither, so its hash
      gate cannot see the fix.
- [ ] 7.4 Byte-identical corpus check against the 1.1 baseline: zero drift except
      the `all_mtypes.pbrt` fixture, from group 5.
- [ ] 7.5 `tests/pbrt` hostless sweep green against the recorded pre-existing
      failures; `ruff check src/skinny`.
- [ ] 7.6 Confirm the albedo-shaped reduction is acceptable for an extinction
      coefficient, which pbrt types `SpectrumType::Unbounded`, or record the
      limit.

## 8. Documentation

- [ ] 8.1 Remove `KnownBugs.md` item 3. Correct item 2: `reflectance` and
      `radius` leave the frozen-divergence list.
- [ ] 8.2 `docs/PbrtImport.md` records the promoting layer's two rules, the
      presence-vs-readability split, and skipped-vs-approximate.
- [ ] 8.3 `openspec validate --strict`; pre-merge review.

## Notes

`KnownBugs.md` item 3 names six parameters. The measured surface is **seven** —
it misses `radius`. Group 5 deletes that one rather than hardening it, so the
change fixes six and removes one.
