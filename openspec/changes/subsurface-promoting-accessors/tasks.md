# Tasks: subsurface-promoting-accessors

Groups 1–2 are prerequisites: D1 is unsafe until `get_spectrum_texture` notes.

## 1. Record the defect

- [x] 1.1 Capture the byte-identity baseline: import every parity-corpus and
      confirming-suite scene on both flavours, hash the `.usda` / `.mtlx` output.
      Run it in ONE worktree — the emitted `.usda` embeds absolute asset paths,
      so a texture-bearing scene hashes differently elsewhere.
- [x] 1.2 Write the structural gate as a FAILING test: an AST check that
      `resolve_material` and its helpers call no `float()`-on-token `ParamSet`
      accessor. Reuse `_reads_in` from `tests/pbrt/test_material_resolve.py`.
      Expect it to fail on the `subsurface` branch only.
- [x] 1.3 Write the silent-degradation tests, all FAILING today:
      a named spectrum on a spectrum lane produces no note (`status == EXACT`);
      `"spectrum reflectance" "metal-Au-eta"` silently yields gold's RGB;
      `"blackbody sigma_a" [6500]` yields `[6500, 6500, 6500]`;
      `"spectrum sigma_a" [400 .1 700 .9]` yields `[400.0, 0.1, 700.0]`.
- [x] 1.4 Write the precedence tests: a texture-bound `sigma_a` with a numeric
      `sigma_s` selects the explicit-sigma branch AND degrades both members.
      Assert the branch and both values, not that the import survives.

## 2. `get_spectrum_texture` gains a note path (prerequisite for group 3)

- [x] 2.1 Note every degradation: unrecognised name, spectrum file reference, and
      a resolvable texture whose value the target drops. Reuse
      `spectra.looks_like_spectrum_file` and the wording from
      `_named_spectrum_scalar`. A recognised name on a valid lane stays unnoted.
- [x] 2.2 Make the substitution lane-aware — the spectrum-side mirror of
      `_IOR_PARAM_NAMES`. A named metal's reflectance is legal on a
      reflectance-like lane, and degrades with a note on a coefficient lane.
- [x] 2.3 Confirm zero corpus drift: the only named-spectrum material parameter
      in the corpus is `all_mtypes.pbrt:83`, a recognised name on a reflectance
      lane. If any other scene drifts, stop and re-measure before continuing.

## 3. The subsurface branch joins the promoting layer

- [x] 3.1 `sigma_a`, `sigma_s`, `reflectance`, `mfp` read through
      `get_spectrum_texture`; `g` and `scale` through `get_float_texture`.
- [x] 3.2 Presence comes from `p.get(<name>) is not None`, value from the
      accessor — two separate reads (design D3).
- [x] 3.3 σ pair coherence: if either member is unusable, both degrade to the
      default pair with one note (design D4).
- [x] 3.4 Correct the `_IOR_PARAM_NAMES` comment — it claims every other float
      parameter through `get_float_texture` is a roughness; `g` and `scale` make
      that false. Record that accepting a texture there is deliberately more
      permissive than pbrt, which `ErrorExit`s.

## 4. One resolution for `reflectance`

- [x] 4.1 Resolve `reflectance` once, outside the mtlx flavour gate; feed both
      the `subsurface_color` lobe and the coefficient chain.
- [x] 4.2 Use `(0.5, 0.5, 0.5)` as the present-but-unusable default (design D8).
      The lobe keeps `[1.0, 1.0, 1.0]` for an ABSENT `reflectance` — a different
      question. No test can catch this choice; it must be made deliberately.
- [x] 4.3 Note order is pinned by construction rather than by a separate test:
      `_resolve_medium_colour` is called at each coefficient's own resolution
      site, so a note lands where the read happens. One note per binding is
      asserted directly (the accessor's note is suppressed for a texture on a
      medium lane, where the "cannot vary" note is the real reason).

## 5. Delete the phantom `radius` read

- [x] 5.1 Remove the `radius` read: pbrt's `SubsurfaceMaterial::Create` has no
      such parameter. Derive `subsurface_radius` from the resolved `mfp`, which
      is the same physical quantity.
- [x] 5.2 Confirmed: no corpus or suite scene authors a subsurface `radius` (the
      corpus hit was `subsurface_infinite.pbrt:19`, a `Shape "sphere" "float
      radius"` — a different parameter). Two tests encoded the phantom read and
      were updated to the derived-from-`mfp` contract.

## 6. Skipped vs approximate

- [x] 6.1 A texture-bound coefficient records `SKIPPED`; a spectral reduction
      records `APPROX` (design D5).
- [x] 6.2 Confirm no corpus scene starts failing
      `test_corpus_scene_imports_cleanly`, which asserts `report.count("skipped")
      == 0`.

## 7. Gates

- [x] 7.1 The structural gate from 1.2 passes.
- [x] 7.2 Every test from 1.3 and 1.4 passes.
- [x] 7.3 Add fixtures with recorded values for the `blackbody` and
      inline-sampled-spectrum forms. The corpus contains neither, so its hash
      gate cannot see the fix.
- [x] 7.4 Byte-identical corpus check against the 1.1 baseline: 75 of 76
      identical. Only the `all_mtypes.pbrt` fixture moved, on both flavours —
      it authors a `radius` (group 5), unrecognised named spectra (the widened
      status escalation), and gained the two new non-numeric cases (7.3). It has
      no test consumers.
- [x] 7.5 `tests/pbrt`: 843 passed, 16 failed — diffed against the pre-change
      run, the SAME 16 (5 corpus scenes whose assets live only in the primary
      checkout, and an absolute-vs-relative `.mtlx` path assertion). No new
      failure. `ruff check src/skinny` clean.
- [x] 7.6 Confirmed: the reduction does NOT clamp. An inline sampled spectrum
      with values ~50-80 reduces to ~49-88, so an unbounded extinction
      coefficient survives it. No limit to record.

## 8. Documentation

- [x] 8.1 Remove `KnownBugs.md` item 3. Correct item 2: `reflectance` and
      `radius` leave the frozen-divergence list.
- [x] 8.2 `docs/PbrtImport.md` records the promoting layer's two rules, the
      presence-vs-readability split, and skipped-vs-approximate.
- [x] 8.3 `openspec validate --strict`; pre-merge review.

## Notes

`KnownBugs.md` item 3 names six parameters. The measured surface is **seven** —
it misses `radius`. Group 5 deletes that one rather than hardening it, so the
change fixes six and removes one.

## Implementation notes

- `_resolve_medium_colour` became the one owner for a medium coefficient's
  resolution AND its note. The first cut had the accessor and the branch each
  noting a texture binding, which produced two notes for one binding and broke
  the spec's "exactly one note".
- The status escalation widened from `"unresolved/unsupported"` to `"used
  default"`, so a named-spectrum degradation escalates EXACT→APPROX like a
  texture one. Safe: the only scenes carrying an unrecognised name are in
  `all_mtypes.pbrt`, which has no test consumers.
- `subsurface_radius` is derived from the resolved `mfp` — the same physical
  quantity — rather than dropped, so the mtlx lobe still exists.

## 9. Pre-merge self-review fold

- [x] 9.1 The AST gate covered `resolve_material` and
      `subsurface_medium_overrides` only. `_resolve_medium_colour` — the function
      that actually resolves every medium coefficient — was NOT checked, so a raw
      read reintroduced there would have passed. The gate now covers it and both
      promoting accessors.
- [x] 9.2 Added a sensitivity control: the detector is run over a reconstructed
      pre-change read and must flag it. A gate that cannot fail is decoration.
- [x] 9.3 The widened `used default` escalation reaches EVERY material type, not
      just subsurface — `_named_spectrum_scalar` serves `eta` everywhere, so an
      unrecognised named eta on a `dielectric` now reports APPROX. Intended, but
      it was stated only obliquely; now pinned by a test and recorded in
      `docs/PbrtImport.md`.
- [x] 9.4 Verified no other consumer of `subsurface_radius` breaks: it is packed
      by `material_pack.py:473` and consumed by `mtlx_std_surface.slang` as the
      per-channel scattering radius — the same physical quantity as `mfp`, and
      the default is unchanged ([1,1,1], since MFP_DEFAULT is 1.0). No corpus or
      suite scene authors `mfp` on a subsurface material either
      (`subsurface_infinite.pbrt` authors only sigma_a/sigma_s), so nothing moves.
