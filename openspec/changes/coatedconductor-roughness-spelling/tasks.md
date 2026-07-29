# Tasks: coatedconductor-roughness-spelling

Group 1 comes first for a reason: the material is ungated today, so a fix landed
before the scene exists is a fix nothing protects.

## 1. Gate the material before fixing it

- [x] 1.1 Capture the byte-identity baseline: import every parity-corpus and
      confirming-suite scene on both flavours, hash the `.usda` / `.mtlx`.
      Run it in ONE worktree — the emitted `.usda` embeds absolute asset paths.
- [x] 1.2 Add `tests/assets/suite/mat_coated_metal/` via
      `tests/assets/suite/_gen/build.py`: a `coatedconductor` with a near-mirror
      metal (`conductor.roughness` 0.02) under a satin coat
      (`interface.roughness` 0.3), so reading the wrong parameter visibly blurs
      the reflection. Register it in `tests/pbrt/corpus/manifest.json` with a
      `_mtlx` sibling carrying an `equivalence` disposition.
- [x] 1.3 Generate its pbrt-truth EXR:
      `python tests/pbrt/regen_refs.py --scene mat_coated_metal --res 256 --spp <n>`
      (needs the pinned pbrt at `~/projects/pbrt-v4/build/pbrt`).
- [x] 1.4 Write the FAILING discrimination test: under both flavours the resolved
      metal roughness comes from `conductor.roughness`, and it is NOT equal to
      the coat's. Expect the `usd` flavour to fail today.
- [x] 1.5 Write the FAILING anisotropy test: `conductor.uroughness` /
      `conductor.vroughness` with different values resolve to an unreduced pair.
      Expect both flavours to fail today.

## 2. One calibration chain, parameterised by spelling

- [x] 2.1 `_resolve_roughness` takes a `prefix`, defaulting to `""`. It reads
      `{prefix}roughness` / `{prefix}uroughness` / `{prefix}vroughness`.
- [x] 2.2 `remaproughness` stays UNPREFIXED — pbrt reads one per material
      (`GetOneBool`), governing every roughness on it.
- [x] 2.3 Delete the hand-rolled `conductor.roughness` block in the
      `coatedconductor` branch; it is the chain minus the anisotropic pair, which
      is precisely why the pair was dropped.

## 3. The coatedconductor branch reads pbrt's spelling

- [x] 3.1 Both flavours resolve the metal from
      `_resolve_roughness(..., prefix="conductor.")`. The flavour gate goes.
- [x] 3.2 Delete the top-level `roughness` read for this material type. Do NOT
      add it as a fallback — pbrt renders such a scene at `conductor.roughness`'s
      default of 0, so a fallback would import an image pbrt does not render
      (design D1).
- [x] 3.3 Leave `coateddiffuse` alone. Its top-level `roughness` read matches
      `CoatedDiffuseMaterial::Create`; the two coated types are asymmetric in
      pbrt. Add a test pinning that asymmetry so a later "consistency" cleanup
      cannot quietly unify them.
- [x] 3.4 `coat_roughness` keeps its scalar `interface.roughness` read.
      `interface.uroughness`/`vroughness` stay unread — the lobe is scalar and
      has nowhere to put anisotropy. Record it as a known gap rather than
      inventing a reduction (design D3).

## 4. Gates

- [x] 4.1 The discrimination test from 1.4 passes on both flavours.
- [x] 4.2 The anisotropy test from 1.5 passes; the two adapters emit their own
      documented reductions (usd geometric mean + note, mtlx mean +
      `specular_anisotropy`).
- [x] 4.3 A hostless source gate: the roughness calibration arithmetic appears
      only in the shared resolver, not in any material branch.
- [x] 4.4 Byte-identity against the 1.1 baseline: only `all_mtypes.pbrt` moves.
      It authors `conductor.roughness` 0.02 alongside `roughness` 0.44 precisely
      to exhibit this drift. If anything else moves, stop and re-measure.
- [x] 4.5 `tests/pbrt` hostless sweep against the recorded pre-existing failures;
      `ruff check src/skinny`.
- [x] 4.6 GPU parity + suite gates for the new scene, on the developer's machine
      (Metal backend, one guarded process at a time). Record the measured
      baselines rather than loosening a tolerance.

## 5. Documentation

- [x] 5.1 Remove `KnownBugs.md` item 1 — and correct its premise in the removal
      note: it claims the top-level `roughness` "drives the coat", which is true
      of `coateddiffuse` and false of `coatedconductor`, where pbrt reads no
      top-level roughness at all.
- [x] 5.2 `docs/PbrtImport.md` records the per-type spelling table and the
      asymmetry between the two coated types.
- [x] 5.3 `openspec validate --strict`; pre-merge review.

## What the implementation changed about the plan

Two things the artifacts got wrong, both corrected above and folded back into
`proposal.md` / `design.md` / the spec delta:

1. **pbrt does not ignore a top-level `roughness` on a `coatedconductor` — it
   refuses the scene** (`Error: "roughness": unused parameter`, pinned pbrt v4).
   This strengthens D1: a fallback would import a value from a scene pbrt will
   not render at all.
2. **The suite scene points the other way round.** 1.2 asked for a near-mirror
   metal under a satin coat, which assumed the buggy path read the *coat's*
   value into the metal. It reads the top-level spelling, which the scene cannot
   carry (see 1) — so with it absent the old path fell to its default of 0. The
   committed scene is therefore a ROUGH metal (`conductor.roughness` 0.45) under
   a SHARP coat (`interface.roughness` 0.02), and the old read renders a mirror.

Also: 1.3's command line is `--scene suite:mat_coated_metal` (the `suite:` form);
a bare `--scene mat_coated_metal` looks in `tests/pbrt/corpus/`.

Two consequences worth keeping:

- Deleting the hand-rolled block took the LAST bare `ParamSet` method call out of
  `resolve_material`'s own body. `test_the_read_gate_is_sensitive` had asserted
  one was there as its negative control, so it now points at the helpers that
  still make one.
- `tests/pbrt/fixtures/all_mtypes.pbrt`'s anisotropy block used the top-level
  `uroughness`/`vroughness`, which this change stops reading. It now authors the
  `conductor.` pair with the top-level pair beside it, so the block still covers
  the anisotropic path AND pins that the top-level spelling is ignored. Both its
  `usd` and `mtlx` hashes move as a result; every other scene is byte-identical.

## Codex pre-merge review (5.3)

Five findings; four folded in, one consciously dismissed.

- **[HIGH] the calibration was not single-owner** — `interface.roughness`
  reached `coat_roughness` raw, bypassing `remaproughness`. Verified against
  `pbrt-v4 src/pbrt/materials.cpp:351`, which remaps it: **real bug**, and one
  `coateddiffuse` did not have. FIXED — `_calibrate_roughness` now owns the
  arithmetic and the coat calls it (design D3 rewritten).
- **[HIGH] a second spelling owner remained** — `references_texture` scans
  `_TEXTURABLE`, which listed only the top-level `roughness`, so a texture-bound
  `conductor.roughness` connected on a UV-less shape that never got synthesized
  default UVs. FIXED by registering the spelling. The deeper fix codex asks for
  (texture requirements flowing out of the *resolved* material instead of a
  second `ParamSet` rescan) is a real seam problem and a follow-up, not this
  change.
- **[MEDIUM] the non-vacuity assertion was detached from the scene** — it used
  inline source, so the committed scene could decay to equal roughnesses and
  still pass. FIXED: `test_the_committed_coated_metal_scene_still_discriminates`
  parses the on-disk `.pbrt` and pins both properties.
- **[MEDIUM] two distinct roughness axis textures collapse to the first** —
  true, and identical for the top-level pair, so it predates the prefixed
  spellings and affects every material equally. DISMISSED for this change and
  recorded as a known gap in `docs/PbrtImport.md`.
- **[LOW] CHANGELOG** — FIXED.

## Measured results

| Gate | Before | After |
|------|--------|-------|
| `mat_coated_metal` pbrt-truth, path\|wavefront | relMSE 0.4287 / FLIP 0.2392 | **0.03792 / 0.03630** |
| plain-vs-MaterialX equivalence | relMSE 0.08661 / FLIP 0.06609 | **0.009194 / 0.01407** |

All 19 combos pass on both flavours at the scene's 0.06/0.08 tolerance, so no
per-combo `baselines` were needed; `path|megakernel` vs the anchor is exactly 0
(mega ≡ wave). The `_mtlx` sibling sits closer to pbrt than the plain one
(0.02452 vs 0.03792).

**Residual, out of scope and recorded as `KnownBugs.md` item 4:** the coated
metal renders 1.60× brighter and less saturated than pbrt's. Localized by
measurement to the COAT: the uncoated gold spheres of `mat_conductor` measure
0.923× and 0.978× under the same lighting and the environment background matches
to 0.999×, so neither the metal response nor the lighting is at fault. skinny
adds a Fresnel coat lobe over an unattenuated base (`coatAttenuation` is the
identity, because the importer sets `coat_color = [1,1,1]`), where pbrt's
`CoatedConductorBxDF` layers — and layering makes a coated metal DARKER.

Two hypotheses died on the way and are recorded so they are not re-run: the
"zero the clearcoat" probe (it removes skinny's coat while pbrt keeps its own,
so it measures the wrong difference), and the metal's RGB response (the
named-metal F0 table reproduces the spectrally-integrated normal-incidence
reflectance to within 0.3% luminance for Au/Ag/Al/Cu). `--spectral` does not
close it either (0.04112 vs RGB 0.03792). Independent of the roughness spelling.

## Notes

Measured against the tree at `39e45fe`, not assumed:

- pbrt-v4 `CoatedConductorMaterial::Create` reads `interface.roughness` (+u/v)
  for the coat and `conductor.roughness` (+u/v) for the metal, and **no
  top-level `roughness`**. `CoatedDiffuseMaterial::Create` **does** read the
  top-level spelling.
- `coatedconductor` appears in exactly one file in the repo —
  `tests/pbrt/fixtures/all_mtypes.pbrt` — which has no test consumers.
  `mat_plastic` and `samp_many_lights` use `coateddiffuse`, which this change
  does not touch.
- The anisotropic conductor spellings are read by **neither** flavour today, so
  that half of the defect is not a divergence but a silent drop on both paths.
