# Tasks: coatedconductor-roughness-spelling

Group 1 comes first for a reason: the material is ungated today, so a fix landed
before the scene exists is a fix nothing protects.

## 1. Gate the material before fixing it

- [ ] 1.1 Capture the byte-identity baseline: import every parity-corpus and
      confirming-suite scene on both flavours, hash the `.usda` / `.mtlx`.
      Run it in ONE worktree — the emitted `.usda` embeds absolute asset paths.
- [ ] 1.2 Add `tests/assets/suite/mat_coated_metal/` via
      `tests/assets/suite/_gen/build.py`: a `coatedconductor` with a near-mirror
      metal (`conductor.roughness` 0.02) under a satin coat
      (`interface.roughness` 0.3), so reading the wrong parameter visibly blurs
      the reflection. Register it in `tests/pbrt/corpus/manifest.json` with a
      `_mtlx` sibling carrying an `equivalence` disposition.
- [ ] 1.3 Generate its pbrt-truth EXR:
      `python tests/pbrt/regen_refs.py --scene mat_coated_metal --res 256 --spp <n>`
      (needs the pinned pbrt at `~/projects/pbrt-v4/build/pbrt`).
- [ ] 1.4 Write the FAILING discrimination test: under both flavours the resolved
      metal roughness comes from `conductor.roughness`, and it is NOT equal to
      the coat's. Expect the `usd` flavour to fail today.
- [ ] 1.5 Write the FAILING anisotropy test: `conductor.uroughness` /
      `conductor.vroughness` with different values resolve to an unreduced pair.
      Expect both flavours to fail today.

## 2. One calibration chain, parameterised by spelling

- [ ] 2.1 `_resolve_roughness` takes a `prefix`, defaulting to `""`. It reads
      `{prefix}roughness` / `{prefix}uroughness` / `{prefix}vroughness`.
- [ ] 2.2 `remaproughness` stays UNPREFIXED — pbrt reads one per material
      (`GetOneBool`), governing every roughness on it.
- [ ] 2.3 Delete the hand-rolled `conductor.roughness` block in the
      `coatedconductor` branch; it is the chain minus the anisotropic pair, which
      is precisely why the pair was dropped.

## 3. The coatedconductor branch reads pbrt's spelling

- [ ] 3.1 Both flavours resolve the metal from
      `_resolve_roughness(..., prefix="conductor.")`. The flavour gate goes.
- [ ] 3.2 Delete the top-level `roughness` read for this material type. Do NOT
      add it as a fallback — pbrt renders such a scene at `conductor.roughness`'s
      default of 0, so a fallback would import an image pbrt does not render
      (design D1).
- [ ] 3.3 Leave `coateddiffuse` alone. Its top-level `roughness` read matches
      `CoatedDiffuseMaterial::Create`; the two coated types are asymmetric in
      pbrt. Add a test pinning that asymmetry so a later "consistency" cleanup
      cannot quietly unify them.
- [ ] 3.4 `coat_roughness` keeps its scalar `interface.roughness` read.
      `interface.uroughness`/`vroughness` stay unread — the lobe is scalar and
      has nowhere to put anisotropy. Record it as a known gap rather than
      inventing a reduction (design D3).

## 4. Gates

- [ ] 4.1 The discrimination test from 1.4 passes on both flavours.
- [ ] 4.2 The anisotropy test from 1.5 passes; the two adapters emit their own
      documented reductions (usd geometric mean + note, mtlx mean +
      `specular_anisotropy`).
- [ ] 4.3 A hostless source gate: the roughness calibration arithmetic appears
      only in the shared resolver, not in any material branch.
- [ ] 4.4 Byte-identity against the 1.1 baseline: only `all_mtypes.pbrt` moves.
      It authors `conductor.roughness` 0.02 alongside `roughness` 0.44 precisely
      to exhibit this drift. If anything else moves, stop and re-measure.
- [ ] 4.5 `tests/pbrt` hostless sweep against the recorded pre-existing failures;
      `ruff check src/skinny`.
- [ ] 4.6 GPU parity + suite gates for the new scene, on the developer's machine
      (Metal backend, one guarded process at a time). Record the measured
      baselines rather than loosening a tolerance.

## 5. Documentation

- [ ] 5.1 Remove `KnownBugs.md` item 1 — and correct its premise in the removal
      note: it claims the top-level `roughness` "drives the coat", which is true
      of `coateddiffuse` and false of `coatedconductor`, where pbrt reads no
      top-level roughness at all.
- [ ] 5.2 `docs/PbrtImport.md` records the per-type spelling table and the
      asymmetry between the two coated types.
- [ ] 5.3 `openspec validate --strict`; pre-merge review.

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
