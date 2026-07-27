# Tasks: flat-material-field-table

## 1. Baseline

- [ ] 1.1 Capture, from the current packers, a permanent golden of name → byte
      offset for every field of `FlatMaterialParams` and `StdSurfaceParams`,
      scalar and MSL. Source of names: the docstring map at
      `renderer.py:627-672` plus the argument list.
- [ ] 1.2 Capture packed bytes for a spread of materials (pbrt corpus +
      confirming suite) as the byte-identity target.
- [ ] 1.3 Enumerate the 31 override key strings the packer reads and every
      site that authors one.

## 2. Field table

- [ ] 2.1 Add the table to `slang_layout.py`: row offsets derived, lane
      assignment declared, pinned against 1.1.
- [ ] 2.2 Transposition gate test: name → offset golden, with a negative
      control that swaps two same-typed fields and must fail.

## 3. Name-keyed packing

- [ ] 3.1 Rewrite `pack_flat_material` and `pack_std_surface_params` to take a
      mapping; MSL variants consume the same table.
- [ ] 3.2 Byte-identity test against 1.2.
- [ ] 3.3 **Report-only** unknown-key survey over the pbrt corpus and the
      confirming suite. List every key not in the table.
- [ ] 3.4 Resolve each finding from 3.3 (fix author, or add key), then enable
      hard rejection.

## 4. Collapse the alias tables

- [ ] 4.1 Verify each of the 7 entries on which `_STD_SURFACE_TO_FLAT` (5) and
      `_STD_SURFACE_TO_FLAT_PACK` (12) disagree; record the verdict per entry.
- [ ] 4.2 Make all three tables projections of the field table.
- [ ] 4.3 Turn the two "keep in sync" comments into assertions.

## 5. Merge ordering

- [ ] 5.1 Order the `skinnyOverrides` merge before the derivations; remove the
      re-run of `_derive_opacity_from_subsurface` at `usd_loader.py:1246-1253`.
- [ ] 5.2 Test: each derivation runs once, results identical to the current
      double-derivation output.

## 6. Gates

- [ ] 6.1 `ruff check src/`; full hostless `pytest`.
- [ ] 6.2 GPU: confirming-scene suite authoring-equivalence gate (plain USD ≡
      MaterialX) green — this is the gate that would catch a vocabulary slip.
- [ ] 6.3 Parity matrix dual gate unchanged.
- [ ] 6.4 Docs: `docs/Architecture.md` byte-layout section, `docs/PbrtImport.md`
      override key vocabulary.
- [ ] 6.5 `openspec validate flat-material-field-table --strict`.

## Note

Tasks 2.2 and 3.2 are only enforced on Vulkan-capable hosts until
`renderer-pure-core-extraction` lands, because these packers sit above
`import vulkan` in `renderer.py`. Prefer landing that change first.
