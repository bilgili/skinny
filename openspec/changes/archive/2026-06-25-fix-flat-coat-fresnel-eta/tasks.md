## 1. Reproduce & confirm root cause

- [x] 1.1 Render `assets/dragon_removed.usda` on Metal/Path; confirm dark region
  (coat on, mean ≈ 102) vs uniform (coat off, mean ≈ 241).
- [x] 1.2 Trace `pCoat = coat·fresnelDielectric(NdotV, coatIOR)`; confirm the eta
  convention (`sinT2 = eta²·sin²θ` ⇒ `eta = n_i/n_t`) and that passing `coatIOR`
  raw gives spurious TIR past ~42°.
- [x] 1.3 Cross-check the correct convention against the glass-refraction
  (`entering ? 1/ior`) and subsurface (`1.0/ior` "entering") sites.

## 2. Fix

- [x] 2.1 `flat_material.slang` `sample()`: `coatIOR` → `1.0 / m.coatIOR`.
- [x] 2.2 `flat_material.slang` `evaluate()`: same.
- [x] 2.3 `flat_lobes.slang` `flatBsdfResponse()`: same (keeps sample/evaluate pdf
  consistent for MIS / NEE / BDPT / ReSTIR).

## 3. Shaders

- [x] 3.1 No committed `.spv` to regen — Vulkan auto-compiles `.slang`→`.spv` at
  runtime, content-hashed in `<build>/spv_cache` (`vk_compute._compile_slang`),
  so the source change is picked up on next run. Metal compiles in-process.
- [x] 3.2 Change is response/selection-only (coat-guarded `m.coat > 0`); no
  struct/layout/stride change, so packing and the non-coat path are byte-stable.

## 4. Verify

- [x] 4.1 Re-rendered `dragon_removed.usda` on Metal/Path: coated mean
  **239.25** vs uncoated **240.60** (was **101.95** broken) — dark region gone.
- [x] 4.2 Added `tests/test_flat_coat_fresnel.py` (Vulkan harness). Fails on the
  reintroduced bug (oblique coated/base ratio 0.044) and passes with the fix
  (3/3). Pins the entering-eta convention.
- [x] 4.3 Existing flat-lobe gates green (`test_flat_rich_inputs.py` +
  `test_lobe_samplers.py`, 22 passed) — non-coated materials unaffected.

## 5. Docs

- [x] 5.1 `CHANGELOG.md` entry (Unreleased → Fixed).
- [x] 5.2 Noted the coat Fresnel entering-eta convention in the flat-material
  section of `docs/Architecture.md`, cross-linking the glass/subsurface convention.

## 6. Validate

- [x] 6.1 `openspec validate fix-flat-coat-fresnel-eta --strict`.
