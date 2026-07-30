# Change: coatedconductor-roughness-spelling

## Why

`KnownBugs.md` item 1 records that the two export flavours read a different
parameter for a `coatedconductor`'s base-metal roughness, so the same pbrt scene
renders a differently-rough metal through each path.

The item states the defect but gets its premise wrong. It says the top-level
`roughness` "drives the *coat*". It does not. pbrt-v4's
`CoatedConductorMaterial::Create` reads **no top-level `roughness` at all**:

| Lobe | pbrt-v4 reads |
|------|---------------|
| coat | `interface.roughness`, `interface.uroughness`, `interface.vroughness` |
| metal | `conductor.roughness`, `conductor.uroughness`, `conductor.vroughness` |

A top-level `roughness` on a `coatedconductor` is a parameter pbrt does not
merely ignore — it **refuses the scene**. Measured against the pinned pbrt v4:

```
Error: mat_coated_metal.pbrt:16:173: "roughness": unused parameter.
```

So the UsdPreviewSurface path is not reading the *coat's* roughness into the
metal. It is reading a parameter that no valid pbrt scene carries, which is a
stronger form of the phantom read that `subsurface-promoting-accessors` deleted
for `radius`: `radius` at least appeared on pbrt shapes, whereas this one makes
the scene unrenderable.

The two coated types are **asymmetric in pbrt**, which is why this is easy to get
wrong: `CoatedDiffuseMaterial::Create` *does* read top-level `roughness` for its
coat. skinny's `coateddiffuse` handling is therefore correct and must not change.

Measured against the current tree, the branch has three defects, not one:

| Lobe | pbrt-v4 | skinny `usd` | skinny `mtlx` |
|------|---------|--------------|---------------|
| metal roughness | `conductor.roughness` | **top-level `roughness`** | `conductor.roughness` |
| metal anisotropy | `conductor.uroughness`/`vroughness` | **not read** | **not read** |
| coat roughness | `interface.roughness` | `interface.roughness` | `interface.roughness` |

The anisotropic spellings are dropped silently on **both** flavours, so an
anisotropic coated metal imports isotropic with no note.

**Nothing gates any of this.** `coatedconductor` appears in exactly one file —
`tests/pbrt/fixtures/all_mtypes.pbrt` — which has no test consumers. No corpus
scene and no confirming-suite scene uses the material, so both the flavour
divergence and the dropped anisotropy are invisible to every gate in the repo.

## What Changes

**Both flavours read `conductor.roughness` for the metal.** The flavour gate goes
away; the divergence with it.

**The top-level `roughness` read is deleted for this material type.** pbrt ignores
it, so reading it is skinny inventing behaviour.

**The metal's anisotropic spellings are read.** `conductor.uroughness` /
`conductor.vroughness` flow through the same unreduced `ResolvedRoughness` path
the top-level pair already uses, so each adapter applies its own reduction policy.

**One calibration chain, parameterised by prefix.** `_resolve_roughness` gains a
prefix so `conductor.` and the top-level spelling share it, instead of the branch
hand-rolling a scalar read with its own texture handling and its own remap call.

**A confirming-suite scene gates it.** The material is currently ungated, so the
fix ships with a scene that discriminates it: a coated metal whose coat and metal
roughnesses differ, so reading the wrong one is visible.

## Impact

- Affected specs: `pbrt-material-resolution`
- Affected code: `src/skinny/pbrt/materials.py` (the `coatedconductor` branch and
  `_resolve_roughness`)
- Affected docs: `KnownBugs.md` (item 1 resolved, and its premise corrected),
  `docs/PbrtImport.md`
- Affected assets: a new `tests/assets/suite/mat_coated_metal/` scene plus its
  pbrt-truth EXR, and `tests/pbrt/corpus/manifest.json`
- `tests/pbrt/fixtures/all_mtypes.pbrt` output changes — it authors both
  `conductor.roughness` 0.02 and `roughness` 0.44 precisely to exhibit this drift.
  It has no test consumers.
- **No other scene moves**: no corpus or suite scene uses `coatedconductor`. The
  `coateddiffuse` scenes (`mat_plastic`, `samp_many_lights`) are untouched,
  because their top-level `roughness` read is correct.

## Non-Goals

- `interface.eta` on the coated types, and `diffusetransmission`'s
  `transmittance`. Those are `KnownBugs.md` item 2 — a different defect (a
  one-sided *read*, not a wrong parameter) spanning a different set of materials.
  Fixing half of it here would leave the item true but differently scoped.
- pbrt's layered-BxDF controls `thickness`, `albedo`, `g`, `maxdepth` and
  `nsamples`, which skinny reads on neither coated type. That is a fidelity gap
  in the coat model, not a spelling defect.
