# Design: coatedconductor-roughness-spelling

## Context

pbrt-v4 spells a coated material's roughnesses differently per material type.

```
CoatedDiffuseMaterial::Create      roughness / uroughness / vroughness   -> coat
CoatedConductorMaterial::Create    interface.roughness (+u/v)            -> coat
                                   conductor.roughness (+u/v)            -> metal
                                   (no top-level roughness at all)
```

skinny's resolver mirrors the first correctly and the second three ways wrong:

| Lobe | pbrt-v4 | `usd` flavour | `mtlx` flavour |
|------|---------|---------------|----------------|
| metal roughness | `conductor.roughness` | top-level `roughness` | `conductor.roughness` |
| metal anisotropy | `conductor.uroughness`/`vroughness` | not read | not read |
| coat roughness | `interface.roughness` | `interface.roughness` | `interface.roughness` |

The metal read is behind a flavour gate. The gate's comment says the top-level
`roughness` "drives the coat", which is true of `coateddiffuse` and false here.

The mtlx side also hand-rolls its read rather than using the shared chain:

```python
rv = get_float_texture(p, "conductor.roughness", 0.0, ...)
if rv.is_tex:
    lobes["roughness"] = ResolvedRoughness(ParamValue(0.5, rv.tex))
    notes.append("roughness texture connected; perceptual remap not applied ...")
else:
    remap = p.bool("remaproughness", True)
    lobes["roughness"] = ResolvedRoughness(
        ParamValue(alpha_to_usd_roughness(pbrt_roughness_to_alpha(rv.const, remap))))
```

That is `_resolve_roughness`'s body, minus the anisotropic pair — which is
exactly why the anisotropic spellings are the ones that got dropped.

## Goals

- One parameter spelling per lobe, identical on both flavours.
- No read of a parameter pbrt does not define for the material type.
- The metal roughness uses the same calibration chain as every other roughness.
- A gate that fails if the spelling regresses.

## Decision

### D1 — Both flavours read `conductor.roughness`; the top-level read is deleted

The flavour gate exists to freeze a divergence, not to preserve a behaviour worth
keeping. Both sides converge on pbrt's spelling.

Deleting the top-level read is the second half and the less obvious one. A
parameter pbrt ignores has no correct value to import, so reading it cannot be
"the usd approximation" — it is a different material. This is the same call
`subsurface-promoting-accessors` made for `radius`, and the rule the spec already
carries: *the resolver SHALL NOT read a parameter pbrt itself does not define for
that material type.*

**Discarded alternative:** keep reading the top-level `roughness` as a fallback
when `conductor.roughness` is absent. It looks forgiving and is not: pbrt renders
such a scene with the metal at `conductor.roughness`'s default of 0, so a
fallback would import a *different image* than pbrt renders, silently, for a
scene pbrt itself accepts.

### D2 — `_resolve_roughness` takes a prefix, and the branch stops hand-rolling

The chain is `roughness`/`uroughness`/`vroughness` → texture promotion →
`remaproughness` → alpha → unreduced `ResolvedRoughness`. `conductor.` needs the
identical chain over prefixed names.

```python
_resolve_roughness(params, notes, prefix="conductor.", ...)
```

One owner, so the anisotropic pair cannot be dropped for one caller and honoured
for another, and the "texture connected; perceptual remap not applied" note is
worded once.

`remaproughness` stays **unprefixed**: pbrt reads one per material, not one per
layer (`GetOneBool("remaproughness", true)`), and it governs both roughnesses.

**Discarded alternative:** a second copy of the chain for the conductor lane.
That is the arrangement that produced this bug — the copy was written without the
anisotropic pair and drifted immediately.

### D3 — The coat keeps `interface.roughness`, unprefixed by this change

Both flavours already read `interface.roughness` for `coat_roughness`, which
matches pbrt. It stays a scalar read here: routing it through the prefixed chain
would also pick up `interface.uroughness`/`vroughness`, and `coat_roughness` is a
scalar lobe with nowhere to put anisotropy. Recording that as a known gap is
honest; inventing a reduction for it is not.

### D4 — The material gets a gate before it gets a fix

`coatedconductor` appears in one fixture with no test consumers. Nothing in the
corpus or the confirming suite renders it, so today both the divergence and the
dropped anisotropy are invisible.

A confirming-suite scene must **discriminate** the defect, not merely exercise
the material: coat and metal roughness have to differ enough that reading the
wrong one changes the image. A near-mirror metal (`conductor.roughness` 0.02)
under a satin coat (`interface.roughness` 0.3) does that — reading 0.3 into the
metal blurs the reflection visibly.

The scene needs a pbrt-truth EXR from the pinned pbrt v4 build, plus the plain
and `-mtlx` authoring-equivalence pair the suite's gate classes expect. Both
flavours reading one spelling is what makes that equivalence gate meaningful for
this material — today it would pass vacuously, because no scene reaches it.

## Risks

- **The suite scene is the load-bearing artifact.** If it is authored with equal
  coat and metal roughness, every gate passes whichever parameter is read, and
  the change ships unprotected. Assert the discrimination directly: the imported
  metal roughness must differ from the coat's.
- **Anisotropy reduction differs per target.** UsdPreviewSurface collapses the
  pair to a geometric mean and notes it; standard_surface keeps a mean plus
  `specular_anisotropy`. Reading the pair for the first time on this material
  means both adapters start emitting their reduction for it, so the `.usda` and
  `.mtlx` will differ in the expected, already-specified way — not a new
  divergence, but it will show up in the fixture diff.
- **`all_mtypes.pbrt` moves.** It authors `conductor.roughness` 0.02 alongside
  `roughness` 0.44 specifically to exhibit this drift, so its output must change.
  Confirm no other scene does before accepting the diff.
- **Regen needs the pinned pbrt binary and a GPU.** `regen_refs.py` shells out to
  `~/projects/pbrt-v4/build/pbrt`, and the parity gate renders. Those steps run on
  the developer's machine, not in a hostless sweep.
