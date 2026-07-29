# Change: subsurface-promoting-accessors

## Why

A pbrt `Material "subsurface"` crashes the importer when it binds a parameter to
a texture or to a named spectrum.

```
Material "subsurface" "texture reflectance" "sometex"
ValueError: could not convert string to float: 'sometex'
```

`KnownBugs.md` item 3 records this. A measured sweep over every material type,
every parameter, both binding kinds and both flavours finds **26 crashing
combinations across seven parameters** — and clears every other material type.

| Parameter | Accessor today | Flavour | In pbrt-v4 |
|-----------|----------------|---------|------------|
| `sigma_a`, `sigma_s` | `ParamSet.rgb` | both | `GetSpectrumTextureOrNull` |
| `reflectance` | `ParamSet.rgb` | both | `GetSpectrumTextureOrNull` |
| `mfp` | `ParamSet.rgb` | both | `GetSpectrumTexture` |
| `g`, `scale` | `ParamSet.floats` | both | `GetOneFloat` |
| `radius` | `ParamSet.rgb` | mtlx only | **absent — pbrt has no such parameter** |

`subsurface` is the **only** branch that reads pbrt parameters through the raw
accessors instead of the promoting ones (`get_float_texture`,
`get_spectrum_texture`). Those exist for exactly this case: pbrt `ErrorExit`s on
an unusable binding, skinny degrades and records the loss in the import report.

**The crash is the visible half. The silent half is worse.** The raw `rgb`
accessor does not classify the parameter type, so a legal pbrt binding it *can*
parse produces garbage instead of an error:

| Binding on `sigma_a` | Today | Correct |
|----------------------|-------|---------|
| `"blackbody sigma_a" [6500]` | `[6500, 6500, 6500]` | `[1.042, 0.984, 1.035]` |
| `"spectrum sigma_a" [400 .1 700 .9]` | `[400.0, 0.1, 700.0]` | `[0.869, 0.461, 0.181]` |

The second row is the raw wavelength/value tokens read as an RGB triple. No
crash, no note, `status == EXACT`. No corpus scene uses either form, so no gate
sees it.

**`reflectance` is read twice on the mtlx flavour** — once promoted, for the
`subsurface_color` lobe, and once raw, inside the coefficient chain. That is the
defect `subsurface-eta-single-owner` removed for `eta`, left on a second
parameter.

## What Changes

**The subsurface branch joins the promoting layer.** Six parameters read the way
the other ten branches already do. No binding raises.

**`get_spectrum_texture` gains the note path its float sibling already has.** It
currently substitutes silently: `"spectrum reflectance" "metal-Au-eta"` yields
gold's reflectance RGB with no note and `status == EXACT`. Without this, routing
the subsurface branch through it converts 26 loud crashes into ~20 quiet wrong
values — a worse failure mode. The substitution also becomes lane-aware, mirroring
`_IOR_PARAM_NAMES` on the float side: a named metal's reflectance is a legal
substitution on a reflectance lane and meaningless on an absorption coefficient.

**Presence stays independent of readability.** pbrt selects its precedence branch
with `GetSpectrumTextureOrNull` — on presence, not readability. A promoting
accessor returns a default rather than `None`, so presence is read from the raw
parameter and the value from the accessor.

**An unusable σ pair degrades as a unit.** pbrt `ErrorExit`s on a half-authored
pair. Substituting a default for one member only would pair one material's σ_a
with another's σ_s.

**A texture-bound coefficient is `SKIPPED`, not `APPROX`.** pbrt evaluates these
per intersection; skinny's imported medium is homogeneous. The construct is not
approximated, it is unrepresentable, and the CLI should exit non-zero.

**The phantom `radius` read is deleted.** pbrt's `SubsurfaceMaterial::Create`
never reads `radius`. Hardening it would cement a parameter pbrt ignores.

**The gate becomes structural.** An AST check over the resolver asserts that no
`float()`-on-a-token accessor (`.rgb`, `.floats`, `.float`, `.int`, `.ints`) is
used for a material parameter. A behavioural sweep on top proves the accessors
degrade *and note*.

## Impact

- Affected specs: `pbrt-material-resolution`
- Affected code: `src/skinny/pbrt/materials.py` — the `subsurface` branch,
  `subsurface_medium_overrides`, and `get_spectrum_texture` (all eleven branches
  gain its note path)
- Affected docs: `KnownBugs.md` (item 3 resolved; item 2 loses `reflectance`),
  `docs/PbrtImport.md`
- **Not crash-only.** `rgb`- and `float`-typed bindings are byte-identical.
  `blackbody` and inline-sampled-spectrum bindings change, from garbage to a
  correct reduction. No corpus or suite scene uses those forms, so the corpus
  hash gate cannot see the fix — it needs its own fixtures with recorded values.
- The `get_spectrum_texture` note path is drift-free on the corpus: the only
  named-spectrum material parameter in it is a *recognised* name
  (`all_mtypes.pbrt:83`), which stays an exact substitution and stays unnoted.

## Non-Goals

- **Honouring** a texture-bound coefficient. The imported medium is homogeneous.
  A spatially varying interior is its own change.
- The remaining frozen flavour divergences in `KnownBugs.md` item 2
  (`transmittance`, `interface.eta`). This change removes only `reflectance`,
  because fixing its crash forces the decision, and `radius`, by deletion.
- Per-material reflectance SPDs (`spectral-measured-material`).
