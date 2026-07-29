# Design: subsurface-promoting-accessors

## Context

`materials.py` has two layers for reading a pbrt parameter.

- **Raw accessors** — `ParamSet.rgb`, `.floats`, `.float`. They call `float()` on
  the token values. A texture name or a spectrum name raises. A `blackbody` or an
  inline sampled spectrum does **not** raise; it returns the raw tokens.
- **Promoting accessors** — `get_float_texture`, `get_spectrum_texture`. They
  classify the parameter type first, then substitute or degrade.

Ten of the eleven material branches use the promoting layer. `subsurface` does
not. A measured sweep (11 types × 19 parameters × 2 binding kinds × 2 flavours)
finds 26 crashing combinations, all in that branch; every other type is clean.

The promoting layer is itself uneven. `get_float_texture` notes every
degradation. `get_spectrum_texture` notes **none**:

| Binding on a spectrum lane | Value | Note | Status |
|---|---|---|---|
| texture, unresolvable | default | yes | APPROX |
| texture, resolvable but lobe dropped | default | **none** | EXACT |
| `"spectrum x" "glass-BK7"` | default | **none** | EXACT |
| `"spectrum x" "metal-Au-eta"` | **gold's reflectance RGB** | **none** | EXACT |

## Goals

- No pbrt parameter binding raises, on any material type.
- No pbrt parameter binding degrades silently.
- The precedence branch a material selects does not change.
- One resolution per parameter, feeding every consumer.

## Decision

### D1 — The subsurface branch joins the promoting layer

`sigma_a`, `sigma_s`, `reflectance` and `mfp` read through
`get_spectrum_texture`; `g` and `scale` through `get_float_texture`. This is the
layer the other ten branches already use.

**Discarded alternative:** guard each raw read with a type check, as the old
`_subsurface_overrides` did for `eta`. That is six hand-rolled guards, each of
which must stay in step with the promoting accessors' own rules. The previous
change deleted exactly one such guard.

### D2 — `get_spectrum_texture` gains a note path, and the substitution is lane-aware

D1 is **not safe without this**. Routing six parameters into an accessor that
degrades silently would convert 26 loud crashes into ~20 quiet wrong values.

Two rules, both mirroring what `get_float_texture` already does:

- **Note every degradation.** An unrecognised name, a spectrum file reference,
  and a texture whose value is dropped each append one note and escalate to
  APPROX. A *recognised* name stays an exact substitution and stays unnoted.
- **Substitute only where the substitution means something.**
  `_IOR_PARAM_NAMES` restricts the float side so a glass IOR cannot land in a
  roughness lane. The spectrum side needs the mirror: a named metal's reflectance
  RGB is legal on a reflectance-like lane and meaningless on an absorption or
  scattering coefficient, where it must degrade with a note instead.

This is one owner and it repairs all eleven branches, not the six being moved.

**Drift:** none on the corpus. Its only named-spectrum material parameter is
`all_mtypes.pbrt:83` `"spectrum reflectance" "metal-Au-eta"` — recognised, on a
reflectance lane, so unnoted before and after.

### D3 — Presence is read from the parameter, value from the accessor

`subsurface_coefficients` picks its branch on presence:

```python
if name:                                           # (1) named preset
elif sigma_a is not None and sigma_s is not None:  # (2) explicit sigma
elif reflectance is not None:                      # (3) Jensen inversion
else:                                              # (4) Wholemilk defaults
```

A promoting accessor returns the caller's default, never `None`. Feeding its
result straight in would make `sigma_a` and `sigma_s` *always* non-`None`, so
**every** subsurface material would take branch 2 and the other three would
become unreachable.

So two questions get two reads: **was it authored?** — `p.get("sigma_a") is not
None`, a syntactic fact; **what is its value?** — the promoting accessor.

This matches pbrt exactly: `SubsurfaceMaterial::Create` branches on
`GetSpectrumTextureOrNull`, which returns non-null for a texture binding too.
Treating an unreadable parameter as absent would diverge from pbrt and silently
swap the physical model.

### D4 — An unusable σ pair degrades as a unit

pbrt refuses a half-authored pair outright:

```
ErrorExit(loc, "Provided \"sigma_a\" parameter without \"sigma_s\".");
```

Under D3 alone, a texture-bound `sigma_a` with a numeric `sigma_s` pairs
Wholemilk's σ_a with the author's σ_s. Those come from different materials. With
a dense authored `sigma_s`, the single-scattering albedo approaches 1 and the
mean free path collapses; the interior walk saturates `VOLUME_MAX_SCATTERS` and
returns a dark blob, slowly.

So: if **either** member is unusable, both degrade to the default pair, with one
note that names the unusable member and states the pair was replaced together.
Branch selection still follows presence.

Note that degrading both members is numerically identical to branch 4. The
distinction between "keep branch 2" and "fall through" only exists in the mixed
case — which is the case this rule governs.

**Discarded alternative:** a sentinel type carrying "present but unusable" into
`subsurface_coefficients`. That pushes a third state into a pure function that
should keep taking values; the resolver already knows which reads degraded.

### D5 — A texture-bound coefficient is SKIPPED; a spectral reduction is APPROX

Two different losses are being conflated.

- **Named or inline spectrum → RGB** is a bounded fidelity loss, the same one
  every other branch takes. APPROX with a note.
- **A texture-bound coefficient** is not an approximation. pbrt evaluates it per
  intersection; skinny's imported medium is homogeneous, so the authored data has
  nowhere to go and the substituted default bears no relation to it.

`SKIPPED` is the existing vocabulary for an unrepresentable construct, and it is
cheap: a material-level SKIPPED only reaches `report.add`, so the material is
still authored and nothing renders differently, while `report.has_unsupported()`
makes the CLI exit non-zero. No corpus scene regresses — they all crash today.

### D6 — The phantom `radius` read is deleted, not hardened

pbrt's `SubsurfaceMaterial::Create` reads `name`, `sigma_a`, `sigma_s`,
`reflectance`, `mfp`, `g`, `eta`, `scale`, the roughnesses, `displacement` and
`remaproughness`. There is no `radius`; pbrt's only `radius` parameters are on
shapes. skinny reads one at materials.py:607 and feeds the mtlx
`subsurface_radius` lobe from it.

Hardening it would cement a read pbrt ignores and keep a flavour gate alive for
it. Deleting it takes the surface from seven parameters to six and removes an
entry from `KnownBugs.md` item 2 for free.

**Measured cost:** no corpus or suite scene authors a subsurface `radius`. The
one hit in the corpus is `subsurface_infinite.pbrt:19`, a `Shape "sphere" "float
radius"` — a different parameter. Only `all_mtypes.pbrt` authors it, and that
fixture has no test consumers.

`subsurface_radius` is the physical mean free path, which the resolved `mfp`
already carries; derive the lobe from it rather than dropping the lobe.

### D7 — The gate is structural, not a parameter list

"Every parameter a type reads" has no machine-readable source. Any behavioural
sweep is a hand-written list of names, so the seventh parameter added with a raw
accessor **passes** it. That is the failure mode the gate exists to prevent.

So the load-bearing gate is an AST check, reusing the `_reads_in` machinery
already in `tests/pbrt/test_material_resolve.py`: inside `resolve_material` and
`subsurface_medium_overrides`, the only `ParamSet` methods called may be the
non-raising ones (`get`, `bool`, `string`, `__contains__`). The `float()`-calling
accessors — `.rgb`, `.floats`, `.float`, `.int`, `.ints` — may not appear. That
check needs no parameter names and cannot rot.

A behavioural sweep sits on top to prove the accessors degrade *and note*, which
the AST check cannot see.

### D8 — `reflectance` is resolved once, and the default conflict is decided

`reflectance` has two readers today: `get_spectrum_texture` for the
`subsurface_color` lobe, inside the mtlx flavour gate, and `ParamSet.rgb` for the
coefficient chain, outside it. One resolution, outside the gate, feeds both.

The two readers carry **different defaults** — the lobe uses `[1.0, 1.0, 1.0]`
(materials.py:604), the chain uses `(0.5, 0.5, 0.5)` (subsurface.py:144). One
resolution can hold only one. The default is reachable only when the parameter is
present but unusable, which crashes today, so **no test can tell which is
picked** — it has to be decided here.

Take `(0.5, 0.5, 0.5)`: it is pbrt's meaningful mid-albedo for the inversion the
value feeds, and a mid-grey preview closure is sane where an unusable binding
degraded. The lobe's `[1.0, 1.0, 1.0]` stays the default for an **absent**
`reflectance`, which is a different question and keeps its bytes.

This changes what the flavour gate claims. Its stated effect is "no value, no
note, no EXACT→APPROX escalation" on the USD path. For `reflectance` that was
already half false: the USD path *does* read it, for the chain — the gate
suppressed the note, not the read. After this change the USD path reports a
degrading `reflectance` as the mtlx path does. No working scene changes, because
every binding that would newly produce a note raises today.

## Risks

- **Note order is unguarded on this path.** `test_notes_are_in_read_order` pins
  one `conductor` case and one `coatedconductor` case. **Nothing pins the
  subsurface path on either flavour**, so a reordering lands green. Moving
  `reflectance` out of the flavour gate changes when its note is appended. Pin
  the order as part of this change rather than relying on a net that is not there.
- **The corpus gate is blind to the fix.** No corpus or suite scene uses a
  `blackbody` or inline-sampled-spectrum binding on a medium parameter, so the
  hash diff cannot show the garbage-to-correct change. Those forms need explicit
  fixtures with recorded values.
- **`_IOR_PARAM_NAMES` commentary.** Its comment states that every other float
  parameter through `get_float_texture` is a roughness. Adding `g` and `scale`
  makes that false.
- **`g`/`scale` are `GetOneFloat` in pbrt**, which would `ErrorExit` on a texture.
  Accepting one is deliberately more permissive than pbrt, per the best-effort
  charter; the note should say the parameter is not texture-able in pbrt.
- **`SpectrumType::Unbounded`.** pbrt types these coefficients unbounded, while
  `get_spectrum_texture` distinguishes only illuminant from albedo. Confirm the
  albedo-shaped reduction is acceptable for an extinction coefficient, or record
  the limit.
